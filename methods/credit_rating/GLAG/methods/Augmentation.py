import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from tqdm import tqdm
import numpy as np


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()

def info_nce_loss(z1, z2, temperature=0.5):

    if z1.size(0) == 0: return torch.tensor(0.0, device=z1.device, requires_grad=True)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # sim matrix: [M, M], M is number of valid nodes
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    
    target = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(sim_matrix, target)
    return loss


class GAT(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, nhead, nhead_out, 
                 activation, feat_drop, attn_drop, negative_slope, residual, norm, concat_out=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out
        self.gat_layers.append(GATConv(in_dim, num_hidden, heads=nhead, dropout=attn_drop, negative_slope=negative_slope, concat=True))
        for l in range(1, num_layers - 1):
            self.gat_layers.append(GATConv(num_hidden * nhead, num_hidden, heads=nhead, dropout=attn_drop, negative_slope=negative_slope, concat=True))
        self.gat_layers.append(GATConv(num_hidden * nhead, out_dim, heads=nhead_out, dropout=attn_drop, negative_slope=negative_slope, concat=concat_out))
        self.feat_drop = nn.Dropout(feat_drop)
        if norm == "layernorm":
            self.norms = nn.ModuleList([nn.LayerNorm(num_hidden * nhead) for _ in range(num_layers - 1)])
            self.norms.append(nn.LayerNorm(out_dim * nhead_out if concat_out else out_dim))
        else:
            self.norms = None

    def forward(self, x, edge_index):
        h = x
        for l in range(self.num_layers):
            h = self.feat_drop(h)
            h = self.gat_layers[l](h, edge_index)
            if self.norms is not None: h = self.norms[l](h)
            if self.activation is not None and l != self.num_layers - 1: h = self.activation(h)
        return h


class AugmentationEncoder(nn.Module):
    def __init__(self, in_dim, supp_in_dim, num_hidden, num_layers, nhead, activation, 
                 feat_drop, attn_drop, negative_slope, residual, norm,
                 mask_rate=0.5, remask_rate=0.5, num_remasking=3, 
                 alpha_l=3, lam=1.0,temperature=0.5):
        super(AugmentationEncoder, self).__init__()
        self._mask_rate = mask_rate
        self._remask_rate = remask_rate
        self._num_remasking = num_remasking
        self._alpha_l = alpha_l
        self._lam = lam
        self._temperature = temperature
        

        enc_num_hidden = num_hidden // nhead
        
        # 1. Encoder for Original Features
        # Input: in_dim (e.g. 1433) -> Output: num_hidden
        self.encoder = GAT(
            in_dim, enc_num_hidden, enc_num_hidden, num_layers, nhead, nhead,
            activation, feat_drop, attn_drop, negative_slope, residual, norm, concat_out=True
        )
        
        # 2. Encoder for Supplemented Features

        self.encoder_supp = GAT(
            supp_in_dim, enc_num_hidden, enc_num_hidden, num_layers, nhead, nhead,
            activation, feat_drop, attn_drop, negative_slope, residual, norm, concat_out=True
        )
        
        # 3. Decoder
        # Input: num_hidden -> Output: supp_in_dim 
        self.decoder = GAT(
            num_hidden, num_hidden, supp_in_dim, 2, nhead, 1,
            activation, feat_drop, attn_drop, negative_slope, residual, norm, concat_out=True
        )
        
        # 4. Projector for Contrastive
        self.projector = nn.Sequential(nn.Linear(num_hidden, 256), nn.PReLU(), nn.Linear(256, num_hidden))
        
        # Tokens
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, num_hidden))
        self.encoder_to_decoder = nn.Linear(num_hidden, num_hidden, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    def encoding_mask_noise(self, x, mask_rate):
        num_nodes = x.size(0)
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        
        out_x = x.clone()
        out_x[mask_nodes] = 0.0
        out_x[mask_nodes] += self.enc_mask_token
        return out_x, mask_nodes

    def random_remask(self, rep, remask_rate):
        num_nodes = rep.size(0)
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[:num_remask_nodes]
        
        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token
        return rep

    def forward(self, x_orig, x_supp, edge_index, supp_avail_mask):
        # -------------------------------------------------------
        # Branch A: Original (Masked) -> Encoder_Orig
        # -------------------------------------------------------
        x_orig_masked, mask_nodes = self.encoding_mask_noise(x_orig, self._mask_rate)
        h_orig = self.encoder(x_orig_masked, edge_index) # [N, num_hidden]

        # -------------------------------------------------------
        # Branch B: Supplemented (Clean) -> Encoder_Supp
        # -------------------------------------------------------

        h_supp = self.encoder_supp(x_supp, edge_index)   # [N, num_hidden]

        # -------------------------------------------------------
        # Loss 1: Contrastive
        # -------------------------------------------------------
        z_orig = self.projector(h_orig)
        z_supp = self.projector(h_supp)
        
        z_orig_valid = z_orig[supp_avail_mask]
        z_supp_valid = z_supp[supp_avail_mask]
        
        loss_contrast = info_nce_loss(z_orig_valid, z_supp_valid,self._temperature)

        # -------------------------------------------------------
        # Loss 2: Reconstruction (Target: x_supp)
        # -------------------------------------------------------
        origin_rep = self.encoder_to_decoder(h_orig)
        loss_rec_all = 0
        
        is_masked = torch.zeros(x_orig.size(0), dtype=torch.bool, device=x_orig.device)
        is_masked[mask_nodes] = True
        target_nodes_mask = is_masked & supp_avail_mask
        
        if target_nodes_mask.sum() > 0:
            for i in range(self._num_remasking):
                rep = origin_rep.clone()
                rep = self.random_remask(rep, self._remask_rate)
                recon = self.decoder(rep, edge_index) # Output dim: [N, supp_in_dim]
                
     
                x_rec = recon[target_nodes_mask]
                x_target = x_supp[target_nodes_mask]
                
                loss_rec_all += sce_loss(x_rec, x_target, self._alpha_l)
        else:
             loss_rec_all = torch.tensor(0.0, device=x_orig.device, requires_grad=True)

        loss = loss_rec_all + self._lam * loss_contrast
        return loss, loss_rec_all, loss_contrast

    def embed(self, x, edge_index):
        return self.encoder(x, edge_index)

    @torch.no_grad()
    def generate_virtual_supp(self, x_orig, edge_index):
 
        self.eval() 
   
        h_orig = self.encoder(x_orig, edge_index)
        
        rep = self.encoder_to_decoder(h_orig)
        
      
        x_supp_generated = self.decoder(rep, edge_index)
        
        return x_supp_generated

def linear_probing(model, x, edge_index, y, mask_dict, device):
    model.eval()
    with torch.no_grad():
        emb = model.embed(x.to(device), edge_index.to(device))
    
    in_feat = emb.shape[1]
    num_classes = y.max().item() + 1
    classifier = nn.Linear(in_feat, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    test_acc = 0
    
    for epoch in range(100): 
        classifier.train()
        out = classifier(emb)
        loss = criterion(out[mask_dict['train']], y[mask_dict['train']])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            classifier.eval()
            pred = classifier(emb).argmax(dim=1)
            val_acc = (pred[mask_dict['val']] == y[mask_dict['val']]).sum().item() / mask_dict['val'].sum().item()
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                test_acc = (pred[mask_dict['test']] == y[mask_dict['test']]).sum().item() / mask_dict['test'].sum().item()
    return test_acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    path = "citation_benchmark/Planetoid"
    dataset = Planetoid(path, "Cora", transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    

    supp_dim = 512
    num_nodes = data.x.size(0)
    
    x_supp_full = torch.randn(num_nodes, supp_dim, device=device) 
    
    supp_avail_mask = torch.rand(num_nodes, device=device) > 0.5
    
   
    x_supp = torch.zeros_like(x_supp_full)
    x_supp[supp_avail_mask] = x_supp_full[supp_avail_mask]
    
    print(f"Original Dim: {data.x.shape[1]}, Supplemented Dim: {x_supp.shape[1]}")
    print(f"Nodes with Supplemented Features: {supp_avail_mask.sum().item()} / {num_nodes}")
    
 
    model = AugmentationEncoder(
        in_dim=dataset.num_features,   # 1433
        supp_in_dim=supp_dim,          # 512
        num_hidden=512,                # Latent Space Dim
        num_layers=2,
        nhead=4,
        activation=nn.PReLU(),
        feat_drop=0.2,
        attn_drop=0.1,
        negative_slope=0.2,
        residual=False,
        norm="layernorm",
        mask_rate=0.5,
        lam=1.0
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)
    
    print("Start Training...")
    model.train()
    for epoch in tqdm(range(100)):
        loss, l_rec, l_con = model(data.x, x_supp, data.edge_index, supp_avail_mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Rec: {l_rec.item():.4f}, Con: {l_con.item():.4f}")
            
    print("Start Evaluation...")
    mask_dict = {'train': data.train_mask, 'val': data.val_mask, 'test': data.test_mask}
    acc = linear_probing(model, data.x, data.edge_index, data.y, mask_dict, device)
    print(f"Final Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()