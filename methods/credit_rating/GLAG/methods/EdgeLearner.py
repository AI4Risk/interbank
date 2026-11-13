import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from .base_model import PretrainModel
from methods.credit_rating.GLAG.utils.args import get_args
from tqdm import tqdm
import os


class SimpleGraphLearner(nn.Module):

    def __init__(self, input_dim, epsilon=0.1,topk=10, num_heads=4):
        super(SimpleGraphLearner, self).__init__()
        self.epsilon = epsilon
        self.topk = topk
        self.num_heads = num_heads
        self.weight_tensor = torch.Tensor(num_heads, input_dim)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def forward(self, x):
        N, D = x.shape
        x_expand = x.unsqueeze(0).expand(self.num_heads, N, D)
        w_expand = self.weight_tensor.unsqueeze(1)
        x_weighted = x_expand * w_expand
        x_weighted_norm = F.normalize(x_weighted, p=2, dim=-1)
        attention = torch.bmm(x_weighted_norm, x_weighted_norm.transpose(1, 2))
        attention = torch.mean(attention, dim=0)
        

        if self.topk is not None and self.topk < N:
   
            topk_vals, topk_inds = torch.topk(attention, self.topk, dim=-1)
            
   
            mask = torch.zeros_like(attention)
        
            mask.scatter_(-1, topk_inds, 1.0)
        
            attention = attention * mask        

        mark_off = torch.lt(attention, self.epsilon)
        adj = attention.masked_fill(mark_off, 0)
        adj = (adj + adj.t()) / 2
        row_sum = torch.sum(adj, dim=1, keepdim=True) + 1e-6
        adj_normalized = adj / row_sum
        return adj, adj_normalized

class GatedGraphLearner(nn.Module):

    def __init__(self, input_dim1, input_dim2, fusion_dim, epsilon=0.1,topk=10, num_heads=4):
        super(GatedGraphLearner, self).__init__()
        self.epsilon = epsilon
        self.num_heads = num_heads
        self.topk=topk

        self.linear1 = nn.Linear(input_dim1, fusion_dim)
        self.linear2 = nn.Linear(input_dim2, fusion_dim)
        self.gate_linear = nn.Linear(fusion_dim * 2, fusion_dim)
        
        self.weight_tensor = torch.Tensor(num_heads, fusion_dim)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def forward(self, x1, x2):

        h1 = torch.tanh(self.linear1(x1))
        h2 = torch.tanh(self.linear2(x2))
   
        cat_features = torch.cat([h1, h2], dim=-1)
        z = torch.sigmoid(self.gate_linear(cat_features))
        x_fused = z * h1 + (1 - z) * h2  
      
        N, D = x_fused.shape
        x_expand = x_fused.unsqueeze(0).expand(self.num_heads, N, D)
        w_expand = self.weight_tensor.unsqueeze(1)
        x_weighted = x_expand * w_expand
        x_weighted_norm = F.normalize(x_weighted, p=2, dim=-1)
        attention = torch.bmm(x_weighted_norm, x_weighted_norm.transpose(1, 2))
        attention = torch.mean(attention, dim=0)


        if self.topk is not None and self.topk < N:
            topk_vals, topk_inds = torch.topk(attention, self.topk, dim=-1)
            mask = torch.zeros_like(attention)
            mask.scatter_(-1, topk_inds, 1.0)
            attention = attention * mask

        mark_off = torch.lt(attention, self.epsilon)
        adj = attention.masked_fill(mark_off, 0)
        adj = (adj + adj.t()) / 2
        row_sum = torch.sum(adj, dim=1, keepdim=True) + 1e-6
        adj_normalized = adj / row_sum
        
        return adj, adj_normalized, x_fused

class GraphLearner(nn.Module):
    def __init__(self, args, nfeat2, max_iter=10, epsilon=0.4, topk=5, eta=0.1, 
                 smooth_item=0.2, sparse_item=0.1, diff_threshold=4e-5, lam=0.8):
        super(GraphLearner, self).__init__()
        
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.eta = eta
        self.smooth_item = smooth_item
        self.sparse_item = sparse_item
        self.diff_threshold = diff_threshold
        self.lam = lam 
        # self.cut_off = cut_off
        
        
        pretrained = PretrainModel(args) 
   
        self.gnn_encoder = pretrained.gnn
        
        self.classifier = pretrained.class_head

        nhid = args.hid_dim 

        self.graph_learner = GatedGraphLearner(args.input_dim, nfeat2, fusion_dim=nhid, epsilon=epsilon,topk=topk)

        self.graph_learner2 = SimpleGraphLearner(nhid, epsilon=epsilon,topk=topk)

    def load_model_weights(self, weights_path):
        if os.path.exists(weights_path):
            model_weights = torch.load(weights_path, weights_only=True)
            
  
            self.gnn_encoder.load_state_dict(model_weights['gnn'])
            print("GNN backbone weights loaded.")

            try:
                self.classifier.load_state_dict(model_weights['class_head'])
                print("Class head weights loaded.")
            except:
                print("Class head structure changed or mismatch. Initializing class head from scratch (Recommended for fine-tuning).")
                
        else:
            print(f"Warning: gnn weights file not found at {weights_path}. Skipping loading.")

    def _disable_gnn_normalization(self):

        gnn_module = self.gnn_encoder
        for module in gnn_module.modules():
            if 'Conv' in module.__class__.__name__:
                module.normalize = False
                print(f"Disabled normalization for {module}")

    def compute_graph_regularization(self, adj, x):
        row_sum = torch.sum(adj, dim=1)
        D = torch.diag(row_sum)
        L = D - adj
        loss_smooth = torch.trace(torch.mm(x.t(), torch.mm(L, x))) / (x.shape[0]**2)
        loss_sparse = torch.sum(torch.pow(adj, 2)) / (adj.shape[0]**2)
        return loss_smooth, loss_sparse

    def forward(self, x1, x2, initial_edge_index, return_adj=False):

        N = x1.shape[0]
  
        adj_0 = to_dense_adj(initial_edge_index, max_num_nodes=N)[0]

        row_sum = torch.sum(adj_0, dim=1, keepdim=True) + 1e-6
        adj_0 = adj_0 / row_sum
        
 
        cur_raw_adj, cur_adj, x_fused = self.graph_learner(x1, x2)
        first_raw_adj, first_adj = cur_raw_adj, cur_adj
        
        curr_x = x_fused 
        graph_loss = 0
        prev_adj = cur_adj
        iter_count = 0 
        final_edge_index = None

        for t in range(self.max_iter):
            iter_count += 1

            if t == 0:
                learned_adj = cur_adj
                raw_learned_adj = cur_raw_adj
            else:

                raw_learned_adj, learned_adj = self.graph_learner2(curr_x)

            if t > 0:
                adj_diff = torch.norm(learned_adj - prev_adj, p='fro') ** 2
                adj_norm = torch.norm(learned_adj, p='fro') ** 2 + 1e-6
                if (adj_diff / adj_norm) < self.diff_threshold:
                    break
            prev_adj = learned_adj


            combined_learned_adj = self.eta * learned_adj + (1 - self.eta) * first_adj
            
            mask_nonzero = combined_learned_adj > 1e-9
            vals = combined_learned_adj[mask_nonzero]
            if vals.numel() > 0: 
                mu = vals.mean()
                std = vals.std()
                
           
                alpha = 0.5 
                cutoff = mu + alpha * std
                
                clean_learned_adj = combined_learned_adj.clone()
            
                clean_learned_adj[clean_learned_adj < cutoff] = 0
            else:
                clean_learned_adj = combined_learned_adj.clone()

            final_adj_dense = self.lam * adj_0 + (1 - self.lam) * clean_learned_adj
     
            final_edge_index, final_edge_weight = dense_to_sparse(final_adj_dense)

  
            # x_hidden = self.gnn_encoder(x1, final_edge_index, edge_weight=final_edge_weight)

            x_hidden = self.gnn_encoder(x1, final_edge_index)
            
            curr_x = x_hidden 
            
            l_smooth, l_sparse = self.compute_graph_regularization(raw_learned_adj, x_fused)
            graph_loss += self.smooth_item * l_smooth + self.sparse_item * l_sparse

   
        output = self.classifier(curr_x)
        
        if return_adj:
            return output, graph_loss / iter_count, final_edge_index
        
        return output, graph_loss / iter_count

