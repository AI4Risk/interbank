import torch
from torch import nn
from models import GraphTransformer, GCN, GAT
from models.TGAR import TGAR
from torch_geometric.data import Data
from prompt import DirectionPrompt, DistancePrompt, AugmentPrompt, AdaptPrompt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import numpy as np

class PretrainModel(nn.Module):
    def __init__(self, args):
        super(PretrainModel, self).__init__()
        self.args = args
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        if args.gnn_type == "GAT":
            self.gnn = GAT(input_dim=args.input_dim, hid_dim=args.hid_dim, \
                           num_layer=args.num_layer, drop_ratio=args.droprate).to(device)
        elif args.gnn_type == "GCN":
            self.gnn = GCN(input_dim=args.input_dim, hid_dim=args.hid_dim, \
                           num_layer=args.num_layer, drop_ratio=args.droprate).to(device)
        elif args.gnn_type == "TGAR":
            self.gnn = TGAR(args, 1, 4).to(device)
        elif args.gnn_type == "GT":
            self.gnn = GraphTransformer(input_dim=args.input_dim, hid_dim=args.hid_dim,
                                        num_layer=args.num_layer, drop_ratio=args.droprate).to(device)

        self.dist_head = nn.Sequential(
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(),
            nn.Linear(args.hid_dim, 4),
            nn.Softmax(dim=1)
        )
        self.dire_head = nn.Sequential(
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(),
            nn.Linear(args.hid_dim, 3),
            nn.Softmax(dim=1)
        )
        self.load_model_weights(args.model_path)


    def forward(self, x, y, edge_index, batch=None):
        emb = self.gnn(x, edge_index)
        num_samples = 4548  
        total_nodes = 4548
        row = np.random.randint(0, total_nodes, size=num_samples)
        col = np.random.randint(0, total_nodes, size=num_samples)

        emb_i, emb_j = emb[row], emb[col]
        e_ij = emb_i - emb_j + emb_i * emb_j
        # Direction prediction
        dire_logits = self.dire_head(e_ij)
        true_diff = y[row] - y[col]
        true_dire = torch.where(true_diff < 0, torch.tensor(0, device=true_diff.device),
                                torch.where(true_diff == 0, torch.tensor(1, device=true_diff.device),
                                                            torch.tensor(2, device=true_diff.device)))
        loss_dire = F.cross_entropy(dire_logits, true_dire)
        # Distance prediction
        dist_logits = self.dist_head(torch.abs(e_ij))
        true_diff = abs(true_diff)
        loss_dist = F.cross_entropy(dist_logits,true_diff)
        return loss_dire, loss_dist

    def load_model_weights(self, model_path):
        if os.path.exists(model_path):
            model_weights = torch.load(model_path)
            self.gnn.load_state_dict(model_weights['gnn'])
            print("model weights loaded.")
        else:
            print(f"Warning: gnn weights file not found at {model_path}. Skipping loading.")

class DownstreamModel(nn.Module):
    def __init__(self, args):
        super(DownstreamModel, self).__init__()
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        if args.gnn_type == "GAT":
            self.gnn = GAT(input_dim=args.input_dim, hid_dim=args.hid_dim, \
                           num_layer=args.num_layer, drop_ratio=args.droprate)
        elif args.gnn_type == "GCN":
            self.gnn = GCN(input_dim=args.input_dim, hid_dim=args.hid_dim, \
                           num_layer=args.num_layer, drop_ratio=args.droprate)
        elif args.gnn_type == "TGAR":
            self.gnn = TGAR(args, 1, 4)
        elif args.gnn_type == "GT":
            self.gnn = GraphTransformer(input_dim=args.input_dim, hid_dim=args.hid_dim,
                                        num_layer=args.num_layer, drop_ratio=args.droprate)

        self.adapt = AdaptPrompt(args.input_dim, args.output_dim)
        self.augment = AugmentPrompt(args.input_dim, args.aug_num_in)

        self.dist_prompt = AugmentPrompt(args.hid_dim, args.aug_num_hid)
        self.dire_prompt = AugmentPrompt(args.hid_dim, args.aug_num_hid)

        self.dist_head = nn.Sequential(
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(),
            nn.Linear(args.hid_dim, 4),
            nn.Softmax(dim=1)
        )
        self.dire_head = nn.Sequential(
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(),
            nn.Linear(args.hid_dim, 3),
            nn.Softmax(dim=1)
        )
        self.load_model_weights(args.model_path)

    def forward(self, Q1_input, Q2_input, selected_idxes, remaining_idxes):
        # input
        input_x = Q1_input.x.clone()
        input_y = Q1_input.y.clone()
        input_edge_index = Q1_input.edge_index.clone()

        input_x[selected_idxes] = self.augment.compute(Q2_input.x[selected_idxes].clone())
        adapt_rmn = self.adapt.compute(Q1_input, Q2_input, selected_idxes, remaining_idxes)
        input_x[remaining_idxes] = self.augment.compute(adapt_rmn)
        input_y[selected_idxes] = Q2_input.y[selected_idxes].clone()
        input = Data(x=input_x, edge_index=input_edge_index, y=input_y).to(self.device)

        emb = self.gnn(input.x, input.edge_index)

        ## class-centers
        Q1_x = Q1_input.x.clone()
        Q1_y = Q1_input.y.clone()
        Q1_edge_index = Q1_input.edge_index.clone()
        Q1_aug_x =self.augment.compute(Q1_x)
        Q1_emb = self.gnn(Q1_aug_x, Q1_edge_index)
        Q1_class_centers = torch.vstack(
            [Q1_emb[Q1_y == i].mean(dim=0) for i in range(self.args.output_dim)]
        )
        few_class_centers = torch.vstack(
            [emb[selected_idxes][input.y[selected_idxes] == i].mean(dim=0) for i in range(self.args.output_dim)]
        )
        sim = F.cosine_similarity(emb[remaining_idxes].unsqueeze(1),few_class_centers.unsqueeze(0),dim=-1)
        input.y[remaining_idxes] = torch.argmax(sim,dim=1) # pseudo labels for the remaining nodes
        Q2_class_centers = torch.vstack(
            [emb[input.y == i].mean(dim=0) for i in range(self.args.output_dim)]
        )
        class_centers = torch.cat((Q1_class_centers, Q2_class_centers))
        ##################

        delta = emb[selected_idxes].unsqueeze(1) - class_centers + emb[selected_idxes].unsqueeze(1) * class_centers
        dire_emb = self.dire_prompt.compute(delta)
        dire_logits = self.dire_head(dire_emb)
        dire_probs = F.softmax(dire_logits, dim=2)
        dist_emb = self.dist_prompt.compute(delta)
        dist_logits = self.dist_head(torch.abs(dist_emb))
        dist_probs = F.softmax(dist_logits, dim=2)

        num_nodes = delta.size(0)
        combined_probs = torch.zeros((num_nodes, 4, 7), device=delta.device)  
        out = torch.zeros((num_nodes, 4), device=delta.device)


        for idx in range(8):
            class_idx = idx % 4
            for dire_idx in range(3):  # 0, 1, 2
                for dist_idx in range(4):  # 0, 1, 2, 3
                    product = (dire_idx - 1) * dist_idx  # turn the direction value into -1, 0, 1
                    combined_probs[:, class_idx, product + 3] += (
                            dire_probs[:, idx, dire_idx] * dist_probs[:, idx, dist_idx])

        for class_idx in range(4):
            pre_out = combined_probs[:, class_idx, 3 - class_idx:7 - class_idx].squeeze(1)
            out += F.softmax(pre_out, dim=1)

        out = F.softmax(out, dim=1)
        loss_classify = F.cross_entropy(out, input_y[selected_idxes])

        return loss_classify

    def inference(self, Q1_input, Q2_input, selected_idxes, remaining_idxes, test_idxes):
        # input
        input_x = Q1_input.x.clone()
        input_y = Q1_input.y.clone()
        input_edge_index = Q1_input.edge_index.clone()

        input_x[selected_idxes] = self.augment.compute(Q2_input.x[selected_idxes].clone())
        adapt_rmn = self.adapt.compute(Q1_input, Q2_input, selected_idxes, remaining_idxes)
        input_x[remaining_idxes] = self.augment.compute(adapt_rmn)
        input_y[selected_idxes] = Q2_input.y[selected_idxes].clone()
        input = Data(x=input_x, edge_index=input_edge_index, y=input_y).to(self.device)

        emb = self.gnn(input.x, input.edge_index)

        ## class-centers
        Q1_x = Q1_input.x.clone()
        Q1_y = Q1_input.y.clone()
        Q1_edge_index = Q1_input.edge_index.clone()
        Q1_aug_x =self.augment.compute(Q1_x)
        Q1_emb = self.gnn(Q1_aug_x, Q1_edge_index)

        Q1_class_centers = torch.vstack(
            [Q1_emb[Q1_y == i].mean(dim=0) for i in range(self.args.output_dim)]
        )
        few_class_centers = torch.vstack(
            [emb[selected_idxes][input.y[selected_idxes] == i].mean(dim=0) for i in range(self.args.output_dim)]
        )
        sim = F.cosine_similarity(emb[remaining_idxes].unsqueeze(1),few_class_centers.unsqueeze(0),dim=-1)
        input.y[remaining_idxes] = torch.argmax(sim,dim=1)
        Q2_class_centers = torch.vstack(
            [emb[input.y == i].mean(dim=0) for i in range(self.args.output_dim)]
        )
        class_centers = torch.cat((Q1_class_centers, Q2_class_centers))
        ##################

        if self.args.exp_obj == 'shot_num':
            delta = emb[test_idxes].unsqueeze(1) - class_centers + emb[test_idxes].unsqueeze(1) * class_centers
        else:
            delta = emb[remaining_idxes].unsqueeze(1) - class_centers + emb[remaining_idxes].unsqueeze(1) * class_centers

        dire_emb = self.dire_prompt.compute(delta)
        dire_logits = self.dire_head(dire_emb)
        dire_probs = F.softmax(dire_logits, dim=2)
        dist_emb = self.dist_prompt.compute(delta)
        dist_logits = self.dist_head(torch.abs(dist_emb))
        dist_probs = F.softmax(dist_logits, dim=2)

        num_nodes = delta.size(0)
        combined_probs = torch.zeros((num_nodes, 4, 7), device=delta.device)  
        out = torch.zeros((num_nodes, 4), device=delta.device)

       
        for idx in range(8):
            class_idx = idx % 4
            for dire_idx in range(3):  # 0, 1, 2
                for dist_idx in range(4):  # 0, 1, 2, 3
                    product = (dire_idx - 1) * dist_idx 
                    combined_probs[:, class_idx, product + 3] += (
                            dire_probs[:, idx, dire_idx] * dist_probs[:, idx, dist_idx])
        for class_idx in range(4):
            pre_out = combined_probs[:, class_idx, 3 - class_idx:7 - class_idx].squeeze(1)
            out += F.softmax(pre_out, dim=1)

        out = F.softmax(out, dim=1)
        pred = torch.argmax(out, dim=1)

        return out, pred

    def load_model_weights(self, model_path):
        model_weights = torch.load(model_path)
        self.gnn.load_state_dict(model_weights['gnn'])
        print("model weights loaded.")

    def plot_distribution(self, Q_str, Q1_input, Q2_input, selected_idxes, remaining_idxes):
        model_path = f"pretrained_gnn/weights_{Q_str}.pth"
        model_weights = torch.load(model_path)
        self.gnn.load_state_dict(model_weights['gnn'])
        self.augment.load_state_dict(model_weights['augment'])

        input_x = Q1_input.x.clone()
        input_y = Q1_input.y.clone()
        input_edge_index = Q1_input.edge_index.clone()
        input_x[remaining_idxes] = self.adapt.compute(Q1_input, Q2_input, selected_idxes, remaining_idxes)
        input_x = self.augment.compute(input_x)
        input_y[selected_idxes] = Q2_input.y[selected_idxes].clone()
        input = Data(x=input_x, edge_index=input_edge_index, y=input_y).to(self.device)

        emb = self.gnn(input.x, input.edge_index)  # (4548, 256)
        emb_np = emb.cpu().detach().numpy()

        Q1_x = Q1_input.x.clone()
        Q1_y = Q1_input.y.clone()
        Q1_edge_index = Q1_input.edge_index.clone()
        Q1_aug_x = self.augment.compute(Q1_x)
        Q1_emb = self.gnn(Q1_aug_x, Q1_edge_index)
        Q1_class_centers = torch.vstack(
            [Q1_emb[Q1_y == i].mean(dim=0) for i in range(self.args.output_dim)]
        )
        few_class_centers = torch.vstack(
            [emb[selected_idxes][input.y[selected_idxes] == i].mean(dim=0) for i in range(self.args.output_dim)]
        )
        sim = F.cosine_similarity(emb[remaining_idxes].unsqueeze(1), few_class_centers.unsqueeze(0), dim=-1)
        input.y[remaining_idxes] = torch.argmax(sim, dim=1)
        Q2_class_centers = torch.vstack(
            [emb[input.y == i].mean(dim=0) for i in range(self.args.output_dim)]
        )

        ## tsne-method
        combined_data = np.vstack(
            (emb_np, Q1_class_centers.cpu().detach().numpy(), Q2_class_centers.cpu().detach().numpy()))

        n_samples = emb_np.shape[0]
        tsne = TSNE(n_components=2, perplexity=2, n_iter=1000, random_state=42)
        combined_2d_tsne = tsne.fit_transform(combined_data)

        emb_2d_tsne = combined_2d_tsne[:n_samples]
        Q1_class_centers_2d_tsne = combined_2d_tsne[n_samples:n_samples + 4]
        Q2_class_centers_2d_tsne = combined_2d_tsne[n_samples + 4:]

        y_np = Q2_input.y.cpu().detach().numpy()

        np.savez('distri_plot_var.npz',emb_2d_tsne=emb_2d_tsne,Q1_class_centers_2d_tsne=Q1_class_centers_2d_tsne,Q2_class_centers_2d_tsne=Q2_class_centers_2d_tsne,y_np=y_np)
        

        # colors_light = ['#FF6666', '#00CC00', '#6699FF', '#FFB366']
        colors_light = ['#00CC00', '#6699FF', '#FFB366', '#FF6666']
        # colors_dark = ['#FF0000', '#008000', '#0000FF', '#FF8000']
        colors_dark = ['#008000', '#0000FF', '#FF8000', '#FF0000']

        center_markers = ['^', 's']

        plt.figure(figsize=(8, 6))
        for i in range(4):
            plt.scatter(emb_2d_tsne[y_np == i, 0], emb_2d_tsne[y_np == i, 1],
                        color=colors_light[i], marker='o', alpha=1.0, s=5)
        for i in range(4):
            plt.scatter(Q1_class_centers_2d_tsne[i, 0], Q1_class_centers_2d_tsne[i, 1],
                        color=colors_dark[i], marker=center_markers[0], alpha=1.0, s=200)
            plt.scatter(Q2_class_centers_2d_tsne[i, 0], Q2_class_centers_2d_tsne[i, 1],
                        color=colors_dark[i], marker=center_markers[1], alpha=1.0, s=200)

        rating = ['A','B','C','D']

        for i, color in enumerate(colors_light):
            plt.scatter([], [], color=color, label=f'Rating {rating[i]}')

        plt.scatter([], [], color='k', marker=center_markers[0], label=r'$\mathbf{Z}^{\omega}$')
        plt.scatter([], [], color='k', marker=center_markers[1], label=r'$\mathbf{Z}^{\gamma}$')
        plt.legend(loc='upper right', prop={'size': 12}, ncol=6, frameon=False, bbox_to_anchor=(1.00, 1.05))
        plt.axis('off')
        # plt.tight_layout()
        plt.savefig(f'./results/C1/plot/tsne/{Q_str}_emb_visual.pdf', dpi=300)

