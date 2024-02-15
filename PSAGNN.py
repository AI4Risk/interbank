import torch
import torch.nn.functional as F
from numpy import *
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv # GCN


class PSAGNNNet(torch.nn.Module):
    def __init__(self, num_feature, num_label, hiddim, droprate,hidlayers):
        super(PSAGNNNet, self).__init__()
        self.GCN1 = GCNConv(num_feature, hiddim)
        self.GCN = GATConv(hiddim, hiddim)
        self.GCN2 = GCNConv(hiddim, num_label)
        self.mlp = nn.Sequential(
            nn.Linear(num_feature, num_feature*2),
            nn.ReLU(),
            nn.Linear(num_feature*2, 1)
        )
        self.att = nn.MultiheadAttention(embed_dim=num_feature, num_heads=1)
        
        
        self.num_feature=num_feature
        self.num_label=num_label
        self.hiddim=hiddim
        self.hidlayers=hidlayers
        self.dropout = torch.nn.Dropout(p=droprate)

        self.fcs = nn.ModuleList()

        for i in range(self.hidlayers-1):
            self.fcs.append(nn.Linear(self.hiddim, self.num_label))

    def forward(self, data,drop,num_runs,p):
        x, edge_index = data.x, data.edge_index
        self.fcs.append(nn.Linear(self.num_feature, self.num_label))
        self.fcs.append(nn.Linear(self.hiddim, self.num_label))

        x=(x-x.mean(dim=0,keepdims=True))/x.std(dim=0,keepdims=True)

        x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
        x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)        

        row, col = edge_index
        ratio=p
        n_reserve =  int(ratio * len(edge_index[0]))
        index_edge_drop=[]
        for i in range(num_runs):   
            if p == 0 :
                index_edge_drop.append(index_edge_drop)
            else:
                edge_rep = x[i][row]-x[i][col]
                # att_output,_=self.att(edge_rep,edge_rep,edge_rep)
                # edge_score = (att_output * edge_rep).mean(1)
                edge_score = self.mlp(edge_rep).view(-1)
                min_val = torch.min(edge_score)
                max_val = torch.max(edge_score)
                edge_score = (edge_score - min_val) / (max_val - min_val)
                idx_drop = torch.multinomial(edge_score, n_reserve,replacement=False)
                edge_index_drop=edge_index[:, idx_drop]
                index_edge_drop.append(edge_index_drop)
                del edge_score

        x = x.view(-1, x.size(-1))
        run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(num_runs, device=edge_index.device).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)

        x = self.GCN1(x, run_edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        for i in range(self.hidlayers):
            x = self.GCN(x, run_edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.GCN2(x, run_edge_index)
        del  run_edge_index

        x=x.view(num_runs, -1, x.size(-1))
        out=[]
        for i in range(num_runs):
            out.append(F.log_softmax(x[i], dim=1))
        
        return drop,out,index_edge_drop
