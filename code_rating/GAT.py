import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv # GCN


class Model():
    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=5e-4)

    def fit(self, batch):
        self.optimizer.zero_grad()
        out = self.model(batch)
        loss = F.nll_loss(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        self.optimizer.step()

    def test(self, testbatch):
        self.model.eval()
        out = self.model(testbatch)
        predb = out.cpu().max(dim=1).indices[:testbatch.batch_size]
        trub = torch.Tensor.cpu(testbatch.y)[:testbatch.batch_size]
        self.model.train()

        return trub, predb


class GAT(torch.nn.Module):
    def __init__(self, num_feature, num_label, hiddim, droprate,hidlayers,p):
        super(GAT, self).__init__()
        self.model_name = "GAT"
        self.GAT1 = GATConv(num_feature, hiddim)
        self.GAT2 = GATConv(hiddim, num_label)
        
        self.num_feature=num_feature
        self.num_label=num_label
        self.hiddim=hiddim
        self.hidlayers=hidlayers
        self.dropout = torch.nn.Dropout(p=droprate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = x.view(-1, x.size(-1))

        x = self.GAT1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.GAT2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x