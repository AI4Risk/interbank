import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv # GCN
from tqdm import tqdm

 
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
        # loss.requires_grad = True
        loss.backward()
        self.optimizer.step()

    def test(self, testbatch):
        self.model.eval()
        out = self.model(testbatch)
        predb = out.cpu().max(dim=1).indices[:testbatch.batch_size]
        trub = torch.Tensor.cpu(testbatch.y)[:testbatch.batch_size]
        self.model.train()

        return trub, predb


class TGAR(torch.nn.Module):
    def __init__(self, batch_size, num_feature, num_label, hiddim, droprate,hidlayers,p, hyper_k):
        super(TGAR, self).__init__()
        self.model_name = "TGAR" 
        # hyper parameters
        self.batch_size = batch_size
        self.num_feature = num_feature
        self.num_label = num_label
        self.hiddim = hiddim 
        self.hidlayers = hidlayers
        self.dropout = torch.nn.Dropout(p=droprate)
        self.hyper_k = hyper_k

        # layers

        self.fcs = nn.ModuleList()

        # hyper-feature transition
        self.GATConv11 = GATConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        self.GATConv12 = GATConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        self.GATConv13 = GATConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        self.GATConv21 = GATConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        self.GATConv22 = GATConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        self.GATConv23 = GATConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        
        self.TransConv1 = TransformerConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        self.TransConv2 = TransformerConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)


        # fcnn layer 4 * hiddim    hiddim represents m in paper
        for i in range(4):
            self.fcs.append(nn.Linear(self.num_feature, self.hiddim))

        self.fcs.append(nn.Linear(self.hiddim * 3, 1))

        self.fcsK = nn.Linear(self.hiddim, self.num_feature)

        # fcnn layer 4 * hiddim    hiddim represents m in paper
        for i in range(4):
            self.fcs.append(nn.Linear(self.num_feature, self.hiddim))

        # Gain todo: dim?
        self.fcs.append(nn.Linear(self.hiddim * 3, 1))

        # todo: output layer
        self.fcs.append(nn.Linear(self.hiddim, self.num_label)) 

    def forward(self, data):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x, edge_index = data.x, data.edge_index     

        x = x.view(-1, x.size(-1))  # batch_size * num_feature(4)

        K = self.contextAttentionLayer(x, edge_index, 0)
        K = self.fcsK(K)
        K = self.contextAttentionLayer(K, edge_index, 1)

        output = self.fcs[-1](K)

        output = F.log_softmax(output, dim=1)

        return output
    

    # differential aggregation operator
    def diffAggr(self, X1, X2):
        concatenated = torch.cat([X1, X2, X1 - X2], dim=1) 
        return concatenated
    

    def contextAttentionLayer(self, x, edge_index, layer_num):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        innerLayerNum = 5

        x = x.to(device)
        # todo activation and dropout
        # FCNN
        x1 = self.fcs[0 + layer_num * innerLayerNum](x)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        x2 = self.fcs[1 + layer_num * innerLayerNum](x)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        x3 = self.fcs[2 + layer_num * innerLayerNum](x)
        x3 = F.relu(x3)
        x3 = self.dropout(x3)
        x4 = self.fcs[3 + layer_num * innerLayerNum](x)
        x4 = F.relu(x4)
        x4 = self.dropout(x4)


        # hyper-feature transition 
        if layer_num == 0:
            x1 = self.GATConv11(x1, edge_index)
            x2 = self.GATConv12(x2, edge_index)
            x3 = self.GATConv13(x3, edge_index)
        elif layer_num == 1:
            x1 = self.GATConv21(x1, edge_index)
            x2 = self.GATConv22(x2, edge_index)
            x3 = self.GATConv23(x3, edge_index)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)   
        x2 = F.relu(x2)
        x2 = self.dropout(x2)   
        x3 = F.relu(x3)
        x3 = self.dropout(x3)   

        # output Fc
        output_Fc = torch.mul(x1, x2)

        # output F 
        output_F = x3

        # Graph Fusion Representation
        if layer_num == 0:
            m = self.TransConv1(output_Fc, edge_index)
        elif layer_num == 1:
            m = self.TransConv2(output_Fc, edge_index)
        m = F.relu(m)
        m = self.dropout(m)
        m = torch.softmax(m, dim = 1)

        # Binomial Gain Learning stage 1
        z = torch.mul(m, output_F)

        # Binomial Gain Learning stage 2
        G = self.fcs[4 + layer_num * innerLayerNum].to(device)(self.diffAggr(z, x4))
        G = torch.sigmoid(G)
        G = self.dropout(G)

        leaky_relu = nn.LeakyReLU(0.25)    

        # bernoulli Fusion
        Y = G * z + (1 - G) * x4
        Y = leaky_relu(Y)

        return Y