import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv # GCN
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,f1_score,recall_score,classification_report


class Model():
    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.num_runs=args.num_runs
        self.alpha=args.alpha
        self.beta=args.beta
        self.gamma=args.gamma
        
    def fit(self, epoch, batch, drop_all):
        #print("1")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        bl= batch.n_id.numpy()


        drop_half=drop_all[-1].cpu().numpy()
        drop_batch=torch.LongTensor(drop_half[bl]).unsqueeze(0).unsqueeze(0).float().to(device)
        optimizer.zero_grad()

        #print(self.num_runs)
        
        drop,out = self.model(batch,drop_batch,self.num_runs)
        loss = F.nll_loss(out[0], batch.y)
        drop_batch=drop[0]
        for i in range(1,self.num_runs):
            loss2 = F.nll_loss(out[i], batch.y)
            if loss > loss2 :
                loss=loss2
                drop_batch=drop[i]
        
        drop_half[bl]=drop_batch.cpu().numpy()
        drop_all=torch.cat([drop_all.cpu(),torch.LongTensor(drop_half).unsqueeze(0)],dim=0)
        del drop_half

        drop_batch1=drop_batch.int()
        drop0=drop_batch1[batch.edge_index[0]]
        drop1=drop_batch1[batch.edge_index[1]]        
        drop2=drop0+drop1        
        edge0=batch.edge_index[0][drop2<2]
        edge1=batch.edge_index[1][drop2<2]
        edge0x=batch.y[edge0]
        edge1x=batch.y[edge1]
        smooth=edge0x-edge1x
        smooth_feat=(1/2*smooth*smooth).mean()

        drop_batch_not= ~drop_batch
        bx=batch.x[drop_batch_not]
        by=batch.y[drop_batch_not]
        bx0=bx[by==0]
        bx0m=torch.abs(bx0-bx0.mean(0))
        bx1=bx[by==1]
        bx1m=torch.abs(bx1-bx1.mean(0))
        bx2=bx[by==2]
        bx2m=torch.abs(bx2-bx2.mean(0))
        bx3=bx[by==3]
        bx3m=torch.abs(bx3-bx3.mean(0))
        bm=(bx0m.sum()+bx1m.sum()+bx2m.sum()+bx3m.sum())/(len(batch.y)*900)

        loss=self.alpha*loss+self.beta*smooth_feat+self.gamma*bm

        loss.backward()
        optimizer.step()

        if(epoch % 100 == 0):
            self.model.eval()
            bl= batch.n_id.numpy()
            drop_half=drop_all[-1].cpu().numpy()
            drop_batch=torch.LongTensor(drop_half[bl]).unsqueeze(0).unsqueeze(0).float().to(device)
            _, out = self.model(batch,drop_batch,self.num_runs)
            pred=out[0]
            pred =  pred.max(dim=1).indices
            correct = int(pred.eq(batch.y).sum().item())
            pred = torch.Tensor.cpu(pred)
            tru = torch.Tensor.cpu(batch.y)
            pred = F.one_hot(pred, 4)
            tru = F.one_hot(tru, 4)
            self.model.train()
        
        return drop_all


    def test(self,testbatch,drop_all):
        pred=[]
        tru=[]    
        outall=[]
        yy=[]

        self.model.eval()
        drop_half2=drop_all[-1].cpu().numpy()
        bl2= testbatch.n_id.numpy()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        drop_batch2=torch.LongTensor(drop_half2[bl2]).unsqueeze(0).unsqueeze(0).float().to(device)
        _, out = self.model(testbatch,drop_batch2,self.num_runs)
        predb=out[0]
        outall.append(torch.Tensor.cpu(predb))
        predb = predb.max(dim=1).indices
        
        yy.append(torch.Tensor.cpu(testbatch.y))
        predb = torch.Tensor.cpu(predb).tolist()
        trub = torch.Tensor.cpu(testbatch.y).tolist()
        pred.extend(predb)
        tru.extend(trub)

        return tru, pred


class SAGNNNet(torch.nn.Module):
    def __init__(self, num_feature, num_label, hiddim, droprate,hidlayers,p):
        super(SAGNNNet, self).__init__()
        self.GCN1 = GCNConv(num_feature, hiddim)
        self.GCN = GATConv(hiddim, hiddim)
        self.GCN2 = GCNConv(hiddim, num_label)
        
        self.num_feature=num_feature
        self.num_label=num_label
        self.hiddim=hiddim
        self.hidlayers=hidlayers
        self.dropout = torch.nn.Dropout(p=droprate)
        self.p=p

        self.fcs = nn.ModuleList()

        for i in range(self.hidlayers-1):
            self.fcs.append(nn.Linear(self.hiddim, self.num_label))

    def forward(self, data,drop_half,num_runs):
        x, edge_index = data.x, data.edge_index
        self.fcs.append(nn.Linear(self.num_feature, self.num_label))
        self.fcs.append(nn.Linear(self.hiddim, self.num_label))
        
        drop = drop_half.mean(dim=0)
        for i in range(1,num_runs):
            x1 = data.x.unsqueeze(0).expand(1, -1, -1).clone()
            drop1 = torch.bernoulli(torch.ones([x1.size(0), x1.size(1)], device=x1.device)*self.p)
            drop= torch.cat([drop,drop1],dim=0).bool()
        x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
        x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)     
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
        
        return drop,out