import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
import powerlaw


class Model():
    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.num_runs=args.num_runs
        self.aaa=args.a
        self.bbb=args.b
        self.ccc=args.c
        self.p=args.p
        self.batchsize=args.batchsize
        self.epochs=args.epochs
        
    def fit(self, epoch, batch, drop_all,m,edge_drop,edge0, edge1):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        bl= batch.n_id.numpy()

        drop_half=drop_all[-1].cpu().numpy()
        x1 = batch.x.unsqueeze(0).expand(1, -1, -1).clone()

        if epoch<1/2*self.epochs:
            drop = torch.bernoulli(torch.ones([x1.size(0), x1.size(1)], device=x1.device)*self.p)
            for i in range(1,self.num_runs):
                drop1 = torch.bernoulli(torch.ones([x1.size(0), x1.size(1)], device=x1.device)*self.p)
                drop= torch.cat([drop,drop1],dim=0).bool()
        elif m<1/2*self.epochs:
            drop=drop_all[m].cpu().numpy()
            m=m+1
            drop=torch.LongTensor(drop[bl]).bool().unsqueeze(0).to(device)
            for i in range(1,self.num_runs):        
                drop1 = drop_all[m].cpu().numpy()
                drop1=torch.LongTensor(drop1[bl]).bool().unsqueeze(0).to(device)
                m=m+1
                drop= torch.cat([drop,drop1],dim=0).bool()
        else:        
            drop_smooth=np.array([0] * 9096)
            for i in range(len(edge_drop[0])):
                drop_smooth[bl[edge_drop[0][i]]]=1
                drop_smooth[bl[edge_drop[1][i]]]=1
            drop_smooth=torch.LongTensor(drop_smooth[bl]).bool().unsqueeze(0).to(device) 

            de=degree(edge0) +degree(edge1)      
            min_de = torch.min(de)
            max_de = torch.max(de)
            minmax_de = abs(de - min_de) / (max_de - min_de)
            minmax_y=batch.y/4
            deplusy=(minmax_de+minmax_y)/2
            
            deyrank = np.argpartition(deplusy.cpu().numpy(), int(self.p * 4548))
            idx_drop_deyrank = deyrank[:int(self.p * 4548)]
            drop_perfer=np.array([0] * 4548)
            drop_perfer[idx_drop_deyrank]=1
            drop_perfer=torch.LongTensor(drop_perfer[bl]).bool().unsqueeze(0).to(device) 
            
            drop=drop_all[-1].cpu().numpy()
            drop=torch.LongTensor(drop[bl]).bool().unsqueeze(0).to(device)
            drop0=drop
            for i in range(1,self.num_runs):
                drop1 = torch.bernoulli((drop_smooth.float()+drop_perfer.float())/2)
                drop1 = torch.bernoulli((drop1.float()++drop0.float())/2)
                drop= torch.cat([drop,drop1],dim=0).bool()

        optimizer.zero_grad()
        drop,out,index_edge_drop = self.model(batch,drop,self.num_runs,self.p)
        loss = F.nll_loss(out[0], batch.y)
        drop_batch=drop[0]
        edge_drop=index_edge_drop[0]
        for i in range(1,self.num_runs):
            loss2 = F.nll_loss(out[i], batch.y)
            if loss > loss2 :
                loss=loss2
                drop_batch=drop[i]
                edge_drop=index_edge_drop[i]
        
        drop_half[bl]=drop_batch.cpu().numpy()
        drop_all=torch.cat([drop_all.cpu(),torch.LongTensor(drop_half).unsqueeze(0)],dim=0)
        del drop_half

        #print(loss)
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
        #print(smooth_feat)     
        
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
        bm=(bx0m.sum()+bx1m.sum()+bx2m.sum()+bx3m.sum())/(len(batch.y)*1000)
        bm1=abs(len(bx1)+len(bx3)-len(bx0)-len(bx2))/len(batch.y)
        #print(bm1)

        # 计算节点的度
        degrees = degree(edge0)+degree(edge1)
        # Fit the power-law model to the data
        fit = powerlaw.Fit(degrees.cpu())
        # Retrieve the power-law exponent
        alpha = fit.power_law.alpha

        loss=(loss+self.aaa*smooth_feat+self.bbb*bm+self.ccc*alpha)/(1+self.aaa+self.bbb+self.ccc)

        loss.backward()
        optimizer.step()

        return drop_all,m,edge_drop,edge0, edge1



    def test(self,testbatch,drop_all,x):

        pred=[]
        tru=[]    
        outall=[]
        yy=[]

        self.model.eval()
        drop_half2=drop_all[-1].cpu().numpy()
        drop_all2=torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device)*0)[0].cpu().numpy()
        bl2= testbatch.n_id.numpy()
        drop_batch2=torch.LongTensor(drop_all2[bl2]).bool().unsqueeze(0)
        _, out,_ = self.model(testbatch,drop_batch2,1,0)
        predb=out[0]
        outall.append(torch.Tensor.cpu(predb))
        predb = predb.max(dim=1).indices
        
        predb = torch.Tensor.cpu(predb).tolist()
        trub = torch.Tensor.cpu(testbatch.y).tolist()
        pred.extend(predb)
        tru.extend(trub)

        return tru, pred
