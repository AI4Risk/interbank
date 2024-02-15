import numpy as np
import time
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,f1_score,recall_score,classification_report
from utils import load_data, preprocess 
from PSAGNN import PSAGNNNet
from model import Model
import pandas as pd

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hiddim', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--droprate', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--year', type=int, default=2023,
        choices=[2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023], help='year')
parser.add_argument('--Q', type=int, default=1,
        choices=[1,2,3,4], help='Quarter')
parser.add_argument('--attack_rate', type=float, default=0.25, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=1000, help='Number of epochs to train.')
parser.add_argument('--batchsize', type=int, default=4548, help='batchsize')
parser.add_argument('--numneighbors', type=int, default=-1, help='numneighbors')
parser.add_argument('--hidlayers', type=int, default=1, help='hidlayers')
parser.add_argument('--num_runs', type=int, default=20, help='num_runs')
parser.add_argument('--a', type=float, default=0.6, help='weight of label similarity')
parser.add_argument('--b', type=float, default=0.3, help='weight of feature similarity')
parser.add_argument('--c', type=float, default=0.1, help='weight of scale-free')
parser.add_argument('--p', type=float, default=0.5, help='parameter p')
args = parser.parse_args()
numneighbors=args.numneighbors
batchsize=args.batchsize
attack_rate=args.attack_rate
epochs=args.epochs
num_runs=args.num_runs
p=args.p

label_to_index,labels,features,edge_index=load_data(args.year,args.Q,args.attack_rate)
train_mask,test_mask=preprocess()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = Data(x = features, edge_index = edge_index.t().contiguous(), y = labels).to(device)
sagnn = PSAGNNNet(features.shape[1], len(label_to_index), hiddim=args.hiddim, droprate=args.droprate,hidlayers=args.hidlayers).to(device)
model = Model(sagnn, args, device)

loader = NeighborLoader(
    data,
    num_neighbors=[args.numneighbors],
    batch_size=args.batchsize,
    input_nodes=train_mask,
)
testload=NeighborLoader(
    data,
    num_neighbors=[args.numneighbors],
    batch_size=args.batchsize,
    input_nodes=test_mask,
)

x = data.x.unsqueeze(0).expand(1, -1, -1).clone()
drop_all= torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device)*p)
#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

i=0
m=0
for (batch,testbatch) in zip(tqdm(loader),tqdm(testload)):
    i=i+1
    if i>1:
        break
    best_acc=0
    edge_drop=[]
    edge0, edge1=0, 0
    for epoch in tqdm(range(args.epochs)):  
        drop_all,m,edge_drop, edge0, edge1=model.fit(epoch,batch,drop_all,m,edge_drop,edge0, edge1)
        tru, pred=model.test(testbatch,drop_all,x)
        accuracy=accuracy_score(tru, pred)
        if accuracy>best_acc:
            best_acc=accuracy
            print("epoch {}".format(epoch))
            print(accuracy)
            # with open('output'+str(args.dataset)+'Q'+str(args.Q)+'-attack'+str(args.attack_rate)+'.txt', 'a') as f:
            #     print("epoch {}".format(epoch),file = f)
            #     print(accuracy,file = f)
    print(best_acc)    
   



