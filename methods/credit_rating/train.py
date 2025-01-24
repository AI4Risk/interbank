import numpy as np
import time
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from utils import load_data, preprocess 
from GCN import GCN, Model
from GAT import GAT
from TGAR import TGAR
import pandas as pd


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hiddim', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--droprate', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--year', type=int, default=2020)
parser.add_argument('--quarter', type=int, default=2)
parser.add_argument('--epochs', type=int,  default=1000, help='Number of epochs to train.')
parser.add_argument('--batchsize', type=int, default=4548, help='batchsize')
parser.add_argument('--numneighbors', type=int, default=30, help='numneighbors')
parser.add_argument('--hidlayers', type=int, default=2, help='hidlayers')
parser.add_argument('--net', type=str, default="GCN", choices=["GCN", "GAT", "TGAR"])
args = parser.parse_args()

label_to_index,labels,features,edge_index=load_data(args.year,args.quarter)
train_mask,test_mask=preprocess()
# Normalize
scaler = MinMaxScaler()
features = features.numpy()
scaler.fit(features[:int(0.5*features.shape[0])])
features_norm = scaler.transform(features)
features_norm = torch.FloatTensor(features_norm)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = Data(x = features_norm, edge_index = edge_index.t().contiguous(), y = labels).to(device)

if (args.net == "GCN"):
    gnnnet = GCN(features.shape[1], len(label_to_index), hiddim=args.hiddim, droprate=args.droprate,hidlayers=args.hidlayers,p=1).to(device)
elif (args.net == "GAT"):
    gnnnet = GAT(features.shape[1], len(label_to_index), hiddim=args.hiddim, droprate=args.droprate,hidlayers=args.hidlayers,p=1).to(device)
elif (args.net == "TGAR"):
    gnnnet = TGAR(args.batchsize, features.shape[1], len(label_to_index), hiddim=args.hiddim, droprate=args.droprate,hidlayers=args.hidlayers,p=1,hyper_k=4).to(device)

model = Model(gnnnet, args, device)

loader = NeighborLoader(
    data,
    num_neighbors=[-1],
    batch_size=args.batchsize,
    input_nodes=train_mask,
)
testload=NeighborLoader(
    data,
    num_neighbors=[-1],
    batch_size=args.batchsize,
    input_nodes=test_mask,
)

data.n_id = torch.arange(data.num_nodes)
best_acc=0
for epoch in tqdm(range(args.epochs)):  

    # batch-level training
    for batch in loader:
        model.fit(batch)
    
    # batch-level inference
    with torch.no_grad():
        tru_list, pred_list = list(), list()
        for testbatch in testload:
            tru, pred=model.test(testbatch) # [128, 1]
            tru_list.append(tru)
            pred_list.append(pred)
        tru_torch = torch.cat(tru_list, dim=0)

        pre_torch = torch.cat(pred_list, dim=0)
        accuracy=accuracy_score(tru_torch, pre_torch)
        if accuracy>best_acc:
            best_acc=accuracy
            report = classification_report(tru_torch, pre_torch, digits=5)
            print(report) 

print_year = args.year
print_quarter = args.quarter

with open('./results/classification_report_' + str(print_year) + "Q" + str(print_quarter) + '.txt', 'a') as f:
    f.write("\n" + str(gnnnet.model_name) + " epochs " + str(args.epochs) + "\n")
    f.write(report)
    f.write("\n")
# with open('./credit_rating_results_TGAR/classification_report_' + str(print_year) + "Q" + str(print_quarter) + '.txt', 'a') as f:
#     f.write("\n" + str(gnnnet.model_name) + " epochs " + str(args.epochs) + "\n")
#     f.write(report)
#     f.write("\n")