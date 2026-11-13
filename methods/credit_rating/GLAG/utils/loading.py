
import torch
from torch_geometric.data import Data
import json
import numpy as np

def feature_norm(features):
    features = (features - features.mean(dim=0, keepdims=True)) / \
    features.std(dim=0, keepdims=True)
    features = torch.sign(features) * torch.abs(features) ** 1.2
    return features

def load_data_for_quarter(year:int,Q:int,if_norm:bool,norm_type:str='minmax'):
    
    folder = "datasets/"
    # credit_label= {'1':0,'2':1,'3':2,'4':3}
    features= []
    edge_index= []

    edge_weight=[]

    sriskrs = []
    sriskvs = []
    sriskms = []
    labels=[]
    index_dict = dict()

    with open(folder+f"nodes/{year}Q{Q}.csv","r") as f:
        srisk_contain_flag=0
        nodes = f.readlines()
        j=0
        for node in nodes:
            node = node.strip("\n")
            if j==0:
                feature_names = node.split(',')
                if "srisk_value" in feature_names or "srisk_ratio" in feature_names:
                    srisk_contain_flag=1
                j=1
                continue
            node_info = node.split(',')
            index_dict[node_info[0]] = len(index_dict)
            if srisk_contain_flag:
                features.append([float(i) for i in node_info[1:-3]])
                labels.append(int(node_info[-3])-1)
                ### 加载srisk
                sriskr = node_info[-2]
                sriskv = node_info[-1]
                sriskr = float(sriskr) if sriskr != '' else np.nan
                sriskv = float(sriskv) if sriskv != '' else np.nan
                sriskrs.append(sriskr)
                sriskvs.append(sriskv)
                if np.isnan(sriskr):
                    sriskms.append(0)
                else:
                    sriskms.append(1)

            else:
                features.append([float(i) for i in node_info[1:-1]])
                labels.append(int(node_info[-1])-1)


    with open(folder+f"edges/edge_{year}Q{Q}.csv","r") as f:
        edges = f.readlines()
        j=0
        for edge in edges:
            if j==0:
                j=1
                continue
            edge = edge.strip('\n')
            start,end,weight = edge.split(',')
            if start in index_dict and end in index_dict:
                edge_index.append([index_dict[start], index_dict[end]])
                edge_index.append([index_dict[end], index_dict[start]])

                weight=float(weight)
                edge_weight.append(weight)
                edge_weight.append(weight)

    features = torch.FloatTensor(features)
    if if_norm:
        features = (features - features.mean(dim=0, keepdims=True)) / \
        features.std(dim=0, keepdims=True)
        if norm_type=='power':
            features = torch.sign(features) * torch.abs(features) ** 1.2
        elif norm_type=='minmax':
            fmin = features.min(dim=0, keepdims=True).values
            fmax = features.max(dim=0, keepdims=True).values
            features = (features-fmin)/(fmax-fmin)

    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(features)
    edge_index = torch.LongTensor(edge_index)

    edge_weight = torch.FloatTensor(edge_weight)

    sriskrs = torch.FloatTensor(sriskrs)
    sriskvs = torch.FloatTensor(sriskvs)
    sriskms = torch.LongTensor(sriskms)

    return features,labels,edge_index,edge_weight,sriskrs,sriskvs,sriskms

