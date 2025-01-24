import torch
import numpy as np
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data
import os

def load_two_quarter(time_str1,time_str2):
    

    folder=os.path.join(os.getcwd(),"datasets/")

    year1,Q1= time_str1.split('-')
    year2,Q2 = time_str2.split('-')

    cites1 = folder+"edges/edge_"+str(year1)+"Q"+str(Q1)+".csv"
    content1 = folder+"nodes/"+str(year1)+"Q"+str(Q1)+".csv"


    content2 = folder+"nodes/"+str(year2)+"Q"+str(Q2)+".csv"

    credit_label= {'1':0,'2':1,'3':2,'4':3}
    index_dict = dict()
    

    features = []
    labels = []

    with open(content1,"r") as f:
        nodes = f.readlines()
        j=0
        for node in nodes:
            node=node.strip('\n')
            if j==0:
                j=1
                continue
            node_info = node.split(',')    
            index_dict[node_info[0]] = len(index_dict)
            features.append([float(i) for i in node_info[1:-1]])
                
            label_str = node_info[-1]
            labels.append(credit_label[label_str])
          

    with open(content2,"r") as f:
        nodes = f.readlines()
        j=0
        for node in nodes:
            node=node.strip('\n')
            if j==0:
                j=1
                continue
            node_info = node.split(',')  
            index_dict[str(int(node_info[0])+24271)] = len(index_dict) 

            features.append([float(i) for i in node_info[1:-1]])
                
            label_str = node_info[-1]
            labels.append(credit_label[label_str])
  

    edge_index = []

    with open(cites1,"r") as f:
        edges = f.readlines()
        j=0
        for edge in edges:
            if j==0:
                j=1
                continue
            edge=edge.strip('\n')
            
            start, end,_ = edge.split(',')
            if start in index_dict and end in index_dict :
                edge_index.append([index_dict[start],index_dict[end]])
                edge_index.append([index_dict[end],index_dict[start]])

    
    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(features)
    edge_index =  torch.LongTensor(edge_index)
    return features,labels,edge_index

def load_one_quarter(time_str):

    year,Q = time_str.split('-')
    


    folder=os.path.join(os.getcwd(),"datasets/")

    cites = folder + "edges/edge_"+str(year)+"Q"+str(Q)+".csv"
    content  = folder + "nodes/"+ str(year)+"Q"+str(Q)+".csv"

    index_dict = dict()
 

    features = []
    labels = []
    edge_index = []
    credit_label= {'1':0,'2':1,'3':2,'4':3}

    with open(content,"r") as f:
        nodes = f.readlines()
        j=0
        for node in nodes:
            node=node.strip('\n')
            if j==0:
                j=1
                continue
            node_info = node.split(',')    
            index_dict[node_info[0]] = len(index_dict)
            features.append([float(i) for i in node_info[1:-1]])
                
            label_str = node_info[-1]
            labels.append(credit_label[label_str])
          
        
    with open(cites,"r") as f:
        edges = f.readlines()
        j=0
        for edge in edges:
            if j==0:
                j=1
                continue
            edge=edge.strip('\n')
            
            start, end,_ = edge.split(',')
            if start in index_dict and end in index_dict :
                edge_index.append([index_dict[start],index_dict[end]])
                edge_index.append([index_dict[end],index_dict[start]])

    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(features)
    edge_index =  torch.LongTensor(edge_index)
    return features,labels,edge_index

# every graph is undirected
def get_interbank_graph_list(data, subgraphs=400, num_q = 1):
    x = data.x.detach()
    edge_index = data.edge_index
    edge_index_undirected = to_undirected(edge_index)
    data = Data(x=x, edge_index=edge_index_undirected)
    input_dim = data.num_features
    out_dim = 4
    graph_list = list(ClusterData(data=data, num_parts=subgraphs*num_q))
    return input_dim, out_dim, graph_list

