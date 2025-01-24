import torch
import numpy as np


def load_data(year,Q,attack_rate):
    if Q==1:
        if attack_rate!=0:
            cites1 = "datasets/edges"+str(attack_rate)+"/edge_"+str(int(year)-1)+"Q3.csv"
            content1 = "datasets/nodes"+str(attack_rate)+"/"+str(int(year)-1)+"Q3.csv"
        else:
            cites1 = "datasets/edges/edge_"+str(year-1)+"Q3.csv"
            content1 = "datasets/nodes/"+str(year-1)+"Q3.csv"

        cites2 = "datasets/edges/edge_"+str(int(year)-1)+"Q4.csv"
        content2 = "datasets/nodes/"+str(int(year)-1)+"Q4.csv"
    elif Q==2:
        if attack_rate!=0:
            cites1 = "datasets/edges"+str(attack_rate)+"/edge_"+str(int(year)-1)+"Q4.csv"
            content1 = "datasets/nodes"+str(attack_rate)+"/"+str(int(year)-1)+"Q4.csv"
        else:
            cites1 = "datasets/edges/edge_"+str(int(year)-1)+"Q4.csv"
            content1 = "datasets/nodes/"+str(int(year)-1)+"Q4.csv"

        cites2 = "datasets/edges/edge_"+str(year)+"Q1.csv"
        content2 = "datasets/nodes/"+str(year)+"Q1.csv"
    else:
        if attack_rate!=0:
            cites1 = "datasets/edges"+str(attack_rate)+"/edge_"+str(year)+"Q"+str(int(Q)-2)+".csv"
            content1 = "datasets/nodes"+str(attack_rate)+"/"+str(year)+"Q"+str(int(Q)-2)+".csv"
        else:
            cites1 = "datasets/edges/edge_"+str(year)+"Q"+str(int(Q)-2)+".csv"
            content1 = "datasets/nodes/"+str(year)+"Q"+str(int(Q)-2)+".csv"

        cites2 = "datasets/edges/edge_"+str(year)+"Q"+str(int(Q)-1)+".csv"
        content2 = "datasets/nodes/"+str(year)+"Q"+str(int(Q)-1)+".csv"

    
    index_dict = dict()
    label_to_index = dict()

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
            # features.append([float(i) for i in node_info[1:-1]])
            if len(node_info)>72:
                features.append([float(i) for i in node_info[1:-3]])
            else:
                features.append([float(i) for i in node_info[1:-1]])

            # label_str = node_info[-1]
            if len(node_info)>72:
                label_str = node_info[-3]
            else:
                label_str = node_info[-1]
            if(label_str not in label_to_index.keys()):
                label_to_index[label_str] = len(label_to_index)
            labels.append(label_to_index[label_str])


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
            # features.append([float(i) for i in node_info[1:-1]])
            if len(node_info)>72:
                features.append([float(i) for i in node_info[1:-3]])
            else:
                features.append([float(i) for i in node_info[1:-1]])

            # label_str = node_info[-1]
            if len(node_info)>72:
                label_str = node_info[-3]
            else:
                label_str = node_info[-1]
            if(label_str not in label_to_index.keys()):
                label_to_index[label_str] = len(label_to_index)
            labels.append(label_to_index[label_str])

    edge_index = []

    with open(cites1,"r") as f:
        edges = f.readlines()
        j=0
        for edge in edges:
            if j==0:
                j=1
                continue
            edge=edge.strip('\n')
            if attack_rate!=0:
                start, end = edge.split(',')
            else:
                start, end,_ = edge.split(',')
            if start in index_dict and end in index_dict :
                edge_index.append([index_dict[start],index_dict[end]])
                edge_index.append([index_dict[end],index_dict[start]])  
    

    with open(cites2,"r") as f:
        edges = f.readlines()
        j=0
        for edge in edges:
            if j==0:
                j=1
                continue
            edge=edge.strip('\n')
            start, end, weight = edge.split(',')
            start=str(int(start)+24271)
            end=str(int(end)+24271)  
            if start in index_dict and end in index_dict :
                edge_index.append([index_dict[start],index_dict[end]])
                edge_index.append([index_dict[end],index_dict[start]]) 
    
    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(features)
    edge_index =  torch.LongTensor(edge_index)
    return label_to_index,labels,features,edge_index


def preprocess():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    mask = torch.arange(9096)
    train_mask = mask[:4548]
    test_mask = mask[4548:]
    return train_mask,test_mask
