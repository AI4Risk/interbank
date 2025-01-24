import torch
import numpy as np

def load_data_for_quarter(year, quarter, Scaling="power", dir='graph_data'):
    if (quarter != 4):
        Bank = "../../../datasets/nodes/" + \
            str(year) + "Q" + str(quarter) + ".csv"
        
        Edge = "../../../datasets/edges/edge_" + \
            str(year) + "Q" + str(quarter) + ".csv"
        if year == 2023 and quarter == 1:
            Target = "../../../datasets/nodes/" + \
                str(year) + "Q" + str(quarter) + ".csv"
        else:
            Target = "../../../datasets/nodes/" + \
                str(year) + "Q" + str(quarter + 1) + ".csv"
    elif (quarter == 4):
        Bank = "../../../datasets/nodes/" + \
            str(year) + "Q" + str(quarter) + ".csv"
        
        Edge = "../../../datasets/edges/edge_" + \
            str(year) + "Q" + str(quarter) + ".csv"
        Target = "../../../datasets/nodes/" + \
            str(year + 1) + "Q" + str(1) + ".csv"

    index_dict = dict()
    features = []
    labels = []
    edge_index = []
    edge_attr = []
    sriskrs = []
    sriskvs = []
    sriskms = []

    with open(Bank, "r") as f:
        nodes = f.readlines()
        j = 0
        for node in nodes:
            node = node.strip('\n')
            if j == 0:
                j = 1
                continue
            node_info = node.split(',')
            index_dict[node_info[0]] = len(index_dict)
            # features.append([float(i) for i in node_info[1:]])
            features.append([float(i) for i in node_info[1:-3]])
            # label = int(node_info[-3])
            # sriskr = float(node_info[-2]) if node_info[-2] != '' else np.nan
            # sriskv = float(node_info[-1]) if node_info[-1] != '' else np.nan
            # labels.append(label - 1)
            # sriskrs.append(sriskr)  
            # sriskvs.append(sriskv)
            # if np.isnan(sriskr):
            #     sriskms.append(0)
            # else:
            #     sriskms.append(1)

    with open(Edge, "r") as f:
        edges = f.readlines()
        j = 0
        for edge in edges:
            if j == 0:
                j = 1
                continue
            edge = edge.strip('\n')
            start, end, weight = edge.split(',')
            if start in index_dict and end in index_dict:
                edge_index.append([index_dict[start], index_dict[end]])
                edge_index.append([index_dict[end], index_dict[start]])
                weight = float(weight)
                edge_attr.append(weight)
                edge_attr.append(weight)

    with open(Target, "r") as f:
            targets = f.readlines()
            j = 0
            for target in targets:
                if j == 0:
                    j = 1
                    continue
                target = target.strip('\n')
                # label, sriskr, sriskv = target.split(',')[1:4]
                label, sriskr, sriskv = target.split(',')[-3:]
                label = int(label)
                sriskr = float(sriskr) if sriskr != '' else np.nan
                sriskv = float(sriskv) if sriskv != '' else np.nan
                labels.append(label - 1)
                sriskrs.append(sriskr)
                sriskvs.append(sriskv)
                if np.isnan(sriskr):
                    sriskms.append(0)
                else:
                    sriskms.append(1)
    

    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(features)
    edge_index = torch.LongTensor(edge_index)
    edge_attr = torch.FloatTensor(edge_attr)
    sriskrs = torch.FloatTensor(sriskrs)
    sriskvs = torch.FloatTensor(sriskvs)
    sriskms = torch.LongTensor(sriskms)

    sriskrs = sriskrs / 100
    sriskvs = (sriskvs - sriskvs[sriskms == 1].mean()
               ) / sriskvs[sriskms == 1].std()

    edge_attr = torch.unsqueeze(edge_attr, dim=0)
    edge_attr = edge_attr.permute(1, 0)

    features = (features - features.mean(dim=0, keepdims=True)) / \
        features.std(dim=0, keepdims=True)

    if (Scaling == "power"):
        features = torch.sign(features) * torch.abs(features) ** 1.2
    elif (Scaling == "minmax"):
        fmin = features.min(dim=0, keepdims=True).values
        fmax = features.max(dim=0, keepdims=True).values
        features = (features-fmin)/(fmax-fmin)

    return labels, features, edge_index, sriskrs, sriskvs, sriskms

def load_data_temporal(year_from, quarter_from, year_to, quarter_to, Scaling="power", dir='graph_data'):

    if (year_from > year_to or (year_from == year_to and quarter_from > quarter_to)):
        print("the input year and quarter error!")
        return [], [], [], []

    time_steps = 0
    labels_temporal = []
    features_temporal = []
    edge_index_temporal = []
    sriskrs_temporal = []
    sriskvs_temporal = []
    sriskms_temporal = []

    while (True):
        time_steps += 1
        labels, features, edge_index, sriskrs, sriskvs, sriskms = load_data_for_quarter(year_from, quarter_from, Scaling, dir)

        labels = torch.unsqueeze(labels, dim=0)
        features = torch.unsqueeze(features, dim=0)
        sriskrs = torch.unsqueeze(sriskrs, dim=0)
        sriskvs = torch.unsqueeze(sriskvs, dim=0)
        sriskms = torch.unsqueeze(sriskms, dim=0)

        labels_temporal.append(labels)
        features_temporal.append(features)
        edge_index_temporal.append(edge_index)
        sriskrs_temporal.append(sriskrs)
        sriskvs_temporal.append(sriskvs)
        sriskms_temporal.append(sriskms)

        if (year_from == year_to and quarter_from == quarter_to):
            break

        if (quarter_from == 4):
            quarter_from = 1
            year_from += 1
        else:
            quarter_from += 1

    labels_temporal = torch.cat(labels_temporal, dim=0)
    features_temporal = torch.cat(features_temporal, dim=0)
    sriskrs_temporal = torch.cat(sriskrs_temporal, dim=0)
    sriskvs_temporal = torch.cat(sriskvs_temporal, dim=0)
    sriskms_temporal = torch.cat(sriskms_temporal, dim=0)
    
    for i in range(len(edge_index_temporal)):
        edge_index_temporal[i] = edge_index_temporal[i].t().contiguous()

    return labels_temporal, features_temporal, edge_index_temporal, time_steps, sriskrs_temporal, sriskvs_temporal, sriskms_temporal

def load_contagion_list_temporal(year_from, quarter_from, year_to, quarter_to, dir='contagion_data'):
    time_steps = 0
    contagion_list_temporal = []

    if (year_from > year_to or (year_from == year_to and quarter_from > quarter_to)):
        print("the input year and quarter error!")
        return

    while (True):
        time_steps += 1

        file_path = "../../../datasets/{}/contagion_".format(dir) + \
            str(year_from) + "_" + str(quarter_from) + ".pth"

        contagion_list_temporal.append(torch.load(file_path))

        if (year_from == year_to and quarter_from == quarter_to):
            break

        if (quarter_from == 4):
            quarter_from = 1
            year_from += 1
        else:
            quarter_from += 1

    contagion_list_temporal = torch.stack(contagion_list_temporal)

    return contagion_list_temporal