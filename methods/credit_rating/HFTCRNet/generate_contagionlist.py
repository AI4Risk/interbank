import torch
import numpy as np
import random
import os
# import sys
from utils import load_data_for_quarter

def get_contagion_list(year_from, quarter_from, year_to, quarter_to):
    time_steps = 0
    base_dir = '../../../datasets/contagion_data'
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    if (year_from > year_to or (year_from == year_to and quarter_from > quarter_to)):
        print("the input year and quarter error!")
        return
    while (True):
        time_steps += 1
        contagion_list = get_contagion_list_for_quarter(
            year_from, quarter_from)
        file_path = base_dir + "/contagion_" + \
            str(year_from) + "_" + str(quarter_from) + ".pth"
        torch.save(contagion_list, file_path)
        ###
        print('year {} quarter {} done'.format(year_from,quarter_from))
        ###
        if (year_from == year_to and quarter_from == quarter_to):
            break
        if (quarter_from == 4):
            quarter_from = 1
            year_from += 1
        else:
            quarter_from += 1
        

def get_contagion_list_for_quarter(year, quarter):
    max_path_length = 8
    max_pathlist_length = 8
    edge_attr, single_edge_index = load_data_edgeWeight_quarter(year, quarter)
    sort_edges = sort_edges_by_weight(single_edge_index, edge_attr)
    adjacency = adjacency_list(sort_edges, max_path_length)
    all_paths = get_all_paths_from_edge_index(adjacency, single_edge_index.max(
    ).item() + 1, max_path_length, max_pathlist_length)
    all_paths = torch.IntTensor(all_paths)
    feature = load_data_for_quarter(year, quarter)[1]
    new_row = torch.full((1, 70), fill_value=0, dtype=torch.float)
    feature_extended = torch.cat((feature, new_row), dim=0)
    result = all_paths.clone()
    result = result.to(torch.float32)
    result = torch.unsqueeze(result, dim=3)
    result = result.expand(-1, -1, -1, 70)
    all_paths = all_paths.long()
    result = feature_extended[all_paths]

    return result

def load_data_edgeWeight_quarter(year, quarter, data_dir='graph_data'):
    if (quarter != 4):
        # Edge = "../../../datasets/{}/edge_Q/edge_".format(data_dir) + \
            # str(year) + "Q" + str(quarter) + ".csv"
        Edge = "../../../datasets/edges/edge_" + \
            str(year) + "Q" + str(quarter) + ".csv"
    elif (quarter == 4):
        # Edge = "../../../datasets/{}/edge_Q/edge_".format(data_dir) + \
        #     str(year) + "Q" + str(quarter) + ".csv"
        Edge = "../../../datasets/edges/edge_" + \
            str(year) + "Q" + str(quarter) + ".csv"
    edge_attr = []
    single_edge_index = []
    with open(Edge, "r") as f:
        edges = f.readlines()
        j = 0
        for edge in edges:
            if j == 0:
                j = 1
                continue
            edge = edge.strip('\n')
            start, end, weight = edge.split(',')
            weight = float(weight)
            start = int(start)
            end = int(end)
            single_edge_index.append([start, end])
            edge_attr.append(weight)
    edge_attr = torch.FloatTensor(edge_attr)
    single_edge_index = torch.IntTensor(single_edge_index)
    edge_attr = torch.unsqueeze(edge_attr, dim=0)
    edge_attr = edge_attr.permute(1, 0)
    return edge_attr, single_edge_index

def sort_edges_by_weight(edge_index, edge_weights):
    num_nodes = edge_index.max().item() + 1
    edge_groups = [[] for _ in range(num_nodes)]
    for i, (start, end) in enumerate(edge_index):
        weight = edge_weights[i]
        edge_groups[start].append((end, weight))
    for group in edge_groups:
        random.shuffle(group)
    return edge_groups

def get_all_paths_from_edge_index(truncated_list, num_nodes, max_path_length, max_pathlist_length):
    all_paths = []
    visited = [False] * num_nodes
    for node in range(num_nodes):
        current_path = []
        one_node_paths = []
        find_paths(truncated_list, node, visited, current_path,
                   one_node_paths, max_path_length)
        for sublist in one_node_paths:
            if len(sublist) < max_path_length:
                sublist.extend([4548] * (max_path_length - len(sublist)))
        if len(one_node_paths) < max_pathlist_length:
            missing_paths = max_pathlist_length - len(one_node_paths)
            one_node_paths.extend([[4548] * max_path_length] * missing_paths)
        all_paths.append(one_node_paths[0:max_pathlist_length])
    return all_paths

def adjacency_list(sorted_edge_groups, max_path_length):
    adj_list = []
    for edges in sorted_edge_groups:
        adj_list.append([(int)(end) for end, _ in edges])
    truncated_list = [sublist[:max_path_length] if len(
        sublist) > max_path_length else sublist for sublist in adj_list]
    return truncated_list

def find_path(graph, node, visited, current_path, one_node_paths, max_path_length):
    visited[node] = True
    current_path.append(node)
    current_length = len(current_path)
    if current_length == max_path_length or len(graph[node]) == 0:
        one_node_paths.append(current_path.copy())
        visited[node] = False
        current_path.pop()
        return True
    flag = False
    for neighbor in graph[node]:
        if not visited[neighbor]:
            flag = find_path(graph, neighbor, visited,
                             current_path, one_node_paths, max_path_length)
            if flag:
                break
    visited[node] = False
    current_path.pop()
    return flag

def find_paths(graph, node, visited, current_path, one_node_paths, max_path_length):
    current_path.append(node)
    for neighbor in graph[node]:
        find_path(graph, neighbor, visited, current_path,
                  one_node_paths, max_path_length)
    if len(graph[node]) == 0:
        one_node_paths.append(current_path)

if __name__ == '__main__':
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # _path = os.path.abspath(os.path.join(current_dir, '../../..'))
    # sys.path.append(_path)
    get_contagion_list(2016, 1, 2023, 1)