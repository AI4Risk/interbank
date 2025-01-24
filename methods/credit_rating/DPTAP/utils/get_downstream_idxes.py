import math
import torch
import numpy as np
import random

def get_idxes(num_nodes, labels, shot_num, train_ratio=0.5):

    assert num_nodes == labels.shape[0]
    shuffled_indices = np.random.permutation(num_nodes)

    train_size = int(num_nodes * train_ratio)
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    train_labels = labels[train_indices]
    indices_per_class = {i: np.where(train_labels == i)[0].astype(int) for i in [0, 1, 2, 3]}
    min_len = min(len(indices) for indices in indices_per_class.values())

    selected_idxes = []
    remaining_idxes = []

    for i in range(len(indices_per_class)):
        class_idxes = indices_per_class[i]
        class_idxes = np.random.permutation(class_idxes)

        if min_len < shot_num:
            sel_num = min_len
            print(
                f"Not enough for class {i} to select {shot_num} finetuning samples, so only sample {sel_num}.")
        else:
            sel_num = shot_num

        selected_idxes.extend(train_indices[class_idxes[:sel_num]].tolist())
        remaining_idxes.extend(train_indices[class_idxes[sel_num:]].tolist())

    remaining_idxes.extend(test_indices.tolist())

    return selected_idxes, remaining_idxes, test_indices


def get_idxes_case_study(num_nodes, Q1_edge_index, Q2_edge_index, Q1_labels, Q2_labels, shot_num, train_ratio=0.5, n_folds=5):

    assert num_nodes == Q2_labels.shape[0]
    case_idx = 901 # 901-SBNY

    Q1_hop_1 = set()
    Q1_hop_2 = set()
    for edge in Q1_edge_index:
        if edge[0].item() == case_idx and edge[1] != case_idx:
            Q1_hop_1.add(edge[1].item())
    for edge in Q1_edge_index:
        if edge[0].item() in Q1_hop_1 and edge[0].item() != case_idx:
            Q1_hop_2.add(edge[1].item())
    Q1_neighbors = list(Q1_hop_1.union(Q1_hop_2)) # 1,2-order neighbors
    print("len(Q1_neighbors):", len(Q1_neighbors))

    Q2_hop_1 = set()
    Q2_hop_2 = set()
    for edge in Q2_edge_index:
        if edge[0].item() == case_idx and edge[1] != case_idx:
            Q2_hop_1.add(edge[1].item())
    for edge in Q2_edge_index:
        if edge[0].item() in Q2_hop_1 and edge[0].item() != case_idx:
            Q2_hop_2.add(edge[1].item())
    Q2_neighbors = list(Q2_hop_1.union(Q2_hop_2)) # 1,2-order neighbors
    print("len(Q2_neighbors):", len(Q2_neighbors))
    
    common_neighbors = [idx for idx in Q1_neighbors if idx in Q2_neighbors]
    print("len(common_neighbors)/len(Q2_neighbors):", len(common_neighbors)/len(Q2_neighbors))

    shuffled_indices = np.random.permutation(num_nodes) # 4548
    shuffled_idxes = [idx for idx in shuffled_indices if idx not in Q2_neighbors] # 3965
    
    num_nodes = len(shuffled_idxes)
    train_size = int(num_nodes * train_ratio)
    train_indices = shuffled_idxes[:train_size]
    test_indices = shuffled_idxes[train_size:]
    train_labels = Q2_labels[train_indices]
    indices_per_class = {i: np.where(train_labels == i)[0].astype(int) for i in [0, 1, 2, 3]}
    min_len = min(len(indices) for indices in indices_per_class.values())
    selected_idxes = []
    remaining_idxes = []
    for i in range(len(indices_per_class)):
        class_idxes = indices_per_class[i]
        class_idxes = np.random.permutation(class_idxes)
        if min_len < shot_num:
            sel_num = min_len
            print(
                f"Not enough for class {i} to select {shot_num} finetuning samples, so only sample {sel_num}.")
        else:
            sel_num = shot_num
        selected_idxes.extend(np.take(train_indices,class_idxes[:sel_num].tolist()))
        remaining_idxes.extend(np.take(train_indices,class_idxes[sel_num:].tolist()))
        
    remaining_idxes.extend(test_indices)
    
    random.shuffle(Q2_neighbors)
    fold_size = len(Q2_neighbors) // n_folds
    folds = [Q2_neighbors[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    
    sel_idx_list = [] # few-shot sample
    rmn_idx_list = [] # remaining nodes (instead of few-shot)
    drop_idx_list = [] # rating drop nodes in neighbors

    drop_all_idxes = [idx for idx in Q2_neighbors if Q2_labels[idx]>Q1_labels[idx]]
    drop_num = len(drop_all_idxes)
    print("drop_num:", drop_num)

    for i in range(n_folds):
        train_fold = folds[i]
        train_neighbors = [idx for idx in train_fold]
        rmn_folds = [fold for j, fold in enumerate(folds) if j != i]
        rmn_neighbors = [idx for fold in rmn_folds for idx in fold]

        ## case_study(a)
        # sel_idxes = [idx for idx in selected_idxes if idx not in rmn_neighbors]
        # sel_idxes.extend(train_neighbors)
        # sel_idxes = [idx for idx in sel_idxes if idx != case_idx]
        # sel_idxes = list(set(sel_idxes))
        ####

        ## case_study(b)
        sel_idxes = [idx for idx in selected_idxes if idx not in rmn_neighbors]
        sel_idxes.extend(train_neighbors)
        sel_idxes.append(case_idx)
        sel_idxes = list(set(sel_idxes))
        ####

        rmn_idxes = [idx for idx in shuffled_indices if idx not in sel_idxes]
        drop_idxes = [idx for idx in rmn_neighbors if Q2_labels[idx] > Q1_labels[idx]]

        assert case_idx in sel_idxes
        # assert case_idx not in sel_idxes  # case_study(a)
        
        # check class distribution
        # count = torch.zeros(4) 
        # for idx in sel_idxes:
            # count[Q2_labels[idx]] += 1
        # print(count)

        sel_idx_list.append(sel_idxes)
        rmn_idx_list.append(rmn_idxes)
        drop_idx_list.append(drop_idxes)

    return sel_idx_list, rmn_idx_list, drop_idx_list, Q2_neighbors


def get_idxes_imb_y(num_nodes, labels, shot_num, class_ratios, train_ratio=0.5):

    assert num_nodes == labels.shape[0]
    class_ratios_list = list(map(int, class_ratios.split()))
    # class_ratios_list = [33,33,33]  # for scenario with zero
    shot_num = 4 * shot_num  # 4-class

    shuffled_indices = np.random.permutation(num_nodes)
    train_size = int(num_nodes * train_ratio)
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    train_labels = labels[train_indices]
    indices_per_class = {i: np.where(train_labels == i)[0].astype(int) for i in [0, 1, 2, 3]}
    numList = [math.floor(len(indices_per_class[i]) / ratio * 100) if ratio != 0 else 0 for i, ratio in enumerate(class_ratios_list)]
    min_num = min(numList)
    sel_num = min(min_num, shot_num)

    selected_idxes = []
    remaining_idxes = []

    for i, ratio in enumerate(class_ratios_list):
        class_idxes = indices_per_class[i]
        class_idxes = np.random.permutation(class_idxes)
        num_selected = math.floor(sel_num * ratio / 100)

        if num_selected < int(shot_num * ratio / 100):
            print(f"Not enough for class {i} to select {int(shot_num * ratio / 100)} finetuning samples, so only sample {num_selected}.")

        selected_idxes.extend(class_idxes[:num_selected].tolist())
        remaining_idxes.extend(class_idxes[num_selected:].tolist())

    remaining_idxes.extend(test_indices.tolist())

    return selected_idxes, remaining_idxes, test_indices


def get_idxes_imb_d(num_nodes, edge_index, shot_num, dense_sparse_ratio, degree_threshold=5, train_ratio=0.5):

    dense_sparse_ratio_list = list(map(int, dense_sparse_ratio.split()))
    assert sum(dense_sparse_ratio_list) == 100
    assert degree_threshold > 0
    shot_num = 4 * shot_num  # 4-class

    degrees = np.zeros(num_nodes, dtype=int)
    for edge in edge_index:
        degrees[edge[0].item()] += 1
        degrees[edge[1].item()] += 1

    dense_nodes = np.where(degrees >= degree_threshold)[0]
    sparse_nodes = np.where(degrees < degree_threshold)[0]

    shuffled_indices = np.random.permutation(num_nodes)
    train_size = int(num_nodes * train_ratio)
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    train_dense_nodes = np.intersect1d(train_indices, dense_nodes)
    train_sparse_nodes = np.intersect1d(train_indices, sparse_nodes)

    num_dense = len(train_dense_nodes)
    num_sparse = len(train_sparse_nodes)

    min_num = min(math.floor(num_dense/dense_sparse_ratio_list[0]*100), math.floor(num_sparse/dense_sparse_ratio_list[1]*100))
    sel_num = min(min_num, shot_num)

    num_dense_selected = math.floor(sel_num * dense_sparse_ratio_list[0] / 100)
    num_sparse_selected = math.floor(sel_num * dense_sparse_ratio_list[1] / 100)

    if num_dense_selected < int(shot_num * dense_sparse_ratio_list[0] / 100):
        print(
            f"Not enough({int(shot_num * dense_sparse_ratio_list[0] / 100)}) dense nodes with degree_threshold {degree_threshold}, only sample {num_dense_selected}.")
    if num_sparse_selected < int(shot_num * dense_sparse_ratio_list[1] / 100):
        print(
            f"Not enough({int(shot_num * dense_sparse_ratio_list[1] / 100)}) sparse nodes with degree_threshold {degree_threshold}, only sample {num_sparse_selected}.")

    selected_dense_idxes = np.random.choice(train_dense_nodes, num_dense_selected, replace=False)
    selected_sparse_idxes = np.random.choice(train_sparse_nodes, num_sparse_selected, replace=False)
    selected_idxes = np.concatenate((selected_dense_idxes, selected_sparse_idxes))
    remaining_idxes = np.setdiff1d(train_indices, selected_idxes)
    remaining_idxes = np.concatenate((remaining_idxes, test_indices))

    return selected_idxes, remaining_idxes, test_indices


def substitute_(features, labels, shot_num):
    """
    two quarters of features and labels
    """
    assert features.shape[0] == labels.shape[0]
    Q_num_nodes = int(0.5 * features.shape[0])
    Q1_features, Q2_features = features[:Q_num_nodes], features[Q_num_nodes:]
    Q1_labels, Q2_labels = labels[:Q_num_nodes], labels[Q_num_nodes:]
    assert Q1_features.shape[0] == Q2_features.shape[0] and Q1_labels.shape[0] == Q2_labels.shape[0]

    indices_per_class = {i: np.where(Q2_labels == i)[0].astype(int) for i in [0, 1, 2, 3]}
    # Q2_selected_idxes=[]
    selected_idxes = []
    remaining_idxes = []
    for i in range(len(indices_per_class)):
        class_idxes = indices_per_class[i]
        class_idxes = np.random.permutation(class_idxes)
        # print("1")
        if len(class_idxes) < shot_num:
            raise ValueError(f"Not enough for class {i} to select {shot_num} finetuning samples.")
        selected_idxes.extend(class_idxes[:shot_num].tolist())
        remaining_idxes.extend(class_idxes[shot_num:].tolist())


    substitute_nodes = Q2_features[selected_idxes].clone()
    Q1_features[selected_idxes] = substitute_nodes


    Q1_labels = Q2_labels.clone()

    selected_idxes = torch.tensor(selected_idxes, dtype=torch.long)
    remaining_idxes = torch.tensor(remaining_idxes, dtype=torch.long)

    return Q1_features, Q1_labels, selected_idxes, remaining_idxes