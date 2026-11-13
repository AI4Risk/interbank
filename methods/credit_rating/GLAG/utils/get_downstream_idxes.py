import math
import torch
import numpy as np
import random
import json



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
