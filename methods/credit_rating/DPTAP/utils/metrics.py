

import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from scipy.integrate import trapz

def tolerance_metric(matrix, tolerance=1):

    total_samples = matrix.sum()
    correct_with_tolerance = 0

    correct_with_tolerance += np.trace(matrix)


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and abs(i - j) <= tolerance:
                correct_with_tolerance += matrix[i, j]
    

    accuracy = correct_with_tolerance / total_samples
    error_rate = 1 - accuracy

    return accuracy,error_rate



def create_cost_matrix(n, a=2, b=2):

    M_cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M_cost[i, j] = dire(i, j, a) * dist(i, j, b)
            # print(M_cost[i, j])
        # print("\n")
    return M_cost

def dist(i, j, n=2): 
    return abs(i - j) ** n

def dire(i, j, k=2): 
    if i < j:
        return 1
    elif i > j:
        return k
    else:
        return 0


def compute_norm_cost(M_conf):

    p_dire=2
    p_dist=2
    sample_size = np.sum(M_conf)

    class_num = M_conf.shape[0]

    M_cost = create_cost_matrix(class_num,p_dire,p_dist)
    C_mul = M_cost * M_conf
    max_cost = np.max(M_cost)
    C_norm = np.sum(C_mul) / (sample_size * max_cost)
    return C_norm


def recall_r4_at_k(logits,true,k=2):

    probs = F.softmax(logits,dim=-1)
  
    sorted_probs_idx = torch.argsort(probs,dim=1,descending=True)
    assert logits.shape[0]==true.shape[0]
    TP=0
    FN=0
    for i in range(logits.shape[0]):
        if true[i]==3:
            if 3 in sorted_probs_idx[i,0:k]:
                TP+=1
            else:
                FN+=1
    recall = TP/(TP+FN)
    return recall

