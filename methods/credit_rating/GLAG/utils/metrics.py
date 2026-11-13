import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# from scipy.integrate import trapz

def FAcc(y_true,y_pred, v_safe=1, v_risky=0):
    """

    可以把这个指标命名为 "Prudential Accuracy" (审慎准确率) 或者 "Risk-Aware Fuzzy Accuracy"。

    :param y_pred,y_true: 
    :param v_safe: 高估风险时的容忍度 (Prediction > Truth)。建议设为 1。
    :param v_risky: 低估风险时的容忍度 (Prediction < Truth)。建议设为 0 (严格)。
    """
    matrix = confusion_matrix(y_true,y_pred)

    total_samples = matrix.sum()
    correct_fuzzy = 0
    
    rows, cols = matrix.shape # rows=True, cols=Pred
    
    for i in range(rows):      # i: True Label
        for j in range(cols):  # j: Predicted Label
            
            # 情况 1: 预测值 >= 真实值 (Over-prediction or Exact match)
            # 这里的方向是“安全”的，我们使用宽松的 v_safe
            if j >= i and (j - i) <= v_safe:
                correct_fuzzy += matrix[i, j]
                
            # 情况 2: 预测值 < 真实值 (Under-prediction)
            # 这里的方向是“危险”的，我们使用严格的 v_risky
            elif j < i and (i - j) <= v_risky:
                correct_fuzzy += matrix[i, j]
                
    acc = correct_fuzzy / total_samples
    return acc

def RiskCaptureRecall_k(logits,true,k=2):
    '''
    RiskCaptureRecall_k 的 Docstring
    指标：RCR@k
    :param logits: 说明
    :param true: 说明
    :param k: 说明
    '''
    probs = F.softmax(torch.FloatTensor(logits),dim=-1)
    
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

# def Recall_down(y_prev, y_curr_true, y_curr_pred):
#     """
#     方向性降级召回率 (Directional Downgrade Recall)
    
#     含义：在所有真实发生评级下降的样本中，模型正确预测出“会下降”的比例。
#     只要预测值比上一季度大（变坏），就算检出成功，不要求预测值和真实值完全相等。
#     """
#     # 1. 找出真实发生降级的样本索引 (Ground Truth Downgrades)
#     # 假设标签越大越差 (0:Good -> 3:Bad)
#     gt_downgrade_mask = (y_curr_true > y_prev)
    
#     # 如果本季度没有任何真实降级样本，返回 None 或 0
#     if np.sum(gt_downgrade_mask) == 0:
#         return 0.0
    
#     # 2. 获取这些样本的预测值和上一季度标签
#     relevant_preds = y_curr_pred[gt_downgrade_mask]
#     relevant_prevs = y_prev[gt_downgrade_mask]
    
#     # 3. 判定是否“检出”：预测值也显示恶化 (Pred > Prev)
#     # 这就是你说的：方向正确，就+1
#     successful_captures = (relevant_preds > relevant_prevs)
    
#     # 4. 计算召回率
#     recall = np.sum(successful_captures) / np.sum(gt_downgrade_mask)
    
#     return recall


# def MDA(y_prev, y_true, y_pred):
#     """ 
#     Mean Directional Accuracy (Relative to Previous State)
#     计算预测的变动方向是否等于真实的变动方向 (相对于上一季度)
    
#     Args:
#         y_prev: 上一季度的真实标签
#         y_true: 本季度的真实标签
#         y_pred: 本季度的预测标签
#     """
#     # 计算真实变动方向 (-1, 0, 1)
#     direction_true = np.sign(y_true - y_prev)
#     # 计算预测变动方向 (-1, 0, 1)
#     direction_pred = np.sign(y_pred - y_prev)
    
#     # 计算方向一致的比例
#     return np.mean((direction_true == direction_pred).astype(int))


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


def CSCE(M_conf):
    p_dire=2
    p_dist=2
    sample_size = np.sum(M_conf)

    class_num = M_conf.shape[0]

    M_cost = create_cost_matrix(class_num,p_dire,p_dist)
    C_mul = M_cost * M_conf
    max_cost = np.max(M_cost)
    C_norm = np.sum(C_mul) / (sample_size * max_cost)
    return C_norm




