
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot


class DirectionPrompt(nn.Module):
    def __init__(self, dim: int):
        super(DirectionPrompt, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(1, dim))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weights)

    def compute(self, x: Tensor):

        return x * self.weights



class DistancePrompt(nn.Module):
    def __init__(self, dim: int):
        super(DistancePrompt, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(1, dim))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weights)

    def compute(self ,x :Tensor):
  
        return x * self.weights



class AugmentPrompt(nn.Module):
    def __init__(self, dim: int, num: int):
        super(AugmentPrompt, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(dim, num))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weights)

    def compute(self, x: Tensor):
        attn_scores = F.softmax(torch.matmul(x, self.weights), dim=-1)
        weighted_prompts = torch.matmul(attn_scores, self.weights.T)  # (batch_size, dim)
        return x + weighted_prompts

class AdaptPrompt(nn.Module):
    def __init__(self, dim: int, num: int):
        super(AdaptPrompt, self).__init__()
        self.num = num
        self.dim = dim


    def compute(self, Q1_input, Q2_input, selected_idxes, remaining_idxes):
        Q1_x_few = Q1_input.x[selected_idxes].clone()
        Q1_y_few = Q1_input.y[selected_idxes].clone()
        Q2_x_few = Q2_input.x[selected_idxes].clone()
        Q1_x_rmn = Q1_input.x[remaining_idxes].clone()
        delta = torch.zeros((self.num, self.dim)).to(Q1_x_rmn.device)
        class_centers = torch.vstack([Q1_x_few[Q1_y_few == i].mean(dim=0) for i in range(self.num)])
        sim = F.cosine_similarity(Q1_x_rmn.unsqueeze(1),class_centers.unsqueeze(0),dim=-1)
        sim = torch.softmax(sim, dim=-1)
        for i in range(self.num):
            mask_i = (Q1_y_few == i)
            delta_i = Q2_x_few[mask_i] - Q1_x_few[mask_i]
            delta[i] = delta_i.mean(dim=0)

        Q1_x_rmn += torch.matmul(sim, delta)

        return Q1_x_rmn
