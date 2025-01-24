
import torch
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_random(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

def center_embedding(input, index, label_num):
    device=input.device
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index.unsqueeze(1).expand(-1, input.size(1)), src=input)
    class_counts = torch.bincount(index, minlength=label_num).unsqueeze(1).to(dtype=input.dtype, device=device)

    # Take the average embeddings for each class
    # If directly divided the variable 'c', maybe encountering zero values in 'class_counts', such as the class_counts=[[0.],[4.]]
    # So we need to judge every value in 'class_counts' one by one, and seperately divided them.
    # output_c = c/class_counts
    for i in range(label_num):
        if(class_counts[i].item()==0):
            continue
        else:
            c[i] /= class_counts[i]

    return c, class_counts


def act(x=None, act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        return torch.nn.LeakyReLU() if x is None else F.leaky_relu(x)
    elif act_type == 'tanh':
        return torch.nn.Tanh() if x is None else torch.tanh(x)
    elif act_type == 'relu':
        return torch.nn.ReLU() if x is None else F.relu(x)
    elif act_type == 'sigmoid':
        return torch.nn.Sigmoid() if x is None else torch.sigmoid(x)
    elif act_type == 'softmax':

        return torch.nn.Softmax(dim=-1) if x is None else F.softmax(x, dim=-1)
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")

def mask_edge(graph, mask_prob):
    E = graph.num_edges()
    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


# graph transformation: drop edge
def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = Data(edge_index=torch.concat((nsrc, ndst), 0))
    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def plot(tune_epochs, quarter_metrics, experiment, metric):
    """
    plot and save metric-epochs figures
    
    parameters:
    tune_epochs
    quarter_metrics:  dict()
        usage:quarter_metrics=dict();
             quarter_metrics[time_list[t+1]] = list();
             quarter_metrics[time_list[t+1]].appends(metric)
    experiment: string [experiment type] A1,B1,C1,without_fewshot...
    metric: string [metric type] accuracy/recall_r4/tolerance_error/norm_cost

    Return:
    None figures are at: results/{metric}-epochs.png
    """
    epochs = list(range(1, tune_epochs + 1))
    plt.figure(figsize=(60, 20))


    num_rows = 2
    num_cols = 4
    years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

    for i, year in enumerate(years):
        plt.subplot(num_rows, num_cols, i + 1)


        for quarter, metrics in quarter_metrics.items():
            if quarter.startswith(year):
                plt.plot(epochs, metrics, linestyle='-', label=f'Quarter {quarter}')

  
        plt.title(f'{metric} vs. Epochs for {year}')
        plt.xlabel('Epochs')
        plt.ylabel(metric)

        plt.xlim(0, tune_epochs+1)
        if metric=="norm_cost" or metric=="tolerance_error":
            plt.ylim()
        else:
            plt.ylim(0, 1.0)

        plt.legend() 
        plt.grid(True)  

   
    plt.tight_layout()
    plt.savefig(f'results/{experiment}/{metric}-epochs.png', dpi=300)