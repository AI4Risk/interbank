import torch
import scipy.sparse as sp
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops
import scipy.sparse.linalg as spla
from torch_geometric.utils import  to_undirected
import random
import os


def seed_everything(seed: int = 42):
    # Python
    random.seed(seed)
    # OS environment
    os.environ["PYTHONHASHSEED"] = str(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU / Multi-GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(f"[Info] Global seed set to {seed}")

def init_time_list():
    years=[2016,2017,2018,2019,2020,2021,2022,2023]
    quarters=[1,2,3,4]
    time_str=[]
    for y in years:
        for q in quarters:
            time_str.append(f"{y}-{q}")
    return time_str

def read_text(filename):
    with open(filename,'r',encoding='utf-8') as f:
        text = f.read()
    return text



def edge_index_to_sparse_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Convert a dense edge_index to a coalesced sparse adjacency matrix."""
    undirected_edge_index = to_undirected(edge_index)
    values = torch.ones(undirected_edge_index.size(1), device=device)
    adj = torch.sparse_coo_tensor(undirected_edge_index, values, (num_nodes, num_nodes), device=device)
    return adj.coalesce()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.FloatTensor(indices, values, shape)
    return torch.sparse_coo_tensor(indices,values,shape)

def create_normalized_adjacency(edge_index, num_nodes):
    A = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).astype(float)
    A = A + sp.eye(num_nodes)  # Â = A + I
    rowsum = np.array(A.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    A_normalized = d_mat_inv_sqrt @ A @ d_mat_inv_sqrt
    return sparse_mx_to_torch_sparse_tensor(A_normalized)

def laplacian_positional_encoding(edge_index, num_nodes, pos_enc_dim):
    A = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).astype(float)
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = np.power(degrees, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = sp.diags(D_inv_sqrt, format='csr')
    L = sp.eye(num_nodes) - D_mat_inv_sqrt @ A @ D_mat_inv_sqrt
    eigvals, eigvecs = spla.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    idx = eigvals.argsort()
    eigvecs = eigvecs[:, idx]
    pe = torch.from_numpy(eigvecs[:, 1:pos_enc_dim+1].real).float()
    return pe

def re_features(adj, features, K):
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])
    for i in range(features.shape[0]):
        nodes_features[i, 0, 0, :] = features[i]
    x = features + torch.zeros_like(features)
    for i in range(K):
        x = torch.matmul(adj, x)
        for index in range(features.shape[0]):
            nodes_features[index, 0, i + 1, :] = x[index]        
    nodes_features = nodes_features.squeeze()
    return nodes_features

