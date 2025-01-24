
import torch.nn.functional as F
import torch

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

# loss function: sig
def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss

def kl_divergence(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)

    kl_div = (p * (p / q).log()).sum(dim=-1)
    return kl_div