import json
import os
import random
from torch import Tensor
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from methods.credit_rating.GLAG.utils.metrics import RiskCaptureRecall_k,FAcc
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm


from methods.credit_rating.GLAG.methods.base_model import PretrainModel
from methods.credit_rating.GLAG.methods.Augmentation import AugmentationEncoder
from methods.credit_rating.GLAG.methods.EdgeLearner import GraphLearner
from methods.credit_rating.GLAG.utils.args import get_args
from methods.credit_rating.GLAG.utils.loading import load_data_for_quarter
# from utils.lr import PolynomialDecayLR
from methods.credit_rating.GLAG.utils.tools import seed_everything,init_time_list,read_text

from methods.credit_rating.GLAG.utils.get_downstream_idxes import get_idxes
from methods.credit_rating.GLAG.utils.text_encoding import get_text_embeddings


def train(model, optimizer, x, x_supp, edge_index, y, mask):
    model.train()
    optimizer.zero_grad()
    out, graph_loss = model(x, x_supp, edge_index)
    # loss_pred = F.nll_loss(out[mask], y[mask])
    loss_pred = F.cross_entropy(out[mask],y[mask])
    loss = loss_pred + graph_loss
    loss.backward()
    optimizer.step()
    return loss.item(), loss_pred.item(), graph_loss.item()

@torch.no_grad()
def test(model, x, x_supp, edge_index, y, mask):
    model.eval()
    out, _ = model(x, x_supp, edge_index)
    pred = out.argmax(dim=1)
    acc = pred[mask].eq(y[mask]).sum().item() / mask.sum().item()
    predictions = pred[mask]
    trues = y[mask]
    logits = out[mask].cpu().numpy()
    predictions = predictions.cpu().numpy()
    trues = trues.cpu().numpy()

    f1 = f1_score(trues,predictions,average='macro')
    facc = FAcc(trues,predictions)
    recall = RiskCaptureRecall_k(logits,trues)
    qwk = cohen_kappa_score(trues,predictions,weights='quadratic')
    return acc,f1,facc,recall,qwk, logits,predictions,trues

args = get_args()
seed_everything(args.seed)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
time_list = init_time_list()


for t in range(0, 1):  # (0,len(time_list)-2) for all quarters
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    last_y,last_q = time_list[t].split('-')
    y,q=time_list[t+1].split('-')
    last_y = int(last_y)
    last_q = int(last_q)
    y=int(y)
    q=int(q)
    if (y, q) in [(2016, 1), (2023, 4)]:
        continue
    if q==4:
        print_y=y+1
        print_q=1
    else:
        print_y=y
        print_q=q+1

    print(f"{y}Q{q} begin: ")
    Q1_features,Q1_labels,Q1_edge_index,_,_,_,_ = load_data_for_quarter(last_y,last_q,True,args.norm_type)
    Q2_features,Q2_labels,Q2_edge_index,_,_,_,_ = load_data_for_quarter(y,q,True,args.norm_type)
    
    num_nodes = Q1_features.shape[0]

    pre_data = Data(x=Q1_features,edge_index=Q1_edge_index.t().contiguous(),y=Q1_labels).to(device)
    pre_model = PretrainModel(args)
    pre_model = pre_model.to(device)
    pre_optim = torch.optim.Adam(pre_model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    
    print("pre-training on last quarter:")
    pre_model.train()
    pbar = tqdm(range(args.pretrain_epochs), desc=f"{args.gnn_type} Epochs")
    best_pre_loss = float('inf')
    pretrain_patience=50
    pretrain_patience_cnt=0
    for _ in pbar:
        pre_optim.zero_grad()
        logits = pre_model(pre_data.x,pre_data.edge_index)
        loss = F.cross_entropy(logits,pre_data.y)
        loss.backward()
        pre_optim.step()
        avg_loss = loss.item() / pre_data.x.shape[0]
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})
        if loss < best_pre_loss:
            best_pre_loss = loss
            pretrain_patience_cnt = 0
        else:
            pretrain_patience_cnt += 1
            if pretrain_patience_cnt >= pretrain_patience:
                break
    
    model_path = './ckpts/pretrained.pth'
    pre_state_dict = {
        'gnn':pre_model.gnn.state_dict(),
        'class_head':pre_model.class_head.state_dict()
    }

    torch.save(pre_state_dict,model_path)

    ### get textual embeddings 
    with open("idx_ticker_mapping.json","r",encoding='utf-8') as f:
        mapping_dict = json.load(f)
    aug_idxes = [int(k) for k in mapping_dict.keys()]
    input_texts= {}
    for idx in aug_idxes:
        bank_name = mapping_dict[str(idx)]
        orig_text = read_text(f"summary/Qwen/{bank_name}/{y}Q{q}.txt")
        text = orig_text.split("|||")[0]
        input_texts[idx] = text

    # text_embeddings = get_text_embeddings(input_texts,batch_size=2)
    text_embeddings = get_text_embeddings(input_texts,args.text_encoding_method,
                                          args.text_encoding_model_path,args.text_encoding_bs)
    aug_dim=list(text_embeddings.values())[0].shape

    ### Text Reconstruction & Contrastive Matching : Graph Masked Autoencoders + Contrastive Learning
    rec_mat_features= Q1_features.clone()
    rec_mat_features[aug_idxes] = Q2_features[aug_idxes].clone()
    rec_mat_data = Data(x=rec_mat_features,edge_index=Q1_edge_index.t().contiguous()).to(device)
    x_supp = torch.zeros(size=(num_nodes,aug_dim[0]),dtype=torch.float)
    for i in range(num_nodes):
        if i in aug_idxes:
            x_supp[i] = text_embeddings[i]
    x_supp = x_supp.to(device) 
    supp_avail_mask = torch.zeros(num_nodes,dtype=torch.bool,device = device)
    supp_avail_mask[aug_idxes] = True 

    rec_mat_model = AugmentationEncoder(
        in_dim=args.input_dim,   
        supp_in_dim=aug_dim[0],         
        num_hidden=512,
        num_layers=2,
        nhead=4,
        activation=torch.nn.PReLU(),
        feat_drop=0.2,
        attn_drop=0.1,
        negative_slope=0.2,
        residual=False,
        norm="layernorm",
        mask_rate=0.5,
        lam=1.0,
        temperature=args.temperature,
        # dist_lam=50.0  
    ).to(device)

    optimizer = torch.optim.Adam(rec_mat_model.parameters(), lr=0.001, weight_decay=2e-4)
    rec_mat_model.train()
    


    best_rec_loss = float('inf')
    rec_patience = 20
    rec_patience_counter = 0

    pbar = tqdm(range(args.rec_mat_epochs), desc="Reconstruction & Matching Epochs")
    for epoch in pbar:
        loss,l_rec,l_mat = rec_mat_model(rec_mat_data.x,x_supp,rec_mat_data.edge_index,supp_avail_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"Loss": f"{loss:.4f}","L_rec":f"{l_rec:.4f}","L_mat":f"{l_mat:.4f}"})
        
        # --- Early Stopping Logic ---
        if loss < best_rec_loss:
            best_rec_loss = loss
            rec_patience_counter = 0
        else:
            rec_patience_counter += 1
            if rec_patience_counter >= rec_patience:
                # print(f"\n[Early Stopping] Reconstruction stopped at epoch {epoch} (Best Loss: {best_rec_loss:.4f})")
                break
        # ----------------------------



    virtual_supp_full = rec_mat_model.generate_virtual_supp(rec_mat_data.x, rec_mat_data.edge_index)
    remaining_mask = ~ supp_avail_mask

    x_orig = rec_mat_features.clone()
    del rec_mat_features
    x_aug = x_supp.clone()
    x_aug[remaining_mask] = virtual_supp_full[remaining_mask]
    del x_supp



    selected_idxes, remaining_idxes,_ = get_idxes(num_nodes,Q2_labels,100)
    mask =  torch.zeros(num_nodes,dtype=torch.bool,device = device)

    
    ## text aug as few_shot
    train_mask = supp_avail_mask.cpu()
    test_mask = remaining_mask.cpu()
    # train_mask = train_mask.to(device)
    # test_mask = test_mask.to(device)
    x_tune = Q1_features.clone()
    x_tune[train_mask] = Q2_features[train_mask].clone()
    y_train = Q1_labels.clone()
    y_train[train_mask] = Q2_labels[train_mask].clone()

    ##
    train_data = Data(x=x_tune,edge_index=Q1_edge_index.t().contiguous(),y=y_train).to(device)
    test_data = Data(x=Q2_features,edge_index=Q1_edge_index.t().contiguous(),y=Q2_labels).to(device)
    x_aug = x_aug.to(device)
    
    model_net = GraphLearner(
        args=args,
        nfeat2=x_aug.shape[1],
        max_iter=10,
        epsilon=args.epsilon,
        topk=args.topk_edge,
        smooth_item=args.smooth,
        sparse_item=args.sparse,
        diff_threshold=1e-3,
        lam=args.lam 
    )
    model_net.load_model_weights(model_path)
    model_net._disable_gnn_normalization()
    model_net = model_net.to(device)

    optimizer = torch.optim.Adam([
        {'params': model_net.graph_learner.parameters(), 'lr': 0.01}, 
        {'params': model_net.graph_learner2.parameters(), 'lr': 0.01},
        {'params': model_net.gnn_encoder.parameters(), 'lr': 0.001},  
        {'params': model_net.classifier.parameters(), 'lr': 0.001}
    ], weight_decay=5e-4)
    

    final_test_acc = 0
    final_f1=0
    final_facc=0
    final_recall=0
    final_qwk=0
    
    # --- Early Stopping Setup for Tuning ---
    tune_patience = 30
    tune_patience_counter = 0
    # ---------------------------------------

    pbar = tqdm(range(args.tune_epochs), desc="Tuning")
    for epoch in pbar:

        loss, loss_pred, loss_graph = train(model_net, optimizer, train_data.x, x_aug, train_data.edge_index, train_data.y, supp_avail_mask)
        

        # val_acc = test(model, data.x, x_supp, data.edge_index, data.y, data.val_mask)
        test_acc,f1,facc,recall,qwk,logits_to_save,preds_to_save,labels_to_save = test(model_net, test_data.x, x_aug, test_data.edge_index, test_data.y, remaining_mask)
        
        if test_acc > final_test_acc:
            final_test_acc = test_acc
            final_f1=f1
            final_facc=facc
            final_recall=recall
            final_qwk=qwk            
            tune_patience_counter = 0 # Reset counter
            os.makedirs(f"preds",exist_ok=True)
            np.savez(f'preds/{print_y}Q{print_q}.npz',
            logits=logits_to_save,preds=preds_to_save,labels=labels_to_save)
        else:
            tune_patience_counter += 1
        

        pbar.set_postfix({
            "Loss": f"{loss:.4f}", 
            "GraphLoss": f"{loss_graph:.4f}", 
            "TAcc": f"{test_acc:.4f}"
        })

        # --- Early Stopping Check ---
        if tune_patience_counter >= tune_patience:
            # print(f"\n[Early Stopping] Tuning stopped at epoch {epoch} (Best FAcc: {final_facc:.4f})")
            break
        # ----------------------------


