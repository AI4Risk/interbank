import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Args.')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--edge_update_method", type=str, default="HES_GSL", help="edge update baseline method name")
    parser.add_argument("--text_encoding_method", type=str, default="Qwen",choices=["Qwen","word2vec","Finbert", "Sentence-BERT"])
    parser.add_argument("--text_encoding_model_path", type=str, default="embedding_models/")
    parser.add_argument("--text_encoding_bs", type=int, default=32)
    parser.add_argument("--gnn_type", type=str, default="GT", help="type of GNN for backbone to use,choice = GCN, GAT, GT")
    parser.add_argument("--num_layer", type=int, default=2, help="layer num for gnn")
    parser.add_argument("--input_dim", type=int, default=70, help="input dimension")
    parser.add_argument("--hid_dim", type=int, default=256, help="hidden dimension")
    parser.add_argument("--num_class", type=int, default=4, help="hidden dimension")
    parser.add_argument("--norm_type",type=str,default='power')
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate ")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay ")
    parser.add_argument("--pretrain_epochs", type=int, default=1000, help="number of epochs in update.")
    parser.add_argument("--rec_mat_epochs", type=int, default=200, help="reconstruction & matching epochs")
    # parser.add_argument("--net_update_epochs", type=int, default=500, help="net update epochs")
    parser.add_argument("--tune_epochs", type=int, default=100, help="number of epochs in finetune.")
    parser.add_argument('--droprate', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    
    # parser.add_argument('--shot_num', type=int, default=100, help='finetuning samples number per class.')
    parser.add_argument('--backbone_trainable', type=bool, default=False, help='whether the backbone is trainable.')
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--exp', type=str, default="epsilon")
    ## construction & matching
    parser.add_argument('--temperature', type=float, default=0.5)
    
    ## edge
    parser.add_argument('--epsilon', type=float, default=0.4)
    parser.add_argument('--topk_edge', type=int, default=3)
    parser.add_argument('--lam', type=float, default=0.8)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--smooth', type=float, default=0.2)
    parser.add_argument('--sparse', type=float, default=0.3)

    args = parser.parse_args()
    return args

