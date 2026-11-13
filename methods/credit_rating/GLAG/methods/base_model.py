import torch
from torch import nn
from .act import act
import torch.nn.functional as F
from . import GAT,GCN,GraphTransformer
from .rev_gnn import RevGNN
# from models.TGAR import TGAR


import os

class PretrainModel(nn.Module):
    def __init__(self,args):
        super(PretrainModel,self).__init__()
        self.args=args
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        if args.gnn_type=="GAT":
            self.gnn = GAT(input_dim=args.input_dim,hid_dim=args.hid_dim,\
            num_layer=args.num_layer,drop_ratio=args.droprate).to(device)
        elif args.gnn_type=="GCN":
            self.gnn = GCN(input_dim=args.input_dim,hid_dim=args.hid_dim,\
            num_layer=args.num_layer,drop_ratio=args.droprate).to(device)
        # elif args.gnn_type=="TGAR":
        #     self.gnn = TGAR(args,1,4).to(device)
        elif args.gnn_type=="GT":
            self.gnn = GraphTransformer(input_dim=args.input_dim,hid_dim=args.hid_dim,\
            num_layer=args.num_layer,drop_ratio=args.droprate).to(device)
        elif args.gnn_type=='RevGNN':
            self.gnn = RevGNN(
                in_channels=args.input_dim,
                hidden_channels=args.hid_dim,
                out_channels=args.hid_dim,
                num_layers=args.num_layer,
                dropout=args.droprate,
                num_groups=2
            )
        

        self.class_head = nn.Sequential(
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(),
            nn.Dropout(p=args.droprate),
            nn.Linear(args.hid_dim, args.num_class)
        )
        # self.load_model_weights(args.weights_path)

    def forward(self,x,edge_index,batch=None):
        node_emb = self.gnn(x,edge_index,batch)
        out = self.class_head(node_emb)
        return out

    def load_model_weights(self, weights_path):
        if os.path.exists(weights_path):
            model_weights = torch.load(weights_path)
            self.gnn.load_state_dict(model_weights['gnn'])
            self.class_head.load_state_dict(model_weights['class_head'])
            print("model weights from last quarter loaded.")
        else:
            print(f"Warning: gnn weights file not found at {weights_path}. Skipping loading.")



class FinetuneModel(nn.Module):
    def __init__(self,args):
        super(FinetuneModel, self).__init__()
        self.args=args
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        if args.gnn_type=="GAT":
            self.gnn = GAT(input_dim=args.input_dim,hid_dim=args.hid_dim,\
            num_layer=args.num_layer,drop_ratio=args.droprate).to(device)
        elif args.gnn_type=="GCN":
            self.gnn = GCN(input_dim=args.input_dim,hid_dim=args.hid_dim,\
            num_layer=args.num_layer,drop_ratio=args.droprate).to(device)
        # elif args.gnn_type=="TGAR":
        #     self.gnn = TGAR(args,1,4).to(device)
        elif args.gnn_type=="GT":
            self.gnn = GraphTransformer(input_dim=args.input_dim,hid_dim=args.hid_dim,\
            num_layer=args.num_layer,drop_ratio=args.droprate).to(device)
        elif args.gnn_type=='RevGNN':
            self.gnn = RevGNN(
                in_channels=args.input_dim,
                hidden_channels=args.hid_dim,
                out_channels=args.hid_dim,
                num_layers=args.num_layer,
                dropout=args.droprate,
                num_groups=2
            )

        # MLP Head (Linear -> ReLU -> Dropout -> Linear)
        self.class_head = nn.Sequential(
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(),
            nn.Dropout(p=args.droprate),           
            nn.Linear(args.hid_dim, args.num_class)             
        )
        
        # load 
        # self.load_model_weights(args.weights_path)
        

    def forward(self,x,edge_index,batch=None):
        node_emb = self.gnn(x,edge_index,batch)
        out = self.class_head(node_emb)
        return out

    def load_model_weights(self, weights_path):
        if os.path.exists(weights_path):
            model_weights = torch.load(weights_path, weights_only=True)
            

            self.gnn.load_state_dict(model_weights['gnn'])
            print("GNN backbone weights loaded.")


            try:
                self.class_head.load_state_dict(model_weights['class_head'])
                print("Class head weights loaded.")
            except:
                print("Class head structure changed or mismatch. Initializing class head from scratch (Recommended for fine-tuning).")
                
        else:
            print(f"Warning: gnn weights file not found at {weights_path}. Skipping loading.")
