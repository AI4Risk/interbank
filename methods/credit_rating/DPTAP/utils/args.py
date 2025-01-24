import argparse


def get_args():
    parser = argparse.ArgumentParser(description='implementation of C1')
    parser.add_argument('--device', type=int, default=0,
                        help='Which gpu to use if any (default: 0)')
    parser.add_argument('--gnn_type', type=str, default="GT",
                        help='We support gnn like \GCN\ \GAT\ \GT\ \GCov\ \GIN\ \GraphSAGE\, please read ProG.model module')
    parser.add_argument('--input_dim', type=int, default=70,
                        help='feature input dim.')
    parser.add_argument('--hid_dim', type=int, default=256,
                        help='hideen layer of GNN dimensions (default: 256)')
    parser.add_argument('--output_dim', type=int, default=4, help="number of class.")
    parser.add_argument('--batch_size', type=int, default=4548,
                        help='Input batch size for training (default: 4548)')
    parser.add_argument('--gnn_trainable', type=bool, default=False, help='whether the gnn is trainable.')
    parser.add_argument('--update_epochs', type=int, default=300,
                        help='Number of epochs to pretrain (without: 300, tune: 300)')
    parser.add_argument('--tune_epochs', type=int, default=500,
                        help='Number of epochs to prompt tune (tune: 500)')
    parser.add_argument('--exp_obj', type=str, default=None,
                        help='shot_num/class_ratios/dense_sparse_ratios/aug_num/hid_dim/update_epochs')
    parser.add_argument('--shot_num', type=int, default=100, help='Number of shots')
    parser.add_argument('--labeled_num', type=int, default=100, help='Number of labeled data in training set')
    parser.add_argument('--class_ratios', type=str, default="25 25 25 25", help='class ratios')
    parser.add_argument('--degree_threshold', type=int, default=6, help='degree_threshold (default: 6)')
    parser.add_argument('--dense_sparse_ratios', type=str, default="50 50", help='dense_sparse_ratios')
    parser.add_argument('--aug_num', type=int, default=5, help='Number of prompts for augment(input_dim)')
    parser.add_argument('--aug_num_in', type=int, default=5, help='Number of prompts for augment(input_dim), default(5)')
    parser.add_argument('--aug_num_hid', type=int, default=5, help='Number of prompts for augment(hid_dim), default(5)')
    parser.add_argument('--model_path', type=str, default="pretrained_gnn/weights.pth",
                        help='pretrained GNN model weights from the last quarter')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate ')
    parser.add_argument('--lr_adapt', type=float, default=0.001, help='Learning rate of adaptation (tune: 0.01)')
    parser.add_argument('--lr_pretrain', type=float, default=0.001, help='Learning rate of pretrain')
    parser.add_argument('--lr_classify', type=float, default=0.001, help='Learning rate ')
    parser.add_argument('--decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')
    parser.add_argument('--decay_adapt', type=float, default=1e-2,
                        help='Weight decay of adaptation')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='Number of GNN message passing layers (default: 2).')

    parser.add_argument('--droprate', type=float, default=0.5,
                        help='Dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='Graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='How the node features across layers are combined. last, sum, max or concat')

    parser.add_argument('--seed', type=int, default=43, help="Seed for experiment.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for running experiments.")

    args = parser.parse_args()
    return args

