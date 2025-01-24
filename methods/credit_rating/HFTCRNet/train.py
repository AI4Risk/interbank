import torch
import numpy as np
import argparse
import torch
from utils import load_data_temporal, load_contagion_list_temporal
from tqdm import trange
from model import HFTCRNet, Model

def init():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_year_from', type=int, default=2019)
    parser.add_argument('--train_quarter_from', type=int, default=2)
    parser.add_argument('--train_year_to', type=int, default=2021)
    parser.add_argument('--train_quarter_to', type=int, default=1)
    parser.add_argument('--test_year_from', type=int, default=2019)
    parser.add_argument('--test_quarter_from', type=int, default=3)
    parser.add_argument('--test_year_to', type=int, default=2021)
    parser.add_argument('--test_quarter_to', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default="graph_data")
    parser.add_argument('--Scaling', default="power")
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--semi_epoch', type=int, default=1000, help='Number of epochs for semi-supervised training.')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers_lt3', type=int, default=2)
    parser.add_argument('--n_layers_stcgt', type=int, default=2)
    parser.add_argument('--n_layers_arct', type=int, default=2)
    parser.add_argument('--cross_trans_heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)

    args = parser.parse_args()
    return args, device

def load_data(args, device):
    Y_temporal, X_temporal, edge_index_temporal, time_steps, sriskrs_temporal, sriskvs_temporal, sriskms_temporal = load_data_temporal(args.train_year_from, args.train_quarter_from, args.train_year_to, args.train_quarter_to, args.Scaling, args.data_dir)
    Y_temporal_test, X_temporal_test, edge_index_temporal_test, time_steps, sriskrs_temporal_test, sriskvs_temporal_test, sriskms_temporal_test = load_data_temporal(args.test_year_from, args.test_quarter_from, args.test_year_to, args.test_quarter_to, args.Scaling, args.data_dir)

    Y_temporal = Y_temporal.to(device)
    X_temporal = X_temporal.to(device)
    edge_index_temporal = [edge_index.to(device) for edge_index in edge_index_temporal]
    sriskrs_temporal = sriskrs_temporal.to(device)
    sriskvs_temporal = sriskvs_temporal.to(device)
    sriskms_temporal = sriskms_temporal.to(device)

    Y_temporal_test = Y_temporal_test.to(device)
    X_temporal_test = X_temporal_test.to(device)
    edge_index_temporal_test = [edge_index.to(device) for edge_index in edge_index_temporal_test]
    sriskrs_temporal_test = sriskrs_temporal_test.to(device)
    sriskvs_temporal_test = sriskvs_temporal_test.to(device)
    sriskms_temporal_test = sriskms_temporal_test.to(device)
    
    contagion_temporal = load_contagion_list_temporal(args.train_year_to, args.train_quarter_to, args.train_year_to, args.train_quarter_to).to(device)
    contagion_temporal_test = load_contagion_list_temporal(args.test_year_to, args.test_quarter_to, args.test_year_to, args.test_quarter_to).to(device)
    return X_temporal, X_temporal_test, Y_temporal, Y_temporal_test, edge_index_temporal, edge_index_temporal_test, contagion_temporal, contagion_temporal_test, time_steps, sriskrs_temporal, sriskrs_temporal_test, sriskvs_temporal, sriskvs_temporal_test, sriskms_temporal, sriskms_temporal_test

def load_model(args):
    ours = HFTCRNet(args.num_nodes, args.in_channels, args.time_steps, args.d_model, args.d_k, args.d_v, args.n_layers_lt3, args.n_layers_stcgt, args.n_layers_arct, args.n_heads, args.cross_trans_heads)
    model = Model(ours, args)
    return model

def final_print(args, report, epochs):
    print_year = args.test_year_to
    print_quarter = args.test_quarter_to
    if print_quarter == 4:
        print_quarter = 1
        print_year += 1
    else:
        print_quarter += 1
    with open('Report_HFTCRNet.txt', 'a') as f:
        f.write("HFTCRNet {}Q{}\n".format(print_year, print_quarter))
        f.write(report)
        f.write("\n\n")

def train(X, edge, Y, contagion, X_test, edge_test, Y_test, contagion_test, model, args, sriskrs, sriskrs_test, sriskvs, sriskvs_test, sriskms, sriskms_test):
    srisk_temp = None
    with trange(args.epochs, ncols=80) as t:
        for epoch in t:
            model.fit(X, edge, Y, contagion, X_test[:, -1], sriskvs, sriskrs, srisk_temp, sriskms, epoch)
            report, srisk_temp = model.eval(X_test, edge_test, Y_test, contagion_test, sriskvs_test, sriskms_test)
    return report

def main():
    args, device = init()
    X, X_test, Y, Y_test, edge, edge_test, contagion, contagion_test, time_steps, sriskrs, sriskrs_test, sriskvs, sriskvs_test, sriskms, sriskms_test = load_data(args, device)
    epochs = args.epochs
    
    X = X.permute(1, 0, 2)
    X_test = X_test.permute(1, 0, 2)

    Y = Y[-1]
    Y_test = Y_test[-1]

    args.num_nodes = X.size(0)
    args.in_channels = X.size(2)
    args.time_steps = time_steps

    model = load_model(args)
    report = train(X, edge, Y, contagion, X_test, edge_test, Y_test, contagion_test, model, args, sriskrs, sriskrs_test, sriskvs, sriskvs_test, sriskms, sriskms_test)
    final_print(args, report, epochs)
    
if __name__ == "__main__":
    main()