import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from utils.loading import load_one_quarter
from utils.metrics import tolerance_metric, recall_r4_at_k, compute_norm_cost
from utils.get_downstream_idxes import get_idxes, get_idxes_imb_y, get_idxes_imb_d
from model import PretrainModel, DownstreamModel
from utils.args import get_args
from utils.tools import seed_everything
from utils.tools import plot
import os

args = get_args()
seed_everything(args.seed)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

time_list = [
    "2016-1",
    "2016-2",
    "2016-3",
    "2016-4",
    "2017-1",
    "2017-2",
    "2017-3",
    "2017-4",
    "2018-1",
    "2018-2",
    "2018-3",
    "2018-4",
    "2019-1",
    "2019-2",
    "2019-3",
    "2019-4",
    "2020-1",
    "2020-2",
    "2020-3",
    "2020-4",
    "2021-1",
    "2021-2",
    "2021-3",
    "2021-4",
    "2022-1",
    "2022-2",
    "2022-3",
    "2022-4",
    "2023-1",
    "2023-2",
    "2023-3",
    "2023-4"
]

if args.exp_obj != None:
    if args.exp_obj == 'shot_num':
        dir = args.shot_num
        target_dir = f"./results/C1/shot_num/{dir}"
    elif args.exp_obj == 'class_ratios':
        dir = args.class_ratios
        target_dir = f"./results/C1/class_ratios/{dir}"
    elif args.exp_obj == 'dense_sparse_ratios':
        dir = args.dense_sparse_ratios
        target_dir = f"./results/C1/dense_sparse_ratios/{dir}"
    elif args.exp_obj == 'aug_num_in':
        dir = args.aug_num_in
        target_dir = f"./results/C1/aug_num_in/{dir}"
    elif args.exp_obj =='aug_num_hid':
        dir = args.aug_num_hid
        target_dir = f"./results/C1/aug_num_hid/{dir}"
    elif args.exp_obj == 'hid_dim':
        dir = args.hid_dim
        target_dir = f"./results/C1/hid_dim/{dir}"
    elif args.exp_obj == 'update_epochs':
        dir = args.update_epochs
        target_dir = f"./results/C1/update_epochs/{dir}"
    else:
        ValueError("error exp")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        pass
    with open(f'./results/C1/{args.exp_obj}/{dir}/acc_list.txt', 'w') as f:
        pass
    with open(f'./results/C1/{args.exp_obj}/{dir}/recall_r4_list.txt', 'w') as f:
        pass
    with open(f'./results/C1/{args.exp_obj}/{dir}/fuzzy_acc_list.txt', 'w') as f:
        pass
    with open(f'./results/C1/{args.exp_obj}/{dir}/norm_cost_list.txt', 'w') as f:
        pass
    with open(f'./results/C1/{args.exp_obj}/{dir}/pred.txt', 'w') as f:
        pass
    with open(f'./results/C1/{args.exp_obj}/{dir}/true.txt', 'w') as f:
        pass
    with open(f'./results/C1/{args.exp_obj}/{dir}/logits.txt', 'w') as f:
        pass
else:
    target_dir = f"./results/C1/main"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open('./results/C1/main/acc.txt', 'w') as f:
        pass
    with open('./results/C1/main/recall_r4.txt', 'w') as f:
        pass
    with open('./results/C1/main/fuzzy_acc.txt', 'w') as f:
        pass
    with open('./results/C1/main/norm_cost.txt', 'w') as f:
        pass
    with open(f'./results/C1/main/pred.txt', 'w') as f:
        pass
    with open(f'./results/C1/main/true.txt', 'w') as f:
        pass
    with open(f'./results/C1/main/logits.txt', 'w') as f:
        pass

    with open('./results/C1/main/acc_list.txt', 'w') as f:
        pass
    with open('./results/C1/main/recall_r4_list.txt', 'w') as f:
        pass
    with open('./results/C1/main/fuzzy_acc_list.txt', 'w') as f:
        pass
    with open('./results/C1/main/norm_cost_list.txt', 'w') as f:
        pass

quarter_accuracies = dict()
quarter_recall_r4 = dict()
quarter_fuz_acc = dict()
quarter_norm_cost = dict()

if os.path.exists(args.model_path):
    os.remove(args.model_path)
    print(f"model weights from the previous {args.model_path} has been deleted")
else:
    print(f"there is no model weights from the previous {args.model_path}")



for t in range(0, len(time_list) - 1):
    print("train:", time_list[t])
    Q1_features, Q1_labels, Q1_edge_index = load_one_quarter(time_list[t])
    scaler = MinMaxScaler()
    Q1_features=Q1_features.numpy()
    Q1_features = scaler.fit_transform(Q1_features)
    Q1_features = torch.FloatTensor(Q1_features)
    data = Data(x=Q1_features, edge_index=Q1_edge_index.t().contiguous(), y=Q1_labels).to(device)
    model = PretrainModel(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    mask = torch.arange(data.num_nodes)
    train_mask = mask
    test_mask = mask

    train_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=args.batch_size,
        input_nodes=train_mask
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=args.batch_size,
        input_nodes=test_mask,
    )

    for epoch in range(args.update_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss_dire, loss_dist = model(batch.x, batch.y, batch.edge_index)
            loss = loss_dire + loss_dist
            if epoch % 10 == 0:
                print("pretrain: epoch-{:03d} with loss_dire = {:.5f}, loss_dist = {:.5f}".
                      format(epoch, loss_dire.item(), loss_dist.item()))
            loss.backward()
            optimizer.step()

    model_weights = {
        'gnn': model.gnn.state_dict(),
    }

    torch.save(model_weights, "pretrained_gnn/weights.pth")

    print("pretrain/update done.")
    del train_loader, test_loader, data, model

    ## DOWNSTREAM

    Q2_features, Q2_labels, Q2_edge_index = load_one_quarter(time_list[t + 1])
    Q2_features = Q2_features.numpy()
    Q2_features = scaler.transform(Q2_features)
    Q2_features = torch.FloatTensor(Q2_features)

    if args.exp_obj == 'class_ratios':
        selected_idxes, remaining_idxes, test_idxes = get_idxes_imb_y(Q1_features.shape[0], Q2_labels, args.shot_num,
                                                                      args.class_ratios)
    elif args.exp_obj == 'dense_sparse_ratios':
        selected_idxes, remaining_idxes, test_idxes = get_idxes_imb_d(Q1_features.shape[0], Q2_edge_index,
                                                                      args.shot_num, args.dense_sparse_ratios,
                                                                      args.degree_threshold)
    elif args.exp_obj == 'shot_num' or args.exp_obj == 'aug_num_in' or args.exp_obj=='aug_num_hid' or args.exp_obj == 'hid_num' or args.exp_obj == 'update_epochs' or args.exp_obj == None:
        selected_idxes, remaining_idxes, test_idxes = get_idxes(Q1_features.shape[0], Q2_labels, args.shot_num)

    Q1_data = Data(x=Q1_features, edge_index=Q1_edge_index.t().contiguous(), y=Q1_labels).to(device)
    Q2_data = Data(x=Q2_features, edge_index=Q2_edge_index.t().contiguous(), y=Q2_labels).to(device)

    mask = torch.arange(Q1_data.num_nodes)

    Q1_loader = NeighborLoader(
        Q1_data,
        num_neighbors=[-1],
        batch_size=Q1_data.num_nodes,
        input_nodes=mask
    )

    Q2_loader = NeighborLoader(
        Q2_data,
        num_neighbors=[-1],
        batch_size=Q2_data.num_nodes,
        input_nodes=mask
    )

    model = DownstreamModel(args).to(device)
    ################################################# few-shot update###############################################
    if args.gnn_trainable:
        optim_classify = torch.optim.Adam(
            list(model.gnn.parameters()) +
            list(model.augment.parameters()) +
            list(model.dire_prompt.parameters()) + list(model.dist_prompt.parameters()) +
            list(model.dire_head.parameters()) + list(model.dist_head.parameters()),
            lr=args.lr_classify, weight_decay=args.decay)

    else:
        optim_classify = torch.optim.Adam(
            list(model.augment.parameters()) +
            list(model.dire_prompt.parameters()) + list(model.dist_prompt.parameters()) +
            list(model.dire_head.parameters()) + list(model.dist_head.parameters()),
            lr=args.lr_classify, weight_decay=args.decay)


    best_acc = 0
    best_recall_r4 = 0
    best_fuzzy_acc = 0
    best_nc = 1

    quarter_accuracies[time_list[t + 1]] = list()
    quarter_recall_r4[time_list[t + 1]] = list()
    quarter_fuz_acc[time_list[t + 1]] = list()
    quarter_norm_cost[time_list[t + 1]] = list()

    for epoch in range(args.tune_epochs):
        if args.gnn_trainable == False:
            for param in model.gnn.parameters():
                param.requires_grad = False

        for _, (Q1_input, Q2_input) in enumerate(zip(Q1_loader, Q2_loader)):
            loss_classify = model(Q1_input, Q2_input, selected_idxes, remaining_idxes)
            optim_classify.zero_grad()
            loss_classify.backward()
            optim_classify.step()

        model.eval()
        with torch.no_grad():
            tru_list, pred_list, logit_list = list(), list(), list()
            for batch_i, (Q1_input, Q2_input) in enumerate(zip(Q1_loader, Q2_loader)):
                out, pred = model.inference(Q1_input, Q2_input, selected_idxes, remaining_idxes, test_idxes)
                if args.exp_obj == 'shot_num':
                    true = Q2_input.y[test_idxes].cpu()
                else:
                    true = Q2_input.y[remaining_idxes].cpu()

            pred = pred.cpu()
            out = out.cpu()
            cm = confusion_matrix(true, pred)

            acc = accuracy_score(true, pred)
            nc = compute_norm_cost(cm)
            recall_r4 = recall_r4_at_k(out, true)
            fuzzy_acc, _ = tolerance_metric(cm)

            quarter_accuracies[time_list[t + 1]].append(acc)
            quarter_recall_r4[time_list[t + 1]].append(recall_r4)
            quarter_fuz_acc[time_list[t + 1]].append(fuzzy_acc)
            quarter_norm_cost[time_list[t + 1]].append(nc)

            if acc > best_acc:
                best_acc = acc
            if nc < best_nc:
                best_nc = nc
            if recall_r4 > best_recall_r4:
                best_recall_r4 = recall_r4
            if fuzzy_acc > best_fuzzy_acc:
                best_fuzzy_acc = fuzzy_acc

            if epoch % 10 == 0:
                print("Epoch {:03d}|{:03d} Train: loss_classify = {:.5f}, eval_recall_r4 = {:.4f}, eval_acc = {:.4f}, ".
                      format(epoch, args.tune_epochs, loss_classify.item(), recall_r4, acc))
     

 

    print("best acc = {:.4f}, final acc = {:.4f}".format(best_acc, acc))

    if args.exp_obj == None:
        with open('./results/C1/main/acc.txt', 'a', encoding='utf-8') as f:
            f.write(f"{acc:.4f}\n")

        with open('./results/C1/main/recall_r4.txt', 'a', encoding='utf-8') as f:
            f.write(f"{recall_r4:.4f}\n")

        with open('./results/C1/main/fuzzy_acc.txt', 'a', encoding='utf-8') as f:
            f.write(f"{fuzzy_acc:.4f}\n")

        with open('./results/C1/main/norm_cost.txt', 'a', encoding='utf-8') as f:
            f.write(f"{nc:.4f}\n")

        with open('./results/C1/main/pred.txt', 'a', encoding='utf-8') as f:
            f.write(f"{pred}\n")

        with open('./results/C1/main/true.txt', 'a', encoding='utf-8') as f:
            f.write(f"{true}\n")

        with open('./results/C1/main/logits.txt', 'a', encoding='utf-8') as f:
            f.write(f"{out}\n")
    else:
        with open(f'./results/C1/{args.exp_obj}/{dir}/acc.txt', 'a', encoding='utf-8') as f:
            f.write(f"{acc:.4f}\n")

        with open(f'./results/C1/{args.exp_obj}/{dir}/recall_r4.txt', 'a', encoding='utf-8') as f:
            f.write(f"{recall_r4:.4f}\n")

        with open(f'./results/C1/{args.exp_obj}/{dir}/fuzzy_acc.txt', 'a', encoding='utf-8') as f:
            f.write(f"{fuzzy_acc:.4f}\n")

        with open(f'./results/C1/{args.exp_obj}/{dir}/norm_cost.txt', 'a', encoding='utf-8') as f:
            f.write(f"{nc:.4f}\n")

        with open(f'./results/C1/{args.exp_obj}/{dir}/pred.txt', 'a', encoding='utf-8') as f:
            f.write(f"{pred}\n")

        with open(f'./results/C1/{args.exp_obj}/{dir}/true.txt', 'a', encoding='utf-8') as f:
            f.write(f"{true}\n")

        with open(f'./results/C1/{args.exp_obj}/{dir}/logits.txt', 'a', encoding='utf-8') as f:
            f.write(f"{out}\n")

mean_acc = list()
mean_fuzzy_acc = list()
mean_recall_r4 = list()
mean_norm_cost = list()

for epoch in range(args.tune_epochs):
    acc = 0
    fuzzy_acc = 0
    recall_r4 = 0
    norm_cost = 0
    for t in range(len(time_list) - 1):
        acc += quarter_accuracies[time_list[t + 1]][epoch]
        fuzzy_acc += quarter_fuz_acc[time_list[t + 1]][epoch]
        recall_r4 += quarter_recall_r4[time_list[t + 1]][epoch]
        norm_cost += quarter_norm_cost[time_list[t + 1]][epoch]
    mean_acc.append(acc / (len(time_list) - 1))
    mean_fuzzy_acc.append(fuzzy_acc / (len(time_list) - 1))
    mean_recall_r4.append(recall_r4 / (len(time_list) - 1))
    mean_norm_cost.append(norm_cost / (len(time_list) - 1))
sorted_indices_acc = sorted(enumerate(mean_acc), key=lambda x: x[1], reverse=True)
sorted_indices_fuzzy_acc = sorted(enumerate(mean_fuzzy_acc), key=lambda x: x[1], reverse=True)
sorted_indices_recall_r4 = sorted(enumerate(mean_recall_r4), key=lambda x: x[1], reverse=True)
sorted_indices_norm_cost = sorted(enumerate(mean_norm_cost), key=lambda x: x[1], reverse=True)

if args.exp_obj == None:
    save_path = './results/C1/main/'
    exp = 'C1/main'
else:
    save_path = f'./results/C1/{args.exp_obj}/{dir}/'
    exp = f'C1/{args.exp_obj}/{dir}'

for index, value in sorted_indices_acc:
    with open(save_path + 'acc_list.txt', 'a', encoding='utf-8') as f:
        f.write(f"epoch: {index}, mean_acc: {value:.4f}\n")

for index, value in sorted_indices_fuzzy_acc:
    with open(save_path + 'fuzzy_acc_list.txt', 'a', encoding='utf-8') as f:
        f.write(f"epoch: {index}, mean_fuzzy_acc: {value:.4f}\n")

for index, value in sorted_indices_recall_r4:
    with open(save_path + 'recall_r4_list.txt', 'a', encoding='utf-8') as f:
        f.write(f"epoch: {index}, mean_recall_r4: {value:.4f}\n")

for index, value in sorted_indices_norm_cost:
    with open(save_path + 'norm_cost_list.txt', 'a', encoding='utf-8') as f:
        f.write(f"epoch: {index}, mean_norm_cost: {value:.4f}\n")

epochs4plot = args.tune_epochs
plot(epochs4plot, quarter_accuracies, exp, "accuracy")
plot(epochs4plot, quarter_recall_r4, exp, "recall_r4")
plot(epochs4plot, quarter_fuz_acc, exp, "fuzzy_acc")
plot(epochs4plot, quarter_norm_cost, exp, "norm_cost")



