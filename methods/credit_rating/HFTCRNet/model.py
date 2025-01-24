import numpy as np
import torch.nn as nn
import torch
from torch_geometric.nn import GATConv, TransformerConv
import torch.nn.functional as Func
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score, r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([[pos / np.power(10000, 2 * i / d_model) for i in range(d_model)] if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])
        self.pos_table = torch.FloatTensor(pos_table).to(device)

    def forward(self, enc_inputs):
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_ff = d_ff

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.layernorm = nn.LayerNorm(self.d_model)

    def forward(self, input_Q, input_K, input_V):
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads,
                                   self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads,
                                   self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1,
                                   self.n_heads, self.d_v).transpose(1, 2)

        context, attn = ScaledDotProductAttention(self.d_model, self.d_k, self.d_v, self.n_heads, self.d_ff)(Q, K, V) 
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)

        output = self.fc(context)
        return self.layernorm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.fc = nn.Sequential(
            nn.Linear(d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, d_model, bias=False))

        self.layernorm = nn.LayerNorm(self.d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual)

class CrossTransModule(nn.Module):
    def __init__(self, in_channels, out_channels, time_steps, cross_trans_heads):
        super(CrossTransModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_steps = time_steps
        self.cross_trans_heads = cross_trans_heads
        self.transformerConv = TransformerConv(self.in_channels, self.out_channels, self.cross_trans_heads, concat=False)

    def forward(self, X, edge_index_temporal):
        X = X.permute(1, 0, 2)
        node_num = X.shape[1]
        X_result = X.clone()
        zero_tensor = torch.zeros(1, node_num, self.in_channels).to(device)
        X = torch.cat((zero_tensor, X), dim=0)
        for i in range(0, self.time_steps):
            if i == 0:
                X_result[i] = self.transformerConv((X[i], X[i+1]), edge_index_temporal[i])
            else:
                X_result[i] = self.transformerConv((X[i], X[i+1]), edge_index_temporal[i-1])
        X_result = X_result.permute(1, 0, 2)
        return X_result


class FusionScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(FusionScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_ff = d_ff

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / \
            np.sqrt(self.d_k) 
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class FusionMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(FusionMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.h_model = d_model
        self.t_model = self.h_model
        self.y_model = self.h_model + self.t_model

        self.W_Q = nn.Linear(self.h_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.t_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.y_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

    def forward(self, H, T, Y):
        batch_size = H.size(0)
        Q = self.W_Q(H).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(T).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(Y).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        context = FusionScaledDotProductAttention(self.d_model, self.d_k, self.d_v, self.n_heads, self.d_ff)(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        F = self.fc(context)
        return F


class FusionModule(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(FusionModule, self).__init__()
        self.attentionModule = FusionMultiHeadAttention(d_model, d_k, d_v, n_heads, d_ff)

    def forward(self, H, T):
        Y = torch.cat((H, T), dim=2)
        F = self.attentionModule(H, T, Y) + H
        return F

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.enc_self_attn = MultiHeadAttention(
            self.d_model, self.d_k, self.d_v, self.n_heads, self.d_ff)
        self.pos_ffn = PoswiseFeedForwardNet(
            self.d_model, self.d_k, self.d_v, self.n_heads, self.d_ff)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class SpatialContagionTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(SpatialContagionTransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.enc_self_attn = MultiHeadAttention(
            self.d_model, self.d_k, self.d_v, self.n_heads, self.d_ff)
        self.pos_ffn = PoswiseFeedForwardNet(
            self.d_model, self.d_k, self.d_v, self.n_heads, self.d_ff)

    def forward(self, enc_contagion_inputs):
        enc_outputs, _ = self.enc_self_attn(
            enc_contagion_inputs, enc_contagion_inputs, enc_contagion_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs


class HFTCRNet(nn.Module):
    def __init__(self, num_nodes, in_channels, time_steps, d_model, d_k, d_v, n_layers_lt3, n_layers_stcgt, n_layers_arct, n_heads, cross_trans_heads):
        super(HFTCRNet, self).__init__()
        self.num_nodes = num_nodes
        self.time_steps = time_steps
        self.out_channels = 4

        self.d_model = d_model
        self.d_ff = 4 * self.d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.n_layers_lt3 = n_layers_lt3
        self.n_layers_stcgt = n_layers_stcgt
        self.n_layers_arct = n_layers_arct
        self.cross_trans_heads = cross_trans_heads

        self.begin_mlp_up = nn.Linear(in_channels, self.d_model)
        self.begin_mlp_down = nn.Linear(in_channels, self.d_model)

        self.pos_emb = PositionalEncoding(self.d_model)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_k, self.d_v, self.n_heads, self.d_ff) for _ in range(self.n_layers_lt3)])

        self.CrossTransformer = nn.ModuleList([CrossTransModule(self.d_model, self.d_model, self.time_steps, self.cross_trans_heads) for _ in range(self.n_layers_stcgt)])

        self.FusionModules1 = FusionModule(self.d_model, self.d_k, self.d_v, self.n_heads, self.d_ff)
        self.FusionModules2 = FusionModule(self.d_model, self.d_k,  self.d_v, self.n_heads, self.d_ff)

        self.contagionEncoderLayers = nn.ModuleList([SpatialContagionTransformerEncoderLayer(self.d_model, self.d_k, self.d_v, self.n_heads, self.d_ff) for _ in range(self.n_layers_arct)])

        self.out_mlp_fs = nn.Linear(self.d_model, in_channels)
        self.out_mlp_rank = nn.Linear(self.d_model, self.out_channels)
        self.out_mlp_srisk = nn.Linear(self.d_model, 2)

        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, X, edge_index_temporal, C):
        X = self.begin_mlp_up(X)
        H = self.pos_emb(X)
        for i in range(self.n_layers_lt3):
            H, _ = self.layers[i](H)
        C = self.begin_mlp_down(C.squeeze())
        node_num, list_num, path_length, feature_num = C.shape
        C = C.view(node_num*list_num, path_length, feature_num)
        for i in range(self.n_layers_arct):
            C = self.contagionEncoderLayers[i](C)
        C = torch.mean(torch.max(C.view(node_num, list_num, path_length, feature_num), dim=2)[0], dim=1)
        for i in range(self.n_layers_stcgt):
            T = self.CrossTransformer[i](X, edge_index_temporal)
            T = self.dropout(T)
        HT = self.FusionModules1(H, T)
        HT = Func.max_pool1d(HT.transpose(1, 2), kernel_size=self.time_steps).transpose(1, 2).transpose(0, 1)
        HTC = self.FusionModules2(HT, C.unsqueeze(0))

        output = Func.log_softmax(self.out_mlp_rank(HTC.squeeze()), dim=1)
        output_fs = self.out_mlp_fs(HTC.squeeze())
        output_srisk = self.out_mlp_srisk(HTC.squeeze())
        assert torch.isnan(output).sum() == 0
        return output, output_fs, output_srisk


class Model():
    def __init__(self, model, args):
        self.gnn_model = model.to(device)
        self.args = args
        self.lr = args.lr
        self.lambda1 = 0.1
        self.lambda2 = 0.5
        self.lambda3 = 0.5
        self.b_f1 = 0
        self.b_acc = 0
        self.b_r2 = -1
        self.srisk_temp = None
        self.report = None
        self.optimizer = torch.optim.AdamW(
            self.gnn_model.parameters(), lr=self.lr, weight_decay=5e-4)

    def fit(self, X, edge_index_temporal, Y, contagion_list_temporal, fs, sriskv, sriskr, sriskv_semi, sriskm, epoch):
        self.optimizer.zero_grad()
        out, out_fs, out_srisk = self.gnn_model(X, edge_index_temporal, contagion_list_temporal)
        nllloss = Func.nll_loss(out[:], Y[:])
        fs_loss = Func.mse_loss(fs, out_fs)
        sriskv_loss = Func.mse_loss(out_srisk[:, 0][sriskm[-1] == 1], sriskv[-1][sriskm[-1] == 1])
        sriskr_loss = Func.mse_loss(out_srisk[:, 1][sriskm[-1] == 1], sriskr[-1][sriskm[-1] == 1])
        if epoch > self.args.semi_epoch:
            sriskv_loss += Func.mse_loss(out_srisk[:, 0][sriskm[-1] == 0], sriskv_semi[:, 0][sriskm[-1] == 0])
            sriskr_loss += Func.mse_loss(out_srisk[:, 1][sriskm[-1] == 0], sriskv_semi[:, 1][sriskm[-1] == 0])
        loss = nllloss + self.lambda1 * fs_loss + self.lambda2 * sriskv_loss + self.lambda3 * sriskr_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.gnn_model.parameters(), max_norm=5, norm_type=2)
        self.optimizer.step()

    def test(self, X, edge_index_temporal, Y, contagion_list_temporal):
        self.gnn_model.eval()
        out, _, srisk = self.gnn_model(X, edge_index_temporal, contagion_list_temporal)
        predb = out.cpu().max(dim=1).indices[:]
        trub = torch.Tensor.cpu(Y)[:]
        self.gnn_model.train()
        return trub, predb, srisk
    
    def eval(self, X_test, edge_test, Y_test, contagion_test, sriskvs_test, sriskms_test):
        with torch.no_grad():
            tru, pred, srisk = self.test(X_test, edge_test, Y_test, contagion_test)
        r2 = r2_score(sriskvs_test[-1][sriskms_test[-1] == 1].cpu().detach().numpy(), srisk[:, 0][sriskms_test[-1] == 1].cpu().detach().numpy())
        macro_f1 = f1_score(tru, pred, average='macro')
        acc = accuracy_score(tru, pred)
        if macro_f1 > self.b_f1:
            self.b_f1 = macro_f1
            self.b_acc = acc
            self.report = 'Best F1: {:.3f}; Best Accuracy: {:.3f}'.format(self.b_f1, self.b_acc)
        if r2 > self.b_r2:
            self.b_r2 = r2
            self.srisk_temp = srisk
        return self.report, self.srisk_temp
