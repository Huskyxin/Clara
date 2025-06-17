import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import torch.nn.functional as F
import random
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import argparse
import copy
import torch
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.utils import *
import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
import torch.nn.functional as F

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Attention(gnn.MessagePassing):
    """Multi-head DAG attention using PyG interface
    accept Batch data given by PyG
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False, symmetric=False, gnn_type="gcn", **kwargs):

        super().__init__(node_dim=0, aggr='add')

        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.gnn_type = gnn_type

        self.attend = nn.Softmax(dim=-1)

        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
            # self.to_tqk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
            # self.to_tqk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                x,
                edge_index,
                mask_dag_,
                dag_rr_edge_index,
                edge_attr=None,
                ptr=None):

        # Compute value matrix
        v = self.to_v(x)
        x_struct = x

        # Compute query and key matrices
        qk = self.to_qk(x_struct).chunk(2, dim=-1)

        # Compute self-attention
        out = self.propagate(dag_rr_edge_index, v=v, qk=qk, edge_attr=None, size=None
                             )
        out = rearrange(out, 'n h d -> n (h d)')
        return self.out_proj(out)

    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i):
        """Self-attention based on MPNN """
        qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
        qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
        v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)
        attn = (qk_i * qk_j).sum(-1) * self.scale

        attn = utils.softmax(attn, index, ptr, size_i)

        attn = self.attn_dropout(attn)

        return v_j * attn.unsqueeze(-1)

class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation="relu", batch_norm=True, pre_norm=False,
                 gnn_type="gcn", **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = Attention(d_model, nhead, dropout=dropout,
                                   bias=False, gnn_type=gnn_type, **kwargs)
        self.batch_norm = batch_norm
        self.pre_norm = pre_norm
        if batch_norm:
            self.norm1 = nn.Identity(d_model)
            self.norm2 = nn.Identity(d_model)

    def forward(self, x, edge_index, mask_dag_, dag_rr_edge_index,
                edge_attr=None, ptr=None,
                ):

        if self.pre_norm:
            x = self.norm1(x)
        x2 = self.self_attn(
            x,
            edge_index,
            mask_dag_,
            dag_rr_edge_index,
            edge_attr=edge_attr,
            ptr=None,
        )

        x = x + self.dropout1(x2)
        if self.pre_norm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        if not self.pre_norm:
            x = self.norm2(x)
        return x

class GraphTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, x, edge_index, mask_dag_, dag_rr_edge_index,
                edge_attr=None, ptr=None):
        output = x
        for mod in self.layers:
            output = mod(output, edge_index, mask_dag_, dag_rr_edge_index,
                         edge_attr=edge_attr,
                         ptr=None)
        return output

class GraphTransModel(nn.Module):
    def __init__(self, args, **kwargs):
        super(GraphTransModel, self).__init__()
        d_model = args.dim_hidden
        dim_feedforward = 4 * args.dim_hidden
        num_heads = args.num_heads
        dropout = args.gt_dropout
        num_layers = args.num_layers
        batch_norm = True
        gnn_type = "gcn",
        global_pool = 'mean'
        self.dropout = nn.Dropout(0.1)
        self.embedding = nn.Linear(in_features=100,
                                   out_features=d_model,
                                   bias=False)
        self.gnn_type = gnn_type
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)

        self.global_pool = global_pool
        self.pooling = gnn.global_mean_pool
        # if global_pool == 'mean':
        #     self.pooling = gnn.global_mean_pool
        # elif global_pool == 'add':
        #     self.pooling = gnn.global_add_pool
        # elif global_pool == 'max':
        #     self.pooling = gnn.global_max_pool

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )

    def singleTrain(self, data, labels, weight):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        mask_dag_ = None
        output = self.embedding(x)
        output = self.dropout(output)
        output = self.encoder(
            output,
            edge_index,
            mask_dag_,
            edge_index,
            ptr=None,
        )
        
        output = self.pooling(output, data.batch)
        logits = self.classifier(output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            prob = nn.functional.softmax(logits, dim=-1)
            loss = None
            return loss, prob


    def forward(self, data, labels, weight):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        mask_dag_ = None
        output = self.embedding(x)
        output = self.dropout(output)
        output = self.encoder(
            output,
            edge_index,
            mask_dag_,
            edge_index,
            ptr=None,
        )

        output = self.pooling(output, data.batch)

        return output

class TransModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(TransModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids=None, labels=None, weight=None):

        mask = input_ids.ne(self.config.pad_token_id)
        out = self.encoder(input_ids, attention_mask=input_ids.ne(
            self.tokenizer.pad_token_id))
        token_embeddings = out[0]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)
                               ).sum(1) / mask.sum(-1).unsqueeze(-1)  # averege

        return sentence_embeddings

    def singleTrain(self, input_ids=None, labels=None, weight=None):
        mask = input_ids.ne(self.config.pad_token_id)
        out = self.encoder(input_ids, attention_mask=input_ids.ne(
            self.tokenizer.pad_token_id))
        token_embeddings = out[0]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)
                               ).sum(1) / mask.sum(-1).unsqueeze(-1)  # averege

        logits = self.classifier(sentence_embeddings)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            prob = nn.functional.softmax(logits, dim=-1)
            # loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = None
            return loss, prob

class MLP_adapter(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class Context_fuse(nn.Module):
    def __init__(self, args, config):
        """
            embed_dim (int): 输入特征的维度
            num_heads (int): 注意力头的数量
        """
        super(Context_fuse, self).__init__()
        embed_dim, num_heads = args.dim_hidden, 4
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        # 定义Q、K、V的线性变换
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.text_k_linear = nn.Linear(embed_dim, embed_dim)
        self.text_v_linear = nn.Linear(embed_dim, embed_dim)

        self.graph_k_linear = nn.Linear(embed_dim, embed_dim)
        self.graph_v_linear = nn.Linear(embed_dim, embed_dim)
        

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5  # 缩放因子

        self.graph_token_fuse = MLP_adapter(args.dim_hidden, args.dim_hidden//2, args.dim_hidden)
        self.token_graph_fuse = MLP_adapter(args.dim_hidden, args.dim_hidden//2, args.dim_hidden)

        self.query_tokens = nn.Parameter(
            torch.empty(1, args.n_context, args.dim_hidden).normal_(std=0.02))

        self.graph_conv = nn.Conv1d(
            args.n_context * (args.num_layers - args.n_fusion_layers), args.n_context,
            kernel_size=3, padding=1
            )
        self.text_conv = nn.Conv1d(
            args.n_context * (config.num_hidden_layers - args.n_fusion_layers), args.n_context,
            kernel_size=3, padding=1
            )

    def forward(self, text_c, graph_c):
        """
        前向传播
        
        参数:
            x: 输入张量, shape为(batch_size, seq_len, embed_dim)
            
        返回:
            输出张量, shape与输入相同
        """
        
        graph_c = self.graph_conv(graph_c)
        text_c = self.text_conv(text_c)
        batch_size, seq_len, _ = text_c.size()
        
        # 生成Q、K、V
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        q = self.q_linear(query_tokens)

        tk = self.text_k_linear(text_c)  
        tv = self.text_v_linear(text_c)  

        gk = self.graph_k_linear(graph_c)  
        gv = self.graph_v_linear(graph_c)  
        
        # 将Q、K、V分割成多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        tk = tk.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        tv = tv.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        gk = gk.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        gv = gv.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores_t = torch.matmul(q, tk.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        attn_t = F.softmax(scores_t, dim=-1)
        out_t = torch.matmul(attn_t, tv)  # (batch_size, num_heads, seq_len, head_dim)
        out_t = out_t.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        scores_g = torch.matmul(q, gk.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        attn_g = F.softmax(scores_g, dim=-1)
        out_g = torch.matmul(attn_g, gv)  # (batch_size, num_heads, seq_len, head_dim)
        out_g = out_g.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        token_f = self.token_graph_fuse(out_t + out_g)
        graph_f = self.graph_token_fuse(out_t + out_g)

        return token_f, graph_f

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def cross_correlation(Z_v1, Z_v2):
    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())

def correlation_reduction_loss(S):
    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()

class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MHA, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q_1 = nn.Linear(d_model, d_model)
        self.W_k_1 = nn.Linear(d_model, d_model)
        self.W_v_1 = nn.Linear(d_model, d_model)

        self.W_q_2 = nn.Linear(d_model, d_model)
        self.W_k_2 = nn.Linear(d_model, d_model)
        self.W_v_2 = nn.Linear(d_model, d_model)

        # self.W_o = nn.Linear(d_model, d_model)
        self.se = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        self.fuse_weight = nn.Parameter(
            nn.init.constant_(torch.zeros(1), 0.5))
    def forward(self, A, B):
        # 线性变换
        Q_1 = self.W_q_1(A)
        K_1 = self.W_k_1(B)
        V_1 = self.W_v_1(B)
        # 分割多头
        Q_1 = Q_1.view(-1, self.num_heads, self.d_k)
        K_1 = K_1.view(-1, self.num_heads, self.d_k)
        V_1 = V_1.view(-1, self.num_heads, self.d_k)
        # 计算注意力分数
        attention_scores_1 = torch.mul(Q_1, K_1) / (self.d_k ** 0.5)
        attention_weights_1 = F.softmax(attention_scores_1, dim=-1) # n,d
        # 加权求和
        output_1 = torch.mul(attention_weights_1, V_1)
        output_1 = output_1.view(-1, self.d_model)

        # 线性变换
        Q_2 = self.W_q_2(B)
        K_2 = self.W_k_2(A)
        V_2 = self.W_v_2(A)
        # 分割多头
        Q_2 = Q_2.view(-1, self.num_heads, self.d_k)
        K_2 = K_2.view(-1, self.num_heads, self.d_k)
        V_2 = V_2.view(-1, self.num_heads, self.d_k)
        # 计算注意力分数
        attention_scores_2 = torch.mul(Q_2, K_2) / (self.d_k ** 0.5)
        attention_weights_2 = F.softmax(attention_scores_2, dim=-1)
        # 加权求和
        output_2 = torch.mul(attention_weights_2, V_2)
        output_2 = output_2.view(-1, self.d_model)
        # 线性变换
        atten = F.sigmoid(self.se(output_1 + output_2))

        output = output_1 * atten + output_2 * (1 - atten)

        output2 = self.linear2(self.dropout(F.relu(self.linear1(output))))
        output = output + self.dropout2(output2)
        
        return output

class MModalModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(MModalModel, self).__init__()
        self.trans = TransModel(encoder, config, tokenizer, args)
        self.graph = GraphTransModel(args)

        self.context_nodes = nn.ParameterList([
            nn.Parameter(torch.empty(1, args.n_context, args.dim_hidden).normal_(std=0.02))
            for _ in range(args.num_layers - args.n_fusion_layers)
        ])
        
        self.context_tokens = nn.ParameterList([
            nn.Parameter(torch.empty(1, args.n_context, args.dim_hidden).normal_(std=0.02)) 
            for _ in range(config.num_hidden_layers - args.n_fusion_layers)])
        
        self.context_fuse_layer_1 = Context_fuse(args, config)
        self.context_fuse_layer_2 = Context_fuse(args, config)

        # self.fuse_weight = nn.Parameter(
        #     nn.init.constant_(torch.zeros(1), 0.5))
        self.config = config
        self.args = args

        self.fuse_layer = MHA(args.dim_hidden, 4)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 2)
        )
        
    def freeze_module(self):
        for param in self.trans.parameters():
            param.requires_grad = False
        for param in self.graph.parameters():
            param.requires_grad = False
        for param in self.context_fuse_layer_1.parameters():
            param.requires_grad = True
        for param in self.context_fuse_layer_2.parameters():
            param.requires_grad = True
        trainable_modules = [self.context_nodes, self.context_tokens,
                             self.fuse_layer.modules(), self.classifier.modules(),
                             self.trans.classifier.modules(),
                             self.graph.classifier.modules()
                             ]
                            #  
        for module in trainable_modules:
            for item in module:
                item.requires_grad_(True)
        
        # for param in self.trans.encoder.encoder.layer[-1].parameters():
        #     param.requires_grad = True
        # for param in self.trans.encoder.encoder.layer[-2].parameters():
        #     param.requires_grad = True

        # for param in self.graph.encoder.layers[-1].parameters():
        #     param.requires_grad = True
        # for param in self.graph.encoder.layers[-2].parameters():
        #     param.requires_grad = True

    def forward(self, inputs=None, graph = None, labels=None, weight=None, modal = 'graph'):
        if modal == 'graph-contrast':
            loss, prob = self.train_graph(inputs, graph, labels, weight)
            return loss, prob
        elif modal == 'token-contrast':
            loss, prob = self.train_token(inputs, graph, labels, weight)
            return loss, prob
        elif modal == 'graph':
            loss, prob = self.test_graph(inputs, graph, labels, weight)
            return loss, prob
        elif modal == 'token':
            loss, prob = self.test_token(inputs, graph, labels, weight)
            return loss, prob
        elif modal == 'joint':
            loss, prob = self.joint_train(inputs, graph, labels, weight)
            return loss, prob
        elif modal == 'first':
            loss, prob = self.joint_first_train(inputs, graph, labels, weight)
            return loss, prob
        else:
            raise('error')
        
    # def joint_train(self, input_ids=None, graph = None, labels=None, weight=None):
    #     txt_tokens = self.trans.encoder.embeddings(input_ids)
    #     mask = input_ids.ne(self.config.pad_token_id)
    #     batch_size, token_len = mask.size()[0], mask.size()[1]
    #     context_mask = torch.ones((batch_size, self.args.n_context), device = input_ids.device)
    #     mask = torch.cat([context_mask, mask], dim=1)
    #     mask = torch.unsqueeze(mask, dim=1)
    #     mask = torch.unsqueeze(mask, dim=1)
    #     text_prompt_tokens = []
    #     for trans_layer_id in range(self.config.num_hidden_layers - self.args.n_fusion_layers):
    #         batch_token_p = self.context_tokens[trans_layer_id].expand(batch_size, -1, -1)
    #         txt_tokens = torch.cat([batch_token_p, txt_tokens], dim=1)
    #         txt_tokens = self.trans.encoder.encoder.layer[trans_layer_id](txt_tokens, attention_mask=mask)[0]
    #         text_prompt_tokens.append(txt_tokens[:, :self.args.n_context, :])
    #         txt_tokens = txt_tokens[:, self.args.n_context:, :]
        
    #     concatenated_text = torch.cat(text_prompt_tokens, dim=1)
        
    #     x, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch
    #     graph_tokens = self.graph.embedding(x)
    #     graph_tokens = self.graph.dropout(graph_tokens)
    #     edge_index_list = []
    #     for graph_idx in range(batch_size):
    #         # 获取当前图的最后一个节点索引
    #         graph_mask = batch == graph_idx
    #         last_node_idx = graph_mask.nonzero(as_tuple=True)[0][-1].item()
    #         virtual_start_idx = graph.x.size(0) + graph_idx * self.args.n_context
            
    #         # 1. 虚拟节点之间全连接
    #         for i in range(self.args.n_context):
    #             for j in range(i + 1, self.args.n_context):
    #                 v1 = virtual_start_idx + i
    #                 v2 = virtual_start_idx + j
    #                 edge_index_list.append(torch.tensor([[v1, v2], [v2, v1]], device=graph.x.device))
            
    #         # 2. 所有虚拟节点与最后一个原始节点相连
    #         virtual_nodes_indices = torch.arange(virtual_start_idx, 
    #                                         virtual_start_idx + self.args.n_context,
    #                                         device=graph.x.device)
    #         # 创建双向边
    #         sources = torch.cat([
    #             virtual_nodes_indices,
    #             torch.full_like(virtual_nodes_indices, last_node_idx)
    #         ])
    #         targets = torch.cat([
    #             torch.full_like(virtual_nodes_indices, last_node_idx),
    #             virtual_nodes_indices
    #         ])
    #         edge_index_list.append(torch.stack([sources, targets]))
    #     new_edges = torch.cat(edge_index_list, dim=1)
    #     edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
        
        
    #     graph_prompt_tokens = []
    #     for graph_layer_id in range(self.args.num_layers - self.args.n_fusion_layers):
    #         batch_graph_p = self.context_nodes[graph_layer_id].expand(batch_size, -1, -1)
    #         batch_graph_p = batch_graph_p.reshape(-1, batch_graph_p.size(-1))
    #         o_num_nodes = graph_tokens.size()[0]
    #         graph_tokens = torch.cat([graph_tokens, batch_graph_p], dim=0) 
    #         graph_tokens = self.graph.encoder.layers[graph_layer_id](graph_tokens, edge_index, None, edge_index, None, None)
    #         graph_prompt_tokens.append(graph_tokens[o_num_nodes:,:].reshape(batch_size, self.args.n_context, -1))
    #         graph_tokens = graph_tokens[:o_num_nodes,:]
        
    #     concatenated_graph = torch.cat(graph_prompt_tokens, dim=1)

    #     # for trans_layer_id in range(self.args.n_fusion_layers):
    #     text_fuse_1, graph_fuse_1 = self.context_fuse_layer_1(concatenated_text, concatenated_graph)
    #     # text_fuse_2, graph_fuse_2 = self.context_fuse_layer_2(concatenated_text, concatenated_graph)

    #     txt_tokens = torch.cat([text_fuse_1, txt_tokens], dim=1)
    #     txt_tokens = self.trans.encoder.encoder.layer[-1](txt_tokens, attention_mask=mask)[0]
    #     # txt_tokens = txt_tokens[:, self.args.n_context:, :]
    #     # txt_tokens = torch.cat([text_fuse_2, txt_tokens], dim=1)
    #     # txt_tokens = self.trans.encoder.encoder.layer[-1](txt_tokens, attention_mask=mask)[0]
    #     mask = torch.squeeze(mask, dim=1)
    #     mask = torch.squeeze(mask, dim=1)
    #     sentence_embeddings = (txt_tokens * mask.unsqueeze(-1)
    #                            ).sum(1) / mask.sum(-1).unsqueeze(-1)
        

    #     graph_fuse_1 = graph_fuse_1.reshape(-1, batch_graph_p.size(-1))
    #     graph_tokens = torch.cat([graph_tokens, graph_fuse_1], dim=0)
    #     graph_tokens = self.graph.encoder.layers[-1](graph_tokens, edge_index, None, edge_index, None, None)
    #     # graph_tokens = graph_tokens[:o_num_nodes,:]
    #     # graph_fuse_2 = graph_fuse_2.reshape(-1, batch_graph_p.size(-1))
    #     # graph_tokens = torch.cat([graph_tokens, graph_fuse_2], dim=0)
    #     # graph_tokens = self.graph.encoder.layers[-1](graph_tokens, edge_index, None, edge_index, None, None)

    #     new_batch = []
    #     for i in range(batch_size):
    #         new_batch.extend([i] * (self.args.n_context))
    #     new_batch = torch.tensor(new_batch, device=input_ids.device)
    #     batch = torch.cat([graph.batch, new_batch], dim=0)
    #     output = self.graph.pooling(graph_tokens, batch)


    #     emb = self.fuse_layer(sentence_embeddings, output)
    #     logits = self.classifier(emb)
    #     # logits_t = self.trans.classifier(sentence_embeddings)
    #     # logits_g = self.graph.classifier(output)
    #     # logits = self.graph.classifier(emb)

    #     # logits = (logits_t + logits_g) * 0.5
    #     # logits = (self.fuse_weight) * logits_t + (1-(self.fuse_weight))*logits_g
        
    #     if labels is not None:
    #         loss_fct = nn.CrossEntropyLoss(weight=weight)
    #         loss = loss_fct(logits, labels)
    #         return loss, logits
    #     else:
    #         prob = nn.functional.softmax(logits, dim=-1)
    #         loss = None
    #         return loss, prob
        
    def joint_train(self, input_ids=None, graph = None, labels=None, weight=None):
        txt_tokens = self.trans.encoder.embeddings(input_ids)
        mask = input_ids.ne(self.config.pad_token_id)
        batch_size, token_len = mask.size()[0], mask.size()[1]
        context_mask = torch.ones((batch_size, self.args.n_context), device = input_ids.device)
        mask = torch.cat([context_mask, mask], dim=1)
        mask = torch.unsqueeze(mask, dim=1)
        mask = torch.unsqueeze(mask, dim=1)
        text_prompt_tokens = []
        for trans_layer_id in range(self.config.num_hidden_layers - self.args.n_fusion_layers):
            batch_token_p = self.context_tokens[trans_layer_id].expand(batch_size, -1, -1)
            txt_tokens = torch.cat([batch_token_p, txt_tokens], dim=1)
            txt_tokens = self.trans.encoder.encoder.layer[trans_layer_id](txt_tokens, attention_mask=mask)[0]
            text_prompt_tokens.append(txt_tokens[:, :self.args.n_context, :])
            txt_tokens = txt_tokens[:, self.args.n_context:, :]
        
        concatenated_text = torch.cat(text_prompt_tokens, dim=1)
        
        x, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch
        graph_tokens = self.graph.embedding(x)
        graph_tokens = self.graph.dropout(graph_tokens)
        edge_index_list = []
        for graph_idx in range(batch_size):
            # 获取当前图的最后一个节点索引
            graph_mask = batch == graph_idx
            last_node_idx = graph_mask.nonzero(as_tuple=True)[0][-1].item()
            virtual_start_idx = graph.x.size(0) + graph_idx * self.args.n_context
            
            # 1. 虚拟节点之间全连接
            for i in range(self.args.n_context):
                for j in range(i + 1, self.args.n_context):
                    v1 = virtual_start_idx + i
                    v2 = virtual_start_idx + j
                    edge_index_list.append(torch.tensor([[v1, v2], [v2, v1]], device=graph.x.device))
            
            # 2. 所有虚拟节点与最后一个原始节点相连
            virtual_nodes_indices = torch.arange(virtual_start_idx, 
                                            virtual_start_idx + self.args.n_context,
                                            device=graph.x.device)
            # 创建双向边
            sources = torch.cat([
                virtual_nodes_indices,
                torch.full_like(virtual_nodes_indices, last_node_idx)
            ])
            targets = torch.cat([
                torch.full_like(virtual_nodes_indices, last_node_idx),
                virtual_nodes_indices
            ])
            edge_index_list.append(torch.stack([sources, targets]))
        new_edges = torch.cat(edge_index_list, dim=1)
        edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
        
        
        graph_prompt_tokens = []
        for graph_layer_id in range(self.args.num_layers - self.args.n_fusion_layers):
            batch_graph_p = self.context_nodes[graph_layer_id].expand(batch_size, -1, -1)
            batch_graph_p = batch_graph_p.reshape(-1, batch_graph_p.size(-1))
            o_num_nodes = graph_tokens.size()[0]
            graph_tokens = torch.cat([graph_tokens, batch_graph_p], dim=0) 
            graph_tokens = self.graph.encoder.layers[graph_layer_id](graph_tokens, edge_index, None, edge_index, None, None)
            graph_prompt_tokens.append(graph_tokens[o_num_nodes:,:].reshape(batch_size, self.args.n_context, -1))
            graph_tokens = graph_tokens[:o_num_nodes,:]
        
        concatenated_graph = torch.cat(graph_prompt_tokens, dim=1)

        # for trans_layer_id in range(self.args.n_fusion_layers):
        text_fuse_1, graph_fuse_1 = self.context_fuse_layer_1(concatenated_text, concatenated_graph)
        text_fuse_2, graph_fuse_2 = self.context_fuse_layer_2(concatenated_text, concatenated_graph)

        txt_tokens = torch.cat([text_fuse_1, txt_tokens], dim=1)
        txt_tokens = self.trans.encoder.encoder.layer[-2](txt_tokens, attention_mask=mask)[0]
        txt_tokens = txt_tokens[:, self.args.n_context:, :]

        txt_tokens = torch.cat([text_fuse_2, txt_tokens], dim=1)
        txt_tokens = self.trans.encoder.encoder.layer[-1](txt_tokens, attention_mask=mask)[0]
        mask = torch.squeeze(mask, dim=1)
        mask = torch.squeeze(mask, dim=1)
        sentence_embeddings = (txt_tokens * mask.unsqueeze(-1)
                               ).sum(1) / mask.sum(-1).unsqueeze(-1)
        

        graph_fuse_1 = graph_fuse_1.reshape(-1, batch_graph_p.size(-1))
        graph_tokens = torch.cat([graph_tokens, graph_fuse_1], dim=0)
        graph_tokens = self.graph.encoder.layers[-2](graph_tokens, edge_index, None, edge_index, None, None)
        graph_tokens = graph_tokens[:o_num_nodes,:]

        graph_fuse_2 = graph_fuse_2.reshape(-1, batch_graph_p.size(-1))
        graph_tokens = torch.cat([graph_tokens, graph_fuse_2], dim=0)
        graph_tokens = self.graph.encoder.layers[-1](graph_tokens, edge_index, None, edge_index, None, None)

        new_batch = []
        for i in range(batch_size):
            new_batch.extend([i] * (self.args.n_context))
        new_batch = torch.tensor(new_batch, device=input_ids.device)
        batch = torch.cat([graph.batch, new_batch], dim=0)
        output = self.graph.pooling(graph_tokens, batch)

        emb = self.fuse_layer(sentence_embeddings, output)
        logits = self.classifier(emb)
        # logits_t = self.trans.classifier(sentence_embeddings)
        # logits_g = self.graph.classifier(output)
        # logits = self.graph.classifier(emb)

        # logits = (logits_t + logits_g) * 0.5
        # logits = (self.fuse_weight) * logits_t + (1-(self.fuse_weight))*logits_g
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            prob = nn.functional.softmax(logits, dim=-1)
            loss = None
            return loss, prob
    
    def joint_first_train(self, input_ids=None, graph = None, labels=None, weight=None):
        txt_tokens = self.trans.encoder.embeddings(input_ids)
        mask = input_ids.ne(self.config.pad_token_id)
        mask = torch.unsqueeze(mask, dim=1)
        mask = torch.unsqueeze(mask, dim=1)
        for trans_layer_id in range(self.config.num_hidden_layers):
            txt_tokens = self.trans.encoder.encoder.layer[trans_layer_id](txt_tokens, attention_mask=mask)[0]

        mask = torch.squeeze(mask, dim=1)
        mask = torch.squeeze(mask, dim=1)
        sentence_embeddings = (txt_tokens * mask.unsqueeze(-1)
                               ).sum(1) / mask.sum(-1).unsqueeze(-1)   
    
        logits_t = self.trans.classifier(sentence_embeddings)

        x, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch
        graph_tokens = self.graph.embedding(x)
        graph_tokens = self.graph.dropout(graph_tokens)
        for graph_layer_id in range(self.args.num_layers):
            graph_tokens = self.graph.encoder.layers[graph_layer_id](graph_tokens, edge_index, None, edge_index, None, graph.ptr)
        
        output = self.graph.pooling(graph_tokens, graph.batch)
        logits_g = self.graph.classifier(output)

        logits = (logits_g + logits_t) * 0.5
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            prob = nn.functional.softmax(logits, dim=-1)
            loss = None
            return loss, prob        
    
    # 等价于下面
    def train_token(self, input_ids=None, graph = None, labels=None, weight=None):
        txt_tokens = self.trans.encoder.embeddings(input_ids)
        mask = input_ids.ne(self.config.pad_token_id)
        mask = torch.unsqueeze(mask, dim=1)
        mask = torch.unsqueeze(mask, dim=1)
        for trans_layer_id in range(self.config.num_hidden_layers):
            txt_tokens = self.trans.encoder.encoder.layer[trans_layer_id](txt_tokens, attention_mask=mask)[0]
        mask = torch.squeeze(mask, dim=1)
        mask = torch.squeeze(mask, dim=1)
        sentence_embeddings = (txt_tokens * mask.unsqueeze(-1)
                               ).sum(1) / mask.sum(-1).unsqueeze(-1)   


        logits = self.trans.classifier(sentence_embeddings)
        if labels is not None:
            posi_correlation = cross_correlation(sentence_embeddings.t(), sentence_embeddings.t())
            con_loss = correlation_reduction_loss(posi_correlation)

            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss_ce = loss_fct(logits, labels)

            loss = 1.0 * con_loss + loss_ce
            return loss, logits
        else:
            prob = nn.functional.softmax(logits, dim=-1)
            loss = None
            return loss, prob

    def test_token(self, input_ids=None, graph = None, labels=None, weight=None):
        txt_tokens = self.trans.encoder.embeddings(input_ids)
        mask = input_ids.ne(self.config.pad_token_id)
        mask = torch.unsqueeze(mask, dim=1)
        mask = torch.unsqueeze(mask, dim=1)
        for trans_layer_id in range(self.config.num_hidden_layers):
            txt_tokens = self.trans.encoder.encoder.layer[trans_layer_id](txt_tokens, attention_mask=mask)[0]
        mask = torch.squeeze(mask, dim=1)
        mask = torch.squeeze(mask, dim=1)
        sentence_embeddings = (txt_tokens * mask.unsqueeze(-1)
                               ).sum(1) / mask.sum(-1).unsqueeze(-1)   
        logits = self.trans.classifier(sentence_embeddings)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss_ce = loss_fct(logits, labels)
            loss = loss_ce
            return loss, logits
        else:
            prob = nn.functional.softmax(logits, dim=-1)
            loss = None
            return loss, prob
            
    def train_graph(self,input_ids=None, graph = None, labels=None, weight=None):
        x, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch
        graph_tokens = self.graph.embedding(x)
        graph_tokens = self.graph.dropout(graph_tokens)
        for graph_layer_id in range(self.args.num_layers):
            graph_tokens = self.graph.encoder.layers[graph_layer_id](graph_tokens, edge_index, None, edge_index, None, graph.ptr)
        output = self.graph.pooling(graph_tokens, graph.batch)

        logits = self.graph.classifier(output)
        if labels is not None:
            posi_correlation = cross_correlation(output.t(), output.t())
            con_loss = correlation_reduction_loss(posi_correlation)

            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss_ce = loss_fct(logits, labels)

            loss = 0.1 * con_loss + loss_ce
            return loss, logits
        else:
            prob = nn.functional.softmax(logits, dim=-1)
            loss = None
            return loss, prob

    def test_graph(self, input_ids=None, graph = None, labels=None, weight=None):
        x, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch
        graph_tokens = self.graph.embedding(x)
        graph_tokens = self.graph.dropout(graph_tokens)
        for graph_layer_id in range(self.args.num_layers):
            graph_tokens = self.graph.encoder.layers[graph_layer_id](graph_tokens, edge_index, None, edge_index, None, graph.ptr)
        output = self.graph.pooling(graph_tokens, graph.batch)

        logits = self.graph.classifier(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss_ce = loss_fct(logits, labels)
            loss = loss_ce
            return loss, logits
        else:
            prob = nn.functional.softmax(logits, dim=-1)
            loss = None
            return loss, prob

