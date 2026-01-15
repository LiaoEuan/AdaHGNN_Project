# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import dhg
from dhg.nn import HGNNPConv

class MultiScale_EEG_Hypergraph(nn.Module):
    """
    多尺度自适应超图构建器。
    
    功能:
    为每个预定义的空间尺度 (Scale) 独立学习一个超图结构。
    通过可学习的节点嵌入和超边嵌入计算相似度，动态生成连接关系。
    """
    def __init__(self, config):
        super().__init__()
        self.scale_node_nums = config.scale_node_nums
        self.dim = config.model.d_model
        self.hyper_num_list = config.hyper_num
        self.alpha = config.model.hgnn_alpha
        self.k = config.model.k_neighbors
        
        # 为每个尺度创建独立的嵌入
        self.embedhy = nn.ModuleList()
        self.embednod = nn.ModuleList()
        for i in range(len(self.scale_node_nums)):
            n_nodes = self.scale_node_nums[i]
            n_edges = self.hyper_num_list[i]
            self.embedhy.append(nn.Embedding(n_edges, self.dim))
            self.embednod.append(nn.Embedding(n_nodes, self.dim))

    def forward(self, x):
        """
        Returns:
            hyperedge_all: 包含每个尺度超图结构 (hyperedge_index Tensor) 的列表
        """
        hyperedge_all = []
        for i in range(len(self.scale_node_nums)):
            num_nodes, num_hyperedges = self.scale_node_nums[i], self.hyper_num_list[i]
            
            # 生成嵌入并计算相似度矩阵
            hyp_idx = torch.arange(num_hyperedges, device=x.device)
            node_idx = torch.arange(num_nodes, device=x.device)
            
            h_embed = self.embedhy[i](hyp_idx)
            n_embed = self.embednod[i](node_idx)
            
            sim_matrix = torch.mm(n_embed, h_embed.t())
            
            # 稀疏化处理: Top-K 选择
            adj = F.softmax(F.relu(self.alpha * sim_matrix), dim=1)
            mask = torch.zeros(num_nodes, num_hyperedges, device=x.device)
            k = min(adj.size(1), self.k)
            if k > 0:
                _, topk_indices = adj.topk(k, 1)
                mask.scatter_(1, topk_indices, 1)
            
            adj = adj * mask
            adj = torch.where(adj > 0.5, 1.0, 0.0)
            
            # 移除空超边
            adj = adj[:, (adj != 0).any(dim=0)]
            
            if adj.shape[1] > 0:
                node_indices, hyperedge_indices = torch.nonzero(adj, as_tuple=True)
                hypergraph = torch.stack([node_indices, hyperedge_indices])
                hyperedge_all.append(hypergraph)
            else:
                hyperedge_all.append(torch.empty((2, 0), dtype=torch.long, device=x.device))
                
        return hyperedge_all


class HypergraphConvDHG(nn.Module):
    """
    优化的 DHG 超图卷积层 (支持 Batch 处理).
    
    包含:
    1. 特征线性变换。
    2. 批处理超图拼接 (Create Batched Hypergraph)。
    3. NHC (Node-Hyperedge-Consistency) 约束损失计算。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels)
        self.conv = HGNNPConv(out_channels, out_channels)
            
    def _create_batched_hypergraph(self, hg_single, batch_size, num_nodes, device):
        """将单个超图结构复制 B 次并拼接，以实现并行计算"""
        e_list_batched = []
        if hg_single.num_e > 0:
            single_e_list = hg_single.e[0]
            for i in range(batch_size):
                offset = i * num_nodes
                for edge in single_e_list:
                    e_list_batched.append([node + offset for node in edge])
        
        num_total_nodes = batch_size * num_nodes
        return dhg.Hypergraph(num_total_nodes, e_list_batched, device=device)

    def forward(self, x, hg):
        """
        Input: x [B, N, C_in], hg (DHG Hypergraph Object)
        Output: out [B, N, C_out], constraint_loss
        """
        B, N, _ = x.shape
        
        # 1. 线性变换
        x_trans = self.lin(x) # [B, N, C]
        x_reshaped = x_trans.reshape(B * N, self.out_channels)
        
        # 2. 构建 Batch 超图
        hg_batched = self._create_batched_hypergraph(hg, B, N, x.device)
        
        # 3. 计算 NHC 约束损失
        constraint_loss = torch.tensor(0.0, device=x.device)
        if hg_batched.num_e > 0:
            # 计算超边特征
            he_feats = hg_batched.v2e(x_reshaped, aggr="sum")
            
            loss_hyper = 0.0
            if hg_batched.num_e > 1:
                # 超边间距离与相似度约束
                sim = F.cosine_similarity(he_feats.unsqueeze(1), he_feats.unsqueeze(0), dim=-1)
                dist = torch.cdist(he_feats, he_feats, p=2)
                loss_item = sim * dist + (1 - sim) * torch.clamp(4.2 - dist, min=0.0)
                loss_hyper = torch.mean(torch.abs(loss_item))
            
            # 节点重构损失
            node_feats_recon = hg_batched.e2v(he_feats, aggr="mean")
            loss_node = torch.mean(torch.abs(x_reshaped - node_feats_recon))
            
            constraint_loss = loss_node + loss_hyper
        
        # 4. 超图卷积
        out_reshaped = self.conv(x_reshaped, hg_batched)
        out = out_reshaped.reshape(B, N, self.out_channels)
        
        return out, constraint_loss