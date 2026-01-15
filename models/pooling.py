# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class AnatomicalSpatialPooling(nn.Module):
    """
    基于解剖学先验的空间池化层。
    
    作用:
    将精细尺度上的节点特征，根据预定义的解剖学映射 (Mapping)，聚合为粗糙尺度的节点特征。
    实现方式: 矩阵乘法 (N_old x N_new 权重矩阵)。
    """
    def __init__(self, pre_defined_mapping):
        super().__init__()
        
        # 计算节点数
        if not pre_defined_mapping: 
            num_old_nodes = 0
        else: 
            # 找到最大的索引值 + 1 作为旧节点数
            num_old_nodes = max(max(indices) for indices in pre_defined_mapping if indices) + 1
            
        num_new_nodes = len(pre_defined_mapping)
        
        # 构建聚合权重矩阵 (平均池化)
        weight_matrix = torch.zeros(num_old_nodes, num_new_nodes)
        for new_idx, old_indices in enumerate(pre_defined_mapping):
            if old_indices:
                for old_idx in old_indices:
                    weight_matrix[old_idx, new_idx] = 1.0 / len(old_indices)
                    
        # 注册为 Buffer (随模型保存但不参与梯度更新)
        self.register_buffer('pooling_weight', weight_matrix)

    def forward(self, x):
        """
        Input: [B, N_old, D]
        Output: [B, N_new, D]
        """
        # 高效矩阵乘法: (B, D, N_old) @ (N_old, N_new) -> (B, D, N_new)
        x_perm = x.permute(0, 2, 1)
        pooled = torch.matmul(x_perm, self.pooling_weight.to(x.device))
        return pooled.permute(0, 2, 1)