# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import warnings
import dhg
from torch.nn.init import trunc_normal_

from models.layers import LearnableTemporalShift, TemporalStem, GradientReversalLayer
from models.hypergraph import MultiScale_EEG_Hypergraph, HypergraphConvDHG
from models.pooling import AnatomicalSpatialPooling

class TemporalFeatureExtractor(nn.Module):
    """
    时间特征提取器。
    集成: LearnableTemporalShift (对齐) + TemporalStem (CNN主干)。
    """
    def __init__(self, num_channels, d_model, patch_size, time_sample_num):
        super().__init__()
        # 1. 自适应时间对齐
        self.lts = LearnableTemporalShift(num_channels, sampling_rate=128, max_shift_sec=0.2)
        
        # 2. 并行 CNN 提取器 (每通道独立)
        num_patches = time_sample_num // patch_size
        self.channel_extractors = nn.ModuleList([
            TemporalStem(out_planes=d_model, kernel_size=63, patch_size=patch_size, radix=2) 
            for _ in range(num_channels)
        ])
        
        # 3. 时间位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_channels, num_patches, d_model))

    def forward(self, x):
        # x: [B, C, T]
        x = self.lts(x) # 对齐
        
        # 提取特征
        outputs = [ext(x[:, i, :].unsqueeze(1)) for i, ext in enumerate(self.channel_extractors)]
        
        # 堆叠与位置编码: [B, C, D, P] -> [B, C, P, D]
        patched_feats = torch.stack(outputs, dim=1).permute(0, 1, 3, 2)
        feats_with_pos = patched_feats + self.pos_embedding
        
        # 均值池化得到节点特征: [B, C, D]
        return feats_with_pos.mean(dim=2)

class EEG_AdaMSHyper(nn.Module):
    """
    AdaHGNN 主模型。
    包含: 特征提取 -> 多尺度超图构建 -> 层次化卷积与池化 -> 尺度间注意力 -> DANN分类。
    """
    def __init__(self, config, pre_defined_mappings, num_domain_classes):
        super().__init__()
        self.scale_node_nums = config.scale_node_nums
        self.num_scales = len(self.scale_node_nums)
        self.d_model = config.model.d_model
        
        # 组件实例化
        self.feature_extractor = TemporalFeatureExtractor(
            config.dataset.num_channels, self.d_model, 
            config.dataset.patch_size, config.dataset.time_sample_num
        )
        self.hypergraph_builder = MultiScale_EEG_Hypergraph(config)
        
        self.conv_layers = nn.ModuleList([
            HypergraphConvDHG(self.d_model, self.d_model) for _ in self.scale_node_nums
        ])
        
        self.spatial_pooling_layers = nn.ModuleList([
            AnatomicalSpatialPooling(m) for m in pre_defined_mappings
        ])
        
        # 尺度间注意力
        self.inter_scale_attention = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=8, 
            dropout=config.model.dropout, batch_first=True
        )
        self.inter_scale_norm = nn.LayerNorm(self.d_model)

        # 分类器
        fused_dim = self.d_model * self.num_scales + self.d_model
        
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(config.model.dropout),
            nn.Linear(128, config.model.num_classes)
        )
        
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(config.model.dropout),
            nn.Linear(128, num_domain_classes)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear): 
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward(self, x, alpha=1.0):
        # 1. 初始节点特征
        curr_node_feats = self.feature_extractor(x)
        
        # 2. 生成超图结构
        hg_structures = self.hypergraph_builder(x)
        
        # 3. 层次化处理
        total_loss = 0.0
        scale_node_outs = []
        scale_he_feats = []
        
        for i in range(self.num_scales):
            hg_struct = hg_structures[i]
            n_nodes = self.scale_node_nums[i]
            
            # 构建 DHG 对象
            hg = dhg.Hypergraph(n_nodes, device=x.device)
            if hg_struct.numel() > 0:
                # 兼容不同版本 DHG API
                try: hg.add_hyperedges_from_feature_tensor('hyperedge_index', hg_struct)
                except: hg.add_edges_from_pairing(hg_struct.t().cpu().numpy())
            
            # 卷积
            updated_feats, loss = self.conv_layers[i](curr_node_feats, hg)
            total_loss += loss
            
            # 提取超边特征 (用于尺度间交互)
            if hg.num_e > 0:
                try: scale_he_feats.append(hg.v2e(updated_feats, aggr="mean"))
                except: pass # 忽略 Batch 错误
            
            # 记录图级特征
            scale_node_outs.append(updated_feats.mean(dim=1))
            
            # 空间池化
            if i < self.num_scales - 1:
                curr_node_feats = self.spatial_pooling_layers[i](updated_feats)
                
        # 4. 尺度间交互
        if scale_he_feats:
            all_he = torch.cat(scale_he_feats, dim=1)
            attn_out, _ = self.inter_scale_attention(all_he, all_he, all_he)
            pooled_he = self.inter_scale_norm(all_he + attn_out).mean(dim=1)
        else:
            pooled_he = torch.zeros(x.shape[0], self.d_model, device=x.device)
            
        # 5. 最终融合与分类
        fused_nodes = torch.cat(scale_node_outs, dim=1)
        final_feat = torch.cat([fused_nodes, pooled_he], dim=1)
        
        label_out = self.classifier(final_feat)
        domain_out = self.domain_classifier(self.grl(final_feat, alpha))
        
        return label_out, domain_out, total_loss