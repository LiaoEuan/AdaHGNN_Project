# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

class EEGDataset_AVED(Dataset):
    """
    AVED 数据集的自定义 Dataset 类。
    
    功能:
    1. 加载预处理后的 .npz EEG 数据。
    2. 执行通道级 Min-Max 归一化。
    3. 支持训练阶段的随机振幅缩放数据增强。
    4. 返回 EEG 数据、标签以及用于 DANN 的领域标签 (Domain Label)。
    """
    def __init__(self, root, file_name, domain_label=-1):
        self.file_path = os.path.join(root, file_name)
        # 加载数据 (假设数据结构为 {'eeg': [...], 'label': [...]})
        self.data = np.load(self.file_path)
        self.eeg_data = self.data['eeg']
        self.event = self.data['label']
        
        # DANN 所需的领域标签，验证集通常设为 -1
        self.domain_label = torch.tensor(domain_label, dtype=torch.long)

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        # 1. 加载原始数据并转置: [Time, Channel] -> [Channel, Time]
        eeg = torch.tensor(self.eeg_data[idx], dtype=torch.float32).permute(1, 0)
        
        # 2. 通道级 Min-Max 归一化 (缩放到 [0, 1])
        epsilon = 1e-8
        eeg_min, _ = torch.min(eeg, dim=1, keepdim=True)
        eeg_max, _ = torch.max(eeg, dim=1, keepdim=True)
        eeg = (eeg - eeg_min) / (eeg_max - eeg_min + epsilon)

        # 3. 数据增强: 随机振幅缩放 (仅在训练集/已知领域标签时应用)
        if self.domain_label != -1:
            scale_factor = random.uniform(0.8, 1.2)
            eeg = eeg * scale_factor

        # 4. 封装标签
        event = torch.LongTensor([int(self.event[idx])])

        return eeg, event, self.domain_label

def get_loso_datasets(config, all_subject_ids, val_subject_id):
    """
    构建留一法 (Leave-One-Subject-Out) 的训练集和验证集。
    
    Args:
        config: 配置对象
        all_subject_ids: 所有被试 ID 列表
        val_subject_id: 当前作为验证集的被试 ID
        
    Returns:
        train_dataset: 合并后的训练数据集 (ConcatDataset)
        valid_dataset: 验证数据集
        num_domains: 训练集中的源域数量
    """
    train_subject_ids = [sid for sid in all_subject_ids if sid != val_subject_id]
    
    # 创建从 subject_id 到 domain_index (0 ~ N-1) 的映射
    domain_map = {sid: i for i, sid in enumerate(train_subject_ids)}
    
    # 构建训练集列表
    train_datasets = [
        config.dataset.dataset_class(
            config.dataset.root, 
            config.dataset.file_format.format(subject=sid), 
            domain_label=domain_map[sid]
        ) 
        for sid in train_subject_ids
    ]
    train_dataset = ConcatDataset(train_datasets)
    
    # 构建验证集 (domain_label 设为 -1)
    valid_dataset = config.dataset.dataset_class(
        config.dataset.root, 
        config.dataset.file_format.format(subject=val_subject_id), 
        domain_label=-1
    )
    
    num_domains = len(train_subject_ids)
    print(f"LOSO Fold: Training on {num_domains} subjects, Validating on subject {val_subject_id}.")
    print(f"Dataset Sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
    
    return train_dataset, valid_dataset, num_domains


# ==============================================================================
#  DTU 数据集类
# ==============================================================================
class DTU_AAD_Dataset(Dataset):
    """
    DTU AAD 数据集加载器。
    
    特性:
    1. 原始数据包含 66 通道 (64 EEG + 2 Mastoids/EOG)。
    2. 仅保留前 64 个 EEG 通道。
    3. 标签需要从 1/2 转换为 0/1。
    """
    def __init__(self, root, file_name, domain_label=-1):
        # 注意：这里为了接口统一，file_name 已经是格式化好的文件名 (e.g., S1_Dataset_1s.npz)
        self.file_path = os.path.join(root, file_name)
        
        self.data_cache = {}
        self.index_map = []
        # 将 domain_label 转为 Tensor
        self.domain_label = torch.tensor(domain_label, dtype=torch.long)
        
        if not os.path.exists(self.file_path):
            warnings.warn(f"警告: 找不到文件: {self.file_path}")
            return

        try:
            data = np.load(self.file_path, allow_pickle=True)
            
            # 标签转换: 原始 1/2 -> 0/1
            # 注意：data['event_slices'] 可能是嵌套列表，需小心处理
            raw_labels = [int(item[0]) if isinstance(item, (list, np.ndarray)) else int(item) 
                          for item in data['event_slices']]
            direction_labels_tensor = torch.tensor(raw_labels, dtype=torch.long) - 1
            
            # 脑电数据: [N_samples, Channels, TimePoints]
            eeg_data = torch.tensor(data['eeg_slices'], dtype=torch.float32) 
            
            # 预计算归一化 (Min-Max Scaling)
            # 此时 eeg_data 包含 66 通道
            eeg_min = eeg_data.min(dim=-1, keepdim=True)[0]
            eeg_max = eeg_data.max(dim=-1, keepdim=True)[0]
            eeg_normalized = (eeg_data - eeg_min) / (eeg_max - eeg_min + 1e-8)
            
            # 存入缓存
            self.data_cache = {'eeg': eeg_normalized, 'direction_label': direction_labels_tensor}
            
            # 建立索引映射
            for i in range(len(eeg_normalized)):
                self.index_map.append(i)
                
        except Exception as e:
            warnings.warn(f"警告: 加载数据出错 {self.file_path}: {e}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 1. 获取缓存数据
        # eeg_full Shape: [66, 128]
        eeg_full = self.data_cache['eeg'][idx] 
        
        # 2. 【关键】仅切片前 64 个通道
        eeg_64ch = eeg_full[:64, :] 
        
        # 3. 获取标签
        event = self.data_cache['direction_label'][idx].unsqueeze(0) # [1]
        
        # 4. 数据增强 (训练阶段/已知Domain时)
        # 注意: 这里的 eeg_64ch 已经是归一化过的
        if self.domain_label.item() != -1:
            scale_factor = random.uniform(0.8, 1.2)
            eeg_out = eeg_64ch * scale_factor
        else:
            eeg_out = eeg_64ch

        return eeg_out, event, self.domain_label