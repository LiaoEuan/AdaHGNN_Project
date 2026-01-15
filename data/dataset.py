# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

class EEGDataset_AVED(Dataset):
    """
    AVED 数据集的自定义 Dataset 类 (改进版)。
    
    改进点:
    1. 移除了片断级 Min-Max 归一化，保留预处理阶段的 Z-score 分布特征。
    2. 增加了音频数据 (wav) 的加载，这对 AAD 任务至关重要。
    3. 修正了 Label 的维度问题。
    4. 增加了 explicit 的 training 标志来控制数据增强。
    """
    def __init__(self, root, file_name, domain_label=-1, training=False):
        """
        :param root: 数据目录路径
        :param file_name: .npz 文件名
        :param domain_label: 领域标签 (用于 DANN，源域通常为 0, 目标域为 1)
        :param training: bool, 是否为训练模式 (控制数据增强)
        """
        self.file_path = os.path.join(root, file_name)
        
        # 加载数据 (需确保预处理脚本保存了 'eeg', 'label', 'wav')
        # 使用 allow_pickle=True 以防 numpy 版本兼容性问题
        self.data = np.load(self.file_path, allow_pickle=True)
        
        self.eeg_data = self.data['eeg']   # Shape: [N, Time, Channels]
        self.wav_data = self.data['wav']   # Shape: [N, Time, 1] 或 [N, Time]
        self.event = self.data['label']    # Shape: [N]
        
        # DANN 领域标签
        self.domain_label = torch.tensor(domain_label, dtype=torch.long)
        self.training = training

        # 校验数据长度一致性
        assert len(self.eeg_data) == len(self.wav_data) == len(self.event), \
            f"Data length mismatch in {file_name}"

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        # 1. 处理 EEG 数据
        # 原始 shape: [Time, Channel] -> 转换后: [Channel, Time] (符合 PyTorch Conv1d 习惯)
        # 注意：这里直接使用预处理好的 Z-score 数据，不再做 Min-Max
        eeg = torch.tensor(self.eeg_data[idx], dtype=torch.float32).permute(1, 0)
        
        # 2. 处理音频数据
        # 同样转换为 [Audio_Channel, Time] (通常 Audio_Channel=1)
        wav = torch.tensor(self.wav_data[idx], dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0) # [Time] -> [1, Time]
        else:
            wav = wav.permute(1, 0) # [Time, 1] -> [1, Time]

        # 3. 数据增强 (仅在训练模式下)
        # 随机振幅缩放：模拟不同受试者或不同时间的阻抗微小变化
        if self.training:
            scale_factor = random.uniform(0.85, 1.15)
            eeg = eeg * scale_factor
            # 音频通常不需要做振幅缩放，因为它已归一化且是参考信号

        # 4. 封装标签
        # 转换为标量 LongTensor，适应 CrossEntropyLoss
        event = torch.tensor(self.event[idx], dtype=torch.long)

        # 返回格式: (EEG输入, 音频输入), 类别标签, 领域标签
        # 这是一个常见的 Tuple 结构，具体取决于你的 Model forward 接收什么参数
        return eeg, wav, event, self.domain_label

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