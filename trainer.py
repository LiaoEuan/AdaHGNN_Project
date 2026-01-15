# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.tools import AvgMeter

class DannTrainer:
    """
    DANN 训练器 (修复版)。
    
    修复内容:
    - 动态兼容返回 3 个值 (DTU) 和 4 个值 (AVED带音频) 的数据集。
    """
    def __init__(self, model, optimizer, lr_scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        
        self.label_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        
        self.metrics = {
            'total_loss': AvgMeter(), 'label_loss': AvgMeter(),
            'domain_loss': AvgMeter(), 'constraint_loss': AvgMeter(),
            'label_acc': AvgMeter(), 'domain_acc': AvgMeter(),
            'test_label_acc': AvgMeter(), 'test_label_loss': AvgMeter()
        }

    def _reset_metrics(self):
        for m in self.metrics.values(): m.reset()

    def _unpack_batch(self, batch):
        """
        [新增] 内部辅助函数：安全地解包 Batch 数据
        兼容:
            - DTU: (eeg, labels, domains) -> 3 items
            - AVED: (eeg, wav, labels, domains) -> 4 items
        """
        # 将所有数据移至 GPU
        batch_data = [d.to(self.config.device) for d in batch]
        
        wav = None # 初始化音频为 None
        
        if len(batch_data) == 3:
            # Case 1: DTU (无音频)
            eeg, labels, domains = batch_data
        elif len(batch_data) == 4:
            # Case 2: AVED (有音频)
            eeg, wav, labels, domains = batch_data
        else:
            raise ValueError(f"DataLoader 返回了 {len(batch_data)} 个元素，但 Trainer 只支持 3 或 4 个。")
            
        return eeg, wav, labels, domains

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        self._reset_metrics()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
        for i, batch in enumerate(pbar):
            # --- 使用辅助函数解包 ---
            eeg, wav, labels, domains = self._unpack_batch(batch)
            labels = labels.squeeze()
            
            # 动态计算 GRL alpha (0 -> 1)
            p = (epoch * len(dataloader) + i) / (self.config.train.epochs * len(dataloader))
            alpha = 2. / (1. + math.exp(-10 * p)) - 1
            
            self.optimizer.zero_grad()
            
            # 前向传播 (注意：如果你的模型目前还不支持 wav 输入，这里只需传 eeg)
            # 如果后续模型升级为多模态，可以在这里传入 wav
            label_logits, domain_logits, c_loss = self.model(eeg, alpha=alpha)
            
            # 计算 Loss
            loss_label = self.label_criterion(label_logits, labels)
            loss_domain = self.domain_criterion(domain_logits, domains)
            total_loss = loss_label + loss_domain + self.config.model.lambda_constraint * c_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            # 记录指标
            bs = eeg.size(0)
            self.metrics['total_loss'].update(total_loss.item(), bs)
            self.metrics['label_loss'].update(loss_label.item(), bs)
            self.metrics['domain_loss'].update(loss_domain.item(), bs)
            self.metrics['constraint_loss'].update(c_loss.item(), bs)
            
            l_acc = (label_logits.argmax(1) == labels).float().mean().item()
            d_acc = (domain_logits.argmax(1) == domains).float().mean().item()
            self.metrics['label_acc'].update(l_acc, bs)
            self.metrics['domain_acc'].update(d_acc, bs)
            
            pbar.set_postfix({'L_Acc': f"{l_acc:.2%}", 'D_Acc': f"{d_acc:.2%}"})
            
        return self.metrics

    def test_epoch(self, dataloader, epoch):
        self.model.eval()
        self._reset_metrics() 
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Valid]")
        with torch.no_grad():
            for batch in pbar:
                # --- 使用辅助函数解包 ---
                eeg, wav, labels, _ = self._unpack_batch(batch)
                labels = labels.squeeze()
                
                label_logits, _, _ = self.model(eeg, alpha=0)
                loss = self.label_criterion(label_logits, labels)
                
                acc = (label_logits.argmax(1) == labels).float().mean().item()
                self.metrics['test_label_acc'].update(acc, eeg.size(0))
                self.metrics['test_label_loss'].update(loss.item(), eeg.size(0))
                
                pbar.set_postfix({'Acc': f"{self.metrics['test_label_acc'].avg:.2%}"})
        
        if self.lr_scheduler:
            self.lr_scheduler.step(self.metrics['test_label_acc'].avg)
            
        return self.metrics