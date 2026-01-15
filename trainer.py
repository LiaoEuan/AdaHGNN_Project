# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.tools import AvgMeter

class DannTrainer:
    """
    DANN 训练器。
    
    管理:
    1. 前向传播与多任务 Loss 计算 (Label + Domain + Constraint)。
    2. DANN 参数 alpha 的动态调度。
    3. 指标统计与日志记录。
    """
    def __init__(self, model, optimizer, lr_scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        
        self.label_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        
        # 初始化指标记录器
        self.metrics = {
            'total_loss': AvgMeter(), 'label_loss': AvgMeter(),
            'domain_loss': AvgMeter(), 'constraint_loss': AvgMeter(),
            'label_acc': AvgMeter(), 'domain_acc': AvgMeter(),
            'test_label_acc': AvgMeter(), 'test_label_loss': AvgMeter()
        }

    def _reset_metrics(self):
        for m in self.metrics.values(): m.reset()

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        self._reset_metrics()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
        for i, batch in enumerate(pbar):
            eeg, labels, domains = [d.to(self.config.device) for d in batch]
            labels = labels.squeeze()
            
            # 动态计算 GRL alpha (0 -> 1)
            p = (epoch * len(dataloader) + i) / (self.config.train.epochs * len(dataloader))
            alpha = 2. / (1. + math.exp(-10 * p)) - 1
            
            self.optimizer.zero_grad()
            
            # 前向传播
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
        # 测试时通常只关心 label_acc，但也需要 reset 所有以免报错
        self._reset_metrics() 
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Valid]")
        with torch.no_grad():
            for batch in pbar:
                eeg, labels, _ = [d.to(self.config.device) for d in batch]
                labels = labels.squeeze()
                
                label_logits, _, _ = self.model(eeg, alpha=0)
                loss = self.label_criterion(label_logits, labels)
                
                acc = (label_logits.argmax(1) == labels).float().mean().item()
                self.metrics['test_label_acc'].update(acc, eeg.size(0))
                self.metrics['test_label_loss'].update(loss.item(), eeg.size(0))
                
                pbar.set_postfix({'Acc': f"{self.metrics['test_label_acc'].avg:.2%}"})
        
        # 调度器步进
        if self.lr_scheduler:
            self.lr_scheduler.step(self.metrics['test_label_acc'].avg)
            
        return self.metrics