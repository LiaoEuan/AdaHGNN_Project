# -*- coding: utf-8 -*-
import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.config_utils import load_config
from utils.tools import set_seed, print_size, get_lr, save_history_plot
from data import get_loso_datasets, EEGDataset_AVED, DTU_AAD_Dataset
from models.adahgnn import EEG_AdaMSHyper
from trainer import DannTrainer

def run_loso_fold(train_loader, valid_loader, config, num_domains, val_subject_id, fold_save_dir):
    """运行单个 Fold 的训练流程"""
    # 1. 初始化模型
    model = EEG_AdaMSHyper(config, config.all_mappings, num_domains).to(config.device)
    print_size(model)
    
    # 2. 优化器与调度器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.train.lr, 
        weight_decay=config.train.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", 
        patience=config.train.patience, 
        factor=config.train.factor, 
        verbose=True
    )
    
    trainer = DannTrainer(model, optimizer, scheduler, config)
    
    # 3. 训练循环
    best_acc = 0.0
    history = []
    
    for epoch in range(config.train.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        test_metrics = trainer.test_epoch(valid_loader, epoch)
        
        # 记录日志
        curr_acc = test_metrics['test_label_acc'].avg
        info = {
            'epoch': epoch + 1,
            'train_acc': train_metrics['label_acc'].avg,
            'valid_acc': curr_acc,
            'train_loss': train_metrics['total_loss'].avg,
            'lr': get_lr(optimizer)
        }
        history.append(info)
        
        # 保存最佳模型
        if curr_acc > best_acc:
            best_acc = curr_acc
            torch.save(model.state_dict(), os.path.join(fold_save_dir, "best_model.pt"))
            print(f"▲ New Best Acc: {best_acc:.4f}")
            
    return best_acc, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/aved_config.yaml')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    set_seed(config.seed)
    
    dataset_name = getattr(config.dataset, 'name', 'AVED') # 默认为 AVED
    
    if dataset_name == 'DTU' or config.dataset.num_channels == 64:
        print(">>> 正在使用 DTU 数据集加载器")
        config.dataset.dataset_class = DTU_AAD_Dataset
    else:
        print(">>> 正在使用 AVED 数据集加载器")
        config.dataset.dataset_class = EEGDataset_AVED
    
    # 创建实验目录
    os.makedirs(config.dataset.save_dir, exist_ok=True)
    
    # 开始 LOSO 交叉验证
    all_accs = []
    
    for sub_id in range(1, config.dataset.total_sub + 1):
        print(f"\n{'='*20} Validating Subject {sub_id} {'='*20}")
        fold_dir = os.path.join(config.dataset.save_dir, f"fold_S{sub_id}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # 数据准备
        train_set, valid_set, n_domains = get_loso_datasets(
            config, list(range(1, config.dataset.total_sub + 1)), sub_id
        )
        train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers)
        valid_loader = DataLoader(valid_set, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.num_workers)
        
        # 运行 Fold
        acc, history = run_loso_fold(train_loader, valid_loader, config, n_domains, sub_id, fold_dir)
        all_accs.append(acc)
        
        # 保存结果
        pd.DataFrame(history).to_csv(os.path.join(fold_dir, "log.csv"), index=False)
        save_history_plot(history, fold_dir)
        
    print(f"\nFinal Mean Accuracy: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")

if __name__ == "__main__":
    main()