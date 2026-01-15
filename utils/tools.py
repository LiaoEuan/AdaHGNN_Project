# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

class AvgMeter:
    """计算并存储平均值和当前值"""
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = 0, 0, 0

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def print_size(net, keyword=None):
    """打印模型参数量"""
    if net is not None and isinstance(net, torch.nn.Module):
        params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"{net.__class__.__name__} Parameters: {params / 1e6:.6f}M")

def set_seed(seed_value):
    """固定所有随机种子以复现结果"""
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    print(f"Random Seed set to: {seed_value}")

def save_history_plot(history, save_dir, title="Training History"):
    """简单的训练曲线绘图工具"""
    if not history: return
    epochs = [h['epoch'] for h in history]
    accs = [h['test_label_acc'] for h in history]
    
    plt.figure()
    plt.plot(epochs, accs, label='Test Acc')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'history.png'))
    plt.close()