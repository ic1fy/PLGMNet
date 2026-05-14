import numpy as np
import torch
import random
import os
import torch.nn as nn

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def init_xavier(m):
    if hasattr(m, 'weight') and m.weight is not None and not isinstance(m, nn.BatchNorm2d):
        nn.init.xavier_normal_(m.weight)  # 使用 Xavier 初始化权重
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)  # 将偏置初始化为0