import torch
import utils.dataset as dataset
from utils.dataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
from model.PLGMNet import PLGMNet
from loss.hybridloss import hybridLoss
import numpy as np
from datetime import datetime
from utils.model_init import seed_torch
import os

seed_torch()
device = torch.device("cuda:1")

def train():
    date_time = datetime.now()
    date_str = date_time.strftime("%Y-%m-%d_%H-%M-%S")

    train_data = dataset.my_dataset('data/train.txt')
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

    model = PLGMNet().to(device)

    loss_fn = hybridLoss().to(device)

    learning_rate = 1.5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    epoch = 1000
    total_train_step = 0
    save_path = f'save/{model.__class__.__name__}_{date_str}/'
    os.makedirs(save_path, exist_ok=True)
    for i in range(epoch):
        print(f'-------第{i+1}轮训练开始---------')

        model.train()
        for data in train_dataloader:
            imgs, labels = data['image'], data['label']
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 25 == 0:
                print(f"训练次数：{total_train_step}，Loss：{loss.item()}")

        # if (i+1)%10 == 0 and (i+1) > 500:
        torch.save(model.state_dict(), save_path+f"epoch{i+1}.pth")
        print("模型已保存")


if __name__ == "__main__":
    train()
