import torch
from PIL import Image
import torch.nn as nn
from utils.dataset import *
from torch.utils.data import DataLoader
from utils.sod_metrics import evaluate
import utils.dataset as dataset
from datetime import datetime
import os
from utils.FPS import computeTime
from model.PLGMNet import PLGMNet
torch.backends.cudnn.benchmark = True   

device = torch.device("cuda:1")

def normPRED(x):
    MAX = torch.max(x)
    MIN = torch.min(x)

    out = (x - MIN) / (MAX - MIN)

    return out

def save_output(image_name, pred, save_dir):
    predict = pred.to(torch.float32)
    predict = predict.squeeze()
    predict = predict.cpu().detach()

    predict = transforms.ToPILImage()(predict).convert('L')

    predict = predict.resize((200, 200), resample=Image.NEAREST)

    predict.save(save_dir + image_name + '.png')


def test(path):

    date_time = datetime.now()
    date_str = date_time.strftime("%Y-%m-%d_%H-%M-%S")

    test_data = dataset.my_dataset('data/test.txt', is_train=False)
    test_dataloader = DataLoader(test_data, batch_size=32)

    model = PLGMNet()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model = model.to(device)

    save_dir = f'output/{model.__class__.__name__}_{date_str}/'
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            
            imgs = data['image'].to(device)
            imgs_name = data['name']
            outputs = model(imgs)

            for j in range(outputs[0].shape[0]):
                pred = outputs[0][j, 0, :, :]
                pred = normPRED(pred)
                save_output(imgs_name[j], pred, save_dir)

    pred_path = save_dir
    gt_path = 'data/Test/Mask/'
    results = evaluate(pred_path, gt_path)
    for name, score in results.items():
        print(f"{name}: {score}")
    print('\n')


if __name__ == '__main__':


    model = PLGMNet()
    
    path = "PLGMNet.pth"

    computeTime(model, device=device)

    test(path)
