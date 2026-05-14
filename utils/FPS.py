import numpy as np
import torch
import time
from tqdm import tqdm
from thop import profile, clever_format
import torch

def computeTime(model, device='cuda'):
    inputs = torch.randn(32, 3, 384, 384).float()
    model = model.cuda()
    inputs = inputs.cuda()

    model.eval()

    time_spent = []
    for idx in tqdm(range(100)):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 20:
            time_spent.append(time.time() - start_time)
    flops, params = profile(model, inputs=(inputs, ))
    print('Average speed: {:.4f} fps'.format(32 / np.mean(time_spent)))
    print('FLOPs:' + str(flops/1000**3/32) + 'G')
    print('Params:' + str(params/1000**2) + 'M')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    from model.PLGMNet import PLGMNet as Model
    model = Model()

    computeTime(model, device='cuda')
