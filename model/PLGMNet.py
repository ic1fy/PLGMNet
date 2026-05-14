import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from mamba_ssm import Mamba
from model.mymodules import convbnrelu, DSConv3x3, CBAM, LocalMambaLayer, GlobalMambaLayer, MEGM

class PLGIMLayer(nn.Module):
    def __init__(self, dim, p):
        super(PLGIMLayer, self).__init__()
        self.BiPixelMambaLayer = LocalMambaLayer(dim, p)
        self.BiWindowMambaLayer = GlobalMambaLayer(dim, p)
        self.conv = convbnrelu(2*dim, dim, k=3, s=1, p=1)
        
    def forward(self, x):
        x1, cls_token = self.BiPixelMambaLayer(x)
        x2 = self.BiWindowMambaLayer(x, cls_token)
        out = torch.cat((x1, x2), dim=1)
        out = self.conv(out)
        
        return out

class encoding(nn.Module):
    def __init__(self):
        super(encoding, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            DSConv3x3(16, 32, s=2),
            PLGIMLayer(32, 4),
        )

        self.stage3 = nn.Sequential(
            DSConv3x3(32, 64, s=2),
            PLGIMLayer(64, 4),
        )

        self.stage4 = nn.Sequential(
            DSConv3x3(64, 96, s=2),
            PLGIMLayer(96, 4),
        )

        self.stage5 = nn.Sequential(
            DSConv3x3(96, 128, s=2),
            PLGIMLayer(128, 4),
        )



    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        return x1, x2, x3, x4, x5

class fuse_stage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(fuse_stage, self).__init__()
        self.layer = nn.Sequential(
            MEGM(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layer(x)
    
class output_layer(nn.Module):
    def __init__(self, in_channels):
        super(output_layer,self).__init__()
        self.layer = nn.Sequential(
            nn.Dropout2d(p=0.1, inplace=False),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.layer(x)

class PLGMNet(nn.Module):
    def __init__(self):
        super(PLGMNet, self).__init__()
        self.encoding = encoding()
        self.fuse_stage5 = fuse_stage(128, 96)
        self.fuse_stage4 = fuse_stage(96, 64)
        self.fuse_stage3 = fuse_stage(64, 32)
        self.fuse_stage2 = fuse_stage(32, 16)
        self.fuse_stage1 = fuse_stage(16, 3)

        self.output_layer5 = output_layer(96)
        self.output_layer4 = output_layer(64)
        self.output_layer3 = output_layer(32)
        self.output_layer2 = output_layer(16)
        self.output_layer1 = output_layer(3)

        self.Gate1 = CBAM(16)
        self.Gate2 = CBAM(32)
        self.Gate3 = CBAM(64)
        self.Gate4 = CBAM(96)
        self.Gate5 = CBAM(128)

        self.con_AIM5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.con_AIM4 = nn.Conv2d(96, 96, 3, 1, 1)
        self.con_AIM3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.con_AIM2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.con_AIM1 = nn.Conv2d(16, 16, 3, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoding(x)

        x1 = self.con_AIM1(self.Gate1(x1))
        x2 = self.con_AIM2(self.Gate2(x2))
        x3 = self.con_AIM3(self.Gate3(x3))
        x4 = self.con_AIM4(self.Gate4(x4))
        x5 = self.con_AIM5(self.Gate5(x5))

        fuse_x5 = self.fuse_stage5(x5)

        temp = interpolate(fuse_x5, x4.size()[2:])
        fuse_x4 = self.fuse_stage4(x4+temp)

        temp = F.interpolate(fuse_x4, x3.size()[2:])
        fuse_x3 = self.fuse_stage3(x3+temp)

        temp = F.interpolate(fuse_x3, x2.size()[2:])
        fuse_x2 = self.fuse_stage2(x2+temp)

        temp = F.interpolate(fuse_x2, x1.size()[2:])
        fuse_x1 = self.fuse_stage1(x1+temp)

        output_5 = interpolate(self.output_layer5(fuse_x5), x.size()[2:])
        output_4 = interpolate(self.output_layer4(fuse_x4), x.size()[2:])
        output_3 = interpolate(self.output_layer3(fuse_x3), x.size()[2:])
        output_2 = interpolate(self.output_layer2(fuse_x2), x.size()[2:])
        output_1 = interpolate(self.output_layer1(fuse_x1), x.size()[2:])


        return output_1, output_2, output_3, output_4, output_5


interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)


