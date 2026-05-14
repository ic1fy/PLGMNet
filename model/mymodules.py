import torch.nn as nn
import torch
import torch.nn.functional as F
from mamba_ssm import Mamba
import copy
import numpy as np
from torchvision import transforms
from einops import rearrange
from model.vssblock import VSSBlock
from model.transformer import VisionTransformer

class convbnrelu(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, g=1, d=1, bias=False):
        super(convbnrelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k, s, p, groups=g, dilation=d, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.layer(x)

class DWConv(nn.Module):
    def __init__(self, in_channels, k=3, s=2, d=1):
        super(DWConv, self).__init__()
        self.layer = convbnrelu(in_channels, in_channels, k, s, p=d, g=in_channels, d=d)

    def forward(self, x):
        return self.layer(x)
    

class PWConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PWConv, self).__init__()
        self.layer = convbnrelu(in_channels, out_channels, k=1, s=1, p=0, g=1, d=1)

    def forward(self, x):
        return self.layer(x)
    

class DSConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, s=1, d=1):
        super(DSConv3x3, self).__init__()
        self.layer = nn.Sequential(
            DWConv(in_channels, k=3, s=s, d=d),
            PWConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layer(x)
    
class ChannelAttention(nn.Module):   #CA
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):  #SA
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)
    
class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channel)

    def forward(self, x):
        x1_ca = x.mul(self.ca(x))
        x1_sa = x1_ca.mul(self.sa(x1_ca))
        x = x + x1_sa
        return x

    
class GlobalMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.p = p
        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,
        )

     
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)

        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,

        )
       
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
       
    def forward(self, x, cls_token):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C = x.shape[:2]

        assert C == self.dim
   
        img_dims = x.shape[2:]

        H,W = x.shape[2:]

        if H%self.p==0 and W%self.p==0:
            pool_layer = nn.AvgPool2d(self.p, stride=self.p)
            x_div = pool_layer(x)
            x_div = x_div + cls_token
        else:
            print(0)
            x_div = x
        
        NH,NW = x_div.shape[2:]

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        y_norm = torch.flip(x_norm, dims=[1])

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))

        if H%self.p==0 and W%self.p==0:
            x_out = x_out.transpose(-1, -2).reshape(B, C, NH, NW)
            x_out = interpolate(x_out, img_dims)
        else:
            x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
                
        out = x_out + x

        return out
    


class LocalMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.learnable_token_forw = nn.Parameter(torch.randn(1, 1, dim))
        self.learnable_token_backw = nn.Parameter(torch.randn(1, 1, dim))
        self.cnn = convbnrelu(dim, dim, k=p, s=p, p=0)

        self.mamba_forw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,
        )

     
        self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)


        
        self.mamba_backw = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                use_fast_path=False,

        )

      
        # adjust the window size here to fit the feature map
        self.p = p
        self.mamba_forw.out_proj = nn.Identity()
        self.mamba_backw.out_proj = nn.Identity()
       
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C = x.shape[:2]

        assert C == self.dim
        img_dims = x.shape[2:]

        H,W = x.shape[2:]

        if H%self.p==0 and W%self.p==0:             
            x_div = x.reshape(B, C, H//self.p, self.p, W//self.p, self.p).permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.p, self.p)
        else:
            print(0)
            x_div = x
        

        NB = x_div.shape[0]
        NH,NW = H//self.p, W//self.p

        n_tokens = x_div.shape[2:].numel()
   

        x_flat = x_div.reshape(NB, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        y_norm = torch.flip(x_norm, dims=[1])

        # --- 添加 learnable token ---
        token_forw = self.learnable_token_forw.expand(NB, -1, -1)  # (NB, 1, C)
        token_backw = self.learnable_token_backw.expand(NB, -1, -1)  # (NB, 1, C)

        x_norm = torch.cat([x_norm, token_forw], dim=1)  # 增加 learnable token at the end
        y_norm = torch.cat([y_norm, token_backw], dim=1)

        x_mamba = self.mamba_forw(x_norm)
        y_mamba = self.mamba_backw(y_norm)

        x_mamba = x_mamba[:, :-1, :]
        y_mamba = y_mamba[:, :-1, :]

        cls_token = self.out_proj(x_mamba[:, -1:, :] + y_mamba[:,-1:,:])  # 获取 learnable token 的输出

        x_out = self.out_proj(x_mamba + torch.flip(y_mamba, dims=[1]))


        if H%self.p==0 and W%self.p==0:
            x_out = x_out.transpose(-1, -2).reshape(B, NH, NW, C, self.p, self.p).permute(0, 3, 1, 4, 2, 5).contiguous().reshape(B, C, *img_dims)
            cnn_cls = self.cnn(x_out)
            cls_token = cls_token.transpose(-1, -2).reshape(B, NH, NW, C).permute(0,3,1,2) + cnn_cls
        else:
            x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)
        out = x_out + x

        return out, cls_token
    

class MEGM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MEGM, self).__init__()
        self.conv1 = DSConv3x3(in_channels, in_channels, 1, 1)
        self.conv2 = DSConv3x3(in_channels, in_channels, 1, 1)
        self.dconv = DSConv3x3(in_channels, in_channels, 1, 4)
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.avg_pool = nn.AvgPool2d(3, 1, 1)
        self.conv3 = DSConv3x3(in_channels, out_channels, 1, 1)


    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.max_pool(x1) - self.avg_pool(x1)

        x2 = F.avg_pool2d(self.dconv(x), 3, 1, 1)

        x3 = self.conv2(x)

        out = self.conv3(x1+x2+x3)
        out = out + out*F.sigmoid(out-F.avg_pool2d(out, 3, 1, 1))
        return out

interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)


