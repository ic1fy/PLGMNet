import torch
import torch.nn as nn
import loss.pytorch_ssim as pytorch_ssim
from loss.CEL import CEL, IOU
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

class hybridLoss(nn.Module):
    def __init__(self):
        super(hybridLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.ssim_loss = pytorch_ssim.SSIM(window_size=11)
        self.iou = IOU()
        self.cel = CEL()
    
    def forward(self, preds, target):
        total_loss = 0
        for pred in preds:
            loss = self.total_loss(pred, target)
            total_loss = total_loss + loss


        return total_loss

    def total_loss(self, pred, target):
        bce_loss = self.bce_loss(pred, target)
        ssim_loss = 1 - self.ssim_loss(pred, target)
        iou_loss = self.iou(pred, target)
        cel_loss = self.cel(pred, target)

        return ssim_loss + bce_loss + iou_loss + cel_loss


