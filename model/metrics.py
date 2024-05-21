import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import math

class FeatureLoss(nn.Module):
    def __init__(self, vgg_feature_extractor):
        super(FeatureLoss, self).__init__()
        self.vgg_feature_extractor = vgg_feature_extractor

    def forward(self, output, target):
        output_features = self.vgg_feature_extractor(output)
        target_features = self.vgg_feature_extractor(target)
    
        loss = torch.mean(torch.abs(output_features - target_features))
        # print('feature-',loss.item())
        return loss

class PixelwiseLoss(nn.Module):
    def __init__(self):
        super(PixelwiseLoss, self).__init__()

    def forward(self, output, target):
        # Calculate L1 loss
        loss = torch.mean(torch.abs(output - target))
        # print('pixel-',loss.item())
        return loss

class PSNR_metrics(nn.Module):
    def __init__(self):
        super(PSNR_metrics, self).__init__()

    def forward(self, output, target):
        mse = torch.mean((target - output) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        PSNR = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return PSNR


