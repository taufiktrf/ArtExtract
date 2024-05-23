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
        return loss

class PixelwiseLoss(nn.Module):
    def __init__(self):
        super(PixelwiseLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.mean(torch.abs(output - target))
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
    
class RRMSE_metrics(nn.Module):
    def __init__(self):
        super(RRMSE_metrics, self).__init__()
            
    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError("Images must have the same dimensions")
        mse = torch.mean((target - output)**2)
        rmse = torch.sqrt(mse)
        mean_img = torch.mean(output)
        rrmse = rmse / mean_img
        return rrmse


