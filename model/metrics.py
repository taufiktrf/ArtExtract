import torch.nn.functional as F
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
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
        # loss = torch.mean(torch.abs(output_features - target_features))
        output_scaled = output_features / output_features.max()
        target_scaled = target_features / target_features.max()
        
        loss = torch.mean(torch.abs(output_scaled - target_scaled))
        return loss

class PixelwiseLoss(nn.Module):
    def __init__(self):
        super(PixelwiseLoss, self).__init__()

    def forward(self, output, target):
        output_scaled = output / output.max()
        target_scaled = target / target.max()
        loss = torch.mean(torch.abs(output_scaled - target_scaled))
        return loss
    
class BCELogitLoss(nn.Module):
    def __init__(self):
        super(BCELogitLoss, self).__init__()

    def forward(self, output, target):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(output,target)

class PSNR_metrics(nn.Module):
    def __init__(self):
        super(PSNR_metrics, self).__init__()

    def forward(self, output, target):
        output_scaled = output / output.max()
        target_scaled = target / target.max()
        
        mse = torch.mean((target_scaled - output_scaled) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        PSNR = 10 * torch.log10(max_pixel / torch.sqrt(mse))
        return PSNR
    
class RRMSE_metrics(nn.Module):
    def __init__(self):
        super(RRMSE_metrics, self).__init__()
            
    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError("Images must have the same dimensions")
        output_scaled = output / output.max()
        target_scaled = target / target.max()

        mse = torch.mean((target_scaled - output_scaled)**2)
        rmse = torch.sqrt(mse)
<<<<<<< Updated upstream
        mean_img = torch.mean(output_scaled)
        rrmse = rmse / mean_img
=======
        mean_output = torch.mean(output)
        rrmse = rmse / mean_output
>>>>>>> Stashed changes
        return rrmse
    
class SSIM_metrics(nn.Module):
    def __init__(self, size_average=True):
        super(SSIM_metrics, self).__init__()
        self.size_average = size_average

    def forward(self, output, target):
        output_scaled = output / output.max()
        target_scaled = target / target.max()
        output_np = output_scaled.detach().cpu().numpy()
        target_np = target_scaled.detach().cpu().numpy()
        
        if output_np.ndim == 4:  # (N, C, H, W) to (N, H, W, C)
            output_np = np.transpose(output_np, (0, 2, 3, 1))
            target_np = np.transpose(target_np, (0, 2, 3, 1))

        ssim_values = []
        for i in range(output_np.shape[0]):
            ssim_val = ssim(output_np[i], target_np[i], multichannel=True, data_range=target_np[i].max() - target_np[i].min())
            ssim_values.append(ssim_val)
        
        ssim_values = np.array(ssim_values)
        if self.size_average:
            return torch.tensor(ssim_values.mean())
        else:
            return torch.tensor(ssim_values)

