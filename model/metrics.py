import torch.nn.functional as F
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim
import numpy as np
import torch
import math

#Loss Metrics
class FeatureLoss(nn.Module):
    def __init__(self, vgg_feature_extractor):
        super(FeatureLoss, self).__init__()
        self.vgg_feature_extractor = vgg_feature_extractor

    def forward(self, output, target):
        with torch.cuda.amp.autocast():
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
    
class MS_SSIMLoss(nn.Module):
    def __init__(self):
        super(MS_SSIMLoss, self).__init__()

    def forward(self, output, target):
        output = output.unsqueeze(0)
        target = target.unsqueeze(0)
        loss = 1 - ms_ssim(output, target, data_range=1.0, size_average=True)
        return loss

#Evaluation Metrics
class EvalMetrics(nn.Module):
    def __init__(self, feature_extractor, size_average=True):
        super(EvalMetrics, self).__init__()
        self.size_average = size_average
        self.feature_extractor = feature_extractor
        self.loss_fn = nn.MSELoss()

    def psnr(self, output, target):
        mse = torch.mean((target - output) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr_value = 10 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr_value

#     def rrmse(self, output, target):
#         if output.shape != target.shape:
#             raise ValueError("Images must have the same dimensions")
                
#         mse = torch.mean((target - output) ** 2)
#         rmse = torch.sqrt(mse)
#         mean_output = torch.mean(output)
#         rrmse_value = rmse / (mean_output + 1e-8)
#         return rrmse_value
    
    def ssim(self, output, target):
        output_np = output.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
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
        
    def lpips(self, output, target):
        output_features = self.feature_extractor(output)
        target_features = self.feature_extractor(target)
        loss = self.loss_fn(output_features, target_features)
        return loss
    
    def forward(self, output, target):
        psnr_value = self.psnr(output, target)
        # rrmse_value = self.rrmse(output, target)
        lpips_value = self.lpips(output,target)
        ssim_value = self.ssim(output, target)
        return psnr_value, lpips_value, ssim_value

