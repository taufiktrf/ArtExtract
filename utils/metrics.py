import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim


#Loss Metrics
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
        self.ssim_metric = ssim(data_range=1.0)

    def psnr(self, output, target):
        mse = torch.mean((target - output) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr_value = 10 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr_value
    
    def ssim(self, output, target):
        output = output.unsqueeze(0)
        target = target.unsqueeze(0)
        ssim_value = self.ssim_metric(output, target)
        return ssim_value
        
    def lpips(self, output, target):
        output_features = self.feature_extractor(output)
        target_features = self.feature_extractor(target)
        loss = self.loss_fn(output_features, target_features)
        return loss
    
    def forward(self, output, target):
        psnr_value = self.psnr(output, target)
        lpips_value = self.lpips(output,target)
        ssim_value = self.ssim(output, target)
        return psnr_value, lpips_value, ssim_value

