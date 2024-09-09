import torch
import torch.nn as nn
from torchvision import models

class Convlayer(nn.Module):
    def __init__(self):
        super(Convlayer, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.elu = nn.LeakyReLU(0.02)
        
        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.bn_out = nn.BatchNorm2d(1)
        
    def forward(self, x):
        residual = x  
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        
        x = self.conv_out(x)
        x = self.bn_out(x)
        
        x = x + residual
        x = self.elu(x)
        
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = Convlayer()  
        self.conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, gt, refs, thresh=0.7):
        gt = self.conv(gt)
        gt_feat = self.cnn(gt)
        batch_size, c, h, w = gt_feat.shape
        # Shape: [batch_size, c, h, w]
        
        # refs_feat has shape: [num_refs, batch_size, c, h, w]
        refs_feat = torch.stack([self.cnn(ref) for ref in refs], dim=0)
        # Shape: [num_refs, batch_size, c, h, w]
       
        # gt_feat_expanded shape: [batch_size, 1, c, h, w]
        gt_feat_expanded = gt_feat.unsqueeze(1)
        # Shape: [batch_size, 1, c, h, w]
        
        # diff_maps shape: [batch_size, num_refs, c, h, w]
        diff_maps = torch.abs(gt_feat_expanded - refs_feat)
        # Shape: [batch_size, num_refs, c, h, w]
        
        # Sum the differences across channels
        # Shape: [batch_size, num_refs, h, w]
        total_dissim_map = diff_maps.sum(dim=2)

        # Min across all reference images (num_refs) for each batch, for every pixel (h, w)
        min_dissim = total_dissim_map.min(dim=1, keepdim=True)[0]  # Min across num_refs
        min_dissim = min_dissim.view(batch_size, 1, h, w)  # Shape: [batch_size, 1, h, w]
        
        # Max across all reference images (num_refs) for each batch, for every pixel (h, w)
        max_dissim = total_dissim_map.max(dim=1, keepdim=True)[0]  # Max across num_refs
        max_dissim = max_dissim.view(batch_size, 1, h, w)  # Shape: [batch_size, 1, h, w]

        normalized_dissim_map = (total_dissim_map - min_dissim) / (max_dissim - min_dissim + 1e-8)
        # Shape: [batch_size, num_refs, h, w]
        
        normalized_dissim_map = torch.clamp(normalized_dissim_map, min=0.1, max=0.9)
        
        # Thresholding
        dissim_map = torch.where(normalized_dissim_map > thresh, normalized_dissim_map, torch.zeros_like(normalized_dissim_map))
        # Shape: [batch_size, num_refs, h, w]
        
        # Find the index of the maximum dissimilarity value per pixel
        _, ref_idx_map = torch.max(dissim_map, dim=1)
        # ref_idx_map Shape: [batch_size, h, w]
        
        ref_idx_map = ref_idx_map.unsqueeze(1)
        # Shape: [batch_size, 1, h, w]
        
        collage = torch.zeros((batch_size, 1, h, w), device=gt.device)
        # Shape: [batch_size, 1, h, w]
        
        for ref_idx in range(len(refs)):
            ref_img = refs_feat[:, ref_idx, :, :, :]
            # Shape: [batch_size, c, h, w]
            
            mask = (ref_idx_map == ref_idx)
            # Shape: [batch_size, 1, h, w]
            
            collage = torch.where(mask, ref_img, collage)
        
        # Shape: [batch_size, 1, h, w]
        return dissim_map, collage
