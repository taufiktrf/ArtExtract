import torch.nn as nn
import torch.nn.functional as F
import torch


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

