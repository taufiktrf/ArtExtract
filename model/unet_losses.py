import torch.nn as nn
import torch.nn.functional as F
import torch


class FeatureLoss(nn.Module):
    def __init__(self, vgg_feature_extractor):
        super(FeatureLoss, self).__init__()
        self.vgg_feature_extractor = vgg_feature_extractor
        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        return self.criterion(output, target)


class PixelwiseLoss(nn.Module):
    def __init__(self):
        super(PixelwiseLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        return self.criterion(output, target)

