import torch.nn as nn

class FeatureLoss(nn.Module):
    def __init__(self, vgg_feature_extractor):
        super(FeatureLoss, self).__init__()
        self.vgg_feature_extractor = vgg_feature_extractor
        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        generated_features = self.vgg_feature_extractor(output)
        target_features = self.vgg_feature_extractor(target)
        return self.criterion(generated_features, target_features)

class PixelwiseLoss(nn.Module):
    def __init__(self):
        super(PixelwiseLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        return self.criterion(output, target)

