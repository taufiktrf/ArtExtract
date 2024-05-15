import torch.nn.functional as F
import torch.nn as nn
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder_conv1 = InceptionBlock(3, 64)
        self.encoder_conv2 = InceptionBlock(64, 128)
        self.encoder_conv3 = InceptionBlock(128, 256)
        self.encoder_conv4 = InceptionBlock(256, 512)
        self.encoder_conv5 = InceptionBlock(512,1024)
        
        # Decoder
        self.decoder_conv1 = InceptionBlock(1024, 512)
        self.decoder_conv2 = InceptionBlock(512, 256)
        self.decoder_conv3 = InceptionBlock(256, 128)
        self.decoder_conv4 = InceptionBlock(128, 64)
        self.decoder_conv5 = nn.Conv2d(64, 8, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.prelu = nn.PReLU(0.2)

    def forward(self, x):
        # Encoder
        x1 = self.prelu(self.encoder_conv1(x))
        x2 = self.pool(x1)
        x2 = self.prelu(self.encoder_conv2(x2))
        x3 = self.pool(x2)
        x3 = self.prelu(self.encoder_conv3(x3))
        x4 = self.pool(x3)
        x4 = self.prelu(self.encoder_conv4(x4))
        x5 = self.pool(x4)
        x5 = self.prelu(self.encoder_conv5(x5))
        
        # Decoder
        x = self.up(x5)
        x = self.prelu(self.decoder_conv1(x))
        x = torch.cat([x, x4], dim=1)
        x = self.up(x4)
        x = self.prelu(self.decoder_conv1(x))
        x = torch.cat([x, x3], dim=1)
        x = self.up(x)
        x = self.prelu(self.decoder_conv2(x))
        x = torch.cat([x, x2], dim=1)
        x = self.up(x)
        x = self.prelu(self.decoder_conv3(x))
        x = torch.cat([x, x1], dim=1)
        x = self.decoder_conv4(x)
        x = self.decoder_conv5(x)
        return x
    
class InceptionBlock(nn.Module):  
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # The paper didn't mention about wher ethe batch norm and dropout layer were applied
        self.block1 = nn.Conv2d(in_channels, out_channels, 1)
        self.block2 = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Conv2d(1, 3, 3))
        self.block3 = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Conv2d(1, 5, 5))
        self.block4 = nn.Sequential(nn.MaxPool2d(3, 3), nn.Conv2d(in_channels, 1, 1))
        self.batchNorm = nn.BatchNorm2d()
        self.dropout = nn.Dropout2d(0.5)
        
    def forward(self, x):
        x = batchNorm(x)
        out_block1 = self.block1(x)
        out_block2 = self.block2(x)
        out_block3 = self.block3(x)
        out_block4 = self.block4(x)
        
        concat_block = torch.cat([out_block1, out_block2, out_block3, out_block4], dim=1)
        concat_block = dropout(concat_block)
        return concat_block
        
