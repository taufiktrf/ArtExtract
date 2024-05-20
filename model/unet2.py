import torch.nn as nn
import torch
    
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
        self.up1 = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.up2 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.up3 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.up4 = nn.ConvTranspose2d(128,64,2,stride=2)
        
        self.prelu = nn.PReLU(1,0.2)

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
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.prelu(self.decoder_conv1(x))

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.prelu(self.decoder_conv2(x))
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.prelu(self.decoder_conv3(x))
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.prelu(self.decoder_conv4(x))
        x = self.prelu(self.decoder_conv5(x))
        
        return x
    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        
        self.block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        )
        
        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        
        self.batchNorm = nn.BatchNorm2d(out_channels * 4)  
        # Applying batch norm after concatenation
        self.dropout = nn.Dropout2d(0.5)
        self.final_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        
    def forward(self, x):
        out_block1 = self.block1(x)
        out_block2 = self.block2(x)
        out_block3 = self.block3(x)
        out_block4 = self.block4(x)
        
        concat_block = torch.cat([out_block1, out_block2, out_block3, out_block4], dim=1)
        concat_block = self.batchNorm(concat_block)
        concat_block = self.dropout(concat_block)
        final_out = self.final_conv(concat_block)
        return final_out
        
