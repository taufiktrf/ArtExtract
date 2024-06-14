# class InceptionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(InceptionBlock, self).__init__()        
#         self.block1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
#             nn.BatchNorm2d(out_channels),
#             nn.Dropout2d(0.5),
#             nn.PReLU(out_channels,0.2)
#         )
        
#         self.block2 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.Dropout2d(0.5),
#             nn.PReLU(out_channels,0.2)
#         )
        
        # self.block3 = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(out_channels),
        #     nn.Dropout2d(0.5),
        #     nn.PReLU(out_channels,0.2)
        # )
        
#         self.block4 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.Dropout2d(0.5),
#             nn.PReLU(out_channels,0.2)
#         )
        
#         self.final_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        
    # def forward(self, x):
    #     out_block1 = self.block1(x)
        # out_block2 = self.block2(x)
        # out_block3 = self.block3(x)
#         out_block4 = self.block4(x)
        
#         concat_block = torch.cat([out_block1, out_block2, out_block3, out_block4], dim=1)
#         final_out = self.final_conv(concat_block)
        # return out_block1