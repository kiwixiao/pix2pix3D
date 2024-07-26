import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(Generator, self).__init__()
        # Encoder (downsampling) blocks
        self.enc1 = self.conv_block(in_channels, features)
        self.enc2 = self.conv_block(features, features*2)
        self.enc3 = self.conv_block(features*2, features*4)
        self.enc4 = self.conv_block(features*4, features*8)
        
        # Decoder (upsampling) blocks
        self.dec1 = self.upconv_block(features*8, features*4)
        self.dec2 = self.upconv_block(features*8, features*2)
        self.dec3 = self.upconv_block(features*4, features)
        self.dec4 = self.upconv_block(features*2, out_channels)
        
        # Final convolution to produce output
        self.final = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        """Convolutional block with batch normalization and ReLU activation"""
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def upconv_block(self, in_c, out_c):
        """Upconvolutional block with batch normalization and ReLU activation"""
        return nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Decoder path with skip connections
        d1 = self.dec1(e4)
        d2 = self.dec2(torch.cat([d1, e3], dim=1))
        d3 = self.dec3(torch.cat([d2, e2], dim=1))
        d4 = self.dec4(torch.cat([d3, e1], dim=1))
        
        return self.final(d4)

class Discriminator(nn.Module):
    def __init__(self, in_channels, features):
        super(Discriminator, self).__init__()
        # Convolutional blocks for discrimination
        self.conv1 = self.conv_block(in_channels, features)
        self.conv2 = self.conv_block(features, features*2)
        self.conv3 = self.conv_block(features*2, features*4)
        self.conv4 = self.conv_block(features*4, features*8)
        # Final convolution to produce output
        self.final = nn.Conv3d(features*8, 1, kernel_size=1)

    def conv_block(self, in_c, out_c):
        """Convolutional block with batch normalization and LeakyReLU activation"""
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.final(x)