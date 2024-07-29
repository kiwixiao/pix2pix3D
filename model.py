import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ensure x1 and x2 have the same spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=32):
        super(UNet3D, self).__init__()
        self.encoder1 = ConvBlock(in_channels, features)
        self.encoder2 = ConvBlock(features, features * 2)
        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.encoder4 = ConvBlock(features * 4, features * 8)
        
        self.bottleneck = ConvBlock(features * 8, features * 16)
        
        self.up4 = UpBlock(features * 16, features * 8)
        self.up3 = UpBlock(features * 8, features * 4)
        self.up2 = UpBlock(features * 4, features * 2)
        self.up1 = UpBlock(features * 2, features)
        
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        b = self.bottleneck(e4)
        
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        
        return self.final_conv(d1)

class Discriminator(nn.Module):
    def __init__(self, in_channels, features=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self._conv_block(in_channels, features, stride=2),
            self._conv_block(features, features*2, stride=2),
            self._conv_block(features*2, features*4, stride=2),
            self._conv_block(features*4, features*8, stride=2),
            nn.Conv3d(features*8, 1, kernel_size=1)
        )

    def _conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, backbone, in_channels, out_channels, features=64):
        super(Generator, self).__init__()
        self.backbone = backbone
        if backbone == 'unet':
            self.model = UNet3D(in_channels, out_channels, features)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.final_activation = nn.Tanh()

    def forward(self, x):
        x = self.model(x)
        return self.final_activation(x)

class ComboLoss(nn.Module):
    def __init__(self, weight=0.5, smooth=1):
        super(ComboLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = (pred + 1) / 2  # Transform from [-1, 1] to [0, 1]
        pred = pred.contiguous()
        target = target.contiguous()    
        intersection = (pred * target).sum(dim=2).sum(dim=2).sum(dim=2)
        dice_loss = 1 - ((2. * intersection + self.smooth) / 
                    (pred.sum(dim=2).sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2).sum(dim=2) + self.smooth))
        
        bce_loss = F.binary_cross_entropy_with_logits((pred + 1) / 2, target, reduction='mean')
        
        combo_loss = (self.weight * dice_loss) + ((1 - self.weight) * bce_loss)
        return combo_loss.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = (pred + 1) / 2  # Transform from [-1, 1] to [0, 1]
        pred = pred.contiguous()
        target = target.contiguous()
        
        true_pos = (pred * target).sum(dim=2).sum(dim=2).sum(dim=2)
        false_neg = (target * (1-pred)).sum(dim=2).sum(dim=2).sum(dim=2)
        false_pos = ((1-target) * pred).sum(dim=2).sum(dim=2).sum(dim=2)
        
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha*false_neg + self.beta*false_pos + self.smooth)
        focal_tversky = (1 - tversky)**self.gamma
        
        return focal_tversky.mean()