import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, stride=1, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
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
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=64):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = ConvBlock(in_channels, features)
        self.down1 = DownBlock(features, features * 2)
        self.down2 = DownBlock(features * 2, features * 4)
        self.down3 = DownBlock(features * 4, features * 8)
        self.down4 = DownBlock(features * 8, features * 16)
        self.up1 = UpBlock(features * 16, features * 8)
        self.up2 = UpBlock(features * 8, features * 4)
        self.up3 = UpBlock(features * 4, features * 2)
        self.up4 = UpBlock(features * 2, features)
        self.outc = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=64):
        super(ResNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(features, features, 2)
        self.layer2 = self._make_layer(features, features * 2, 2, stride=2)
        self.layer3 = self._make_layer(features * 2, features * 4, 2, stride=2)
        self.layer4 = self._make_layer(features * 4, features * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(features * 8, out_channels)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class Generator(nn.Module):
    def __init__(self, backbone, in_channels, out_channels, features=64):
        super(Generator, self).__init__()
        self.backbone = backbone
        if backbone == 'unet':
            self.model = UNet3D(in_channels, out_channels, features)
        elif backbone == 'resnet':
            self.model = ResNet3D(in_channels, out_channels, features)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.final_activation = nn.Tanh()

    def forward(self, x):
        x = self.model(x)
        return self.final_activation(x)

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
        
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        
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