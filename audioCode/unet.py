

import torch
import torch.nn as nn


# ================= Unet ========================
# Ref: https://www.kaggle.com/code/balraj98/unet-for-building-segmentation-pytorch
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels - out_channels, in_channels - out_channels, kernel_size=2,
                                                stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        cut = x.shape[3] if x.shape[3] <= skip_input.shape[3] else skip_input.shape[3]
        x = torch.cat([x, skip_input[:,:,:,:cut]], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_classes=2, up_sample_mode='bilinear'):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Down sampling Path
        self.down_conv1 = DownBlock(in_ch, 8)
        self.down_conv2 = DownBlock(8, 16)
        self.down_conv3 = DownBlock(16, 32)
        self.down_conv4 = DownBlock(32, 64)

        # Bottleneck
        self.double_conv = DoubleConv(64, 128)
        # Sampling Path
        self.up_conv4 = UpBlock(64 + 128, 64, self.up_sample_mode)
        self.up_conv3 = UpBlock(32 + 64, 32, self.up_sample_mode)
        self.up_conv2 = UpBlock(16 + 32, 16, self.up_sample_mode)
        self.up_conv1 = UpBlock(24, 32, self.up_sample_mode)

        # Final Convolution
        self.conv_last = nn.Conv2d(32, 64, kernel_size=1)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=2)


    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)
        return x
