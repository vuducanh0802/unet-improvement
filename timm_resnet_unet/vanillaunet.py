import pathlib
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# import common_utility.typing_compat as T


def SingleConv(in_channels, out_channels, dilation=1, dropout=0.):
    layers = [
        ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, dilation=dilation)),
        ("bn", nn.BatchNorm2d(out_channels)),
        ("relu", nn.ReLU(inplace=True)),
    ]
    if dropout != 0:
        layers.append(("dropout", nn.Dropout2d(dropout)))
    return nn.Sequential(OrderedDict(layers))


def DoubleConv(in_channels, out_channels, mid_channels=None):
    if mid_channels is None:
        mid_channels = out_channels
    return nn.Sequential(
        OrderedDict(
            [
                ("conv1", SingleConv(in_channels, mid_channels)),
                ("conv2", SingleConv(mid_channels, out_channels))
            ]
        )
    )


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    An implementation of UNet. This implementation works with all input sizes; however,
    it is recommended that image sizes are divisible by 16.
    References:
        https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
        https://github.com/milesial/Pytorch-UNet
    Args:
        in_channels:    number of input channels
        out_channels:   number of output channels
        init_features:  number of channels in the first block. Each subsequent block
                        has twice the number of channels of the previous block. This
                        number decide how big the model is.
        bilinear:       whether to use Bilinear or ConvTranspose for upsampling.
        bottleneck:     a customized bottleneck instead of using the default bottleneck.
                        Number of input channels must be init_features*8; and number of
                        output channels must be init_features*8 when bilinear is True and
                        init_features*16 when bilinear is False. This is useful if you
                        want to use Dilated Convolution or Dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        bilinear: bool = True,
        bottleneck: nn.Module = None,
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.bilinear = bilinear

        features = init_features
        self.stem = DoubleConv(in_channels, features)
        self.down1 = Down(features, features*2)
        self.down2 = Down(features*2, features*4)
        self.down3 = Down(features*4, features*8)
        factor = 2 if bilinear else 1
        if bottleneck is None:
            self.down4 = Down(features*8, features*16 // factor)
        else:
            self.down4 = nn.Sequential(
                nn.MaxPool2d(2),
                bottleneck
            )
        self.up1 = Up(features*16, features*8 // factor, bilinear)
        self.up2 = Up(features*8, features*4 // factor, bilinear)
        self.up3 = Up(features*4, features*2 // factor, bilinear)
        self.up4 = Up(features*2, features, bilinear)
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.stem(x)
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

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()