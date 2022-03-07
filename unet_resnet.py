"""Implementation of UNet-Resnet18"""
"""Architecture based on: https://www.researchgate.net/publication/345841396/figure/fig5/AS:959983546019840@1605889321027/Architecture-of-ResNet-UNet-The-structure-of-the-ResNet-UNet-uses-the-traditional-UNet.png """
import pathlib
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
import common_utility.typing_compat as T


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
        bilinear: T.Optional[bool] = True,
        bottleneck: T.Optional[nn.Module] = None,
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.bilinear = bilinear

        features = init_features
        factor = 2 if bilinear else 1

        self.up1 = Up(512 + 256 , 512 // factor, bilinear)
        self.up2 = Up(256 + 128 , 256 // factor , bilinear)
        self.up3 = Up(64  + 128 , 256 // factor, bilinear)
        self.up4 = Up(64  + 128 , 128, bilinear)

        self.base_model = models.resnet18(pretrained=True)        
        self.base_layers = list(self.base_model.children())  
                      
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = self.convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.layer1_1x1 = self.convrelu(64, 64, 1, 0)       
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.layer2_1x1 = self.convrelu(128, 128, 1, 0)  
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.layer3_1x1 = self.convrelu(256, 256, 1, 0)  
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = self.convrelu(512, 512, 1, 0)  

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_original_size0 = self.convrelu(3, 64, 3, 1)
        self.conv_original_size1 = self.convrelu(64, 64, 3, 1)
        self.conv_original_size2 = self.convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, features, 1)

    def convrelu(self, in_channels, out_channels, kernel, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
  
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = self.up1(layer4, layer3)

        layer2 = self.layer2_1x1(layer2)
        x = self.up2(x, layer2)

        layer1 = self.layer1_1x1(layer1)
        x = self.up3(x, layer1)

        layer0 = self.layer0_1x1(layer0)
        x = self.up4(x, layer0)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)

        x = self.conv_original_size2(x)
        out = self.conv_last(x)

        return out
