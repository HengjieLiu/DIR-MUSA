'''
Extra building block for DualPRNet: the residual block used in PR module

In complementary to the following classes in musa/registration_models/utils_unet_blocks.py
    UnetEncoderBlock
    UnetDecoderBlock
    UnetOutBlock
'''

import torch
import torch.nn as nn


class ConvBlock_PRres(nn.Module):
    """
    The residual block used in PR module
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        bias: bool = True,
        use_residual: bool=True,
        relu_inplace: bool = False,
        instance_norm: bool = False,
        ) -> None:

        super().__init__()

        # self.use_residual = use_residual # residual is always used
        self.instance_norm = instance_norm
        
        # merge channel numbers, no residual connection (in current implementation will reduce channel # by half)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # channel numbers kept the same, with residual connection
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # instance norm
        if self.instance_norm:
            self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

        # activation
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=relu_inplace)

    def forward(self, x):

        # conv1 (no residual)
        out = self.conv1(x)
        if self.instance_norm:
            out = self.norm1(out)
        out = self.activation(out)

        # conv2 (with residual)
        residual = out

        out = self.conv2(out)
        if self.instance_norm:
            out = self.norm2(out)
        
        out += residual
        out = self.activation(out)
        
        return out
