'''
Basic UNet blocks (with residual connection) for nnU-Net implementation

Code Reference:
    https://github.com/Project-MONAI/MONAI/blob/fdd07f36ecb91cfcd491533f4792e1a67a9f89fc/monai/networks/nets/dynunet.py
    https://github.com/Project-MONAI/MONAI/blob/fdd07f36ecb91cfcd491533f4792e1a67a9f89fc/monai/networks/blocks/dynunet_block.py
Diagram:
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/nnunet_pyt 

classes:
    UnetEncoderBlock
    UnetDecoderBlock
    UnetOutBlock
    
Modified and tested by:
    H.LIU

Note:
    This code is used in both ResUNet (resunet_v1.py) and DualPRNet (dualprnet_v1.py)
'''

import torch
import torch.nn as nn


class UnetEncoderBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        bias: bool = True,
        downsample: bool=False,
        use_residual: bool=True,
        relu_inplace: bool = False,
        instance_norm: bool = False,
    ) -> None:

        super().__init__()

        self.use_residual = use_residual
        self.instance_norm = instance_norm
        
        # conv1
        if not downsample:
            # no downsampling with stride=1
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        else:
            # downsampling with stride=2
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias)
        
        # conv2
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # instance norm
        if self.instance_norm:
            self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

        # activation
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=relu_inplace)

        # residual skip connection (conv3)
        if self.use_residual:
            if not downsample:
                # no downsampling with stride=1
                self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
            else:
                # downsampling with stride=2
                self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=bias)
            # instance norm
            if self.instance_norm:
                self.norm3 = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        if not self.use_residual:
            out = self.conv1(x)
            if self.instance_norm:
                out = self.norm1(out)
            out = self.activation(out)
            out = self.conv2(out)
            if self.instance_norm:
                out = self.norm2(out)
            out = self.activation(out)
        else:
            residual = x
            out = self.conv1(x)
            if self.instance_norm:
                out = self.norm1(out)
            out = self.activation(out)
            out = self.conv2(out)
            if self.instance_norm:
                out = self.norm2(out)
            # Second activation is applied after adding the residual
            
            residual = self.conv3(residual)
            if self.instance_norm:
                residual = self.norm3(residual)
            out += residual
            out = self.activation(out)
        
        return out


class UnetDecoderBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        bias: bool = True,
        trans_bias: bool=True,
        relu_inplace: bool = False,
        instance_norm: bool = False,
    ) -> None:

        super().__init__()

        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=trans_bias)
        self.conv_block = UnetEncoderBlock(in_channels=out_channels*2, out_channels=out_channels, bias=bias, downsample=False, use_residual=False, relu_inplace=relu_inplace, instance_norm=instance_norm)
    
    def forward(self, x, skip):
        out = self.conv_transpose(x)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetOutBlock(nn.Module):
    """"
    Modification:
        Use 3x3 convolution instead of 1x1 convolution
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        bias: bool = True,
    ) -> None:

        super().__init__()
        
        # conv
        # self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    
    def forward(self, x):
        out = self.conv(x)
        return out