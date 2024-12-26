'''
Customized registration model: VoxelMorph

Original version:
    Original code retrieved from:
        https://github.com/xi-jia/LKU-Net

    Original paper:
        @article{jia2022lkunet,
        title={U-Net vs Transformer: Is U-Net Outdated in Medical Image Registration?},
        author={Jia, Xi and Bartlett, Joseph and Zhang, Tianyang and Lu, Wenqi and Qiu, Zhaowen and Duan, Jinming},
        journal={arXiv preprint arXiv:2208.04939},
        year={2022}
        }

    Code Reference:
        https://github.com/xi-jia/LKU-Net/blob/main/LKU-Net_3D_OASIS/LKU-Net-Full-Resolution/Models.py
            SHA: https://github.com/xi-jia/LKU-Net/blob/c304a36d099a084ebaba743de387fab816c182c2/LKU-Net_3D_OASIS/LKU-Net-Full-Resolution/Models.py


Modified version:
    Modified and tested by:
        H.LIU
    Modification notes:
        Added:
            class LKUNet_disp
                Registration network using LKUNet and SpatialTransformer
                Similar to VoxelMorph, but with LKUNet as the backbone
        Modified:
            class UNet -> class LKUNet
                1). lk_size and lk_padding:
                    use specified lk_size and calculate lk_padding automatically
                2). Remove Softsign activation in the output layer
                    to be compatible with the unnormalized DVF and SpatialTransformer
                    Softsign is used when DVF is normalized to [-1,1]
                3). Add arguments:
                    output_bias: if True, enable bias for the output layer
                    output_init: if Ture, use voxelmorph's small flow initialization
                    For original LKUNet: output_bias = False, output_init = False
                    For updated version: output_bias = True, output_init = True
        Not modified:
            class LK_encoder
'''

### H.LIU import for updated SpatialTransformer
from ... import utils_warp

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.init as init

from typing import List, Tuple, Dict, Union, Any


class LK_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False, batchnorm=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.batchnorm = batchnorm
        
        super(LK_encoder, self).__init__()
        
        self.layer_regularKernel = self.encoder_LK_encoder(self.in_channels, self.out_channels, kernel_size = 3, stride=1, padding=1, bias=self.bias, batchnorm = self.batchnorm)
        self.layer_largeKernel = self.encoder_LK_encoder(self.in_channels, self.out_channels, kernel_size = self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias, batchnorm = self.batchnorm)
        self.layer_oneKernel = self.encoder_LK_encoder(self.in_channels, self.out_channels, kernel_size = 1, stride=1, padding=0, bias=self.bias, batchnorm = self.batchnorm)
        self.layer_nonlinearity = nn.PReLU()
        # self.layer_batchnorm = nn.BatchNorm3d(num_features = self.out_channels)
    def encoder_LK_encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer
    def forward(self, inputs):
        # print(self.layer_regularKernel)
        regularKernel = self.layer_regularKernel(inputs)
        largeKernel = self.layer_largeKernel(inputs)
        oneKernel = self.layer_oneKernel(inputs)
        # if self.layer_indentity:
        outputs = regularKernel + largeKernel + oneKernel + inputs
        # else:
        # outputs = regularKernel + largeKernel + oneKernel
        # if self.batchnorm:
            # outputs = self.layer_batchnorm(self.layer_batchnorm)
        return self.layer_nonlinearity(outputs)


class LKUNet(nn.Module):
    """
    Based on https://github.com/xi-jia/LKU-Net/blob/main/Models.py class UNet

    Modification notes:
        1). lk_size and lk_padding:
            use specified lk_size and calculate lk_padding automatically
        2). Remove Softsign activation in the output layer
            to be compatible with the unnormalized DVF and SpatialTransformer
            Softsign is used when DVF is normalized to [-1,1]
        3). Add arguments:
            output_bias: if True, enable bias for the output layer
            output_init: if Ture, use voxelmorph's small flow initialization
            For original LKUNet: output_bias = False, output_init = False
            For updated version: output_bias = True, output_init = True
    """
    def __init__(self, in_channel, n_classes, start_channel, lk_size, output_bias=False, output_init=False):

        super(LKUNet, self).__init__()

        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        bias_opt = True

        # H.LIU modified
        # define the large kernel size and calculate padding
        self.lk_size = lk_size 
        self.lk_padding = int((lk_size-1)/2)

        # H.LIU commented
        # self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        # self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        # self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        # self.ec3 = LK_encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=5, stride=1, padding=2, bias=bias_opt)
        # self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        # self.ec5 = LK_encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=5, stride=1, padding=2, bias=bias_opt)
        # self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        # self.ec7 = LK_encoder(self.start_channel * 8, self.start_channel * 8, kernel_size=5, stride=1, padding=2, bias=bias_opt)
        # self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 8, stride=2, bias=bias_opt)
        # self.ec9 = LK_encoder(self.start_channel * 8, self.start_channel * 8, kernel_size=5, stride=1, padding=2, bias=bias_opt)

        # H.LIU modified
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = LK_encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=self.lk_size, stride=1, padding=self.lk_padding, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = LK_encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=self.lk_size, stride=1, padding=self.lk_padding, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = LK_encoder(self.start_channel * 8, self.start_channel * 8, kernel_size=self.lk_size, stride=1, padding=self.lk_padding, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec9 = LK_encoder(self.start_channel * 8, self.start_channel * 8, kernel_size=self.lk_size, stride=1, padding=self.lk_padding, bias=bias_opt)
        
        self.dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)

        # H.LIU commented
        # self.dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # # self.dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        # H.LIU modified
        # output layer enable bias setting
        self.dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=output_bias)
        
        # H.LIU added
        # small initialization as used in voxelmorph
        if output_init:
            # Assuming dc9 has a single Conv3d layer at index 0
            conv_layer = self.dc9[0]
            # Initialize weights with a normal distribution
            init.normal_(conv_layer.weight, mean=0, std=1e-5)
            # Initialize bias with zeros
            if conv_layer.bias is not None:
                init.constant_(conv_layer.bias, 0)

        self.up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            # H.LIU. commented 
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Softsign())

            # H.LIU. modified
            # remove nn.Softsign() to be compatible with the unnormalized DVF and SpatialTransformer settings
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                )

        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        d3 = torch.cat((self.up4(d2), e0), 1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)

        f_xy = self.dc9(d3)
        #f_yx = self.dc10(d3)

        return f_xy#, f_yx


class LKUNet_disp(nn.Module):
    '''
    Registration network using LKUNet and SpatialTransformer
    Similar to VoxelMorph, but with LKUNet as the backbone
    Naming:
        disp: dense displacement prediction, in contrast to diffeomorphic field prediction (diff)
    Notes:
        1). SpatialTransformer:
            updated SpatialTransformer: musa.utils_warp.SpatialTransformer
        2). Add arguments (pass on to LKUNet):
            lk_size, output_bias, output_init
    '''

    def __init__(
        self,
        inshape: Tuple[int, int, int],
        persistent_grid: bool = False,
        in_channel=2, 
        n_classes=3, 
        start_channel=8, 
        lk_size=5, 
        output_bias=True, 
        output_init=True
    ) -> None:

        super(LKUNet_disp, self).__init__()     

        ### set up SpatialTransformer
        self.transformer = utils_warp.SpatialTransformer(inshape, persistent=persistent_grid)
        
        ### set up LKUNet
        self.unet_model = LKUNet(in_channel, n_classes, start_channel, lk_size, output_bias, output_init)  

    def forward(self, moving: torch.FloatTensor, fixed: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Inputs:
            moving: moving image tensor (moving/source) -> x
            fixed: fixed image tensor (fixed/target) -> y
        Outputs:
            deformed: deformed image tensor (deformed/moved/warped) -> y_pred
            dvf: deformation vector field (dvf/warp) -> dvf
        """
        x = moving
        y = fixed

        # register (predict dvf)
        dvf = self.unet_model(x, y)

        # warp the moving image (x) with dvf
        # default settings: (mode='bilinear', padding_mode='border', align_corners=True)
        y_pred = self.transformer(x, dvf)

        return y_pred, dvf
