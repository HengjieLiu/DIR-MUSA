
'''
Customized registration model: LapIRN

Original version:
    Original code retrieved from:
        https://github.com/cwmok/LapIRN

    Original paper:
        Mok, T.C.W., Chung, A.C.S. (2020). Large Deformation Diffeomorphic Image Registration with Laplacian Pyramid Networks. 
        In: Martel, A.L., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2020. MICCAI 2020. 
        Lecture Notes in Computer Science(), vol 12263. Springer, Cham. https://doi.org/10.1007/978-3-030-59716-0_21
    
    Code reference:
        https://github.com/cwmok/LapIRN/blob/master/Code/miccai2020_model_stage.py
            SHA: https://github.com/cwmok/LapIRN/blob/c17fb7564ca752d37e97baf12cafff44e01da668/Code/miccai2020_model_stage.py
        https://github.com/cwmok/LapIRN/blob/master/Code/Functions.py
            SHA: https://github.com/cwmok/LapIRN/blob/f2b25e22914582b289304dbd928790d079d52887/Code/Functions.py

Modified version:
    Modified and tested by:
        H.LIU
    Functions to keep:
        Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1 (modified)
        Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl2 (modified)
        Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl3 (modified)
        PreActBlock (not modified)
    Modification notes:
        1). deformation calculation and image warping:
            use updated SpatialTransformer
            The following changes are needed:
                remove original SpatialTransformer and grid
                remove range_flow
                remove nn.Softsign() in the output layer: the predicited displacement field is not normalized to [-1, 1]
        2). return both dvf and the deformed image, when is_train=False
            Note:
                the return order is (dvf, deformed), which different from other models (defromed, dvf)
                input order (x, y) is the same as other models (moving, fixed)
    
'''

### H.LIU import for updated SpatialTransformer
from ... import utils_warp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1(nn.Module):
    
    # def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4): # H.LIU commented
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, inshape=None): # H.LIU modified
        super(Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        
        # H.LIU commented
        # self.range_flow = range_flow
        # self.is_train = is_train
        # self.imgshape = imgshape
        # self.grid_1 = generate_grid(self.imgshape)
        # self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()
        # self.transform = SpatialTransform().cuda()

        # H.LIU modified
        self.is_train = is_train
        self.inshape = inshape
        self.transformer = utils_warp.SpatialTransformer(self.inshape)
 
        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                               padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer


    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            )
                # nn.Softsign()) # H.LIU removed
        return layer

    def forward(self, x, y):

        cat_input = torch.cat((x, y), 1)
        cat_input = self.down_avg(cat_input)
        cat_input_lvl1 = self.down_avg(cat_input)

        start_s, end_s = self.in_channel//2, self.in_channel
        down_y = cat_input_lvl1[:, start_s:end_s, :, :, :]

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl1)
        e0 = self.down_conv(fea_e0)
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)

        # H.LIU commented
        # output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        # warpped_inputx_lvl1_out = self.transform(x, output_disp_e0_v.permute(0, 2, 3, 4, 1), self.grid_1)
        
        # H.LIU modified
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1))
        # warpped_inputx_lvl1_out = self.transformer(x, output_disp_e0_v)
        x_down = self.down_avg(self.down_avg(x))
        warpped_inputx_lvl1_out = self.transformer(x_down, output_disp_e0_v)

        if self.is_train is True:
            return output_disp_e0_v, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        else:
            # H.LIU commented
            # return output_disp_e0_v
            # H.LIU modified
            return output_disp_e0_v, warpped_inputx_lvl1_out


class Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl2(nn.Module):
    # def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4, model_lvl1=None): # H.LIU commented
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, inshape=None, model_lvl1=None): # H.LIU modified
        super(Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        
        # H.LIU commented
        # self.range_flow = range_flow
        # self.is_train = is_train
        # self.imgshape = imgshape
        # self.model_lvl1 = model_lvl1
        # self.grid_1 = generate_grid(self.imgshape)
        # self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()
        # self.transform = SpatialTransform().cuda()

        # H.LIU modified
        self.is_train = is_train
        self.model_lvl1 = model_lvl1
        self.inshape = inshape
        self.transformer = utils_warp.SpatialTransformer(self.inshape)

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            )
                # nn.Softsign()) # H.LIU removed
        return layer

    def forward(self, x, y):
        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        lvl1_disp, _, _, lvl1_v, lvl1_embedding = self.model_lvl1(x, y)
        lvl1_disp_up = self.up_tri(lvl1_disp) * 2.

        x_down = self.down_avg(x)
        y_down = self.down_avg(y)
        
        # H.LIU commented
        # warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)
        # H.LIU modified
        warpped_x = self.transformer(x_down, lvl1_disp_up)

        cat_input_lvl2 = torch.cat((warpped_x, y_down, lvl1_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl2)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl1_embedding

        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        
        # H.LIU commented
        # output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        # compose_field_e0_lvl1 = lvl1_disp_up + output_disp_e0_v
        # warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)
        # H.LIU modified
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1))
        compose_field_e0_lvl1 = lvl1_disp_up + output_disp_e0_v
        # warpped_inputx_lvl1_out = self.transformer(x, compose_field_e0_lvl1)
        warpped_inputx_lvl1_out = self.transformer(x_down, compose_field_e0_lvl1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y_down, output_disp_e0_v, lvl1_v, e0
        else:
            # H.LIU commented
            # return compose_field_e0_lvl1
            # H.LIU modified
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out


class Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl3(nn.Module):
    # def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4, model_lvl2=None): # H.LIU commented
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, inshape=None, model_lvl2=None): # H.LIU modified
        super(Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        
        # H.LIU commented
        # self.range_flow = range_flow
        # self.is_train = is_train
        # self.imgshape = imgshape
        # self.model_lvl2 = model_lvl2
        # self.grid_1 = generate_grid(self.imgshape)
        # self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()
        # self.transform = SpatialTransform().cuda()

        # H.LIU modified
        self.is_train = is_train
        self.model_lvl2 = model_lvl2
        self.inshape = inshape
        self.transformer = utils_warp.SpatialTransformer(self.inshape)

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


    def unfreeze_modellvl2(self):
        # unFreeze model_lvl2 weight
        print("\nunfreeze model_lvl2 parameter")
        for param in self.model_lvl2.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            )
                # nn.Softsign()) # H.LIU removed
        return layer

    def forward(self, x, y):
        # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
        lvl2_disp, _, _, lvl2_v, lvl1_v, lvl2_embedding = self.model_lvl2(x, y)
        lvl2_disp_up = self.up_tri(lvl2_disp) * 2.
        
        # H.LIU commented
        # warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)
        # H.LIU modified
        warpped_x = self.transformer(x, lvl2_disp_up)

        cat_input = torch.cat((warpped_x, y, lvl2_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl2_embedding
        e0 = self.resblock_group_lvl1(e0)
        e0 = self.up(e0)
        
        # H.LIU commented
        # output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        # compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up
        # warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)
        # H.LIU modified
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1))
        compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up
        warpped_inputx_lvl1_out = self.transformer(x, compose_field_e0_lvl1)
        
        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
        else:
            # H.LIU commented
            # return compose_field_e0_lvl1
            # H.LIU modified
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )


    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(out, negative_slope=0.2))


        out += shortcut
        return out
