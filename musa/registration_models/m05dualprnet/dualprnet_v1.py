'''
Customized registration model: DualPRNet

Description:
    Reimplementation of DualPRNet by H.LIU
    We only reimplemented DualPRNet (without the 3D correlation layer)
    DualPRNet++ (with 3D correlation layer) is not reimplemented

Reference:
    Official implementation:
        https://github.com/kangmiao15/dual-stream-prnet-plus
    Original paper:
        Miao Kang, Xiaojun Hu, Weilin Huang, Matthew R. Scott, Mauricio Reyes, Dual-stream pyramid registration network, 
        Medical Image Analysis, Volume 78, 2022, 102379, ISSN 1361-8415, https://doi.org/10.1016/j.media.2022.102379.

Code reference:
    tensorflow reimplemention of DualPRNet
        https://github.com/JinxLv/reimplemention-of-Dual-PRNet/blob/83f2e42ac6316fac49708a7bc42b303a8ff16af2/network/base_networks.py
'''

### H.LIU import for updated SpatialTransformer and ComposeDVF
from ... import utils_warp
### import unet blocks
from ..unetblocks.utils_unet_blocks import UnetEncoderBlock, UnetDecoderBlock, UnetOutBlock
from .utils_dualpr_blocks import ConvBlock_PRres

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.init as init

from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Any


class DualPRNet(nn.Module):
    """
    Dual stream encoder and dual stream decoder (shared weights)
    PR unit on each resolution level to predict multi-scale DVFs
    
    """
    def __init__(
        self,
        inshape: Tuple[int, int, int], # need to initialize the transformer and composer at all levels
        in_channels: int = 2,
        out_channels: int = 3,
        filters: List[int] = (8, 16, 16, 32, 32),
        filters_pr: List[int] = (8, 16, 16, 32, 32),
        bias: bool = True,
        res_block: bool = True,
        trans_bias: bool = True,
        relu_inplace: bool = False,
        instance_norm: bool = False,
        init_weight: str = 'kaiming+smallflow',
    ):
        super().__init__()

        self.init_weight = init_weight

        ### Initialze SpatialTransformer and ComposeDVF
        inshape_lv0 = inshape
        inshape_lv1 = tuple([d//2 for d in inshape_lv0])
        inshape_lv2 = tuple([d//2 for d in inshape_lv1])
        inshape_lv3 = tuple([d//2 for d in inshape_lv2])
        inshape_lv4 = tuple([d//2 for d in inshape_lv3])

        self.transformer_lv0 = utils_warp.SpatialTransformer(inshape_lv0)
        self.transformer_lv1 = utils_warp.SpatialTransformer(inshape_lv1)
        self.transformer_lv2 = utils_warp.SpatialTransformer(inshape_lv2)
        self.transformer_lv3 = utils_warp.SpatialTransformer(inshape_lv3)
        self.transformer_lv4 = utils_warp.SpatialTransformer(inshape_lv4)

        self.composer_lv0 = utils_warp.ComposeDVF(inshape_lv0)
        self.composer_lv1 = utils_warp.ComposeDVF(inshape_lv1)
        self.composer_lv2 = utils_warp.ComposeDVF(inshape_lv2)
        self.composer_lv3 = utils_warp.ComposeDVF(inshape_lv3)
        self.composer_lv4 = utils_warp.ComposeDVF(inshape_lv4)

        # function for upsampling the DVF by 2
        self.upsample_dvf = utils_warp.dvf_upsample 

        ### shared encoder for dual branch
        # in_channels//2 due to dual branch setting
        self.UnetEncoderBlock_lv0 = UnetEncoderBlock(in_channels//2, out_channels=filters[0], bias=bias, downsample=False, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)
        self.UnetEncoderBlock_lv1 = UnetEncoderBlock(in_channels=filters[0], out_channels=filters[1], bias=bias, downsample=True, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)
        self.UnetEncoderBlock_lv2 = UnetEncoderBlock(in_channels=filters[1], out_channels=filters[2], bias=bias, downsample=True, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)
        self.UnetEncoderBlock_lv3 = UnetEncoderBlock(in_channels=filters[2], out_channels=filters[3], bias=bias, downsample=True, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)
        self.UnetEncoderBlock_lv4 = UnetEncoderBlock(in_channels=filters[3], out_channels=filters[4], bias=bias, downsample=True, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)
        
        ### shared decoder for dual branch
        self.UnetDecoderBlock_lv4 = UnetDecoderBlock(in_channels=filters[4], out_channels=filters[3], bias=bias, trans_bias=trans_bias, relu_inplace=relu_inplace, instance_norm=instance_norm)
        self.UnetDecoderBlock_lv3 = UnetDecoderBlock(in_channels=filters[3], out_channels=filters[2], bias=bias, trans_bias=trans_bias, relu_inplace=relu_inplace, instance_norm=instance_norm)
        self.UnetDecoderBlock_lv2 = UnetDecoderBlock(in_channels=filters[2], out_channels=filters[1], bias=bias, trans_bias=trans_bias, relu_inplace=relu_inplace, instance_norm=instance_norm)
        self.UnetDecoderBlock_lv1 = UnetDecoderBlock(in_channels=filters[1], out_channels=filters[0], bias=bias, trans_bias=trans_bias, relu_inplace=relu_inplace, instance_norm=instance_norm)
        
        ### DVF output via PR module       
        self.pr_lv4 = nn.Sequential(OrderedDict([
            ('conv_block', ConvBlock_PRres(in_channels=filters[4]*2, out_channels=filters_pr[4])),
            ('dvf_out_block', UnetOutBlock(in_channels=filters_pr[4], out_channels=out_channels)),
        ]))
        self.pr_lv3 = nn.Sequential(OrderedDict([
            ('conv_block', ConvBlock_PRres(in_channels=filters[3]*2, out_channels=filters_pr[3])),
            ('dvf_out_block', UnetOutBlock(in_channels=filters_pr[3], out_channels=out_channels)),
        ]))
        self.pr_lv2 = nn.Sequential(OrderedDict([
            ('conv_block', ConvBlock_PRres(in_channels=filters[2]*2, out_channels=filters_pr[2])),
            ('dvf_out_block', UnetOutBlock(in_channels=filters_pr[2], out_channels=out_channels)),
        ]))
        self.pr_lv1 = nn.Sequential(OrderedDict([
            ('conv_block', ConvBlock_PRres(in_channels=filters[1]*2, out_channels=filters_pr[1])),
            ('dvf_out_block', UnetOutBlock(in_channels=filters_pr[1], out_channels=out_channels)),
        ]))
        self.pr_lv0 = nn.Sequential(OrderedDict([
            ('conv_block', ConvBlock_PRres(in_channels=filters[0]*2, out_channels=filters_pr[0])),
            ('dvf_out_block', UnetOutBlock(in_channels=filters_pr[0], out_channels=out_channels)),
        ]))
    

    def forward(self, x):

        # x1: moving
        # x2: fixed
        (x1, x2) = torch.split(x, 1, dim=1)

        x1_history = []
        x2_history = []

        x1 = self.UnetEncoderBlock_lv0(x1); x1_history.append(x1)
        x1 = self.UnetEncoderBlock_lv1(x1); x1_history.append(x1)
        x1 = self.UnetEncoderBlock_lv2(x1); x1_history.append(x1)
        x1 = self.UnetEncoderBlock_lv3(x1); x1_history.append(x1)
        x1 = self.UnetEncoderBlock_lv4(x1)

        x2 = self.UnetEncoderBlock_lv0(x2); x2_history.append(x2)
        x2 = self.UnetEncoderBlock_lv1(x2); x2_history.append(x2)
        x2 = self.UnetEncoderBlock_lv2(x2); x2_history.append(x2)
        x2 = self.UnetEncoderBlock_lv3(x2); x2_history.append(x2)
        x2 = self.UnetEncoderBlock_lv4(x2)

        ### dvf lv4
        # dvf prediction via pr unit
        x = torch.cat((x1, x2), dim=1)
        dvf_lv4 = self.pr_lv4(x)
        # compose with previous (no composing for bottom level)
        dvf_lv4_all = dvf_lv4
        # upsample by 2 for next stage
        dvf_lv4_all_up = self.upsample_dvf(dvf_lv4_all)

        ### dvf lv3
        # get lv3 features from decoder
        x1 = self.UnetDecoderBlock_lv4(x1, x1_history.pop())
        x2 = self.UnetDecoderBlock_lv4(x2, x2_history.pop())
        # warp the moving features
        x1 = self.transformer_lv3(x1, dvf_lv4_all_up)
        # dvf prediction via pr unit
        x = torch.cat((x1, x2), dim=1)
        dvf_lv3 = self.pr_lv3(x)
        # compose with previous
        dvf_lv3_all = self.composer_lv3(dvf_lv4_all_up, dvf_lv3)
        # upsample by 2 for next stage
        dvf_lv3_all_up = self.upsample_dvf(dvf_lv3_all)

        ### dvf lv2
        # get lv2 features from decoder
        x1 = self.UnetDecoderBlock_lv3(x1, x1_history.pop())
        x2 = self.UnetDecoderBlock_lv3(x2, x2_history.pop())
        # warp the moving features
        x1 = self.transformer_lv2(x1, dvf_lv3_all_up)
        # dvf prediction via pr unit
        x = torch.cat((x1, x2), dim=1)
        dvf_lv2 = self.pr_lv2(x)
        # compose with previous
        dvf_lv2_all = self.composer_lv2(dvf_lv3_all_up, dvf_lv2)
        # upsample by 2 for next stage
        dvf_lv2_all_up = self.upsample_dvf(dvf_lv2_all)

        ### dvf lv1
        # get lv1 features from decoder
        x1 = self.UnetDecoderBlock_lv2(x1, x1_history.pop())
        x2 = self.UnetDecoderBlock_lv2(x2, x2_history.pop())
        # warp the moving features
        x1 = self.transformer_lv1(x1, dvf_lv2_all_up)
        # dvf prediction via pr unit
        x = torch.cat((x1, x2), dim=1)
        dvf_lv1 = self.pr_lv1(x)
        # compose with previous
        dvf_lv1_all = self.composer_lv1(dvf_lv2_all_up, dvf_lv1)
        # upsample by 2 for next stage
        dvf_lv1_all_up = self.upsample_dvf(dvf_lv1_all)

        ### dvf lv0
        # get lv0 features from decoder
        x1 = self.UnetDecoderBlock_lv1(x1, x1_history.pop())
        x2 = self.UnetDecoderBlock_lv1(x2, x2_history.pop())
        # warp the moving features
        x1 = self.transformer_lv0(x1, dvf_lv1_all_up)
        # dvf prediction via pr unit
        x = torch.cat((x1, x2), dim=1)
        dvf_lv0 = self.pr_lv0(x)
        # compose with previous
        dvf_lv0_all = self.composer_lv0(dvf_lv1_all_up, dvf_lv0)
        # upsample by 2 for next stage (no upsampling for top level)

        dvf = dvf_lv0_all

        return dvf
        

    def initialize_weights(self):
        """
        Initialize weights
        Ref:
            https://github.com/Project-MONAI/MONAI/blob/fdd07f36ecb91cfcd491533f4792e1a67a9f89fc/monai/networks/nets/dynunet.py
            https://github.com/voxelmorph/voxelmorph/blob/579a995492bddfe9ce38161e58cf260fc155c4fd/voxelmorph/torch/networks.py
        """
        for name, module in self.named_modules():
            if 'kaiming' in self.init_weight:
                if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                    nn.init.kaiming_normal_(module.weight, a=0.2)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            if 'smallflow' in self.init_weight and 'dvf_out_block' in name:
                if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                    init.normal_(module.weight, mean=0, std=1e-5)
                    if module.bias is not None:
                        init.constant_(module.bias, 0)


class DualPRNet_disp(nn.Module):
    """
    Registration network using DualPRNet and SpatialTransformer
    Similar to VoxelMorph, but with DualPRNet as the backbone
    Naming:
        disp: dense displacement prediction, in contrast to diffeomorphic field prediction (diff)
    Notes:
        1). SpatialTransformer:
            updated SpatialTransformer: musa.utils_warp.SpatialTransformer
    """
    def __init__(
        self,
        inshape: Tuple[int, int, int],
        dict_config_unet: Dict[str, Any] = None,
        persistent_grid: bool = False,
        INFO: bool = True,
        DEBUG: bool = False
        ) -> None:

        super().__init__()

        self.INFO = INFO
        self.DEBUG = DEBUG

        ### set up SpatialTransformer
        self.transformer = utils_warp.SpatialTransformer(inshape, persistent=persistent_grid)
        
        ### set up DualPRNet
        # Note: inshape is needed to initialize the transformer and composer at all levels
        if dict_config_unet is not None:
            self.unet_model = DualPRNet(inshape, **dict_config_unet)
        else:
            self.unet_model = DualPRNet(inshape)
        self.unet_model.initialize_weights()
    
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
        
        # concatenate along the channel (NCHWD) dimension
        input_unet = torch.cat([x, y], dim=1)
        
        # register (predict dvf)
        dvf = self.unet_model(input_unet)

        # warp the moving image (x) with dvf
        # default settings: (mode='bilinear', padding_mode='border', align_corners=True)
        y_pred = self.transformer(x, dvf)

        return y_pred, dvf
