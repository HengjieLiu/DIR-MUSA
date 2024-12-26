'''
Customized registration model: ResUNet

Notes:
    The basic setup is similar to VoxelMorph. 
    The main difference is replacing the simple UNet with a residual UNet.
        see architecture diagram here: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/nnunet_pyt

Created and tested by:
    H.LIU
'''

### H.LIU import for updated SpatialTransformer
from ... import utils_warp
### import unet blocks
from ..unetblocks.utils_unet_blocks import UnetEncoderBlock, UnetDecoderBlock, UnetOutBlock

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.init as init

from typing import List, Tuple, Dict, Union, Any


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 3,
        filters: List[int] = (32, 64, 128, 256, 320),
        bias: bool = True,
        res_block: bool = True,
        trans_bias: bool = True,
        relu_inplace: bool = False,
        instance_norm: bool = False,
        init_weight: str = 'kaiming+smallflow',
    ):
        super().__init__()
        
        if len(filters) == 5: # 4 levels of downsampling
            self.levels = 4

            self.UnetEncoderBlock_lv0 = UnetEncoderBlock(in_channels=in_channels, out_channels=filters[0], bias=bias, downsample=False, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)
            self.UnetEncoderBlock_lv1 = UnetEncoderBlock(in_channels=filters[0], out_channels=filters[1], bias=bias, downsample=True, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)
            self.UnetEncoderBlock_lv2 = UnetEncoderBlock(in_channels=filters[1], out_channels=filters[2], bias=bias, downsample=True, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)
            self.UnetEncoderBlock_lv3 = UnetEncoderBlock(in_channels=filters[2], out_channels=filters[3], bias=bias, downsample=True, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)
            self.UnetEncoderBlock_lv4 = UnetEncoderBlock(in_channels=filters[3], out_channels=filters[4], bias=bias, downsample=True, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)

            self.UnetDecoderBlock_lv4 = UnetDecoderBlock(in_channels=filters[4], out_channels=filters[3], bias=bias, trans_bias=trans_bias, relu_inplace=relu_inplace, instance_norm=instance_norm)
            self.UnetDecoderBlock_lv3 = UnetDecoderBlock(in_channels=filters[3], out_channels=filters[2], bias=bias, trans_bias=trans_bias, relu_inplace=relu_inplace, instance_norm=instance_norm)
            self.UnetDecoderBlock_lv2 = UnetDecoderBlock(in_channels=filters[2], out_channels=filters[1], bias=bias, trans_bias=trans_bias, relu_inplace=relu_inplace, instance_norm=instance_norm)
            self.UnetDecoderBlock_lv1 = UnetDecoderBlock(in_channels=filters[1], out_channels=filters[0], bias=bias, trans_bias=trans_bias, relu_inplace=relu_inplace, instance_norm=instance_norm)

        elif len(filters) == 4: # 3 levels of downsampling
            self.levels = 3

            self.UnetEncoderBlock_lv0 = UnetEncoderBlock(in_channels=in_channels, out_channels=filters[0], bias=bias, downsample=False, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)
            self.UnetEncoderBlock_lv1 = UnetEncoderBlock(in_channels=filters[0], out_channels=filters[1], bias=bias, downsample=True, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)
            self.UnetEncoderBlock_lv2 = UnetEncoderBlock(in_channels=filters[1], out_channels=filters[2], bias=bias, downsample=True, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)
            self.UnetEncoderBlock_lv3 = UnetEncoderBlock(in_channels=filters[2], out_channels=filters[3], bias=bias, downsample=True, use_residual=res_block, relu_inplace=relu_inplace, instance_norm=instance_norm)

            self.UnetDecoderBlock_lv3 = UnetDecoderBlock(in_channels=filters[3], out_channels=filters[2], bias=bias, trans_bias=trans_bias, relu_inplace=relu_inplace, instance_norm=instance_norm)
            self.UnetDecoderBlock_lv2 = UnetDecoderBlock(in_channels=filters[2], out_channels=filters[1], bias=bias, trans_bias=trans_bias, relu_inplace=relu_inplace, instance_norm=instance_norm)
            self.UnetDecoderBlock_lv1 = UnetDecoderBlock(in_channels=filters[1], out_channels=filters[0], bias=bias, trans_bias=trans_bias, relu_inplace=relu_inplace, instance_norm=instance_norm)
        else:
            raise ValueError('ERROR in ResUNet __init__: Unsupported number of levels')

        self.UnetOutBlock = UnetOutBlock(in_channels=filters[0], out_channels= out_channels)

        self.init_weight = init_weight
    
    def forward(self, x):
        x_history = []

        if self.levels == 4:
            x = self.UnetEncoderBlock_lv0(x); x_history.append(x)
            x = self.UnetEncoderBlock_lv1(x); x_history.append(x)
            x = self.UnetEncoderBlock_lv2(x); x_history.append(x)
            x = self.UnetEncoderBlock_lv3(x); x_history.append(x)
            x = self.UnetEncoderBlock_lv4(x)

            x = self.UnetDecoderBlock_lv4(x, x_history.pop())
            x = self.UnetDecoderBlock_lv3(x, x_history.pop())
            x = self.UnetDecoderBlock_lv2(x, x_history.pop())
            x = self.UnetDecoderBlock_lv1(x, x_history.pop())
        elif self.levels == 3:
            x = self.UnetEncoderBlock_lv0(x); x_history.append(x)
            x = self.UnetEncoderBlock_lv1(x); x_history.append(x)
            x = self.UnetEncoderBlock_lv2(x); x_history.append(x)
            x = self.UnetEncoderBlock_lv3(x); 

            x = self.UnetDecoderBlock_lv3(x, x_history.pop())
            x = self.UnetDecoderBlock_lv2(x, x_history.pop())
            x = self.UnetDecoderBlock_lv1(x, x_history.pop())
        else:
            raise ValueError('ERROR in ResUNet forward: Unsupported number of levels')
        
        x = self.UnetOutBlock(x)

        return x

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
            if 'smallflow' in self.init_weight and 'UnetOutBlock' in name:
                if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                    init.normal_(module.weight, mean=0, std=1e-5)
                    if module.bias is not None:
                        init.constant_(module.bias, 0)


class ResUNet_disp(nn.Module):
    """
    Registration network using ResUNet and SpatialTransformer
    Similar to VoxelMorph, but with ResUNet as the backbone
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
        
        ### set up ResUNet
        if dict_config_unet is not None:
            self.unet_model = ResUNet(**dict_config_unet)
        else:
            self.unet_model = ResUNet()
        self.unet_model.initialize_weights()
    
    def forward(self, moving: torch.FloatTensor, fixed: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Inputs:
            moving: moving image tensor (moving/source) -> x
            fixed: fixed image tensor (fixed/target) -> y
        Outputs:
            deformed: deformed image tensor (deformed/moved/warped) -> y_pred
            dvf: deformation vector field (dvf/warp) -> dvf

        Naming and order conventions:
            Use the above naming as L_similarity = || y - f(x) ||^2
            We follow the order convention in VoxelMorph, the input order is [moving, fixed] or [source, target]:
                In vxm code class VxmDense():
                    See: https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/networks.py
                    x = torch.cat([source, target], dim=1)
                    y_source = self.transformer(source, pos_flow)
                    return y_source, pos_flow
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
