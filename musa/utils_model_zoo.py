"""
General description:
    Functions for initializing and calling registration models

Function/Class list:
    get_model_v1
    model_register_v1
"""

import os

import numpy as np
import torch


def get_model_v1(inshape, model_type, model_resolution, DEBUG=True, enforce_shape=True):
    '''
    model_type: 
        01voxelmorph-v1
        02resunet-v1
        03lkunet-v1-lk09
        04transmorph-v1
        05dualprnet-v1
        06lapirn-v1
    model_resolution:
        'r1': 2mm isotropic resolution, image size of (160, 160, 192)
        'r2': 4mm isotropic resolution, image size of (80, 80, 96)
    '''
    
    ### Check input
    assert model_type in ['01voxelmorph-v1', '02resunet-v1', '03lkunet-v1-lk09', '04transmorph-v1', '05dualprnet-v1', '06lapirn-v1'], \
        'model_type should be one of the following: 01voxelmorph-v1, 02resunet-v1, 03lkunet-v1-lk09, 04transmorph-v1, 05dualprnet-v1, 06lapirn-v1'
    assert model_resolution in ['r1', 'r2'], 'model_resolution should be one of the following: r1, r2'
    
    if enforce_shape:
        inshape_r1 = (160, 160, 192)
        inshape_r2 = (80, 80, 96)
        if model_resolution == 'r1':
            assert inshape == inshape_r1, 'inshape should be (160, 160, 192) for model_resolution r1'
        elif model_resolution == 'r2':
            assert inshape == inshape_r2, 'inshape should be (80, 80, 96) for model_resolution r2'

    ### Load model
    if model_type == '01voxelmorph-v1':
        from .registration_models.m01voxelmorph.voxelmorph_v1 import VxmDense_disp
        model = VxmDense_disp(inshape)
    elif model_type == '02resunet-v1':
        from .registration_models.m02resunet.resunet_v1 import ResUNet_disp
        model = ResUNet_disp(inshape)
    elif model_type == '03lkunet-v1-lk09':
        from .registration_models.m03lkunet.lkunet_v1 import LKUNet_disp
        lk_size = int(model_type.split('-')[-1].split('lk')[-1])
        model = LKUNet_disp(inshape, lk_size=lk_size)
        if DEBUG: print('Initialized LKUNet_disp, lk_size: ', lk_size)
    elif model_type == '04transmorph-v1':
        from .registration_models.m04transmorph.transmorph_v1 import CONFIGS as CONFIGS_TM
        from .registration_models.m04transmorph.transmorph_v1 import TransMorph
        config_default = CONFIGS_TM['TransMorph']
        
        # Configure the image size and window size
        #     See ref: https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/issues/2
        #     The input image size should be divisible by 32
        #     The window size is recommended to be 1/32 of the image size
        if model_resolution == 'r1':
            config_r1 = config_default.copy_and_resolve_references()
            config_r1.img_size    = (160, 160, 192)
            config_r1.window_size = (5, 5, 6)
            model = TransMorph(config_r1)
        elif model_resolution == 'r2':
            config_r2 = config_default.copy_and_resolve_references()
            config_r2.img_size    = (128, 128, 128)
            config_r2.window_size = (4, 4, 4)
            model = TransMorph(config_r2)
        else:
            raise ValueError("Unsupported model_resolution, must be: r1 or r2")
    elif model_type == '05dualprnet-v1':
        from .registration_models.m05dualprnet.dualprnet_v1 import DualPRNet_disp
        model = DualPRNet_disp(inshape)
    elif model_type == '06lapirn-v1':
        assert model_resolution == 'r1', 'model_resolution should be r1 for 06lapirn-v1'
        from .registration_models.m06lapirn.lapirn_v1 import Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1, \
            Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl3
        # configs
        start_channel = 7
        inshape_r1 = inshape
        inshape_r2 = tuple(d // 2 for d in inshape)
        inshape_r4 = tuple(d // 4 for d in inshape)
        # model
        model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl1(2, 3, start_channel, is_train=True, inshape=inshape_r4)
        model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl2(2, 3, start_channel, is_train=True, inshape=inshape_r2, model_lvl1=model_lvl1)
        model      = Miccai2020_LDR_laplacian_unit_disp_add_unorm_lvl3(2, 3, start_channel, is_train=False, inshape=inshape_r1, model_lvl2=model_lvl2)

    return model


def model_register_v1(inputs, model, model_type):
    '''
    One thing to consider (potentially)
        handle padding and crop for transmorph+r2 here
        neat, but might not be as flexible!!!
    The following work the same for my code:
        deformed, dvf = model(inputs[0], inputs[1])
        deformed, dvf = model(*inputs)
    '''
    if model_type.startswith('01voxelmorph-v1'):
        deformed, dvf = model(inputs[0], inputs[1])
    elif model_type.startswith('02resunet-v1'):
        deformed, dvf = model(inputs[0], inputs[1])
    elif model_type.startswith('03lkunet-v1'):
        deformed, dvf = model(inputs[0], inputs[1])
    elif model_type.startswith('04transmorph-v1'):
        deformed, dvf = model(inputs[0], inputs[1])
    elif model_type.startswith('05dualprnet-v1'):
        deformed, dvf = model(inputs[0], inputs[1])
    elif model_type.startswith('06lapirn-v1'):        
        # Note the return order is reversed from above models, while the input order is the same
        dvf, deformed = model(inputs[0], inputs[1])
    else:
        raise ValueError("Unsupported model_type, got: ", model_type)
    
    return deformed, dvf
