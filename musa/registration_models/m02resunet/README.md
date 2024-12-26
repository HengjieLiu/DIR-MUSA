# Readme for ResUNet
Implementation of the ResUNet model for image registration.  
The basic setup is similar to VoxelMorph. The main difference is replacing the simple UNet with a residual UNet ([nnU-Net for PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/nnunet_pyt)).

### Usage
```python
deformed, dvf = model(moving, fixed)
```

### Files
- **`resunet_v1.py`**  
Similar to voxelmorph_v1.py. The main difference is replacing the simple UNet with a residual UNet ([nnU-Net for PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/nnunet_pyt)).  

- **`../unetblocks/utils_unet_blocks.py`**  
This code is used in both ResUNet (resunet_v1.py) and DualPRNet (dualprnet_v1.py)  
see references:  
    - [monai dynunet](https://github.com/Project-MONAI/MONAI/blob/fdd07f36ecb91cfcd491533f4792e1a67a9f89fc/monai/networks/nets/dynunet.py)  
    - [monai dynunet_block](https://github.com/Project-MONAI/MONAI/blob/fdd07f36ecb91cfcd491533f4792e1a67a9f89fc/monai/networks/blocks/dynunet_block.py)  
    - [nnU-Net for PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/nnunet_pyt)  
