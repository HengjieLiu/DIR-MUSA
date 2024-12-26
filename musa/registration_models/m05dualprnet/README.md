# Readme for DualPRNet
Implementation of the DualPRNet model for image registration.  
The basic setup is similar to VoxelMorph. The main difference is replacing the simple UNet with a DualPRNet.

### Usage
```python
deformed, dvf = model(moving, fixed)
```

### Files
- **`dualprnet_v1.py`**  
Similar to voxelmorph_v1.py. The main difference is replacing the simple UNet with a customized DualPRNet.  
Our current reimplementation is based on both DualPRNet and DualPRNet++. We used redidual connection in DualPRNet++. But the 3D correlation layer in DualPRNet++ is not used.  
See references:  
    - [official dualprnet implementation including dualprnet++](https://github.com/kangmiao15/dual-stream-prnet-plus)  
    - [tensorflow reimplemention of DualPRNet](https://github.com/JinxLv/reimplemention-of-Dual-PRNet/blob/83f2e42ac6316fac49708a7bc42b303a8ff16af2/network/base_networks.py)  

- **`../unetblocks/utils_unet_blocks.py`** 
This code is used in both ResUNet (resunet_v1.py) and DualPRNet (dualprnet_v1.py)  
See references:  
    - [monai dynunet](https://github.com/Project-MONAI/MONAI/blob/fdd07f36ecb91cfcd491533f4792e1a67a9f89fc/monai/networks/nets/dynunet.py)  
    - [monai dynunet_block](https://github.com/Project-MONAI/MONAI/blob/fdd07f36ecb91cfcd491533f4792e1a67a9f89fc/monai/networks/blocks/dynunet_block.py)  
    - [nnU-Net for PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/nnunet_pyt)  

- **`utils_dualpr_blocks.py`**  
class ConvBlock_PRres: residual block used in PR module of DualPRNet
