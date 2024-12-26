# Readme for VoxelMorph
Implementation of the VoxelMorph model for image registration.  

### Usage
```python
deformed, dvf = model(moving, fixed)
```

### Files
- **`voxelmorph_v1.py`**  
A customized version of the VoxelMorph model.  
Key changes:
      1. Use updated SpatialTransformer (see utils_warp.py) 
      2. Simplify the code for direct displacement prediction (non-diffeomorphic version). The code related to scaling-and-squaring for diffeomorphic registration is removed.

- **`voxelmorph_org.py`**  
The original VoxelMorph model sourced from the official VoxelMorph repository. This file is a direct, renamed version of the original pytorch implementation, which can be found here: [Original VoxelMorph Model](https://github.com/voxelmorph/voxelmorph/blob/579a995492bddfe9ce38161e58cf260fc155c4fd/voxelmorph/torch/networks.py).
