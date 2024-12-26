# Readme for TransMorph
Implementation of the TransMorph model for image registration.  

### Usage
```python
deformed, dvf = model(moving, fixed)
```

### Files
- **`transmorph_v1.py`**  
A customized version of the TransMorph model.  
Key changes:
      1. Use updated SpatialTransformer (see utils_warp.py)
      2. add indexing='ij' to torch.meshgrid to get rid of warning


- **`transmorph_org.py`**  
The original TransMorph model sourced from the official TransMorph repository. This file is a direct, renamed version of the original pytorch implementation, which can be found here: [Original TransMorph Model](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/12dd5a1b142161752302f29c9956a95873a97173/TransMorph/models/TransMorph.py).


- **`configs_TransMorph.py`**  
The original TransMorph config file sourced from the official TransMorph repository. This file is a direct copy of: [Original TransMorph config](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/f06934a13c736438541eea12d780c59eeb534d28/TransMorph/models/configs_TransMorph.py).


### Notes for configuration
To configure and use the TransMorph model with different input sizes and resolutions, the following rules should be followed (see musa.utils_model_zoo for details):

#### Configure the image size and window size  
See ref: [Issue #2 from TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/issues/2)
- The input image size should be divisible by 32
- The window size is recommended to be 1/32 of the image size
