# Readme for LKUNet
Implementation of the LKUNet model for image registration.  

### Usage
```python
deformed, dvf = model(moving, fixed)
```

### Files
- **`lkunet_v1.py`**  
A customized version of the LKUNet model.  

- **`lkunet_org.py`**  
The original LKUNet model sourced from the official LKUNet repository. This file is a direct, renamed version of the original pytorch implementation, which can be found here: [Original LKUNet Model](https://github.com/xi-jia/LKU-Net/blob/c304a36d099a084ebaba743de387fab816c182c2/LKU-Net_3D_OASIS/LKU-Net-Full-Resolution/Models.py)
