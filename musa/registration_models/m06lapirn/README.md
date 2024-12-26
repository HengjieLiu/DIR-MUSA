# Readme for LapIRN
Implementation of the LapIRN model for image registration.  

### Usage
```python
deformed, dvf = model(moving, fixed)
```

### Files
- **`lapirn_v1.py`**  
A customized version of the LapIRN model.  

- **`lapirn_org.py`**  
The original LapIRN model sourced from the official LapIRN repository. This file is a direct, renamed version of the original pytorch implementation, which can be found here: [Original LapIRN Model](https://github.com/cwmok/LapIRN/blob/c17fb7564ca752d37e97baf12cafff44e01da668/Code/miccai2020_model_stage.py).
