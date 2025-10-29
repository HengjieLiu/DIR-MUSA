
# Training Scripts

**Note:** LapIRN requires its own training script, see https://github.com/cwmok/LapIRN

## Loss1: Standard Loss (Lsim + Lreg)

- `train_loss1std_1stage.py`: Train with standard loss (MSE or NCC) for 1-stage registration (half/full resolution)
- `train_loss1std_2stage.py`: Train with standard loss (MSE or NCC) for 2-stage registration (stage1: half resolution/weight freezed, stage2: full resolution)

## Loss2: Standard Loss + Dice (Weak) Supervision (Lsim + Lreg + Lseg)

- `train_loss2dice_1stage.py`: Train with Dice loss for 1-stage registration (half/full resolution)
- `train_loss2dice_2stage.py`: Train with Dice loss for 2-stage registration (half resolution/weight freezed, stage2: full resolution)

## Loss3: Standard Loss + MUSA Loss Regularization (Lsim + Lreg + Lmusa)

- `train_loss3musa_1stage.py`: Train with MUSA loss for 1-stage registration (half resolution, full resolution is unstable)
- `train_loss3musa_2stage.py`: Train with MUSA loss for 2-stage registration (half resolution/weight freezed, stage2: full resolution)

> **Note:** For `train_loss3musa_2stage.py`, stage 2 uses standard loss (Lsim + Lreg). In other words, MUSA loss is only used in training the half-resolution models in `train_loss3musa_1stage.py`
