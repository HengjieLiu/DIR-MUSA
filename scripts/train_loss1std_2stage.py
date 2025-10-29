"""
Train registration models with standard loss (e.g., MSE or NCC) for 2-stage registration
(Stage 1: coarse registration at r2 resolution, Stage 2: fine registration at r1 resolution)
Note: 
    1. This script assume the stage1 model is pre-trained at r2 resolution using train_loss1std_1stage.py, 
        and we freeze the stage1 model weights and train the stage2 model at r1 resolution using this script.

    Args:
        --trn-list: Training list filename (.nii.gz)
        --val-list: Validation list filename (.nii.gz)
        --vol-path: Path to training images (volume data)
        --seg-path-o: Path to organ segmentations
        --seg-path-b: Path to bone segmentations
        --model-type: Model type to train
        --model-load-stage1: Model path for weight loading (stage1, pre-trained at r2)
        --model-load-stage2: Model path for weight loading (stage2, or 'from-scratch' for initialization)
        --lr: Learning rate
        --loss-sim-type: Type of similarity loss (mse or ncc)
        --lambda: Weight of deformation loss
        --gpu: GPU ID(s), comma-separated
        --batch-size: Batch size
        --epochs: Number of training epochs
        --steps-per-epoch: Number of training steps per epoch
        --epoch-save: Model save frequency (epochs)
        --epoch-val: Validation frequency (epochs)
        --checkpoint-path: Path to checkpoint for resuming training
        --out-dir: Output directory for models and logs
        --num-workers: Number of data loading workers
        --cudnn: CUDNN mode: det (deterministic), ben (benchmark), default
    Example:
        python train_loss1std_2stage.py \
            --trn-list /path/to/trn_list_inter.txt \
            --val-list /path/to/val_list_inter.txt \
            --vol-path /path/to/vol_path \
            --seg-path-o /path/to/seg_path_o \
            --seg-path-b /path/to/seg_path_b \
            --model-type 01voxelmorph-vf \
            --model-load-stage1 /path/to/stage1_model.pth \
            --model-load-stage2 from-scratch \
            --lr 1e-4 \
            --lambda 1.0 \
            --gpu 0 \
            --batch-size 1 \
            --epochs 500 \
            --steps-per-epoch 100 \
            --epoch-save 10 \
            --epoch-val 10 \
            --out-dir /path/to/out_dir
"""

import os
import sys
import time
import math
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

### Setup path and import musa
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import musa

parser = argparse.ArgumentParser(description='Train DIR-MUSA registration models, standard loss for 2-stage registration (coarse r2 + fine r1)')
parser.add_argument('--trn-list', default='path/to/trn_list_inter.txt', help='Training list filename (.nii.gz)')
parser.add_argument('--val-list', default='path/to/val_list_inter.txt', help='Validation list filename (.nii.gz)')
parser.add_argument('--vol-path', required=True, help='Path to training images (volume data)')
parser.add_argument('--seg-path-o', required=True, help='Path to organ segmentations (only for validation)')
parser.add_argument('--seg-path-b', required=True, help='Path to bone segmentations (only for validation)')
parser.add_argument('--model-type', required=True, choices=['01voxelmorph-vf', '02resunet-vf', '03lkunet-vf-lk09', '04transmorph-vf', '05dualprnet-vf'], help='Model type to train, 06lapirn-vf needs its own training script')
parser.add_argument('--model-load-stage1', required=True, help='Model path for weight loading (stage1, pre-trained at r2 resolution)')
parser.add_argument('--model-load-stage2', required=True, help="Model path for weight loading (stage2), or 'from-scratch' for random initialization")
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--loss-sim-type', required=False, default='mse', choices=['mse', 'ncc'], help='Type of similarity loss')
parser.add_argument('--lambda', required=True, type=float, dest='weight_lambda', help='Weight of deformation loss')
parser.add_argument('--gpu', default='0', help='GPU ID(s), comma-separated')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='Number of training steps per epoch')
parser.add_argument('--epoch-save', type=int, default=10, help='Model save frequency (epochs)')
parser.add_argument('--epoch-val', type=int, default=10, help='Validation frequency (epochs)')
parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to checkpoint for resuming training')
parser.add_argument('--out-dir', required=True, help='Output directory for models and logs')
parser.add_argument('--num-workers', type=int, default=0, help='Number of data loading workers')
parser.add_argument('--cudnn', choices=['det', 'ben', 'default'], default='ben', required=False, help='CUDNN mode: det (deterministic), ben (benchmark), default; default is ben for speed')

args = parser.parse_args()

# Setups for outputs
if args.checkpoint_path is None: # Initialize
    os.makedirs(args.out_dir, exist_ok=True)
    dir_out = args.out_dir
    dir_out_checkpoint = os.path.join(dir_out, 'checkpoint')
    os.makedirs(dir_out_checkpoint, exist_ok=True)
    epoch_start = 0
else: # Load checkpoint if specified
    dir_out = '/'.join(args.checkpoint_path.split('/')[0:-2])
    epoch_start = int(args.checkpoint_path.split('/')[-1][0:4])
    cont_suffix = f'-cont{epoch_start:04d}'
    dir_out_checkpoint = os.path.join(dir_out, f'checkpoint{cont_suffix}')
    os.makedirs(dir_out_checkpoint, exist_ok=True)

print(f"Training started")
print(f"Output directory: {dir_out}")
print(f"Checkpoint directory: {dir_out_checkpoint}")

# Setups for GPU and CUDNN
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda')

if args.cudnn == 'det':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("CUDNN: Deterministic mode enabled")
elif args.cudnn == 'ben':
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print("CUDNN: Benchmark mode enabled")
else:
    print("CUDNN: Using default settings")

# Setups for data
trn_files = musa.utils_io.read_file_list(args.trn_list)
val_files = musa.utils_io.read_file_list(args.val_list)

trn_dataset = musa.utils_dataloader.myDataset_trn(trn_files)
val_dataset = musa.utils_dataloader.myDataset_val(val_files, args.vol_path, args.seg_path_o, args.seg_path_b)

trn_generator = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
val_generator = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

# Setups for model
# Stage 1: coarse registration at r2 resolution (80x80x96)
input_shape_r2 = (80, 80, 96)
# Stage 2: fine registration at r1 resolution (160x160x192)
input_shape_r1 = (160, 160, 192)

# Set up spatial transformer and composer for image warping and DVF composition
spatial_transformer_r1 = musa.warp_modules.SpatialTransformer(input_shape_r1)
spatial_transformer_r1.to(device)
composer_r1 = musa.warp_modules.ComposeDVF(input_shape_r1)
composer_r1.to(device)

# Check if padding is needed for TransMorph (at r2 resolution)
FLAG_PAD = args.model_type.startswith('04transmorph')

# Get models: stage1 (coarse) and stage2 (fine)
model_stage1 = musa.utils_model_zoo.get_model_v1(inshape=input_shape_r2, model_type=args.model_type, model_resolution='r2')
model_stage2 = musa.utils_model_zoo.get_model_v1(inshape=input_shape_r1, model_type=args.model_type, model_resolution='r1')

model_stage1.to(device)
model_stage2.to(device)

model_stage1.eval()  # Stage1 is frozen (pre-trained at r2 resolution)
model_stage2.train() # Stage2 is trainable

# Set up optimizer (only for stage2, since stage1 is frozen)
optimizer = torch.optim.Adam(model_stage2.parameters(), lr=args.lr)

# Set up loss functions
# For 2-stage training, loss is calculated on the final composed result at r1 resolution
if args.loss_sim_type == 'mse':
    func_losses = [musa.utils_loss.MSE().loss]
elif args.loss_sim_type == 'ncc':
    func_losses = [musa.utils_loss.NCC().loss]
else:
    raise ValueError(f"Unsupported loss_sim_type: {args.loss_sim_type}")
be_scale = 1  # BE scale for r1 resolution
print(f"Using BE scale {be_scale} for r1 resolution (final composed result)")
func_losses.append(musa.utils_loss.BE3d(scale=be_scale).loss)

weights = [1, args.weight_lambda]
print(f"Loss weights: similarity=1, deformation={args.weight_lambda}")


###### Initialize training (with support for continue training from checkpoint)

# Load pre-trained stage1 model (coarse registration at r2)
checkpoint_stage1 = torch.load(args.model_load_stage1)
model_stage1.load_state_dict(checkpoint_stage1['model_state_dict'])
print(f"Loaded stage1 model from {args.model_load_stage1}")

if args.checkpoint_path is None: # Initialize
    epoch_start = 0
    
    # Load stage2 model weights or initialize from scratch
    if args.model_load_stage2 == 'from-scratch':
        print(f"Initializing stage2 model from scratch")
    else:
        checkpoint_stage2 = torch.load(args.model_load_stage2)
        model_stage2.load_state_dict(checkpoint_stage2['model_state_dict'])
        print(f"Loaded stage2 model from {args.model_load_stage2}")
    
    # training loss
    list_epoch_loss = []
    list_epoch_loss_sim = []
    list_epoch_loss_dvf_smooth = []
    # validation dice
    dice_scores = {'dice_o': [], 'dice_b': [], 'dice_a': []}
    best_dice_scores = {'dice_o': {'mean': 0, 'std': 0,'epoch': 0}, # for soft organ (seg_o)
                        'dice_b': {'mean': 0, 'std': 0,'epoch': 0}, # for bone (seg_b)
                        'dice_a': {'mean': 0, 'std': 0,'epoch': 0}} # for all (organ + bone)
else: # Load checkpoint if specified
    checkpoint = torch.load(args.checkpoint_path)
    
    # load_state_dict for model and optimizer
    model_stage2.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # epoch_start already retrieved in "Setups for outputs", just check here
    assert epoch_start == checkpoint['epoch'] + 1
    list_epoch_loss = checkpoint.get('list_epoch_loss', [])
    list_epoch_loss_sim = checkpoint.get('list_epoch_loss_sim', [])
    list_epoch_loss_dvf_smooth = checkpoint.get('list_epoch_loss_dvf_smooth', [])
    dice_scores = checkpoint.get('dice_scores', [])
    best_dice_scores = checkpoint.get('best_dice_scores', [])
    
    print(f"\tResuming training from epoch {epoch_start}")
    print(f"\tResuming from checkpoint {args.checkpoint_path}")


                
######################################################################################################################
###### Main loop
######################################################################################################################

###### Start main loop
for epoch in range(epoch_start, args.epochs):
    
    ###### training loop
    ### Set to train
    model_stage2.train()
   
    ### Init for training info
    # loss
    epoch_loss = 0
    epoch_loss_sim  = 0
    epoch_loss_dvf_smooth = 0
    # time
    epoch_step_time = 0
    
    for trn_step, data in enumerate(trn_generator):
        
        ### break for next epoch
        if trn_step >= args.steps_per_epoch:
            break

        step_start_time = time.time()
        
        ### unpack data
        data = [d.to(device) for d in data] # (moving, fixed)
        (moving, fixed) = data
        
        ###### stage1: coarse registration at r2 resolution (frozen)
        with torch.no_grad():
            
            # Deep copy to avoid modifying original tensors
            moving_r2 = copy.deepcopy(moving)
            fixed_r2 = copy.deepcopy(fixed)
            
            # Downsample to r2 resolution
            moving_r2 = musa.warp_modules.vol_downsamplex2(moving_r2)
            fixed_r2 = musa.warp_modules.vol_downsamplex2(fixed_r2)
            
            # Prepare inputs for stage1
            inputs_stage1 = (moving_r2, fixed_r2)
            
            # model_stage1 forward
            if not FLAG_PAD:
                deformed_r2, dvf_r2 = musa.utils_model_zoo.model_register_v1(inputs_stage1, model_stage1, args.model_type)
            else: # special handling for transmorph at r2
                # preprocess-pad
                pad_size = (16,16,24,24,24,24) # reverse order
                inputs_stage1 = [F.pad(d, pad=pad_size) for d in inputs_stage1]

                deformed_r2, dvf_r2 = musa.utils_model_zoo.model_register_v1(inputs_stage1, model_stage1, args.model_type)

                # postprocess-crop
                dvf_r2 = dvf_r2[..., 24:24+80, 24:24+80, 16:16+96]
            
            # Upsample DVF to r1 resolution
            dvf_r1_stage1 = musa.warp_modules.dvf_upsample(dvf_r2)
            
            # Warp original moving image with stage1 DVF
            deformed_r1_stage1 = spatial_transformer_r1(moving, dvf_r1_stage1)
        
        ###### stage2: fine registration at r1 resolution (trainable)
        # Prepare inputs for stage2 (deformed image from stage1, fixed image)
        inputs_stage2 = (deformed_r1_stage1, fixed)
        
        # model_stage2 forward
        deformed_r1_stage2, dvf_r1_stage2 = musa.utils_model_zoo.model_register_v1(inputs_stage2, model_stage2, args.model_type)
        
        # Compose DVFs from stage1 and stage2
        dvf_composed = composer_r1(dvf_r1_stage1, dvf_r1_stage2)
        
        # Warp original moving image with composed DVF (final result)
        deformed_final = spatial_transformer_r1(moving, dvf_composed)

        ### loss calculation
        # Loss is calculated on the final composed result
        loss_sim        = func_losses[0](fixed, deformed_final) * weights[0]
        loss_dvf_smooth = func_losses[1](dvf_composed) * weights[1]
        loss = loss_sim + loss_dvf_smooth

        ### backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### record and log
        epoch_loss            += loss.item()
        epoch_loss_sim        += loss_sim.item()
        epoch_loss_dvf_smooth += loss_dvf_smooth.item()

        epoch_step_time += (time.time() - step_start_time)
        
        ###### END of training loop ######

    ###### log training info
    epoch_loss /= args.steps_per_epoch
    epoch_loss_sim /= args.steps_per_epoch
    epoch_loss_dvf_smooth /= args.steps_per_epoch
    epoch_step_time /= args.steps_per_epoch

    list_epoch_loss.append(epoch_loss)
    list_epoch_loss_sim.append(epoch_loss_sim)
    list_epoch_loss_dvf_smooth.append(epoch_loss_dvf_smooth)
    
    # Organize trn_info
    epoch_info = 'Epoch %d/%d' % (epoch+1, args.epochs)
    time_info = '%.4f sec/step, %.4f sec/epoch' % (epoch_step_time, epoch_step_time*args.steps_per_epoch)
    loss_info = 'loss: %.4e, loss_sim: %.4e, loss_dvf_smooth: %.4e' % \
                (epoch_loss, epoch_loss_sim, epoch_loss_dvf_smooth)
    trn_info = ' - '.join((epoch_info, time_info, loss_info))
    
    ### print and log to all log
    print(trn_info, flush=True)



    ###### validation loop   
    if (epoch+1) % args.epoch_val == 0:
        
        with torch.no_grad():
            
            ### Set to eval
            model_stage2.eval()

            ### Init for validation info
            # dice
            val_dice_scores = {'dice_o': [], 'dice_b': [], 'dice_a': []}
            # time
            val_start_time = time.time()

            for val_step, data in enumerate(val_generator):
                
                ### unpack data
                data = [d.to(device) for d in data] # (moving, fixed, moving_seg_o, fixed_seg_o, moving_seg_b, fixed_seg_b)
                (moving, fixed, moving_seg_o, fixed_seg_o, moving_seg_b, fixed_seg_b) = data
                
                ### convert mask to one-hot
                # Validation is performed on full resolution data only
                moving_seg_o_oh = musa.utils_dataprep.to_onehot_seg_o(moving_seg_o)
                fixed_seg_o_oh  = musa.utils_dataprep.to_onehot_seg_o(fixed_seg_o)
                moving_seg_b_oh = musa.utils_dataprep.to_onehot_seg_b(moving_seg_b)
                fixed_seg_b_oh  = musa.utils_dataprep.to_onehot_seg_b(fixed_seg_b)
                
                ###### stage1: coarse registration at r2 resolution (frozen)
                # Deep copy to avoid modifying original tensors
                moving_r2 = copy.deepcopy(moving)
                fixed_r2 = copy.deepcopy(fixed)
                
                # Downsample to r2 resolution
                moving_r2 = musa.warp_modules.vol_downsamplex2(moving_r2)
                fixed_r2 = musa.warp_modules.vol_downsamplex2(fixed_r2)

                # Prepare inputs for stage1
                inputs_stage1 = (moving_r2, fixed_r2)
                
                # model_stage1 forward
                if not FLAG_PAD:
                    deformed_r2, dvf_r2 = musa.utils_model_zoo.model_register_v1(inputs_stage1, model_stage1, args.model_type)
                else: # special handling for transmorph at r2
                    # preprocess-pad
                    pad_size = (16,16,24,24,24,24) # reverse order
                    inputs_stage1 = [F.pad(d, pad=pad_size) for d in inputs_stage1]

                    deformed_r2, dvf_r2 = musa.utils_model_zoo.model_register_v1(inputs_stage1, model_stage1, args.model_type)

                    # postprocess-crop
                    dvf_r2 = dvf_r2[..., 24:24+80, 24:24+80, 16:16+96]
                    
                # Upsample DVF to r1 resolution
                dvf_r1_stage1 = musa.warp_modules.dvf_upsample(dvf_r2)
                
                # Warp original moving image with stage1 DVF
                deformed_r1_stage1 = spatial_transformer_r1(moving, dvf_r1_stage1)
                
                ###### stage2: fine registration at r1 resolution (frozen during validation)
                # Prepare inputs for stage2
                inputs_stage2 = (deformed_r1_stage1, fixed)
                
                # model_stage2 forward
                deformed_r1_stage2, dvf_r1_stage2 = musa.utils_model_zoo.model_register_v1(inputs_stage2, model_stage2, args.model_type)

                # Compose DVFs from stage1 and stage2
                dvf_composed = composer_r1(dvf_r1_stage1, dvf_r1_stage2)
                
                ### Mask warping with SpatialTransformer
                deformed_seg_o_oh = spatial_transformer_r1(moving_seg_o_oh, dvf_composed, mode='nearest')
                deformed_seg_b_oh = spatial_transformer_r1(moving_seg_b_oh, dvf_composed, mode='nearest')

                ### Calculate dice and get info
                dice_info = musa.utils_dice.dice_val_both((deformed_seg_o_oh, fixed_seg_o_oh), (deformed_seg_b_oh, fixed_seg_b_oh))
                (dice_o, dice_b, dice_a) = dice_info

                ### record and log
                val_dice_scores['dice_o'].append(dice_o.item())
                val_dice_scores['dice_b'].append(dice_b.item())
                val_dice_scores['dice_a'].append(dice_a.item())
            
                ###### END of validation loop ######
            
            ###### log validation info
            # Calculate mean and std for each dice score
            for key in dice_scores.keys():
                mean_score = np.mean(val_dice_scores[key])
                std_score = np.std(val_dice_scores[key])
                dice_scores[key].append((mean_score, std_score)) # Keep track of this epoch's scores
                
                # Check if this is the best mean score so far for the key
                if mean_score > best_dice_scores[key]['mean']:
                    best_dice_scores[key]['mean'] = mean_score
                    best_dice_scores[key]['std']  = std_score
                    best_dice_scores[key]['epoch'] = epoch+1 # FLAG(epoch+1)
                    
            # Organize val_info
            epoch_info = 'Val Epoch %d/%d' % (epoch+1, args.epochs)
            time_info = '%.4f sec/validation' % (time.time()-val_start_time)
            dice_info = 'dice (o/b/a): %.4f, %.4f, %.4f' % \
                (dice_scores['dice_o'][-1][0], dice_scores['dice_b'][-1][0], dice_scores['dice_a'][-1][0])
            # dice_scores['dice_o'][-1][0] [-1] for latest, [0] for mean in (mean, std)
            best_dice_info = 'Best dice (o/b/a): %.4f, %.4f, %.4f' % \
                (best_dice_scores['dice_o']['mean'], best_dice_scores['dice_b']['mean'], best_dice_scores['dice_a']['mean'])          
            val_info = ' - '.join((epoch_info, time_info, dice_info, best_dice_info))
            
            ### print and log to all log
            print(val_info, flush=True)

    ###### Save checkpoint every args.epoch_save epochs
    if (epoch+1) % args.epoch_save == 0:
        checkpoint_path = os.path.join(dir_out_checkpoint, f'{str(epoch+1).zfill(4)}.pth')
        checkpoint = {
            'epoch': epoch, # no +1 here (e.g. 499), but +1 in ###FLAG(epoch+1) in loading checkpoint
            'model_state_dict': model_stage2.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'list_epoch_loss': list_epoch_loss,
            'list_epoch_loss_sim': list_epoch_loss_sim,
            'list_epoch_loss_dvf_smooth': list_epoch_loss_dvf_smooth,
            'dice_scores': dice_scores,
            'best_dice_scores': best_dice_scores,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

###### DONE
