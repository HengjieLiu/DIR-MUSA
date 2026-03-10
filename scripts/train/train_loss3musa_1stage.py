"""
Train registration models with MUSA loss for 1-stage registration (work both on full/half resolution)
Note: 
    1. In practice, we only use MUSA loss on half resolution images (r2, 4mm isotropic), as training on full resolution images unstable for most models.
    2. For half resolution training, we downsample both the images and bone segmentations to r2 resolution for loss calculation
        It is also possible to upsample the dvf to the orignial r1 resolution for loss calculation. In practice, we did not seen much difference.

    Args:
        --trn-list: Training list filename (.nii.gz)
        --val-list: Validation list filename (.nii.gz)
        --vol-path: Path to training images (volume data)
        --seg-path-o: Path to organ segmentations
        --seg-path-b: Path to bone segmentations
        --model-resolution: Input resolution: r1 (2mm) or r2 (4mm)
        --model-type: Model type to train
        --lr: Learning rate
        --loss-sim-type: Type of similarity loss (mse or ncc)
        --lambda: Weight of deformation loss
        --alpha: Weight of MUSA loss
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
        --continue-training: Continue training from checkpoint
    Example:
        python train_loss3musa_1stage.py \
            --trn-list /database/wip/trn_list_inter.txt \
            --val-list /database/wip/val_list_inter.txt \
            --vol-path /database/wip/vol_path \
            --seg-path-o /database/wip/seg_path_o \
            --seg-path-b /database/wip/seg_path-b \
            --model-resolution r2 \
            --model-type 01voxelmorph-vf \
            --lr 1e-4 \
            --lambda 1.0 \
            --alpha 1e3 \
            --gpu 0 \
            --batch-size 1 \
            --epochs 500 \
            --steps-per-epoch 100 \
            --epoch-save 10 \
            --epoch-val 10 \
            --out-dir /database/wip/out_dir \
"""

import os
import sys
import time
import math
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


parser = argparse.ArgumentParser(description='Train DIR-MUSA registration models, MUSA loss for 1-stage registration (work both on full/half resolution, in practice only use half resolution for stability)')
parser.add_argument('--trn-list', default='path/to/trn_list_inter.txt', help='Training list filename (.nii.gz)')
parser.add_argument('--val-list', default='path/to/val_list_inter.txt', help='Validation list filename (.nii.gz)')
parser.add_argument('--vol-path', required=True, help='Path to training images (volume data)')
parser.add_argument('--seg-path-o', required=True, help='Path to organ segmentations (only for validation)')
parser.add_argument('--seg-path-b', required=True, help='Path to bone segmentations (for training MUSA loss and validation)')
parser.add_argument('--model-resolution', required=True, choices=['r1', 'r2', 'r4'], help='Input resolution: r1 (2mm) or r2 (4mm) or r4 (8mm)')
parser.add_argument('--model-type', required=True, choices=['01voxelmorph-vf', '02resunet-vf', '03lkunet-vf-lk09', '04transmorph-vf', '05dualprnet-vf'], help='Model type to train, 06lapirn-vf needs its own training script')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--loss-sim-type', required=False, default='mse', choices=['mse', 'ncc'], help='Type of similarity loss')
parser.add_argument('--lambda', required=True, type=float, dest='weight_lambda', help='Weight of deformation loss')
parser.add_argument('--alpha', required=True, type=float, dest='weight_alpha', help='Weight of MUSA loss')
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

# For MUSA loss, we need bone segmentations for training (bone segmentations are already binarized in dataloader by specifying path_seg_musaloss)
trn_dataset = musa.utils_dataloader.myDataset_trn(trn_files, args.vol_path, path_seg_musaloss=args.seg_path_b)
val_dataset = musa.utils_dataloader.myDataset_val(val_files, args.vol_path, args.seg_path_o, args.seg_path_b)

trn_generator = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
val_generator = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

# Setups for model
if args.model_resolution == 'r1':
    input_shape = (160, 160, 192)
elif args.model_resolution == 'r2':
    input_shape = (80, 80, 96)
elif args.model_resolution == 'r4':
    input_shape = (40, 40, 48)

down_scale = int(args.model_resolution[-1])
down_scale_cnt = int(math.log(down_scale, 2))

# Check if padding is needed for TransMorph at r2 resolution
FLAG_PAD = (args.model_type.startswith('04transmorph') and args.model_resolution == 'r2')

# Set up spatial transformer for validation
spatial_transformer_r1 = musa.warp_modules.SpatialTransformer((160, 160, 192))
spatial_transformer_r1.to(device)
spatial_transformer_r2 = musa.warp_modules.SpatialTransformer((160//2, 160//2, 192//2))
spatial_transformer_r2.to(device)


# Get model
model = musa.utils_model_zoo.get_model_v1(inshape=input_shape, model_type=args.model_type, model_resolution=args.model_resolution)
model.to(device)
model.train()

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Set up loss functions
if args.loss_sim_type == 'mse':
    func_losses = [musa.utils_loss.MSE().loss]
elif args.loss_sim_type == 'ncc':
    func_losses = [musa.utils_loss.NCC().loss]
else:
    raise ValueError(f"Unsupported loss_sim_type: {args.loss_sim_type}")
if args.model_resolution == 'r1':
    be_scale = 1
elif args.model_resolution == 'r2':
    be_scale = 2
elif args.model_resolution == 'r4':
    be_scale = 4
else:
    raise ValueError(f"Unsupported model_resolution: {args.model_resolution}")

print(f"Using BE scale {be_scale} for {args.model_resolution}")
func_losses.append(musa.utils_loss.BE3d(scale=be_scale).loss)

# Add MUSA loss - using bone segmentation for rigidity constraint
func_losses.append(musa.utils_loss.BE3d_masked(scale=be_scale).loss)

weights = [1, args.weight_lambda, args.weight_alpha]
print(f"Loss weights: similarity=1, deformation={args.weight_lambda}, musa={args.weight_alpha}")


###### Initialize training (with support for continue training from checkpoint)
if args.checkpoint_path is None: # Initialize
    epoch_start = 0
    # training loss
    list_epoch_loss = []
    list_epoch_loss_sim = []
    list_epoch_loss_dvf_smooth = []
    list_epoch_loss_musa = []
    # validation dice
    dice_scores = {'dice_o': [], 'dice_b': [], 'dice_a': []}
    best_dice_scores = {'dice_o': {'mean': 0, 'std': 0,'epoch': 0}, # for soft organ (seg_o)
                        'dice_b': {'mean': 0, 'std': 0,'epoch': 0}, # for bone (seg_b)
                        'dice_a': {'mean': 0, 'std': 0,'epoch': 0}} # for all (organ + bone)
else: # Load checkpoint if specified
    checkpoint = torch.load(args.checkpoint_path)
    
    # load_state_dict for model and optimizer
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # epoch_start already retrieved in "Setups for outputs", just check here!!!
    assert epoch_start == checkpoint['epoch'] + 1
    list_epoch_loss = checkpoint.get('list_epoch_loss', [])
    list_epoch_loss_sim = checkpoint.get('list_epoch_loss_sim', [])
    list_epoch_loss_dvf_smooth = checkpoint.get('list_epoch_loss_dvf_smooth', [])
    list_epoch_loss_musa = checkpoint.get('list_epoch_loss_musa', [])
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
    model.train()
   
    ### Init for training info
    # loss
    epoch_loss = 0
    epoch_loss_sim  = 0
    epoch_loss_dvf_smooth = 0
    epoch_loss_musa = 0
    # time
    epoch_step_time = 0
    
    for trn_step, data in enumerate(trn_generator):
        
        ### break for next epoch
        if trn_step >= args.steps_per_epoch:
            break

        step_start_time = time.time()
        
        ### unpack data
        data = [d.to(device) for d in data] # (moving, fixed, moving_seg_musa, fixed_seg_musa), note moving_seg_musa/fixed_seg_musa are already binarized in dataloader
        (moving, fixed, moving_seg_musa, fixed_seg_musa) = data

        ### preprocess-downsample (for r2)
        for i in range(down_scale_cnt):
            moving = musa.warp_modules.vol_downsamplex2(moving)
            fixed  = musa.warp_modules.vol_downsamplex2(fixed)
            moving_seg_musa = musa.warp_modules.vol_downsamplex2(moving_seg_musa)
            fixed_seg_musa = musa.warp_modules.vol_downsamplex2(fixed_seg_musa)

        ### pack inputs
        inputs = (moving, fixed)
        
        ### model forward
        if not FLAG_PAD:
            deformed, dvf = musa.utils_model_zoo.model_register_v1(inputs, model, args.model_type)
        else: # special handling for transmorph&r2
            ## preprocess-pad [paired with crop]
            pad_size = (16,16,24,24,24,24) # reverse order
            inputs = [F.pad(d, pad=pad_size) for d in inputs]

            deformed, dvf = musa.utils_model_zoo.model_register_v1(inputs, model, args.model_type)

            ## postprocess-crop [paired with pad]
            deformed = deformed[..., 24:24+80, 24:24+80, 16:16+96]
            dvf      =      dvf[..., 24:24+80, 24:24+80, 16:16+96]

        ### postprocess-upsample dvf (for r2)
        # Skip postprocess-upsample dvf for r2, loss is calculated on r2 resolution
        # if down_scale_cnt > 0:
        #     dvf_r2 = dvf
        #     dvf_r1 = musa.warp_modules.dvf_upsample(dvf)
        # else:
        #     dvf_r1 = dvf
        
        ### Create MUSA mask from bone segmentation (Use deformed bone segmentation for MUSA loss)
        # note moving_seg_musa/fixed_seg_musa are already binarized in dataloader, so warping with bilinear interpolation is fine!!!
        if args.model_resolution == 'r1':
            deformed_seg_musa = spatial_transformer_r1(moving_seg_musa, dvf, mode='bilinear')
        elif args.model_resolution == 'r2':
            deformed_seg_musa = spatial_transformer_r2(moving_seg_musa, dvf, mode='bilinear')
        else:
            raise ValueError(f"Unsupported model_resolution: {args.model_resolution}")
        
        deformed_seg_musa = deformed_seg_musa.detach() # detach from graph, no backward pass through the mask warping process
        
        
        ### loss calculation
        loss_sim        = func_losses[0](fixed, deformed) * weights[0]
        loss_dvf_smooth = func_losses[1](dvf) * weights[1]
        # weight adjust for portion already included in loss_dvf_smooth 
        #   i.e., adding loss_dvf_smooth and loss_musa together gives the alpha ratio between bones and soft tissues
        loss_musa       = func_losses[2](dvf, deformed_seg_musa) * ((weights[2] - 1) * weights[1]) 
        loss = loss_sim + loss_dvf_smooth + loss_musa

        ### backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### record and log
        epoch_loss            += loss.item()
        epoch_loss_sim        += loss_sim.item()
        epoch_loss_dvf_smooth += loss_dvf_smooth.item()
        epoch_loss_musa       += loss_musa.item()

        epoch_step_time += (time.time() - step_start_time)
        
        ###### END of training loop ######

    ###### log training info
    epoch_loss /= args.steps_per_epoch
    epoch_loss_sim /= args.steps_per_epoch
    epoch_loss_dvf_smooth /= args.steps_per_epoch
    epoch_loss_musa /= args.steps_per_epoch
    epoch_step_time /= args.steps_per_epoch

    list_epoch_loss.append(epoch_loss)
    list_epoch_loss_sim.append(epoch_loss_sim)
    list_epoch_loss_dvf_smooth.append(epoch_loss_dvf_smooth)
    list_epoch_loss_musa.append(epoch_loss_musa)
    
    # Organize trn_info
    epoch_info = 'Epoch %d/%d' % (epoch+1, args.epochs)
    time_info = '%.4f sec/step, %.4f sec/epoch' % (epoch_step_time, epoch_step_time*args.steps_per_epoch)
    loss_info = 'loss: %.4e, loss_sim: %.4e, loss_dvf_smooth: %.4e, loss_musa: %.4e' % \
                (epoch_loss, epoch_loss_sim, epoch_loss_dvf_smooth, epoch_loss_musa)
    trn_info = ' - '.join((epoch_info, time_info, loss_info))
    
    ### print and log to all log
    print(trn_info, flush=True)



    ###### validation loop   
    if (epoch+1) % args.epoch_val == 0:
        
        with torch.no_grad():
            
            ### Set to eval
            model.eval()

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
                # Validation is preformed on full resolution data only, i.e. moving_seg/fixed_seg are not downsampled
                moving_seg_o_oh = musa.utils_dataprep.to_onehot_seg_o(moving_seg_o)
                fixed_seg_o_oh  = musa.utils_dataprep.to_onehot_seg_o(fixed_seg_o)
                moving_seg_b_oh = musa.utils_dataprep.to_onehot_seg_b(moving_seg_b)
                fixed_seg_b_oh  = musa.utils_dataprep.to_onehot_seg_b(fixed_seg_b)
                
                ### preprocess-downsample (for r2)
                for i in range(down_scale_cnt):
                    moving = musa.warp_modules.vol_downsamplex2(moving)
                    fixed  = musa.warp_modules.vol_downsamplex2(fixed)

                ### pack inputs
                inputs = (moving, fixed)
                
                ### model forward
                if not FLAG_PAD:
                    deformed, dvf = musa.utils_model_zoo.model_register_v1(inputs, model, args.model_type)
                else: # special handling for transmorph+r2
                    ## preprocess-pad [paired with crop]
                    pad_size = (16,16,24,24,24,24) # reverse order
                    inputs = [F.pad(d, pad=pad_size) for d in inputs]

                    deformed, dvf = musa.utils_model_zoo.model_register_v1(inputs, model, args.model_type)

                    ## postprocess-crop [paired with pad]
                    dvf = dvf[..., 24:24+80, 24:24+80, 16:16+96]
                
                ### postprocess-upsample
                for i in range(down_scale_cnt):
                    dvf = musa.warp_modules.dvf_upsample(dvf)

                ### Mask warping with SpatialTransformer
                deformed_seg_o_oh = spatial_transformer_r1(moving_seg_o_oh, dvf, mode='nearest')
                deformed_seg_b_oh = spatial_transformer_r1(moving_seg_b_oh, dvf, mode='nearest')

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
                    best_dice_scores[key]['epoch'] = epoch+1 ### FLAG(epoch+1)
                    
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
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'list_epoch_loss': list_epoch_loss,
            'list_epoch_loss_sim': list_epoch_loss_sim,
            'list_epoch_loss_dvf_smooth': list_epoch_loss_dvf_smooth,
            'list_epoch_loss_musa': list_epoch_loss_musa,
            'dice_scores': dice_scores,
            'best_dice_scores': best_dice_scores,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

###### DONE