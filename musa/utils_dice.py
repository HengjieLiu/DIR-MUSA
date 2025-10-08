

import torch

################# local imports #################
from . import utils_basics as utils_basics
#################################################


def dice_val_both(masks_seg_o, masks_seg_b, DEBUG=False):
    # masks_seg_o, masks_seg_b = (deformed_seg_o_oh, fixed_seg_o_oh), (deformed_seg_b_oh, fixed_seg_b_oh)
    
    dice_o = dice_val(masks_seg_o[0], masks_seg_o[1]) # torch.Size([1, 17])
    dice_b = dice_val(masks_seg_b[0], masks_seg_b[1]) # torch.Size([1, 9])
    
    if DEBUG:
        print('\n\n\n')
        print('dice_o: ', dice_o.shape, dice_o)
    
    # remove the 12th element (mandible)
    dice_o = torch.cat((dice_o[:, :11], dice_o[:, 12:]), dim=1) # torch.Size([1, 16])
    
    dice_all = torch.cat((dice_o, dice_b), dim=1) # torch.Size([1, 25])
    
    if DEBUG:
        print('dice_o: ', dice_o.shape, dice_o)
        print('dice_b: ', dice_b.shape, dice_b)
        print('dice_all: ', dice_all.shape, dice_all)
    
    dice_o_mean = torch.mean(dice_o)
    dice_b_mean = torch.mean(dice_b)
    dice_all_mean = torch.mean(dice_all)
    
    if DEBUG:
        print('dice_o_mean: ', dice_o_mean.shape, dice_o_mean)
        print('dice_b_mean: ', dice_b_mean.shape, dice_b_mean)
        print('dice_all_mean: ', dice_all_mean.shape, dice_all_mean)
    
    dice_info = (dice_o_mean, dice_b_mean, dice_all_mean)
    
    return dice_info
    

def dice_val(y_pred, y_true):
    '''
    return all dices (not mean)
    '''
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    
    dice = (2.*intersection) / (union + 1e-5)
    
    return dice



def dice_val_transmorph(y_pred, y_true, num_clus):
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))
