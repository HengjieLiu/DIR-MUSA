"""
General description:
    Loss functions for deformable image registration.

Function/Class list:
    Loss for similarity (unsupervised):
        MSE
        NCC

    Loss for segmentation matching (weakly supervised):
        DiceLoss
    
    Loss for deformation regularization:
        General smoothness:
            Grad3d
            BE3d       
        MUSA loss:
            BE3d_masked

Notes:
    Each class will have a loss function loss(self, y_pred, y_true)
        For deformation regularization loss, y_true is placeholder
    
Reference:
    https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/losses.py
    https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/losses.py
"""


import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nnf


######################################################################################################################
###### Loss for similarity (unsupervised)
######################################################################################################################
class MSE(torch.nn.Module):
    """
    Mean squared error loss.
    Notes:
        Modified to include reduction argument
    """
    def __init__(self, reduction='mean'):
        """       
        Args:
        - reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                           'none': no reduction will be applied,
                           'mean': the sum of the output will be divided by the number of elements in the output,
                           'sum': the output will be summed.
        """
        super(MSE, self).__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)
    
    def loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    
class NCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    Notes:
        Based on https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/losses.py
        Modified: 
            sum_filt = torch.ones([1, 1, *win], device=y_pred.device)
            conv_fn = getattr(nnf, 'conv%dd' % ndims)
        Probably want to add 1 to make it [0,1] in the future
    """

    def __init__(self, win=None):
        super(NCC, self).__init__()
        self.win = win

    def loss(self, y_pred, y_true):

        I = y_true
        J = y_pred

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        # sum_filt = torch.ones([1, 1, *win]).to("cuda")
        sum_filt = torch.ones([1, 1, *win], device=y_pred.device)

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(nnf, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        # return -torch.mean(cc)
        return 1-torch.mean(cc) # add 1 to make it [0,1]


######################################################################################################################
###### Loss for segmentation matching (weakly supervised)
######################################################################################################################

class DiceLoss(nn.Module):
    """
    N-D dice for segmentation matching
    Notes: 
        add 1 to make it [0,1] (compared to orginal voxelmorph)
    """
    
    def __init__(self):
        super(DiceLoss, self).__init__()

    def loss(self, y_pred, y_true):
        
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        
        top = 2 * (y_pred * y_true).sum(dim=vol_axes)
        bottom = torch.clamp((y_pred + y_true).sum(dim=vol_axes), min=1e-5)
        
        mean_dice = torch.mean(top / bottom)
        dice_loss = 1 - mean_dice
        
        return dice_loss



######################################################################################################################
###### Loss for deformation regularization
######################################################################################################################

class Grad3d(torch.nn.Module):
    """
    3-D gradient/diffusion regularization.
    Notes:
        Based on https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/losses.py
        Change default penalty to 'l2'
        Change y/x/z order to x/y/z to be consistent with BE3d
        Forward difference is used for approximating spatial gradient
        Note the unit is voxel not mm
    """

    def __init__(self, penalty='l2', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred, y_true=None):
        dx = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dy = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dx = dx * dx
            dy = dy * dy
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

    
class BE3d(torch.nn.Module):
    """
    3-D bending energy regularization.
        E = ∫ (∂²u/∂x²)² + (∂²u/∂y²)² + (∂²u/∂z²)² + 2[(∂²u/∂x∂y)² + (∂²u/∂y∂z)² + (∂²u/∂x∂z)²] dV
    Notes:
        Forward difference is used for approximating spatial gradient
        Note the unit is voxel not mm, so normalization is needed for difference scale/resolution
            The scale argument is needed to account for resolution/scale change
                If scale is provided, the bending energy is nomalized by 1/scale**2
                e.g., if full resolution images use scale of 1, then half resolution images should use scale of 2.
            The normalization factor is 1/scale**2 instead of 1/scale**4
                This is because both dvf and delta_x/y/z has unit of voxel_size (or scale)
                Therefore, first-order gradients are unitless, second-order gradients have unit [1/voxel_size], energies have unit [1/voxel_size**2]                
    """
    
    def __init__(self, scale=1):
        super(BE3d, self).__init__()
        self.scale = scale
        print('INFO: Initializing BE3d with scale: ', self.scale)

    def loss(self, y_pred, y_true=None):
        
        dvf = y_pred
        
        # first order gradients
        dx = dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :]
        dy = dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :]
        dz = dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1]
        
        # second order gradients
        dxx = dx[:, :, 1:, :, :] - dx[:, :, :-1, :, :]
        dyy = dy[:, :, :, 1:, :] - dy[:, :, :, :-1, :]
        dzz = dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1]
        dxy = dx[:, :, :, 1:, :] - dx[:, :, :, :-1, :]
        dyz = dy[:, :, :, :, 1:] - dy[:, :, :, :, :-1]
        dxz = dx[:, :, :, :, 1:] - dx[:, :, :, :, :-1]

        dfdxx2_mean = torch.mean(dxx ** 2)
        dfdyy2_mean = torch.mean(dyy ** 2)
        dfdzz2_mean = torch.mean(dzz ** 2)
        dfdxy2_mean = torch.mean(dxy ** 2)
        dfdyz2_mean = torch.mean(dyz ** 2)
        dfdxz2_mean = torch.mean(dxz ** 2)
        
        energy = dfdxx2_mean + dfdyy2_mean + dfdzz2_mean + \
            2 * (dfdxy2_mean + dfdyz2_mean + dfdxz2_mean)
        
        energy /= self.scale**2

        return energy

    
class BE3d_masked(torch.nn.Module):
    """
    3-D bending energy regularization with mask.
        E = ∫ (∂²u/∂x²)² + (∂²u/∂y²)² + (∂²u/∂z²)² + 2[(∂²u/∂x∂y)² + (∂²u/∂y∂z)² + (∂²u/∂x∂z)²] dV
    Used specifically for musa loss, where the bony mask is provided as mask.

    Notes:
    Mask erosion is needed:
        (1. to account for shape change when calculating forward difference
        (2. to correctly apply the mask
        erode_masks
            use pytorch's convolution for mask erosion
            support values between 0-1, so that musa_mask can be warped using bilinear interpolation
    """
    def __init__(self, scale=1):
        super(BE3d_masked, self).__init__()
        self.scale = scale
        print('INFO: Initializing BE3d_masked with scale: ', self.scale)
    
    def loss(self, dvf, mask):
        
        # get mask
        mask_dxx, mask_dyy, mask_dzz, mask_dxy, mask_dyz, mask_dxz = self.erode_masks(mask)
        
        # first order gradients
        dx = dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :]
        dy = dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :]
        dz = dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1]
        
        # second order gradients
        dxx = dx[:, :, 1:, :, :] - dx[:, :, :-1, :, :]
        dyy = dy[:, :, :, 1:, :] - dy[:, :, :, :-1, :]
        dzz = dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1]
        dxy = dx[:, :, :, 1:, :] - dx[:, :, :, :-1, :]
        dyz = dy[:, :, :, :, 1:] - dy[:, :, :, :, :-1]
        dxz = dx[:, :, :, :, 1:] - dx[:, :, :, :, :-1]

        ### apply mask
        dxx = dxx * mask_dxx
        dyy = dyy * mask_dyy
        dzz = dzz * mask_dzz
        dxy = dxy * mask_dxy
        dyz = dyz * mask_dyz
        dxz = dxz * mask_dxz
        
        dfdxx2_mean = torch.mean(dxx ** 2)
        dfdyy2_mean = torch.mean(dyy ** 2)
        dfdzz2_mean = torch.mean(dzz ** 2)
        dfdxy2_mean = torch.mean(dxy ** 2)
        dfdyz2_mean = torch.mean(dyz ** 2)
        dfdxz2_mean = torch.mean(dxz ** 2)
        
        energy = dfdxx2_mean + dfdyy2_mean + dfdzz2_mean + \
            2 * (dfdxy2_mean + dfdyz2_mean + dfdxz2_mean)
        
        energy /= self.scale**2

        return energy
    
    
    def erode_masks(self, mask):
        """
        Erode masks in PyTorch.
            
        Returns:
            The eroded masks as a tuple of 3D numpy arrays.
        """
        
        # Prepare the erosion structures
        base = torch.zeros((1, 1, 3, 3, 3), device=mask.device)
        base[0,0,1,1,1] = 1
        # Structures for erosion
        s_erode_xp = base.clone(); s_erode_xp[0, 0, 2, 1, 1] = 1
        s_erode_yp = base.clone(); s_erode_yp[0, 0, 1, 2, 1] = 1
        s_erode_zp = base.clone(); s_erode_zp[0, 0, 1, 1, 2] = 1
        s_erode_xyp = base.clone(); s_erode_xyp[0, 0, 2, 1, 1] = 1; s_erode_xyp[0, 0, 1, 2, 1] = 1
        s_erode_yzp = base.clone(); s_erode_yzp[0, 0, 1, 2, 1] = 1; s_erode_yzp[0, 0, 1, 1, 2] = 1
        s_erode_xzp = base.clone(); s_erode_xzp[0, 0, 2, 1, 1] = 1; s_erode_xzp[0, 0, 1, 1, 2] = 1
        
        
        def apply_erosion(mask, structure):
            """
            Erosion by convolution
            A simplified version only works for binary mask (nearest neighbor interpolation):
                conv_result = nnf.conv3d(mask, structure, padding=1)
                mask = (conv_result == structure.sum().float()).float()
            The following implementaion would work for mask that span [0,1] (bilinear interpolation)
            """
            eroded_mask = nnf.conv3d(mask, structure, padding=1)
            eroded_mask = eroded_mask - torch.sum(structure) + 1
            eroded_mask = nnf.relu(eroded_mask)

            return eroded_mask
        
        # Apply erosion
        mask_dx = apply_erosion(mask, s_erode_xp)
        mask_dy = apply_erosion(mask, s_erode_yp)
        mask_dz = apply_erosion(mask, s_erode_zp)
        
        mask_dxx = apply_erosion(mask_dx, s_erode_xp)
        mask_dyy = apply_erosion(mask_dy, s_erode_yp)
        mask_dzz = apply_erosion(mask_dz, s_erode_zp)
        
        mask_dxy = apply_erosion(mask, s_erode_xyp)
        mask_dyz = apply_erosion(mask, s_erode_yzp)
        mask_dxz = apply_erosion(mask, s_erode_xzp)
        
        # Crop the edges
        mask_dxx = mask_dxx[...,  :-2,  :,    :  ]
        mask_dyy = mask_dyy[...,  :,    :-2,  :  ]
        mask_dzz = mask_dzz[...,  :,    :,    :-2]
        mask_dxy = mask_dxy[...,  :-1,  :-1,  :  ]
        mask_dyz = mask_dyz[...,  :,    :-1,  :-1]
        mask_dxz = mask_dxz[...,  :-1,  :,    :-1]
        
        return mask_dxx, mask_dyy, mask_dzz, mask_dxy, mask_dyz, mask_dxz


