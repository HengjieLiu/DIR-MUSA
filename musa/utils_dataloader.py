"""
General description:
    Functions for data loading and dataset loader for training.

Function/Class list:
    read_file_list
    load_nifti
    save_nifti
    load_vol
    myDataset_trn
    myDataset_val

"""


import os
import itertools
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import nibabel as nib

### local imports
from .utils_basics import numpy_overview


def read_file_list(filename: str, prefix: str = None, suffix: str = None) -> List[str]:
    '''
    Reads a list of files from a line-seperated text file.
    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    Ref:
        https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/py/utils.py: def read_file_list
    Example of trn_list.txt:
        0001
        0002
        0003
        ...
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist


def load_nifti(filename: str, ret_affine: bool = False) -> np.ndarray:
    """
    Load a nifti (nii or nii.gz) file as numpy array
    Parameters:
        filename: filename to load
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    Ref:
        https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/py/utils.py: def read_file_list
    """

    if not os.path.isfile(filename):
        raise ValueError(f"Nifti load error: '{filename}' is not a file.")
    if filename.endswith(('.nii', '.nii.gz')):
        img = nib.load(filename)
        vol = np.array(img.dataobj)
        affine = img.affine
    else:
        raise ValueError(f"Nifti load error: '{filename}' is not a nii or nii.gz file.")

    return (vol, affine) if ret_affine else vol


def save_nifti(array: np.ndarray, filename: str, affine: np.ndarray = None) -> None:
    """
    Save a numpy array to a Nifti (.nii or .nii.gz) file.
    Parameters:
        array: The array to save.
        filename: Output file path.
        affine: Optional affine matrix for spatial orientation.
    Ref:
        https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/py/utils.py: def read_file_list

    Notes:
        When saving with nibabel (see https://nipy.org/nibabel/image_orientation.html)
            the default orientation (affine matrix of identity) is neurology convention: L->R, P->A, I->S (RAS+)
            but the default orientation we use is radiology convention:                  R->L, A->P, I->S (LPS+)
            so the affine_default is used to account for this difference (and viewing with ITK-SNAP)
    """

    affine_default = np.array([[-1.,  0.,  0.,  0.],
                               [ 0., -1.,  0.,  0.],
                               [ 0.,  0., +1.,  0.],
                               [ 0.,  0.,  0., +1.]],
                              dtype = np.float32)

    if filename.endswith(('.nii', '.nii.gz')):
        if affine is None:
            affine = affine_default
        nib.save(nib.Nifti1Image(array, affine), filename)
    else:
        raise ValueError(f'Nifti save error: unknown filetype for {filename}')


def load_vol(fname: str, path: str, add_channel: bool = True, DEBUG: bool = False):
    '''
    Load 3D vol from .nii.gz/.nii or .npy files
    Add dummy channel dimension so that output is: (1, H, W, D)
    '''
    
    if fname.endswith('.nii.gz') or fname.endswith('.nii'):
        path_vol = os.path.join(path, fname)
        vol = nib.load(path_vol).get_fdata()
    elif fname.endswith('.npy'):
        path_vol = os.path.join(path, fname)
        vol = np.load(path_vol)
    elif fname.endswith('.npz'):
        path_vol = os.path.join(path, fname)
        vol = np.load(path_vol)['arr']
    
    else:
        raise ValueError(f"Unsupported file type for {fname}. Supported types are: .nii.gz, .nii, .npy, .npz")

    # Add dummy channel dimension
    if add_channel:
        vol = vol[np.newaxis, ...]

    if DEBUG:
        print(f'DEBUG in load_vol, path_vol: {path_vol}')
        numpy_overview(vol, 'vol')
    
    return vol


class myDataset_trn(Dataset):
    """
    A unified Dataset class to load volumes and corresponding segmentations (optional).
    Supports multiple loss types: 
        - loss 1 (L_sim+L_reg):        volumes only
        - loss 2 (L_sim+L_reg+L_dice): volumes + segmentation for dice loss (multi-class)
        - loss 3 (L_sim+L_reg+L_musa): volumes + segmentation for musaloss (binary)

    The data/loss type is determined by the provided path arguments.
    """
    def __init__(self, trn_list, path_vol, path_seg=None, path_seg_musaloss=None, ftype='.npy', DEBUG=False):

        self.trn_list = trn_list
        self.path_vol = path_vol
        self.path_seg = path_seg
        self.path_seg_musaloss = path_seg_musaloss
        self.ftype = ftype
        self.DEBUG = DEBUG

        # Ensure that both path_seg and path_seg_musaloss are not provided at the same time
        assert not (self.path_seg and self.path_seg_musaloss), \
            "path_seg and path_seg_musaloss cannot be provided simultaneously."

        # Create index pairs (with repeat)
        self.index_pair = list(itertools.product(self.trn_list, repeat=2))
        
        if self.DEBUG:
            print(f'Creating {len(self.index_pair)} pairs from {len(self.trn_list)} samples')

    def __len__(self):
        return len(self.index_pair)
    
    def load_pairs(self, idx, path):
        """ Helper function to load the moving and fixed volumes/segmentations. """
        moving = load_vol(self.index_pair[idx][0] + self.ftype, path, DEBUG=self.DEBUG)
        fixed  = load_vol(self.index_pair[idx][1] + self.ftype, path, DEBUG=self.DEBUG)
        return moving, fixed

    def __getitem__(self, idx):
        moving, fixed = self.load_pairs(idx, self.path_vol)
        moving = torch.from_numpy(moving).float()
        fixed  = torch.from_numpy(fixed).float()

        if self.path_seg is None and self.path_seg_musaloss is None:
            # For loss 1: Only volumes
            return moving, fixed

        elif self.path_seg:
            # For loss 2: Load segmentation (seg)
            moving_seg, fixed_seg = self.load_pairs(idx, self.path_seg)
            moving_seg = torch.from_numpy(moving_seg).long()
            fixed_seg  = torch.from_numpy(fixed_seg).long()

            return moving, fixed, moving_seg, fixed_seg

        elif self.path_seg_musaloss:
            # For loss 3: Load segmentation (seg_musaloss)
            moving_seg_musaloss, fixed_seg_musaloss = self.load_pairs(idx, self.path_seg_musaloss)
            moving_seg_musaloss = torch.from_numpy((moving_seg_musaloss > 0).astype(np.float32))
            fixed_seg_musaloss  = torch.from_numpy((fixed_seg_musaloss > 0).astype(np.float32))

            return moving, fixed, moving_seg_musaloss, fixed_seg_musaloss


class myDataset_val(Dataset):
    """
    A unified Dataset class to load volumes and corresponding segmentations for validation.
    Despite the loss, the validation always loads images with both segmentations (seg_o and seg_b)
    
    The validation pairs are created by splitting the val_list in half and pairing
    corresponding elements from each half.
    """
    def __init__(self, val_list, path_vol, path_seg_o, path_seg_b, ftype='.npy', DEBUG=False):
        
        self.val_list = val_list
        self.path_vol = path_vol
        self.path_seg_o = path_seg_o
        self.path_seg_b = path_seg_b
        self.ftype = ftype
        self.DEBUG = DEBUG
        
        # Create index pairs by splitting val_list in half
        n = len(self.val_list) // 2  # Half the length of the list
        self.index_pair = [(self.val_list[i], self.val_list[i + n]) for i in range(n)]
        
        if self.DEBUG:
            print(f'Creating {len(self.index_pair)} pairs from {len(self.val_list)} samples')
            print('len(self.index_pair) [expecting 25]: ', len(self.index_pair))

    def __len__(self):
        return len(self.index_pair)
    
    def load_pairs(self, idx, path):
        """ Helper function to load the moving and fixed volumes/segmentations. """
        moving = load_vol(self.index_pair[idx][0] + self.ftype, path, DEBUG=self.DEBUG)
        fixed  = load_vol(self.index_pair[idx][1] + self.ftype, path, DEBUG=self.DEBUG)
        return moving, fixed

    def __getitem__(self, idx):
        # Load volumes
        moving, fixed = self.load_pairs(idx, self.path_vol)
        
        # Load original segmentations
        moving_seg_o, fixed_seg_o = self.load_pairs(idx, self.path_seg_o)
        
        # Load binary segmentations
        moving_seg_b, fixed_seg_b = self.load_pairs(idx, self.path_seg_b)

        if self.DEBUG:
            print('\nDEBUG in myDataset_val loading pair: ', self.index_pair[idx][0], self.index_pair[idx][1])
        
        # Convert to tensors
        moving = torch.from_numpy(moving).float()
        fixed  = torch.from_numpy(fixed).float()
        moving_seg_o = torch.from_numpy(moving_seg_o).long()
        fixed_seg_o  = torch.from_numpy(fixed_seg_o).long()
        moving_seg_b = torch.from_numpy(moving_seg_b).long()
        fixed_seg_b  = torch.from_numpy(fixed_seg_b).long()
        
        return moving, fixed, moving_seg_o, fixed_seg_o, moving_seg_b, fixed_seg_b

