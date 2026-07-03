"""
Dataset for seg_cls multi-task method.
Returns (tensor [96,H,W], gland_2d [1,H,W], tumor_2d [1,H,W], label, pid)
  - tensor:    32 slices × 3 ch interleaved (t2w/adc/gland), same as depth-as-channel
  - gland_2d:  2D max-projection of gland mask across 32 slices → [1, 224, 224]
  - tumor_2d:  2D max-projection of tumor mask across 32 slices → [1, 224, 224]
               (all-zero for ciPCa or patients with missing/empty tumor mask)
  - has_tumor: bool flag — whether tumor_2d is non-empty (used to gate seg loss)
"""
import os
import random
import sys

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset import (DATA_ROOT, _fov_crop_and_resize, _resample_inplane,
                     _gland_centroid_2d, _gland_z_centroid,
                     percentile_norm_in_mask, TARGET_SPACING,
                     augment_volume_tensor_scale)


def load_patient_with_tumor(pid, data_root=DATA_ROOT):
    """Same as dataset.load_patient but also loads tumor mask."""
    folder  = os.path.join(data_root, pid)
    t2w_img = nib.load(os.path.join(folder, f'{pid}_t2w.nii.gz'))
    t2w     = t2w_img.get_fdata().astype(np.float32)
    spacing = float(t2w_img.header.get_zooms()[0])
    adc     = nib.load(os.path.join(folder, f'{pid}_adc_reg.nii.gz')).get_fdata().astype(np.float32)
    gland   = nib.load(os.path.join(folder, f'{pid}_gland.nii.gz')).get_fdata().astype(np.float32)
    gland   = (gland > 0.5).astype(np.float32)

    tumor_path = os.path.join(folder, f'{pid}_tumor.nii.gz')
    if os.path.exists(tumor_path):
        tumor = nib.load(tumor_path).get_fdata().astype(np.float32)
        tumor = (tumor > 0.5).astype(np.float32)
    else:
        tumor = np.zeros_like(gland)

    # resample to uniform in-plane spacing
    t2w   = _resample_inplane(t2w,   spacing)
    adc   = _resample_inplane(adc,   spacing)
    gland = _resample_inplane(gland, spacing)
    gland = (gland > 0.5).astype(np.float32)
    tumor = _resample_inplane(tumor, spacing)
    tumor = (tumor > 0.5).astype(np.float32)

    t2w = percentile_norm_in_mask(t2w, gland)
    adc = percentile_norm_in_mask(adc, gland)

    cy, cx = _gland_centroid_2d(gland)
    cz     = _gland_z_centroid(gland)
    return {'t2w': t2w, 'adc': adc, 'gland': gland, 'tumor': tumor,
            'spacing': spacing, 'cy': cy, 'cx': cx, 'cz': cz}


def slice_to_tensor_with_masks(vols, z, target_size=224):
    """Returns (slice_3ch [3,H,W], gland_1ch [1,H,W], tumor_1ch [1,H,W])."""
    arr = np.stack([vols['t2w'][:, :, z],
                    vols['adc'][:, :, z],
                    vols['gland'][:, :, z]], axis=0)
    full = np.concatenate([arr,
                           vols['gland'][:, :, z:z+1].transpose(2, 0, 1),
                           vols['tumor'][:, :, z:z+1].transpose(2, 0, 1)], axis=0)  # [5, H, W]
    full_t = _fov_crop_and_resize(torch.from_numpy(full.astype(np.float32)),
                                  vols['cy'], vols['cx'], vols['spacing'], target_size)
    return full_t[:3], full_t[3:4], full_t[4:5]


class SegClsDataset(Dataset):
    def __init__(self, records, augment=False, aug_scale=False,
                 n_slices=32, input_size=224, data_root=DATA_ROOT):
        self.augment    = augment
        self.aug_scale  = aug_scale
        self.n_slices   = n_slices
        self.input_size = input_size
        self.samples    = []
        for pid, label in records:
            vols = load_patient_with_tumor(pid, data_root)
            has_tumor = bool(vols['tumor'].sum() > 0)
            self.samples.append((pid, label, vols, has_tumor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, label, vols, has_tumor = self.samples[idx]
        D = vols['t2w'].shape[2]
        n = self.n_slices
        start = (D - n) // 2 if D >= n else 0

        img_slices, gland_slices, tumor_slices = [], [], []
        for i in range(n):
            z = start + i
            if 0 <= z < D:
                img_s, gl_s, tu_s = slice_to_tensor_with_masks(vols, z, self.input_size)
            else:
                img_s = torch.zeros(3, self.input_size, self.input_size)
                gl_s  = torch.zeros(1, self.input_size, self.input_size)
                tu_s  = torch.zeros(1, self.input_size, self.input_size)
            img_slices.append(img_s)
            gland_slices.append(gl_s)
            tumor_slices.append(tu_s)

        tensor    = torch.cat(img_slices, dim=0)   # [96, H, W]
        gland_2d  = torch.stack(gland_slices, dim=0).squeeze(1).max(dim=0).values.unsqueeze(0)  # [1, H, W]
        tumor_2d  = torch.stack(tumor_slices, dim=0).squeeze(1).max(dim=0).values.unsqueeze(0)  # [1, H, W]

        if self.aug_scale:
            # apply same spatial transform to all maps together
            combined = torch.cat([tensor, gland_2d, tumor_2d], dim=0)
            combined = augment_volume_tensor_scale(combined)
            tensor   = combined[:96]
            gland_2d = combined[96:97]
            tumor_2d = combined[97:98]
            # re-binarize masks after interpolation
            gland_2d = (gland_2d > 0.5).float()
            tumor_2d = (tumor_2d > 0.5).float()
        elif self.augment:
            combined = torch.cat([tensor, gland_2d, tumor_2d], dim=0)
            from dataset import augment_volume_tensor
            combined = augment_volume_tensor(combined, t2w_int_only=False, no_hflip=True)
            tensor   = combined[:96]
            gland_2d = combined[96:97]
            tumor_2d = combined[97:98]
            gland_2d = (gland_2d > 0.5).float()
            tumor_2d = (tumor_2d > 0.5).float()

        return tensor, gland_2d, tumor_2d, torch.tensor(has_tumor, dtype=torch.bool), label, pid
