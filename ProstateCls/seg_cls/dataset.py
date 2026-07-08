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
from scipy.ndimage import zoom as ndimage_zoom
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset import (DATA_ROOT, _fov_crop_and_resize, _resample_inplane,
                     _gland_centroid_2d, _gland_z_centroid,
                     _gland_bbox_3d, BBOX_MARGIN_MM,
                     percentile_norm_in_mask, TARGET_SPACING,
                     augment_volume_tensor_scale)


def load_patient_with_tumor_bbox_crop(pid, n_slices=32, target_size=224, data_root=DATA_ROOT):
    """3D-bbox-crop variant that also loads tumor mask."""
    folder  = os.path.join(data_root, pid)
    t2w_img = nib.load(os.path.join(folder, f'{pid}_t2w.nii.gz'))
    t2w     = t2w_img.get_fdata().astype(np.float32)
    zooms   = t2w_img.header.get_zooms()
    sp_xy   = float(zooms[0])
    sp_z    = float(zooms[2])

    adc   = nib.load(os.path.join(folder, f'{pid}_adc_reg.nii.gz')).get_fdata().astype(np.float32)
    gland = nib.load(os.path.join(folder, f'{pid}_gland.nii.gz')).get_fdata().astype(np.float32)
    gland = (gland > 0.5).astype(np.float32)

    tumor_path = os.path.join(folder, f'{pid}_tumor.nii.gz')
    if os.path.exists(tumor_path):
        tumor = nib.load(tumor_path).get_fdata().astype(np.float32)
        tumor = (tumor > 0.5).astype(np.float32)
    else:
        tumor = np.zeros_like(gland)

    t2w   = _resample_inplane(t2w,   sp_xy)
    adc   = _resample_inplane(adc,   sp_xy)
    gland = _resample_inplane(gland, sp_xy)
    gland = (gland > 0.5).astype(np.float32)
    tumor = _resample_inplane(tumor, sp_xy)
    tumor = (tumor > 0.5).astype(np.float32)

    t2w = percentile_norm_in_mask(t2w, gland)
    adc = percentile_norm_in_mask(adc, gland)

    margin_px_xy = max(2, int(round(BBOX_MARGIN_MM / TARGET_SPACING)))
    margin_px_z  = max(1, int(round(BBOX_MARGIN_MM / sp_z)))
    y0, y1, x0, x1, z0, z1 = _gland_bbox_3d(gland, margin_px_xy, margin_px_z)

    t2w_c   = t2w[y0:y1, x0:x1, z0:z1]
    adc_c   = adc[y0:y1, x0:x1, z0:z1]
    gland_c = gland[y0:y1, x0:x1, z0:z1]
    tumor_c = tumor[y0:y1, x0:x1, z0:z1]

    Hc, Wc, Dc = t2w_c.shape
    sh, sw, sd = target_size / Hc, target_size / Wc, n_slices / Dc
    t2w_rs   = ndimage_zoom(t2w_c,   (sh, sw, sd), order=1).astype(np.float32)
    adc_rs   = ndimage_zoom(adc_c,   (sh, sw, sd), order=1).astype(np.float32)
    gland_rs = ndimage_zoom(gland_c, (sh, sw, sd), order=1).astype(np.float32)
    gland_rs = (gland_rs > 0.5).astype(np.float32)
    tumor_rs = ndimage_zoom(tumor_c, (sh, sw, sd), order=1).astype(np.float32)
    tumor_rs = (tumor_rs > 0.5).astype(np.float32)

    return {'t2w': t2w_rs, 'adc': adc_rs, 'gland': gland_rs, 'tumor': tumor_rs, 'bbox_crop': True}


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


def slice_to_tensor_with_masks(vols, z, target_size=224, soft_factor=0.0, n_ch_per_slice=3):
    """Returns (img_ch [n_ch,H,W], gland_1ch [1,H,W], tumor_1ch [1,H,W])."""
    gland_sl = vols['gland'][:, :, z]
    eff_mask = gland_sl + (1.0 - gland_sl) * soft_factor
    if n_ch_per_slice == 2:
        arr = np.stack([vols['t2w'][:, :, z] * eff_mask,
                        vols['adc'][:, :, z] * eff_mask], axis=0)
    else:
        arr = np.stack([vols['t2w'][:, :, z] * eff_mask,
                        vols['adc'][:, :, z] * eff_mask,
                        gland_sl], axis=0)
    full = np.concatenate([arr,
                           vols['gland'][:, :, z:z+1].transpose(2, 0, 1),
                           vols['tumor'][:, :, z:z+1].transpose(2, 0, 1)], axis=0)  # [n_ch+2, H, W]
    full_t = _fov_crop_and_resize(torch.from_numpy(full.astype(np.float32)),
                                  vols['cy'], vols['cx'], vols['spacing'], target_size)
    return full_t[:n_ch_per_slice], full_t[n_ch_per_slice:n_ch_per_slice+1], full_t[n_ch_per_slice+1:n_ch_per_slice+2]


class SegClsDataset(Dataset):
    def __init__(self, records, augment=False, aug_scale=False,
                 n_slices=32, input_size=224, soft_mask_factor=0.0, n_ch_per_slice=3,
                 bbox_crop=True, intensity_aug=False, cs_oversample=1, data_root=DATA_ROOT):
        self.augment          = augment
        self.aug_scale        = aug_scale
        self.n_slices         = n_slices
        self.input_size       = input_size
        self.soft_mask_factor = soft_mask_factor
        self.n_ch_per_slice   = n_ch_per_slice
        self.bbox_crop        = bbox_crop
        self.intensity_aug    = intensity_aug
        self.samples    = []
        for pid, label in records:
            if bbox_crop:
                vols = load_patient_with_tumor_bbox_crop(pid, n_slices=n_slices,
                                                          target_size=input_size, data_root=data_root)
            else:
                vols = load_patient_with_tumor(pid, data_root)
            has_tumor = bool(vols['tumor'].sum() > 0)
            self.samples.append((pid, label, vols, has_tumor))
        if cs_oversample > 1:
            cs = [s for s in self.samples if s[1] == 1]
            for _ in range(cs_oversample - 1):
                self.samples.extend(cs)
            import random as _r; _r.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, label, vols, has_tumor = self.samples[idx]
        n   = self.n_slices
        ncp = self.n_ch_per_slice
        sf  = self.soft_mask_factor

        if self.bbox_crop:
            img_slices, gland_slices, tumor_slices = [], [], []
            for z in range(n):
                g   = vols['gland'][:, :, z]
                eff = g + (1.0 - g) * sf
                if ncp == 2:
                    arr = np.stack([vols['t2w'][:, :, z] * eff,
                                    vols['adc'][:, :, z] * eff], axis=0)
                else:
                    arr = np.stack([vols['t2w'][:, :, z] * eff,
                                    vols['adc'][:, :, z] * eff, g], axis=0)
                img_slices.append(torch.from_numpy(arr.astype(np.float32)))
                gland_slices.append(torch.from_numpy(g[np.newaxis].astype(np.float32)))
                tumor_slices.append(torch.from_numpy(vols['tumor'][:, :, z][np.newaxis].astype(np.float32)))
        else:
            D = vols['t2w'].shape[2]
            start = (D - n) // 2 if D >= n else 0
            img_slices, gland_slices, tumor_slices = [], [], []
            for i in range(n):
                z = start + i
                if 0 <= z < D:
                    img_s, gl_s, tu_s = slice_to_tensor_with_masks(vols, z, self.input_size, sf, ncp)
                else:
                    img_s = torch.zeros(ncp, self.input_size, self.input_size)
                    gl_s  = torch.zeros(1, self.input_size, self.input_size)
                    tu_s  = torch.zeros(1, self.input_size, self.input_size)
                img_slices.append(img_s)
                gland_slices.append(gl_s)
                tumor_slices.append(tu_s)

        n_img_ch  = n * ncp
        tensor    = torch.cat(img_slices, dim=0)
        gland_2d  = torch.stack(gland_slices, dim=0).squeeze(1).max(dim=0).values.unsqueeze(0)
        tumor_2d  = torch.stack(tumor_slices, dim=0).squeeze(1).max(dim=0).values.unsqueeze(0)

        if self.aug_scale:
            combined = torch.cat([tensor, gland_2d, tumor_2d], dim=0)
            combined = augment_volume_tensor_scale(combined)
            tensor   = combined[:n_img_ch]
            gland_2d = combined[n_img_ch:n_img_ch+1]
            tumor_2d = combined[n_img_ch+1:n_img_ch+2]
            gland_2d = (gland_2d > 0.5).float()
            tumor_2d = (tumor_2d > 0.5).float()
        elif self.augment:
            combined = torch.cat([tensor, gland_2d, tumor_2d], dim=0)
            from dataset import augment_volume_tensor
            combined = augment_volume_tensor(combined, t2w_int_only=False, no_hflip=True,
                                             intensity_aug=self.intensity_aug)
            tensor   = combined[:n_img_ch]
            gland_2d = combined[n_img_ch:n_img_ch+1]
            tumor_2d = combined[n_img_ch+1:n_img_ch+2]
            gland_2d = (gland_2d > 0.5).float()
            tumor_2d = (tumor_2d > 0.5).float()

        return tensor, gland_2d, tumor_2d, torch.tensor(has_tumor, dtype=torch.bool), label, pid
