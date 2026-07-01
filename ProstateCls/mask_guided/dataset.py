"""
Mask-guided dataset for PI-CAI prostate cancer classification.

Three key differences from the base dataset:
1. ROI crop: gland bounding box (+ margin) → only prostate region fed to model
2. Modality-specific normalization: T2W and ADC normalized separately using
   only pixels within the gland mask (prostate tissue range, not background)
3. Returns (x_96ch, mask_32ch, label, pid) so the model can use the mask
   as an explicit spatial context signal
"""
import csv
import os
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

sys_path_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

DATA_ROOT = '/N/slate/ohjiye/PI-CAI/PI-CAI_reg_processed_filtered'
CSV_PATH  = '/N/slate/ohjiye/PI-CAI/PI-CAI_reg_processed_filtered.csv'

SKIP_PATIENTS = {'10188_1000191', '10448_1000456', '10559_1000571', '10593_1000607'}

TARGET_SPACING_MM = 0.5  # normalize all patients to 0.5 mm/px → 224px = 112 mm FOV


def _resample_and_crop(tensor_chw, src_spacing, target_size=224):
    """Resample [C,H,W] from src_spacing to TARGET_SPACING_MM mm/px, then center-crop or zero-pad to target_size."""
    _, h, w = tensor_chw.shape
    new_h = int(round(h * src_spacing / TARGET_SPACING_MM))
    new_w = int(round(w * src_spacing / TARGET_SPACING_MM))
    if new_h != h or new_w != w:
        tensor_chw = TF.resize(tensor_chw, [new_h, new_w], antialias=True)
    pad_h = max(0, target_size - new_h)
    pad_w = max(0, target_size - new_w)
    if pad_h > 0 or pad_w > 0:
        tensor_chw = TF.pad(tensor_chw, [pad_w // 2, pad_h // 2,
                                          pad_w - pad_w // 2, pad_h - pad_h // 2])
    return TF.center_crop(tensor_chw, [target_size, target_size])


def load_labels(csv_path=CSV_PATH, data_root=DATA_ROOT):
    available = set(os.listdir(data_root)) - SKIP_PATIENTS
    records = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pid = row['patientID']
            if pid in available:
                records.append((pid, 1 if row['case_csPCa'] == 'YES' else 0))
    return records


def get_roi_bbox(gland_3d, margin=16):
    """2D bounding box from max-projection of gland across all slices."""
    proj = gland_3d.max(axis=2)          # [H, W]
    ys, xs = np.where(proj > 0.5)
    H, W = gland_3d.shape[:2]
    if len(ys) == 0:
        return 0, H - 1, 0, W - 1
    return (max(0, int(ys.min()) - margin), min(H - 1, int(ys.max()) + margin),
            max(0, int(xs.min()) - margin), min(W - 1, int(xs.max()) + margin))


def percentile_norm_in_mask(arr, mask, low=1, high=99):
    """Percentile normalization computed only within the gland mask."""
    vals = arr[mask > 0.5]
    if len(vals) < 10:
        vals = arr.ravel()
    lo, hi = np.percentile(vals, low), np.percentile(vals, high)
    arr = np.clip(arr, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32)


def load_patient(pid, data_root=DATA_ROOT):
    """
    Load T2W, ADC, gland mask.
    - Gland mask used as normalization reference (modality-specific)
    - Volumes cropped to gland bounding box + 16px margin
    Returns dict with cropped volumes and in-plane spacing.
    """
    folder  = os.path.join(data_root, pid)
    t2w_img = nib.load(os.path.join(folder, f'{pid}_t2w.nii.gz'))
    t2w     = t2w_img.get_fdata().astype(np.float32)
    spacing = float(t2w_img.header.get_zooms()[0])
    adc   = nib.load(os.path.join(folder, f'{pid}_adc_reg.nii.gz')).get_fdata().astype(np.float32)
    gland = nib.load(os.path.join(folder, f'{pid}_gland.nii.gz')).get_fdata().astype(np.float32)
    gland = (gland > 0.5).astype(np.float32)

    # Modality-specific normalization within mask
    t2w = percentile_norm_in_mask(t2w, gland)
    adc = percentile_norm_in_mask(adc, gland)

    # ROI crop to gland bounding box
    y0, y1, x0, x1 = get_roi_bbox(gland, margin=16)
    t2w   = t2w[y0:y1+1, x0:x1+1, :]
    adc   = adc[y0:y1+1, x0:x1+1, :]
    gland = gland[y0:y1+1, x0:x1+1, :]

    return {'t2w': t2w, 'adc': adc, 'gland': gland}


def slice_to_tensor(vols, z, target_size=224):
    arr = np.stack([vols['t2w'][:, :, z],
                    vols['adc'][:, :, z],
                    vols['gland'][:, :, z]], axis=0)
    tensor = torch.from_numpy(arr)
    return TF.resize(tensor, [target_size, target_size], antialias=True)


def _apply_intensity_aug(tensor, t2w_idx, adc_idx, noise_max, gamma_range, scale_range, shift_max, prob):
    if random.random() > prob:
        std = random.uniform(0.01, noise_max)
        for idx in t2w_idx + adc_idx:
            tensor[idx] = (tensor[idx] + torch.randn_like(tensor[idx]) * std).clamp(0, 1)
    if random.random() > prob:
        for indices in [t2w_idx, adc_idx]:
            gamma = random.uniform(gamma_range[0], gamma_range[1])
            for idx in indices:
                tensor[idx] = tensor[idx].pow(gamma)
    if random.random() > prob:
        for indices in [t2w_idx, adc_idx]:
            sc = random.uniform(scale_range[0], scale_range[1])
            sh = random.uniform(-shift_max, shift_max)
            for idx in indices:
                tensor[idx] = (tensor[idx] * sc + sh).clamp(0, 1)
    return tensor


def augment_volume_tensor(tensor):
    n_ch    = tensor.shape[0]
    t2w_idx = list(range(0, n_ch, 3))
    adc_idx = list(range(1, n_ch, 3))
    if random.random() > 0.5:
        tensor = TF.hflip(tensor)
    if random.random() > 0.5:
        tensor = TF.vflip(tensor)
    tensor = TF.rotate(tensor, random.uniform(-15, 15))
    if random.random() > 0.4:
        h, w = tensor.shape[-2], tensor.shape[-1]
        tensor = TF.affine(tensor, angle=0,
                           translate=[int(random.uniform(-0.08, 0.08)*w),
                                      int(random.uniform(-0.08, 0.08)*h)],
                           scale=1.0, shear=0)
    if random.random() > 0.5:
        tensor = TF.affine(tensor, angle=0, translate=[0, 0], scale=1.0,
                           shear=random.uniform(-6, 6))
    return _apply_intensity_aug(tensor, t2w_idx, adc_idx,
                                noise_max=0.05, gamma_range=(0.85, 1.25),
                                scale_range=(0.90, 1.10), shift_max=0.05, prob=0.4)


class MaskGuidedDataset(Dataset):
    """
    Returns (x, mask, label, pid) where:
      x    [96, 224, 224] — ROI-cropped, mask-normalized T2W/ADC/mask channels
      mask [32, 224, 224] — gland mask per slice (x[2::3]), same spatial augment
    """
    def __init__(self, records, augment=False, n_slices=32,
                 input_size=224, data_root=DATA_ROOT):
        self.augment    = augment
        self.n_slices   = n_slices
        self.input_size = input_size
        self.samples    = []
        for pid, label in records:
            vols = load_patient(pid, data_root)
            self.samples.append((pid, label, vols))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, label, vols = self.samples[idx]
        D = vols['t2w'].shape[2]
        n = self.n_slices

        if D >= n:
            start = (D - n) // 2
            z_range = range(start, start + n)
        else:
            z_range = range(D)

        slices = []
        for z in z_range:
            slices.append(slice_to_tensor(vols, z, self.input_size))
        while len(slices) < n:
            slices.append(torch.zeros(3, self.input_size, self.input_size))

        tensor = torch.cat(slices, dim=0)  # [96, H, W]

        if self.augment:
            tensor = augment_volume_tensor(tensor)

        mask = tensor[2::3]  # [32, H, W] — same spatial transform as tensor
        return tensor, mask, label, pid
