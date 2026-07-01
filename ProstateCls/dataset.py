"""
PI-CAI prostate cancer classification dataset.
Input: T2W + ADC + gland mask → depth-as-channel [n_slices*3, H, W]
Label: csPCa (1) vs ciPCa (0)

Training:  PatientVolumeDataset — 환자 1명 = 1샘플, 전체 볼륨을 채널로 스택
Val/Test:  동일하게 DataLoader 사용
"""
import csv
import os
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


DATA_ROOT = '/N/slate/ohjiye/PI-CAI/PI-CAI_reg_processed_filtered'
CSV_PATH  = '/N/slate/ohjiye/PI-CAI/PI-CAI_reg_processed_filtered.csv'

SKIP_PATIENTS = {'10188_1000191', '10448_1000456', '10559_1000571',  # adc extraction error
                 '10593_1000607'}  # missing t2w.nii.gz


def load_labels(csv_path=CSV_PATH, data_root=DATA_ROOT):
    """Returns list of (patient_id, label) for patients with imaging data."""
    available = set(os.listdir(data_root)) - SKIP_PATIENTS
    records = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pid = row['patientID']
            if pid in available:
                records.append((pid, 1 if row['case_csPCa'] == 'YES' else 0))
    return records


def percentile_norm(arr, low=1, high=99):
    lo, hi = np.percentile(arr, low), np.percentile(arr, high)
    arr = np.clip(arr, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32)


def load_patient(pid, data_root=DATA_ROOT):
    """Load T2W, ADC, gland volumes → each [H, W, D]."""
    folder = os.path.join(data_root, pid)
    t2w   = nib.load(os.path.join(folder, f'{pid}_t2w.nii.gz')).get_fdata().astype(np.float32)
    adc   = nib.load(os.path.join(folder, f'{pid}_adc_reg.nii.gz')).get_fdata().astype(np.float32)
    gland = nib.load(os.path.join(folder, f'{pid}_gland.nii.gz')).get_fdata().astype(np.float32)
    t2w   = percentile_norm(t2w)
    adc   = percentile_norm(adc)
    gland = (gland > 0.5).astype(np.float32)
    return {'t2w': t2w, 'adc': adc, 'gland': gland}


def slice_to_tensor(vols, z, target_size=224):
    """Extract slice z → [3, H, W] tensor."""
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
    """Standard augmentation for [n_slices*3, H, W] depth-as-channel volume."""
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
        tensor = TF.affine(tensor, angle=0, translate=[0,0], scale=1.0,
                           shear=random.uniform(-6, 6))

    return _apply_intensity_aug(tensor, t2w_idx, adc_idx,
                                noise_max=0.05, gamma_range=(0.85, 1.25),
                                scale_range=(0.90, 1.10), shift_max=0.05, prob=0.4)


def augment_volume_tensor_strong(tensor):
    """Stronger augmentation — larger spatial/intensity ranges, higher probabilities."""
    n_ch    = tensor.shape[0]
    t2w_idx = list(range(0, n_ch, 3))
    adc_idx = list(range(1, n_ch, 3))

    if random.random() > 0.5:
        tensor = TF.hflip(tensor)
    if random.random() > 0.5:
        tensor = TF.vflip(tensor)

    tensor = TF.rotate(tensor, random.uniform(-25, 25))

    if random.random() > 0.3:
        h, w = tensor.shape[-2], tensor.shape[-1]
        tensor = TF.affine(tensor, angle=0,
                           translate=[int(random.uniform(-0.12, 0.12)*w),
                                      int(random.uniform(-0.12, 0.12)*h)],
                           scale=1.0, shear=0)

    if random.random() > 0.4:
        tensor = TF.affine(tensor, angle=0, translate=[0,0], scale=1.0,
                           shear=random.uniform(-12, 12))

    return _apply_intensity_aug(tensor, t2w_idx, adc_idx,
                                noise_max=0.08, gamma_range=(0.75, 1.40),
                                scale_range=(0.85, 1.15), shift_max=0.08, prob=0.3)


class PatientVolumeDataset(Dataset):
    """
    Depth-as-channel: 환자 1명 = 1샘플
    모든 슬라이스를 채널로 스택 → [n_slices*3, H, W]
    D < n_slices이면 zero-pad, D > n_slices이면 center crop
    """
    def __init__(self, records, augment=False, aug_strong=False, n_slices=32,
                 input_size=224, data_root=DATA_ROOT):
        self.augment     = augment
        self.aug_strong  = aug_strong
        self.n_slices    = n_slices
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

        tensor = torch.cat(slices, dim=0)  # [n*3, H, W]

        if self.aug_strong:
            tensor = augment_volume_tensor_strong(tensor)
        elif self.augment:
            tensor = augment_volume_tensor(tensor)

        return tensor, label, pid
