"""
SliceWise dataset for MIL-max prostate cancer classification.
Returns (x, label, pid) where x is [N, 3, H, W] — one tensor per slice.
"""
import os
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset import (DATA_ROOT, load_labels, load_patient, slice_to_tensor,
                     augment_volume_tensor, augment_volume_tensor_strong)


class SliceWiseDataset(Dataset):
    """
    One patient = one sample: x is [N, 3, H, W].
    Augmentation applied on interleaved [N*3, H, W] so spatial transforms are
    consistent across slices, then reshaped back to [N, 3, H, W].
    """
    def __init__(self, records, augment=False, aug_strong=False, n_slices=32,
                 input_size=224, data_root=DATA_ROOT):
        self.augment    = augment
        self.aug_strong = aug_strong
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

        # Interleave into [N*3, H, W] for consistent spatial augmentation
        tensor = torch.cat(slices, dim=0)

        if self.aug_strong:
            tensor = augment_volume_tensor_strong(tensor)
        elif self.augment:
            tensor = augment_volume_tensor(tensor)

        # Reshape to [N, 3, H, W]
        tensor = tensor.view(n, 3, self.input_size, self.input_size)
        return tensor, label, pid
