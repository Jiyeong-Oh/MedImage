"""
SliceWise dataset for MIL-max prostate cancer classification.
Returns (x, label, pid) where x is [N, 3, H, W] — one tensor per slice.
"""
import importlib.util
import os
import sys

import torch
from torch.utils.data import Dataset

# Load parent dataset.py under a unique name to avoid circular import
# when train.py adds this directory to sys.path first.
_parent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset.py')
if 'prostatecls_dataset' not in sys.modules:
    _spec = importlib.util.spec_from_file_location('prostatecls_dataset', _parent_path)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules['prostatecls_dataset'] = _mod
    _spec.loader.exec_module(_mod)
from prostatecls_dataset import (DATA_ROOT, load_labels, load_patient, slice_to_tensor,
                                  augment_volume_tensor, augment_volume_tensor_strong,
                                  augment_volume_tensor_scale)


class SliceWiseDataset(Dataset):
    """
    One patient = one sample: x is [N, 3, H, W].
    Augmentation applied on interleaved [N*3, H, W] so spatial transforms are
    consistent across slices, then reshaped back to [N, 3, H, W].
    """
    def __init__(self, records, augment=False, aug_strong=False, aug_scale=False, aug_t2w_only=False,
                 no_hflip=False, gland_z_center=False, n_slices=32,
                 input_size=224, data_root=DATA_ROOT):
        self.augment        = augment
        self.aug_strong     = aug_strong
        self.aug_scale      = aug_scale
        self.aug_t2w_only   = aug_t2w_only
        self.no_hflip       = no_hflip
        self.gland_z_center = gland_z_center
        self.n_slices     = n_slices
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

        if self.gland_z_center:
            start = vols['cz'] - n // 2
        else:
            start = (D - n) // 2 if D >= n else 0

        slices = []
        for i in range(n):
            z = start + i
            if 0 <= z < D:
                slices.append(slice_to_tensor(vols, z, self.input_size))
            else:
                slices.append(torch.zeros(3, self.input_size, self.input_size))

        # Interleave into [N*3, H, W] for consistent spatial augmentation
        tensor = torch.cat(slices, dim=0)

        if self.aug_strong:
            tensor = augment_volume_tensor_strong(tensor, t2w_int_only=self.aug_t2w_only,
                                                  no_hflip=self.no_hflip)
        elif self.aug_scale:
            tensor = augment_volume_tensor_scale(tensor)
        elif self.augment:
            tensor = augment_volume_tensor(tensor, t2w_int_only=self.aug_t2w_only,
                                           no_hflip=self.no_hflip)

        # Reshape to [N, 3, H, W]
        tensor = tensor.view(n, 3, self.input_size, self.input_size)
        return tensor, label, pid
