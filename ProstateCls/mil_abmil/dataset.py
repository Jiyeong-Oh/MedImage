"""
MIL dataset: returns per-patient slice stack [n_slices, nch, H, W].
Gland bbox crop and gland masking are the same as other methods.
"""
import os
import random
import importlib.util as _ilu
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

_pds_spec = _ilu.spec_from_file_location(
    '_parent_dataset',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset.py'))
_pds = _ilu.module_from_spec(_pds_spec)
_pds_spec.loader.exec_module(_pds)
load_patient_bbox_crop = _pds.load_patient_bbox_crop
load_labels            = _pds.load_labels
load_viz_volumes       = _pds.load_viz_volumes
del _pds_spec, _pds, _ilu


class MILDataset(Dataset):
    """
    Each sample is one patient bag: [n_slices, nch, H, W].
    Channels per slice (nch=2): [T2W×gland, ADC×gland]
    Channels per slice (nch=3): [T2W×gland, ADC×gland, gland]
    Tumor mask (max-projected) is returned for optional attn supervision.
    """
    def __init__(self, records, augment=False, n_slices=32, n_ch_per_slice=2, input_size=224,
                 cs_oversample=1):
        self.augment  = augment
        self.n_slices = n_slices
        self.nch      = n_ch_per_slice
        self.samples  = []
        for pid, label in records:
            vols = load_patient_bbox_crop(pid, n_slices=n_slices, target_size=input_size)
            self.samples.append((pid, label, vols))
        if cs_oversample > 1:
            cs = [(pid, lbl, v) for pid, lbl, v in self.samples if lbl == 1]
            for _ in range(cs_oversample - 1):
                self.samples.extend(cs)
            import random as _r; _r.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, label, vols = self.samples[idx]
        n, nch = self.n_slices, self.nch
        H, W   = vols['t2w'].shape[:2]

        # Build [n_slices, nch, H, W]
        slices = []
        for z in range(n):
            g = vols['gland'][:, :, z]
            if nch == 2:
                arr = np.stack([vols['t2w'][:, :, z] * g,
                                vols['adc'][:, :, z] * g], axis=0)
            else:
                arr = np.stack([vols['t2w'][:, :, z] * g,
                                vols['adc'][:, :, z] * g, g], axis=0)
            slices.append(arr)
        tensor = torch.from_numpy(
            np.stack(slices, axis=0).astype(np.float32))  # [n, nch, H, W]

        # Augment: flatten to [n*nch, H, W], apply spatial transform, reshape back
        if self.augment:
            flat = tensor.view(n * nch, H, W)
            flat = self._augment(flat)
            tensor = flat.view(n, nch, H, W)

        tumor_vol = vols.get('tumor', np.zeros_like(vols['gland']))
        tumor_2d  = torch.from_numpy(
            tumor_vol.max(axis=2).astype(np.float32)).unsqueeze(0)  # [1, H, W]

        # Per-slice tumor presence [n_slices] — for attention supervision
        tumor_slices = torch.from_numpy(
            (tumor_vol.max(axis=(0, 1)) > 0).astype(np.float32))  # [n_slices]

        return tensor, tumor_2d, tumor_slices, label, pid

    @staticmethod
    def _augment(t):
        if random.random() < 0.5:
            t = TF.hflip(t)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            t = TF.rotate(t, angle)
        if random.random() < 0.4:
            h, w = t.shape[-2], t.shape[-1]
            t = TF.affine(t, angle=0,
                          translate=[int(random.uniform(-0.1, 0.1) * w),
                                     int(random.uniform(-0.1, 0.1) * h)],
                          scale=1.0, shear=0)
        return t
