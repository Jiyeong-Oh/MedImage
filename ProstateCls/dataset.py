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
from scipy.ndimage import zoom as ndimage_zoom
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


DATA_ROOT = '/N/slate/ohjiye/PI-CAI/PI-CAI_reg_processed_filtered'
CSV_PATH  = '/N/slate/ohjiye/PI-CAI/PI-CAI_reg_processed_filtered.csv'

SKIP_PATIENTS = set()  # new dataset is clean — no patients need skipping

FOV_MM           = 80.0   # fixed physical FOV (mm) centered on gland centroid
TARGET_SPACING   = 0.5    # resample all patients to this in-plane spacing (mm)
BBOX_MARGIN_MM   = 15.0   # physical margin added to every face of the 3D gland bbox


def _resample_inplane(arr_hwz, current_spacing):
    """Resample H×W in-plane to TARGET_SPACING; keep Z (slice) axis unchanged."""
    if abs(current_spacing - TARGET_SPACING) < 1e-3:
        return arr_hwz
    scale = current_spacing / TARGET_SPACING
    return ndimage_zoom(arr_hwz, (scale, scale, 1.0), order=1).astype(arr_hwz.dtype)


def _gland_centroid_2d(gland_3d):
    """2D (cy, cx) centroid of gland mask in pixel space, projected across all slices."""
    proj = gland_3d.sum(axis=2)  # [H, W]
    if proj.sum() == 0:
        return gland_3d.shape[0] / 2.0, gland_3d.shape[1] / 2.0
    H, W = proj.shape
    cy = float(np.average(np.arange(H), weights=proj.sum(axis=1)))
    cx = float(np.average(np.arange(W), weights=proj.sum(axis=0)))
    return cy, cx


def _gland_z_centroid(gland_3d):
    """Weighted-mean Z index of gland (per-slice area as weight)."""
    areas = gland_3d.sum(axis=(0, 1))  # [D]
    if areas.sum() == 0:
        return gland_3d.shape[2] // 2
    return int(round(float(np.average(np.arange(len(areas)), weights=areas))))


def _fov_crop_and_resize(tensor_chw, cy, cx, spacing, target_size=224):
    """Crop FOV_MM×FOV_MM physical region centered on (cy, cx), then resize to target_size."""
    _, H, W = tensor_chw.shape
    half = FOV_MM / (2.0 * spacing)

    y0 = int(round(cy - half))
    y1 = int(round(cy + half))
    x0 = int(round(cx - half))
    x1 = int(round(cx + half))

    pad_top    = max(0, -y0)
    pad_bottom = max(0, y1 - H)
    pad_left   = max(0, -x0)
    pad_right  = max(0, x1 - W)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        tensor_chw = TF.pad(tensor_chw, [pad_left, pad_top, pad_right, pad_bottom])

    y0c, y1c = y0 + pad_top, y1 + pad_top
    x0c, x1c = x0 + pad_left, x1 + pad_left

    tensor_chw = tensor_chw[:, y0c:y1c, x0c:x1c]
    return TF.resize(tensor_chw, [target_size, target_size], antialias=True)


def _gland_bbox_3d(gland_hwz, margin_px_xy, margin_px_z):
    """
    3D bounding box of gland mask with margin on every face, clamped to volume bounds.
    Returns (y0, y1, x0, x1, z0, z1). Falls back to full volume if mask is empty.
    """
    H, W, D = gland_hwz.shape
    ys, xs, zs = np.where(gland_hwz > 0.5)
    if len(ys) == 0:
        return 0, H, 0, W, 0, D
    y0 = max(0, int(ys.min()) - margin_px_xy)
    y1 = min(H, int(ys.max()) + margin_px_xy + 1)
    x0 = max(0, int(xs.min()) - margin_px_xy)
    x1 = min(W, int(xs.max()) + margin_px_xy + 1)
    z0 = max(0, int(zs.min()) - margin_px_z)
    z1 = min(D, int(zs.max()) + margin_px_z + 1)
    return y0, y1, x0, x1, z0, z1


def load_patient_bbox_crop(pid, n_slices=32, target_size=224, data_root=DATA_ROOT):
    """
    3D-bbox-crop variant of load_patient.
    Finds the 3D bounding box of the gland mask, crops the volume,
    then resamples the crop to (target_size, target_size, n_slices) via scipy zoom.
    Returned arrays are already at the final spatial resolution — no per-slice
    FOV crop needed in __getitem__.
    """
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
    if os.path.exists(tumor_path) and os.path.getsize(tumor_path) > 0:
        tumor = (nib.load(tumor_path).get_fdata() > 0.5).astype(np.float32)
    else:
        tumor = np.zeros_like(gland)

    # In-plane resample first (matches root load_patient preprocessing)
    t2w   = _resample_inplane(t2w,   sp_xy)
    adc   = _resample_inplane(adc,   sp_xy)
    gland = _resample_inplane(gland, sp_xy)
    gland = (gland > 0.5).astype(np.float32)
    tumor = _resample_inplane(tumor, sp_xy)
    tumor = (tumor > 0.5).astype(np.float32)

    # Margin in voxels
    margin_px_xy = max(2, int(round(BBOX_MARGIN_MM / TARGET_SPACING)))
    margin_px_z  = max(1, int(round(BBOX_MARGIN_MM / sp_z)))

    y0, y1, x0, x1, z0, z1 = _gland_bbox_3d(gland, margin_px_xy, margin_px_z)

    # 3D crop
    t2w_c   = t2w[y0:y1, x0:x1, z0:z1]
    adc_c   = adc[y0:y1, x0:x1, z0:z1]
    gland_c = gland[y0:y1, x0:x1, z0:z1]
    tumor_c = tumor[y0:y1, x0:x1, z0:z1]

    # Normalise within cropped gland
    t2w_c = percentile_norm_in_mask(t2w_c, gland_c)
    adc_c = percentile_norm_in_mask(adc_c, gland_c)

    # 3D resample: (Hc, Wc, Dc) → (target_size, target_size, n_slices)
    Hc, Wc, Dc = t2w_c.shape
    sc = (target_size/Hc, target_size/Wc, n_slices/Dc)
    t2w_rs   = ndimage_zoom(t2w_c,   sc, order=1).astype(np.float32)
    adc_rs   = ndimage_zoom(adc_c,   sc, order=1).astype(np.float32)
    gland_rs = (ndimage_zoom(gland_c, sc, order=0) > 0.5).astype(np.float32)
    tumor_rs = (ndimage_zoom(tumor_c, sc, order=0) > 0.5).astype(np.float32)

    return {'t2w': t2w_rs, 'adc': adc_rs, 'gland': gland_rs, 'tumor': tumor_rs, 'bbox_crop': True}


def load_viz_volumes(pid, n_slices=32, target_size=224, data_root=DATA_ROOT):
    """Load T2W, ADC, gland, and tumor in bbox_crop space for visualization.
    All arrays are [target_size, target_size, n_slices] float32, spatially consistent."""
    folder  = os.path.join(data_root, pid)
    t2w_img = nib.load(os.path.join(folder, f'{pid}_t2w.nii.gz'))
    t2w     = t2w_img.get_fdata().astype(np.float32)
    sp_xy   = float(t2w_img.header.get_zooms()[0])
    sp_z    = float(t2w_img.header.get_zooms()[2])

    adc   = nib.load(os.path.join(folder, f'{pid}_adc_reg.nii.gz')).get_fdata().astype(np.float32)
    gland = nib.load(os.path.join(folder, f'{pid}_gland.nii.gz')).get_fdata().astype(np.float32)
    gland = (gland > 0.5).astype(np.float32)

    tumor_path = os.path.join(folder, f'{pid}_tumor.nii.gz')
    if os.path.exists(tumor_path) and os.path.getsize(tumor_path) > 0:
        tumor = (nib.load(tumor_path).get_fdata() > 0.5).astype(np.float32)
    else:
        tumor = np.zeros_like(gland)

    t2w   = _resample_inplane(t2w,   sp_xy)
    adc   = _resample_inplane(adc,   sp_xy)
    gland = _resample_inplane(gland, sp_xy); gland = (gland > 0.5).astype(np.float32)
    tumor = _resample_inplane(tumor, sp_xy); tumor = (tumor > 0.5).astype(np.float32)

    margin_px_xy = max(2, int(round(BBOX_MARGIN_MM / TARGET_SPACING)))
    margin_px_z  = max(1, int(round(BBOX_MARGIN_MM / sp_z)))
    y0, y1, x0, x1, z0, z1 = _gland_bbox_3d(gland, margin_px_xy, margin_px_z)

    t2w_c   = t2w[y0:y1, x0:x1, z0:z1]
    adc_c   = adc[y0:y1, x0:x1, z0:z1]
    gland_c = gland[y0:y1, x0:x1, z0:z1]
    tumor_c = tumor[y0:y1, x0:x1, z0:z1]

    t2w_c = percentile_norm_in_mask(t2w_c, gland_c)
    adc_c = percentile_norm_in_mask(adc_c, gland_c)

    Hc, Wc, Dc = t2w_c.shape
    sc = (target_size/Hc, target_size/Wc, n_slices/Dc)
    t2w_rs   = ndimage_zoom(t2w_c,   sc, order=1).astype(np.float32)
    adc_rs   = ndimage_zoom(adc_c,   sc, order=1).astype(np.float32)
    gland_rs = (ndimage_zoom(gland_c, sc, order=0) > 0.5).astype(np.float32)
    tumor_rs = (ndimage_zoom(tumor_c, sc, order=0) > 0.5).astype(np.float32)

    return {'t2w': t2w_rs, 'adc': adc_rs, 'gland': gland_rs, 'tumor': tumor_rs}


def load_labels(csv_path=CSV_PATH, data_root=DATA_ROOT):
    """Returns list of (patient_id, label) for patients with imaging data."""
    available = set(os.listdir(data_root)) - SKIP_PATIENTS
    records = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pid = row['patientID']
            if pid in available:
                records.append((pid, 1 if row['case_csPCa'].strip() == 'YES' else 0))
    return records


def percentile_norm_in_mask(arr, mask, low=1, high=99):
    """Percentile normalization using only within-gland voxels; ignores background."""
    vals = arr[mask > 0.5]
    if len(vals) < 10:
        vals = arr.ravel()
    lo, hi = np.percentile(vals, low), np.percentile(vals, high)
    arr = np.clip(arr, lo, hi)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32)


def load_patient(pid, data_root=DATA_ROOT):
    """Load T2W, ADC, gland volumes → each [H, W, D]. Returns spacing and gland centroid."""
    folder  = os.path.join(data_root, pid)
    t2w_img = nib.load(os.path.join(folder, f'{pid}_t2w.nii.gz'))
    t2w     = t2w_img.get_fdata().astype(np.float32)
    spacing = float(t2w_img.header.get_zooms()[0])
    adc     = nib.load(os.path.join(folder, f'{pid}_adc_reg.nii.gz')).get_fdata().astype(np.float32)
    gland   = nib.load(os.path.join(folder, f'{pid}_gland.nii.gz')).get_fdata().astype(np.float32)
    gland   = (gland > 0.5).astype(np.float32)

    # resample to uniform spacing BEFORE normalization
    t2w   = _resample_inplane(t2w,   spacing)
    adc   = _resample_inplane(adc,   spacing)
    gland = _resample_inplane(gland, spacing)
    gland = (gland > 0.5).astype(np.float32)  # re-binarize after bilinear interpolation
    spacing = TARGET_SPACING

    # within-mask percentile norm: uses only prostate tissue range, ignores background
    t2w = percentile_norm_in_mask(t2w, gland)
    adc = percentile_norm_in_mask(adc, gland)

    cy, cx = _gland_centroid_2d(gland)
    cz     = _gland_z_centroid(gland)
    return {'t2w': t2w, 'adc': adc, 'gland': gland, 'spacing': spacing,
            'cy': cy, 'cx': cx, 'cz': cz}


def slice_to_tensor(vols, z, target_size=224, soft_factor=0.0, n_ch_per_slice=3):
    """Extract slice z → [n_ch_per_slice, H, W] tensor, FOV-cropped centered on gland centroid.
    soft_factor=0: hard masking (outside gland=0). soft_factor>0: outside gland *= soft_factor.
    n_ch_per_slice=2: T2W+ADC only (gland channel dropped). =3: adds gland_sl as 3rd channel."""
    gland_sl = vols['gland'][:, :, z]
    eff_mask = gland_sl + (1.0 - gland_sl) * soft_factor
    if n_ch_per_slice == 2:
        arr = np.stack([vols['t2w'][:, :, z] * eff_mask,
                        vols['adc'][:, :, z] * eff_mask], axis=0)
    else:
        arr = np.stack([vols['t2w'][:, :, z] * eff_mask,
                        vols['adc'][:, :, z] * eff_mask,
                        gland_sl], axis=0)
    return _fov_crop_and_resize(torch.from_numpy(arr), vols['cy'], vols['cx'],
                                vols['spacing'], target_size=target_size)


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


def augment_volume_tensor(tensor, t2w_int_only=False, no_hflip=False, nch_per_slice=3, intensity_aug=False):
    n_ch    = tensor.shape[0]
    s       = nch_per_slice
    t2w_idx = list(range(0, n_ch, s))
    adc_idx = list(range(1, n_ch, s)) if s > 1 else []

    if not no_hflip and random.random() > 0.5:
        tensor = TF.hflip(tensor)

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

    if not intensity_aug:
        return tensor
    int_adc = [] if t2w_int_only else adc_idx
    return _apply_intensity_aug(tensor, t2w_idx, int_adc,
                                noise_max=0.05, gamma_range=(0.85, 1.25),
                                scale_range=(0.90, 1.10), shift_max=0.05, prob=0.4)


def augment_volume_tensor_scale(tensor):
    """Scale-only augmentation (zoom in/out). No rotation, no flip, no intensity changes."""
    scale = random.uniform(0.85, 1.15)
    return TF.affine(tensor, angle=0, translate=[0, 0], scale=scale, shear=0)


def augment_volume_tensor_strong(tensor, t2w_int_only=False, no_hflip=False, nch_per_slice=3, intensity_aug=False):
    n_ch    = tensor.shape[0]
    s       = nch_per_slice
    t2w_idx = list(range(0, n_ch, s))
    adc_idx = list(range(1, n_ch, s)) if s > 1 else []

    if not no_hflip and random.random() > 0.5:
        tensor = TF.hflip(tensor)

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

    if not intensity_aug:
        return tensor
    int_adc = [] if t2w_int_only else adc_idx
    return _apply_intensity_aug(tensor, t2w_idx, int_adc,
                                noise_max=0.08, gamma_range=(0.75, 1.40),
                                scale_range=(0.85, 1.15), shift_max=0.08, prob=0.3)


class PatientVolumeDataset(Dataset):
    """
    Depth-as-channel: 환자 1명 = 1샘플
    모든 슬라이스를 채널로 스택 → [n_slices*3, H, W]
    D < n_slices이면 zero-pad, D > n_slices이면 center crop
    """
    def __init__(self, records, augment=False, aug_strong=False, aug_scale=False, aug_t2w_only=False,
                 no_hflip=False, gland_z_center=False, n_slices=32,
                 input_size=224, soft_mask_factor=0.0, n_ch_per_slice=3,
                 bbox_crop=True, intensity_aug=False, cs_oversample=1, data_root=DATA_ROOT):
        self.augment          = augment
        self.aug_strong       = aug_strong
        self.aug_scale        = aug_scale
        self.aug_t2w_only     = aug_t2w_only
        self.no_hflip         = no_hflip
        self.gland_z_center   = gland_z_center
        self.n_slices         = n_slices
        self.input_size       = input_size
        self.soft_mask_factor = soft_mask_factor
        self.n_ch_per_slice   = n_ch_per_slice
        self.bbox_crop        = bbox_crop
        self.intensity_aug    = intensity_aug
        self.samples    = []
        for pid, label in records:
            if bbox_crop:
                vols = load_patient_bbox_crop(pid, n_slices=n_slices,
                                              target_size=input_size, data_root=data_root)
            else:
                vols = load_patient(pid, data_root)
            self.samples.append((pid, label, vols))
        # Oversample csPCa by adding references (no extra memory — same vols dict reused)
        if cs_oversample > 1:
            cs = [(pid, lbl, v) for pid, lbl, v in self.samples if lbl == 1]
            for _ in range(cs_oversample - 1):
                self.samples.extend(cs)
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, label, vols = self.samples[idx]
        n   = self.n_slices
        sf  = self.soft_mask_factor
        nch = self.n_ch_per_slice

        if self.bbox_crop:
            # vols already (target_size, target_size, n_slices) — just stack channels
            slices = []
            for z in range(n):
                g   = vols['gland'][:, :, z]
                eff = g + (1.0 - g) * sf
                if nch == 2:
                    arr = np.stack([vols['t2w'][:, :, z] * eff,
                                    vols['adc'][:, :, z] * eff], axis=0)
                else:
                    arr = np.stack([vols['t2w'][:, :, z] * eff,
                                    vols['adc'][:, :, z] * eff, g], axis=0)
                slices.append(torch.from_numpy(arr))
            tumor_vol = vols.get('tumor', np.zeros_like(vols['gland']))
            tumor_2d  = torch.from_numpy(
                tumor_vol.max(axis=2).astype(np.float32)
            ).unsqueeze(0)  # [1, H, W]
        else:
            D = vols['t2w'].shape[2]
            if self.gland_z_center:
                start = vols['cz'] - n // 2
            else:
                start = (D - n) // 2 if D >= n else 0

            slices = []
            for i in range(n):
                z = start + i
                if 0 <= z < D:
                    slices.append(slice_to_tensor(vols, z, self.input_size, sf, nch))
                else:
                    slices.append(torch.zeros(nch, self.input_size, self.input_size))
            tumor_2d = torch.zeros(1, self.input_size, self.input_size)

        tensor = torch.cat(slices, dim=0)  # [n*nch, H, W]

        if self.aug_strong:
            tensor = augment_volume_tensor_strong(tensor, t2w_int_only=self.aug_t2w_only,
                                                  no_hflip=self.no_hflip, nch_per_slice=nch,
                                                  intensity_aug=self.intensity_aug)
        elif self.aug_scale:
            tensor = augment_volume_tensor_scale(tensor)
        elif self.augment:
            tensor = augment_volume_tensor(tensor, t2w_int_only=self.aug_t2w_only,
                                           no_hflip=self.no_hflip, nch_per_slice=nch,
                                           intensity_aug=self.intensity_aug)

        return tensor, tumor_2d, label, pid
