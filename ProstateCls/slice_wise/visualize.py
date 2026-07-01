"""
Visualization for slice-wise MIL-max PI-CAI classification results.
Grad-CAM is computed on the slice the model assigned the highest csPCa logit.
"""
import argparse
import os
import re
import sys
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score, confusion_matrix,
                             average_precision_score, precision_recall_curve)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset import load_labels, _gland_centroid_2d, _fov_crop_and_resize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import SliceWiseDataset
from model import build_model

DATA_ROOT = '/N/slate/ohjiye/PI-CAI/PI-CAI_reg_processed_filtered'


def load_tumor_slice(pid, volume_z, target_size=224):
    """Load tumor mask at volume_z with same FOV crop as the dataset."""
    t2w_path   = os.path.join(DATA_ROOT, pid, f'{pid}_t2w.nii.gz')
    tumor_path = os.path.join(DATA_ROOT, pid, f'{pid}_tumor.nii.gz')
    gland_path = os.path.join(DATA_ROOT, pid, f'{pid}_gland.nii.gz')
    spacing = float(nib.load(t2w_path).header.get_zooms()[0])
    gland_3d = (nib.load(gland_path).get_fdata() > 0.5).astype(np.float32)
    cy, cx = _gland_centroid_2d(gland_3d)
    if not os.path.exists(tumor_path) or os.path.getsize(tumor_path) == 0:
        return np.zeros((target_size, target_size), dtype=np.float32)
    tumor_vol = nib.load(tumor_path).get_fdata()
    z = min(volume_z, tumor_vol.shape[2] - 1)
    sl = tumor_vol[:, :, z].astype(np.float32)
    sl = _fov_crop_and_resize(torch.from_numpy(sl[None]), cy, cx, spacing, target_size=target_size)
    return (sl.squeeze().numpy() > 0.5).astype(np.float32)


def get_mil_slice(model, tensor, device):
    """Return (tensor_idx, csPCa_prob) for the slice with highest csPCa logit."""
    model.eval()
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))  # [1, N, 2]
    logits = logits[0]                                  # [N, 2]
    best_i = logits[:, 1].argmax().item()
    prob   = torch.sigmoid(logits[:, 1].max()).item()
    return best_i, prob


class GradCAM:
    def __init__(self, model):
        self.model     = model.backbone
        self.features  = None
        self.gradients = None
        self._fhook = self.model.norm.register_forward_hook(
            lambda m, i, o: setattr(self, 'features', o.detach()))
        self._bhook = self.model.norm.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def __call__(self, slice_tensor, class_idx=1):
        """slice_tensor: [3, H, W] (single slice, no batch dim)"""
        self.model.eval()
        self.model.zero_grad()
        out = self.model(slice_tensor.unsqueeze(0))
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.features).sum(dim=1, keepdim=True))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = F.interpolate(cam, size=(slice_tensor.shape[-2], slice_tensor.shape[-1]),
                            mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

    def remove(self):
        self._fhook.remove()
        self._bhook.remove()


def parse_log(logfile):
    epochs, losses, val_aucs = [], [], []
    with open(logfile) as f:
        for line in f:
            m = re.search(r'Epoch\s+(\d+)/\d+.*loss:\s*([\d.]+).*val AUC:\s*([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                val_aucs.append(float(m.group(3)))
    return np.array(epochs), np.array(losses), np.array(val_aucs)


def get_probs(model, records, device, n_slices):
    ds = SliceWiseDataset(records, augment=False, n_slices=n_slices)
    lbls, probs, pids = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            img, lbl, pid = ds[i]
            logits        = model(img.unsqueeze(0).to(device))      # [1, N, 2]
            prob          = torch.sigmoid(logits[0, :, 1].max()).item()
            probs.append(prob); lbls.append(int(lbl)); pids.append(pid)
    return np.array(lbls), np.array(probs), pids


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    job_id = os.path.basename(args.log).replace('.out', '')

    # ── 1. Learning curve ─────────────────────────────────────────────────────
    epochs, losses, val_aucs = parse_log(args.log)
    best_idx = np.argmax(val_aucs)
    best_ep, best_auc = epochs[best_idx], val_aucs[best_idx]
    print(f"Parsed {len(epochs)} epochs  |  Best val AUC: {best_auc:.4f} @ ep{best_ep}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Job {job_id} — Slice-Wise MIL-Max', fontsize=13)
    ax = axes[0]
    ax.plot(epochs, losses, color='steelblue', lw=1, alpha=0.7, label='Train loss')
    if len(losses) >= 5:
        smooth = np.convolve(losses, np.ones(5)/5, mode='valid')
        ax.plot(epochs[4:], smooth, color='navy', lw=2, label='5-ep mean')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, val_aucs, color='tomato', lw=1, alpha=0.7, label='Val AUC-ROC')
    ax.axhline(best_auc, color='gray', ls='--', alpha=0.5)
    ax.scatter([best_ep], [best_auc], color='red', s=100, zorder=5,
               label=f'Best: {best_auc:.4f} @ ep{best_ep}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('AUC-ROC')
    ax.set_title('Validation AUC-ROC'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    p = os.path.join(args.output_dir, 'learning_curve.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.savefig(p.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(); print(f"Saved: {p}")

    # ── 2. Data splits ────────────────────────────────────────────────────────
    records = load_labels()
    labels  = np.array([r[1] for r in records])
    tv, test_r, ltv, _ = train_test_split(
        records, labels, test_size=args.test_size, stratify=labels, random_state=args.seed)
    _, val_r, _, _ = train_test_split(
        tv, ltv, test_size=args.val_size/(1-args.test_size), stratify=ltv, random_state=args.seed)

    # ── 3. Load model and run inference ───────────────────────────────────────
    model = build_model(num_classes=2, pretrained=False, n_slices=args.n_slices,
                        head_depth=args.head_depth, backbone=args.backbone)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model = model.to(device)
    print(f"Loaded: {args.ckpt}")

    val_lbl,  val_prob,  _    = get_probs(model, val_r,  device, args.n_slices)
    test_lbl, test_prob, pids = get_probs(model, test_r, device, args.n_slices)

    val_auc  = roc_auc_score(val_lbl,  val_prob)
    test_auc = roc_auc_score(test_lbl, test_prob)
    val_ap   = average_precision_score(val_lbl,  val_prob)
    test_ap  = average_precision_score(test_lbl, test_prob)
    print(f"Val  AUC={val_auc:.4f}  AP={val_ap:.4f}")
    print(f"Test AUC={test_auc:.4f}  AP={test_ap:.4f}")

    # ── 4. ROC + PR curve ─────────────────────────────────────────────────────
    val_fpr,  val_tpr,  _        = roc_curve(val_lbl,  val_prob)
    test_fpr, test_tpr, test_thr = roc_curve(test_lbl, test_prob)
    j_idx  = np.argmax(test_tpr - test_fpr)
    best_t = float(test_thr[j_idx])

    preds = (test_prob >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_lbl, preds, labels=[0,1]).ravel()
    sens       = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec       = tn / (tn + fp) if (tn + fp) > 0 else 0
    prevalence = test_lbl.mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f'Job {job_id} — best ckpt ep{best_ep}', fontsize=13)
    ax = axes[0]
    ax.plot(val_fpr,  val_tpr,  'b-', lw=2.5, label=f'Val  AUC={val_auc:.3f}')
    ax.plot(test_fpr, test_tpr, 'r-', lw=2.5, label=f'Test AUC={test_auc:.3f}')
    ax.scatter([test_fpr[j_idx]], [test_tpr[j_idx]], color='red', s=120, zorder=5,
               label=f'Youden thr={best_t:.2f}\nSens={sens:.2f} Spec={spec:.2f}')
    ax.plot([0,1],[0,1],'k--', alpha=0.35)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve'); ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    val_prec,  val_rec,  _ = precision_recall_curve(val_lbl,  val_prob)
    test_prec, test_rec, _ = precision_recall_curve(test_lbl, test_prob)
    ax = axes[1]
    ax.plot(val_rec,  val_prec,  'b-', lw=2.5, label=f'Val  AP={val_ap:.3f}')
    ax.plot(test_rec, test_prec, 'r-', lw=2.5, label=f'Test AP={test_ap:.3f}')
    ax.axhline(prevalence, color='k', ls='--', alpha=0.35, label=f'Random (prev={prevalence:.2f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve'); ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    p = os.path.join(args.output_dir, 'roc_pr_curve.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.savefig(p.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(); print(f"Saved: {p}")

    # ── 5. Confusion matrix ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 4))
    cm_arr = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm_arr, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred ciPCa','Pred csPCa'])
    ax.set_yticklabels(['True ciPCa','True csPCa'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i,j]), ha='center', va='center',
                    fontsize=18, color='white' if cm_arr[i,j] > cm_arr.max()/2 else 'black')
    ax.set_title(f'Confusion Matrix (thr={best_t:.2f})')
    plt.colorbar(im, ax=ax); plt.tight_layout()
    p = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close(); print(f"Saved: {p}")

    # ── 6. Performance table ──────────────────────────────────────────────────
    thresholds = sorted(set([0.2, 0.3, 0.4, 0.5, round(best_t, 2), 0.6, 0.7]))
    lines = [f"=== {job_id} ===",
             f"Best val AUC: {best_auc:.4f} @ epoch {best_ep}",
             f"Val  AP:      {val_ap:.4f}",
             f"Test AUC:     {test_auc:.4f}",
             f"Test AP:      {test_ap:.4f}\n",
             f"{'Threshold':>10} {'Sensitivity':>12} {'Specificity':>12} "
             f"{'Precision':>10} {'F1':>8} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}",
             "-"*75]
    for thr in thresholds:
        p_t = (test_prob >= thr).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(test_lbl, p_t, labels=[0,1]).ravel()
        s  = tp_t/(tp_t+fn_t) if (tp_t+fn_t)>0 else 0
        sp = tn_t/(tn_t+fp_t) if (tn_t+fp_t)>0 else 0
        pr = tp_t/(tp_t+fp_t) if (tp_t+fp_t)>0 else 0
        f  = f1_score(test_lbl, p_t, pos_label=1, zero_division=0)
        marker = ' ← Youden' if abs(thr - best_t) < 0.01 else ''
        lines.append(f"{thr:>10.2f} {s:>12.4f} {sp:>12.4f} {pr:>10.4f} {f:>8.4f} "
                     f"{tp_t:>4} {fp_t:>4} {tn_t:>4} {fn_t:>4}{marker}")
    lines += ["\nPer-patient predictions (test set):",
              f"  {'PatientID':<20} {'Label':<8} {'Prob':>6} {'Pred@0.5':>10} {'Correct':>8}"]
    for pid, lbl, prob in zip(pids, test_lbl, test_prob):
        pred = 'csPCa' if prob >= 0.5 else 'ciPCa'
        true = 'csPCa' if lbl == 1   else 'ciPCa'
        lines.append(f"  {pid:<20} {true:<8} {prob:>6.3f} {pred:>10} "
                     f"{'OK' if pred==true else 'WRONG':>8}")
    txt = "\n".join(lines)
    print(txt)
    p = os.path.join(args.output_dir, 'performance_table.txt')
    with open(p, 'w') as f: f.write(txt); print(f"Saved: {p}")

    # ── 7. Grad-CAM (on MIL-max slice) ───────────────────────────────────────
    if args.gradcam:
        gradcam_dir = os.path.join(args.output_dir, 'gradcam')
        os.makedirs(gradcam_dir, exist_ok=True)
        test_ds = SliceWiseDataset(test_r, augment=False, n_slices=args.n_slices)
        gradcam = GradCAM(model)
        for i, (pid, lbl, prob) in enumerate(zip(pids, test_lbl, test_prob)):
            tensor, _, _ = test_ds[i]   # [N, 3, H, W]
            # Find which slice the model focused on
            best_i, _    = get_mil_slice(model, tensor, device)
            slice_t      = tensor[best_i].to(device)   # [3, H, W]
            heatmap      = gradcam(slice_t)
            t2w          = tensor[best_i, 0].numpy()   # T2W channel of that slice

            # Map tensor slice index back to volume z
            D     = nib.load(os.path.join(DATA_ROOT, pid, f'{pid}_t2w.nii.gz')).shape[2]
            start = (D - args.n_slices) // 2 if D >= args.n_slices else 0
            vz    = start + best_i
            tumor = load_tumor_slice(pid, vz, target_size=t2w.shape[0])

            true  = 'csPCa' if lbl==1 else 'ciPCa'
            pred  = 'csPCa' if prob>=0.5 else 'ciPCa'
            slice_note = f'z={vz} (MIL-max)' if tumor.max() > 0 else f'z={vz} (MIL-max, no tumor)'

            fig, axes = plt.subplots(1, 4, figsize=(17, 4))
            axes[0].imshow(t2w, cmap='gray');  axes[0].set_title(f'T2W ({slice_note})'); axes[0].axis('off')
            axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1); axes[1].set_title('Grad-CAM'); axes[1].axis('off')
            axes[2].imshow(t2w, cmap='gray')
            axes[2].imshow(heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axes[2].set_title('Grad-CAM Overlay'); axes[2].axis('off')
            axes[3].imshow(t2w, cmap='gray')
            if tumor.max() > 0:
                axes[3].imshow(tumor, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
                axes[3].contour(tumor, levels=[0.5], colors='red', linewidths=1.5)
            axes[3].set_title('Tumor Mask' if tumor.max() > 0 else 'Tumor Mask (none)'); axes[3].axis('off')
            fig.suptitle(f'{pid}  GT:{true}  Pred:{pred} (p={prob:.3f})',
                         fontsize=12, color='green' if true==pred else 'red', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(gradcam_dir, f'gradcam_{pid}.png'), dpi=200, bbox_inches='tight')
            plt.close()
        gradcam.remove()
        print(f"Grad-CAM saved to {gradcam_dir}/")

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log',        type=str,   required=True)
    parser.add_argument('--ckpt',       type=str,   required=True)
    parser.add_argument('--output-dir', type=str,   default='figures/run')
    parser.add_argument('--n-slices',   type=int,   default=32)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--val-size',   type=float, default=0.15)
    parser.add_argument('--test-size',  type=float, default=0.15)
    parser.add_argument('--head-depth', type=int,   default=2)
    parser.add_argument('--backbone',   type=str,   default='small',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--no-gradcam', dest='gradcam', action='store_false')
    parser.set_defaults(gradcam=True)
    main(parser.parse_args())
