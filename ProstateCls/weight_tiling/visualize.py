"""
Visualization for PI-CAI classification results (single train/val/test split).
Usage:
    python visualize.py --log logs/9549141.out --ckpt output/split/best.pth

Generates in --output-dir:
  learning_curve.png     — train loss + val AUC per epoch
  roc_pr_curve.png       — ROC + Precision-Recall curve (val & test)
  confusion_matrix.png   — confusion matrix at Youden threshold
  performance_table.txt  — metrics at multiple thresholds + per-patient predictions
  gradcam/               — Grad-CAM overlays (opt-in with --gradcam)
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

# dataset.py is in the parent ProstateCls/ directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PatientVolumeDataset, load_labels
from model import build_model

DATA_ROOT = '/N/slate/ohjiye/PI-CAI/PI-CAI_reg_processed_filtered'


def get_best_display_slice(pid, n_slices=32):
    """Return (tensor_idx, volume_z) for the z with the most tumor area within the sampled range.
    tensor_idx: 0-based index into the 32 stacked slices (used for T2W channel lookup).
    volume_z:   actual z coordinate in the original NIfTI volume."""
    t2w_path   = os.path.join(DATA_ROOT, pid, f'{pid}_t2w.nii.gz')
    tumor_path = os.path.join(DATA_ROOT, pid, f'{pid}_tumor.nii.gz')
    D     = nib.load(t2w_path).shape[2]
    start = (D - n_slices) // 2 if D >= n_slices else 0
    n_eff = min(n_slices, D)
    fallback_ti = n_slices // 2
    fallback_vz = start + fallback_ti
    if not os.path.exists(tumor_path) or os.path.getsize(tumor_path) == 0:
        return fallback_ti, fallback_vz
    tumor_vol = nib.load(tumor_path).get_fdata()
    areas = np.array([tumor_vol[:, :, start + i].sum() for i in range(n_eff)])
    if areas.max() > 0:
        best = int(areas.argmax())
        return best, start + best
    return fallback_ti, fallback_vz


def load_tumor_slice(pid, volume_z, target_size=224):
    """Load tumor mask at volume_z, resized to target_size. Returns zeros if no mask."""
    path = os.path.join(DATA_ROOT, pid, f'{pid}_tumor.nii.gz')
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((target_size, target_size), dtype=np.float32)
    vol = nib.load(path).get_fdata()
    z   = min(volume_z, vol.shape[2] - 1)
    sl  = vol[:, :, z].astype(np.float32)
    sl  = torch.from_numpy(sl[None, None])
    sl  = F.interpolate(sl, size=(target_size, target_size), mode='nearest').squeeze().numpy()
    return (sl > 0.5).astype(np.float32)


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_log(logfile):
    """Returns (epochs, losses, val_aucs) arrays."""
    epochs, losses, val_aucs = [], [], []
    with open(logfile) as f:
        for line in f:
            m = re.search(r'Epoch\s+(\d+)/\d+.*loss:\s*([\d.]+).*val AUC:\s*([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                val_aucs.append(float(m.group(3)))
    return np.array(epochs), np.array(losses), np.array(val_aucs)


# ── Inference ─────────────────────────────────────────────────────────────────

def get_probs(model, records, device, n_slices):
    ds = PatientVolumeDataset(records, augment=False, n_slices=n_slices)
    lbls, probs, pids = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            img, lbl, pid = ds[i]
            p = torch.softmax(model(img.unsqueeze(0).to(device)), dim=1)[0, 1].item()
            probs.append(p); lbls.append(int(lbl)); pids.append(pid)
    return np.array(lbls), np.array(probs), pids


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.features = self.grads = None
        self._hooks = [
            model.norm.register_forward_hook(
                lambda m, i, o: setattr(self, 'features', o.detach())),
            model.norm.register_full_backward_hook(
                lambda m, gi, go: setattr(self, 'grads', go[0].detach())),
        ]

    def __call__(self, tensor, class_idx=1):
        self.model.eval()
        t = tensor.unsqueeze(0).requires_grad_(True)
        self.model(t)[0, class_idx].backward()
        w   = self.grads.mean(dim=[0, 2, 3], keepdim=True)
        cam = F.relu((w * self.features).sum(dim=1, keepdim=True))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = F.interpolate(cam, size=(t.shape[-2], t.shape[-1]),
                            mode='bilinear', align_corners=False)
        self.model.zero_grad()
        return cam.squeeze().cpu().numpy()

    def remove(self):
        for h in self._hooks: h.remove()


# ── Main ──────────────────────────────────────────────────────────────────────

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
    fig.suptitle(f'Job {job_id} — Differential LR (backbone 1e-5 / head 3e-4)', fontsize=13)

    ax = axes[0]
    ax.plot(epochs, losses, color='steelblue', lw=1, alpha=0.7, label='Train loss')
    if len(losses) >= 5:
        smooth = np.convolve(losses, np.ones(5)/5, mode='valid')
        ax.plot(epochs[4:], smooth, color='navy', lw=2, label='5-ep mean')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-Entropy Loss')
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

    # ── 2. Data splits (same seed/sizes as training) ──────────────────────────
    records = load_labels()
    labels  = np.array([r[1] for r in records])
    tv, test_r, ltv, _ = train_test_split(
        records, labels, test_size=args.test_size, stratify=labels, random_state=args.seed)
    _, val_r, _, _ = train_test_split(
        tv, ltv, test_size=args.val_size/(1-args.test_size),
        stratify=ltv, random_state=args.seed)
    print(f"Val:  {len(val_r)}  (csPCa={sum(r[1] for r in val_r)})")
    print(f"Test: {len(test_r)} (csPCa={sum(r[1] for r in test_r)})")

    # ── 3. Load model and run inference ───────────────────────────────────────
    model = build_model(num_classes=2, pretrained=False, n_slices=args.n_slices, head_depth=args.head_depth, backbone=args.backbone)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model = model.to(device)
    print(f"Loaded: {args.ckpt}  (best ep={best_ep})")

    print("Running val inference...")
    val_lbl,  val_prob,  _    = get_probs(model, val_r,  device, args.n_slices)
    print("Running test inference...")
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
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1   = f1_score(test_lbl, preds, pos_label=1, zero_division=0)

    val_prec,  val_rec,  _  = precision_recall_curve(val_lbl,  val_prob)
    test_prec, test_rec, _  = precision_recall_curve(test_lbl, test_prob)
    # random baseline for PR = prevalence
    prevalence = test_lbl.mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f'Job {job_id} — best ckpt ep{best_ep}', fontsize=13)

    # ROC
    ax = axes[0]
    ax.plot(val_fpr,  val_tpr,  'b-', lw=2.5,
            label=f'Val  AUC={val_auc:.3f}')
    ax.plot(test_fpr, test_tpr, 'r-', lw=2.5,
            label=f'Test AUC={test_auc:.3f}')
    ax.scatter([test_fpr[j_idx]], [test_tpr[j_idx]], color='red', s=120, zorder=5,
               label=f'Youden thr={best_t:.2f}\nSens={sens:.2f} Spec={spec:.2f}')
    ax.plot([0,1],[0,1],'k--', alpha=0.35)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve'); ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    # PR
    ax = axes[1]
    ax.plot(val_rec,  val_prec,  'b-', lw=2.5,
            label=f'Val  AP={val_ap:.3f}')
    ax.plot(test_rec, test_prec, 'r-', lw=2.5,
            label=f'Test AP={test_ap:.3f}')
    ax.axhline(prevalence, color='k', ls='--', alpha=0.35,
               label=f'Random (prev={prevalence:.2f})')
    ax.set_xlabel('Recall (Sensitivity)'); ax.set_ylabel('Precision')
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

    # ── 7. Grad-CAM (opt-in) ──────────────────────────────────────────────────
    if args.gradcam:
        gradcam_dir = os.path.join(args.output_dir, 'gradcam')
        os.makedirs(gradcam_dir, exist_ok=True)
        test_ds  = PatientVolumeDataset(test_r, augment=False, n_slices=args.n_slices)
        gcam = GradCAM(model)
        for i, (pid, lbl, prob) in enumerate(zip(pids, test_lbl, test_prob)):
            tensor, _, _ = test_ds[i]
            ti, vz = get_best_display_slice(pid, n_slices=args.n_slices)
            cam   = gcam(tensor.to(device), class_idx=1)
            t2w   = tensor[ti * 3].numpy()
            tumor = load_tumor_slice(pid, vz, target_size=t2w.shape[0])
            true  = 'csPCa' if lbl==1 else 'ciPCa'
            pred  = 'csPCa' if prob>=0.5 else 'ciPCa'
            slice_note = f'z={vz}' if tumor.max() > 0 else f'z={vz} (no tumor)'

            fig, axes = plt.subplots(1, 4, figsize=(17, 4))
            axes[0].imshow(t2w, cmap='gray');  axes[0].set_title(f'T2W ({slice_note})'); axes[0].axis('off')
            axes[1].imshow(cam, cmap='jet', vmin=0, vmax=1); axes[1].set_title('Grad-CAM'); axes[1].axis('off')
            axes[2].imshow(t2w, cmap='gray')
            axes[2].imshow(cam, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axes[2].set_title('CAM Overlay'); axes[2].axis('off')
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
        gcam.remove()
        print(f"Grad-CAM saved to {gradcam_dir}/")

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log',        type=str,   default='logs/9549141.out')
    parser.add_argument('--ckpt',       type=str,   default='output/split/best.pth')
    parser.add_argument('--output-dir', type=str,   default='figures/9549141')
    parser.add_argument('--n-slices',   type=int,   default=32)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--val-size',   type=float, default=0.15)
    parser.add_argument('--test-size',  type=float, default=0.15)
    parser.add_argument('--head-depth',  type=int,   default=2)
    parser.add_argument('--backbone',    type=str,   default='small',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--no-gradcam', dest='gradcam', action='store_false')
    parser.set_defaults(gradcam=True)
    main(parser.parse_args())
