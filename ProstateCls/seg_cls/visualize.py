"""
Visualization for seg_cls PI-CAI classification results.
Usage:
    python visualize.py --log logs/<job>.out --ckpt output/<run>/best.pth

Generates in --output-dir:
  learning_curve.png
  roc_pr_curve.png
  confusion_matrix.png
  performance_table.txt
  gradcam/
"""
import argparse
import os
import re
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score, confusion_matrix,
                             average_precision_score, precision_recall_curve)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset import load_labels, load_viz_volumes
from seg_cls.dataset import SegClsDataset
from seg_cls.model import build_model


def parse_log(logfile):
    epochs, losses, val_aucs = [], [], []
    if not logfile or not os.path.exists(logfile):
        return np.array(epochs), np.array(losses), np.array(val_aucs)
    with open(logfile) as f:
        for line in f:
            m = re.search(r'Epoch\s+(\d+)/\d+.*loss:\s*([\d.]+).*val AUC:\s*([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                val_aucs.append(float(m.group(3)))
    return np.array(epochs), np.array(losses), np.array(val_aucs)


def get_probs(model, records, device, n_slices, nch=2):
    ds = SegClsDataset(records, augment=False, n_slices=n_slices, n_ch_per_slice=nch)
    lbls, probs, pids = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            tensor, gland_2d, _tumor_2d, _has_tumor, lbl, pid = ds[i]
            out, _ = model(tensor.unsqueeze(0).to(device), gland_2d.unsqueeze(0).to(device))
            p = torch.softmax(out, dim=1)[0, 1].item()
            probs.append(p); lbls.append(int(lbl)); pids.append(pid)
    return np.array(lbls), np.array(probs), pids


class GradCAMPerSlice:
    """Per-slice input-gradient saliency for depth-as-channel models."""
    def __init__(self, model, n_ch_per_slice):
        self.model = model
        self.nch   = n_ch_per_slice

    def __call__(self, tensor, gland_2d, class_idx=1):
        self.model.eval()
        self.model.zero_grad()
        x = tensor.unsqueeze(0).detach().requires_grad_(True)
        out, _ = self.model(x, gland_2d.unsqueeze(0))
        out[0, class_idx].backward()
        grad = x.grad.squeeze(0).cpu()
        n = tensor.shape[0] // self.nch
        maps = []
        for zi in range(n):
            g = grad[zi*self.nch:(zi+1)*self.nch].abs().mean(0).numpy()
            g = gaussian_filter(g, sigma=4)
            g -= g.min()
            if g.max() > 1e-8:
                g /= g.max()
            maps.append(g)
        return maps


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    job_id = (os.path.basename(args.log).replace('.out', '')
              if args.log else os.path.basename(os.path.dirname(os.path.abspath(args.ckpt))))

    # ── 1. Learning curve ─────────────────────────────────────────────────────
    epochs, losses, val_aucs = parse_log(args.log)
    if len(epochs) > 0:
        best_idx = np.argmax(val_aucs)
        best_ep, best_auc = epochs[best_idx], val_aucs[best_idx]
        print(f"Parsed {len(epochs)} epochs  |  Best val AUC: {best_auc:.4f} @ ep{best_ep}")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f'Job {job_id} — seg_cls', fontsize=13)
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
    else:
        best_ep, best_auc = 0, 0.0
        print("No training log — skipping learning curve")

    # ── 2. Data splits ────────────────────────────────────────────────────────
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
    nch = 3 if args.add_gland_ch else 2
    model = build_model(num_classes=2, pretrained=False, n_slices=args.n_slices,
                        dropout=args.dropout, backbone=args.backbone, n_ch_per_slice=nch)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False), strict=False)
    model = model.to(device)
    print(f"Loaded: {args.ckpt}  (best ep={best_ep})")

    print("Running val inference...")
    val_lbl,  val_prob,  _    = get_probs(model, val_r,  device, args.n_slices, nch)
    print("Running test inference...")
    test_lbl, test_prob, pids = get_probs(model, test_r, device, args.n_slices, nch)

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

    val_pr,  val_rc,  _  = precision_recall_curve(val_lbl,  val_prob,  pos_label=1)
    test_pr, test_rc, _  = precision_recall_curve(test_lbl, test_prob, pos_label=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Job {job_id} — seg_cls  |  Test AUC={test_auc:.4f}  AP={test_ap:.4f}', fontsize=13)
    ax = axes[0]
    ax.plot(val_fpr,  val_tpr,  color='steelblue', lw=2, label=f'Val  AUC={val_auc:.4f}')
    ax.plot(test_fpr, test_tpr, color='tomato',    lw=2, label=f'Test AUC={test_auc:.4f}')
    ax.scatter([test_fpr[j_idx]], [test_tpr[j_idx]], color='red', s=80, zorder=5,
               label=f'Youden thr={best_t:.3f}')
    ax.plot([0,1],[0,1],'k--',alpha=0.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve'); ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(val_rc,  val_pr,  color='steelblue', lw=2, label=f'Val  AP={val_ap:.4f}')
    ax.plot(test_rc, test_pr, color='tomato',    lw=2, label=f'Test AP={test_ap:.4f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(args.output_dir, 'roc_pr_curve.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.savefig(p.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(); print(f"Saved: {p}")

    # ── 5. Confusion matrix ───────────────────────────────────────────────────
    test_pred = (test_prob >= best_t).astype(int)
    cm = confusion_matrix(test_lbl, test_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['ciPCa','csPCa']); ax.set_yticklabels(['ciPCa','csPCa'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (thr={best_t:.3f})')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    p = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close(); print(f"Saved: {p}")

    # ── 6. Performance table ──────────────────────────────────────────────────
    lines = [f"=== seg_cls  Job: {job_id}  Best val AUC: {best_auc:.4f} @ ep{best_ep} ===",
             f"Val  AUC={val_auc:.4f}  AP={val_ap:.4f}",
             f"Test AUC={test_auc:.4f}  AP={test_ap:.4f}",
             f"\nTest confusion at multiple thresholds (Youden={best_t:.3f}):",
             f"  {'Thr':>10} {'Sens':>12} {'Spec':>12} {'Prec':>10} {'F1':>8} "
             f"{'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}"]
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, best_t, 0.6, 0.7, 0.8]:
        p_t = (test_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_lbl, p_t, labels=[0,1]).ravel()
        s  = tp / (tp+fn) if (tp+fn) > 0 else 0
        sp = tn / (tn+fp) if (tn+fp) > 0 else 0
        pr = tp / (tp+fp) if (tp+fp) > 0 else 0
        f  = f1_score(test_lbl, p_t, pos_label=1, zero_division=0)
        marker = ' ← Youden' if abs(thr - best_t) < 0.01 else ''
        lines.append(f"{thr:>10.2f} {s:>12.4f} {sp:>12.4f} {pr:>10.4f} {f:>8.4f} "
                     f"{tp:>4} {fp:>4} {tn:>4} {fn:>4}{marker}")
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

    # ── 7. Grad-CAM ───────────────────────────────────────────────────────────
    if args.gradcam:
        gradcam_dir = os.path.join(args.output_dir, 'gradcam')
        os.makedirs(gradcam_dir, exist_ok=True)
        test_ds = SegClsDataset(test_r, augment=False, n_slices=args.n_slices, n_ch_per_slice=nch)
        gradcam = GradCAMPerSlice(model, nch)
        for i, (pid, lbl, prob) in enumerate(zip(pids, test_lbl, test_prob)):
            tensor, gland_2d, _tumor_2d, _has_tumor, _, _ = test_ds[i]
            heatmaps = gradcam(tensor.to(device), gland_2d.to(device))
            true  = 'csPCa' if lbl==1 else 'ciPCa'
            pred  = 'csPCa' if prob>=0.5 else 'ciPCa'

            vols      = load_viz_volumes(pid, n_slices=args.n_slices)
            t2w_vol   = vols['t2w']
            tumor_vol = vols['tumor']

            tumor_slices   = [z for z in range(args.n_slices) if tumor_vol[:, :, z].sum() > 0]
            display_slices = tumor_slices if tumor_slices else [args.n_slices // 2]

            for zi in display_slices:
                t2w_sl   = t2w_vol[:, :, zi]
                tumor_sl = tumor_vol[:, :, zi]
                suffix   = f'_z{zi:02d}' if tumor_slices else ''

                fig, axes = plt.subplots(1, 4, figsize=(17, 4))
                axes[0].imshow(t2w_sl, cmap='gray')
                axes[0].set_title(f'T2W (slice {zi})'); axes[0].axis('off')
                axes[1].imshow(heatmaps[zi], cmap='jet', vmin=0, vmax=1)
                axes[1].set_title('Saliency (slice)'); axes[1].axis('off')
                axes[2].imshow(t2w_sl, cmap='gray')
                axes[2].imshow(heatmaps[zi], cmap='jet', alpha=0.5, vmin=0, vmax=1)
                axes[2].set_title('Saliency Overlay'); axes[2].axis('off')
                axes[3].imshow(t2w_sl, cmap='gray')
                if tumor_sl.max() > 0:
                    axes[3].imshow(tumor_sl, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
                    axes[3].contour(tumor_sl, levels=[0.5], colors='red', linewidths=1.5)
                axes[3].set_title('Tumor Mask' if tumor_sl.max() > 0 else 'Tumor Mask (none)')
                axes[3].axis('off')
                fig.suptitle(f'{pid}  GT:{true}  Pred:{pred} (p={prob:.3f})',
                             fontsize=12, color='green' if true==pred else 'red', fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(gradcam_dir, f'gradcam_{pid}{suffix}.png'),
                            dpi=200, bbox_inches='tight')
                plt.close()

            if lbl == 1:
                from PIL import Image as _PIL_Image
                import io as _io
                gif_frames = []
                for si in range(args.n_slices):
                    t2w_i   = t2w_vol[:, :, si]
                    tumor_i = tumor_vol[:, :, si]
                    fig_g, ax_g = plt.subplots(1, 3, figsize=(13, 4))
                    ax_g[0].imshow(t2w_i, cmap='gray')
                    ax_g[0].set_title(f'T2W  {si+1}/{args.n_slices}  (bbox z={si})')
                    ax_g[0].axis('off')
                    ax_g[1].imshow(t2w_i, cmap='gray')
                    ax_g[1].imshow(heatmaps[si], cmap='jet', alpha=0.45, vmin=0, vmax=1)
                    ax_g[1].set_title('Saliency Overlay')
                    ax_g[1].axis('off')
                    ax_g[2].imshow(t2w_i, cmap='gray')
                    if tumor_i.max() > 0:
                        ax_g[2].imshow(tumor_i, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
                        ax_g[2].contour(tumor_i, levels=[0.5], colors='red', linewidths=1.5)
                    ax_g[2].set_title('Tumor Mask' if tumor_i.max() > 0 else 'No Tumor')
                    ax_g[2].axis('off')
                    fig_g.suptitle(f'{pid}  GT:{true}  Pred:{pred} (p={prob:.3f})',
                                   fontsize=11, color='green' if true==pred else 'red', fontweight='bold')
                    plt.tight_layout()
                    buf = _io.BytesIO()
                    fig_g.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    gif_frames.append(_PIL_Image.open(buf).copy())
                    buf.close()
                    plt.close(fig_g)
                gif_frames[0].save(
                    os.path.join(gradcam_dir, f'gradcam_{pid}.gif'),
                    save_all=True, append_images=gif_frames[1:],
                    duration=250, loop=0
                )
        print(f"Grad-CAM saved to {gradcam_dir}/")

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log',        type=str,   default='')
    parser.add_argument('--ckpt',       type=str,   required=True)
    parser.add_argument('--output-dir', type=str,   required=True)
    parser.add_argument('--n-slices',   type=int,   default=32)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--val-size',   type=float, default=0.15)
    parser.add_argument('--test-size',  type=float, default=0.15)
    parser.add_argument('--backbone',   type=str,   default='small',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--dropout',    type=float, default=0.3)
    parser.add_argument('--add-gland-ch',  action='store_true')
    parser.add_argument('--no-gradcam', dest='gradcam', action='store_false')
    parser.set_defaults(gradcam=True)
    main(parser.parse_args())
