"""
Visualization for MIL-ABMIL PI-CAI results.
Generates:
  learning_curve.png, roc_pr_curve.png, confusion_matrix.png,
  performance_table.txt
  attention/   — per-patient attention weight bar charts + per-slice T2W
"""
import argparse
import os
import re
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score, confusion_matrix,
                             average_precision_score, precision_recall_curve)

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)  # mil_abmil/
from dataset import MILDataset, load_labels, load_viz_volumes  # mil_abmil/dataset.py
from model import build_model, SpatialMILModel


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


def _upsample_spatial_attn(sp_hw, out_h, out_w):
    """Upsample [H_a, W_a] spatial attention map to [out_h, out_w], normalize to [0,1]."""
    import torch.nn.functional as _F
    t = torch.tensor(sp_hw).unsqueeze(0).unsqueeze(0).float()
    up = _F.interpolate(t, size=(out_h, out_w), mode='bilinear', align_corners=False)[0, 0]
    up = up.numpy()
    up -= up.min()
    if up.max() > 1e-8:
        up /= up.max()
    return up


def get_probs_and_attn(model, records, device, n_slices, nch=2):
    """Returns labels, probs, slice_attn [N, S], spatial_attn list (None or [S,H,W]), pids.

    For SpatialMILModel: spatial_attn[i] = [S, H_a, W_a] per-slice spatial weights.
    For standard MILModel: spatial_attn[i] = None.
    """
    ds = MILDataset(records, augment=False, n_slices=n_slices, n_ch_per_slice=nch)
    is_spatial = isinstance(model, SpatialMILModel)
    lbls, probs, attns, sp_attns, pids = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            bag, _t2d, _ts, lbl, pid = ds[i]
            logits, attn_w = model(bag.unsqueeze(0).to(device))  # [1,2], [1, S]
            p = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(p)
            lbls.append(int(lbl))
            attns.append(attn_w[0].cpu().numpy())  # [S]
            if is_spatial and model.aggregator.last_spatial_attn is not None:
                # [S, 1, H_a, W_a] → [S, H_a, W_a]
                sp = model.aggregator.last_spatial_attn[0, :, 0].cpu().numpy()
                sp_attns.append(sp)
            else:
                sp_attns.append(None)
            pids.append(pid)
    return np.array(lbls), np.array(probs), np.array(attns), sp_attns, pids


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
        fig.suptitle(f'Job {job_id} — MIL-ABMIL', fontsize=13)
        axes[0].plot(epochs, losses, color='steelblue', lw=1, alpha=0.7)
        if len(losses) >= 5:
            sm = np.convolve(losses, np.ones(5)/5, mode='valid')
            axes[0].plot(epochs[4:], sm, color='navy', lw=2, label='5-ep mean')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss'); axes[0].grid(True, alpha=0.3); axes[0].legend()
        axes[1].plot(epochs, val_aucs, color='tomato', lw=1, alpha=0.7)
        axes[1].axhline(best_auc, color='gray', ls='--', alpha=0.5)
        axes[1].scatter([best_ep], [best_auc], color='red', s=100, zorder=5,
                        label=f'Best: {best_auc:.4f} @ ep{best_ep}')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC-ROC')
        axes[1].set_title('Validation AUC-ROC'); axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0.5, 1.0); axes[1].legend()
        plt.tight_layout()
        p = os.path.join(args.output_dir, 'learning_curve.png')
        plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
        print(f"Saved: {p}")
    else:
        best_ep, best_auc = 0, 0.0
        print("No training log — skipping learning curve")

    # ── 2. Data splits ────────────────────────────────────────────────────────
    records = load_labels()
    labels  = np.array([r[1] for r in records])
    tv, test_r, ltv, _ = train_test_split(
        records, labels, test_size=args.test_size, stratify=labels, random_state=args.seed)
    _, val_r, _, _ = train_test_split(
        tv, ltv, test_size=args.val_size / (1 - args.test_size),
        stratify=ltv, random_state=args.seed)

    # ── 3. Load model ─────────────────────────────────────────────────────────
    if args.spatial_attn:
        args.add_gland_ch = True  # gland channel required for spatial mask inference
    nch = 3 if args.add_gland_ch else 2
    model = build_model(nch=nch, backbone=args.backbone,
                        abmil_hidden=args.abmil_hidden,
                        abmil_dropout=args.abmil_dropout,
                        pretrained=False,
                        spatial_attn=args.spatial_attn)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False),
                          strict=False)
    model = model.to(device)
    is_spatial = isinstance(model, SpatialMILModel)
    print(f"Loaded: {args.ckpt}  (spatial_attn={is_spatial})")

    print("Running val inference...")
    val_lbl, val_prob, _, _, _ = get_probs_and_attn(
        model, val_r, device, args.n_slices, nch)
    print("Running test inference...")
    test_lbl, test_prob, test_attn, test_sp_attn, pids = get_probs_and_attn(
        model, test_r, device, args.n_slices, nch)

    val_auc  = roc_auc_score(val_lbl,  val_prob)
    test_auc = roc_auc_score(test_lbl, test_prob)
    val_ap   = average_precision_score(val_lbl,  val_prob)
    test_ap  = average_precision_score(test_lbl, test_prob)
    print(f"Val  AUC={val_auc:.4f}  AP={val_ap:.4f}")
    print(f"Test AUC={test_auc:.4f}  AP={test_ap:.4f}")

    # ── 4. ROC + PR curve ────────────────────────────────────────────────────
    val_fpr,  val_tpr,  _        = roc_curve(val_lbl,  val_prob)
    test_fpr, test_tpr, test_thr = roc_curve(test_lbl, test_prob)
    j_idx  = np.argmax(test_tpr - test_fpr)
    best_t = float(test_thr[j_idx])
    preds  = (test_prob >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_lbl, preds, labels=[0,1]).ravel()
    sens = tp/(tp+fn) if (tp+fn) > 0 else 0
    spec = tn/(tn+fp) if (tn+fp) > 0 else 0
    f1   = f1_score(test_lbl, preds, pos_label=1, zero_division=0)
    val_prec, val_rec, _   = precision_recall_curve(val_lbl,  val_prob)
    test_prec, test_rec, _ = precision_recall_curve(test_lbl, test_prob)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f'Job {job_id} — MIL-ABMIL  ep{best_ep}', fontsize=13)
    axes[0].plot(val_fpr, val_tpr, 'b-', lw=2.5, label=f'Val  AUC={val_auc:.3f}')
    axes[0].plot(test_fpr, test_tpr, 'r-', lw=2.5, label=f'Test AUC={test_auc:.3f}')
    axes[0].scatter([test_fpr[j_idx]], [test_tpr[j_idx]], color='red', s=120, zorder=5,
                    label=f'Youden thr={best_t:.2f}\nSens={sens:.2f} Spec={spec:.2f}')
    axes[0].plot([0,1],[0,1],'k--', alpha=0.35)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC'); axes[0].legend(loc='lower right'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(val_rec, val_prec, 'b-', lw=2.5, label=f'Val  AP={val_ap:.3f}')
    axes[1].plot(test_rec, test_prec, 'r-', lw=2.5, label=f'Test AP={test_ap:.3f}')
    axes[1].axhline(test_lbl.mean(), color='k', ls='--', alpha=0.35,
                    label=f'Random (prev={test_lbl.mean():.2f})')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('PR Curve'); axes[1].legend(loc='upper right'); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(args.output_dir, 'roc_pr_curve.png')
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Saved: {p}")

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
    plt.savefig(p, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Saved: {p}")

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
    with open(p, 'w') as f: f.write(txt)
    print(f"Saved: {p}")

    # ── 7. Attention weight visualization ────────────────────────────────────
    if args.attn_vis:
        attn_dir = os.path.join(args.output_dir, 'attention')
        os.makedirs(attn_dir, exist_ok=True)
        test_ds = MILDataset(test_r, augment=False, n_slices=args.n_slices, n_ch_per_slice=nch)

        for i, (pid, lbl, prob) in enumerate(zip(pids, test_lbl, test_prob)):
            attn_w  = test_attn[i]             # [n_slices]  ABMIL slice weights
            sp_attn = test_sp_attn[i]          # [S, H_a, W_a] or None
            true    = 'csPCa' if lbl==1 else 'ciPCa'
            pred    = 'csPCa' if prob>=0.5 else 'ciPCa'
            color   = 'green' if true==pred else 'red'

            vols      = load_viz_volumes(pid, n_slices=args.n_slices)
            t2w_vol   = vols['t2w']    # [H, W, n_slices]
            tumor_vol = vols['tumor']
            gland_vol = vols['gland']  # [H, W, n_slices]  for spatial attn clipping
            H_img, W_img = t2w_vol.shape[:2]

            tumor_slices = [z for z in range(args.n_slices) if tumor_vol[:,:,z].sum() > 0]
            top_k        = np.argsort(attn_w)[::-1][:5]  # top 5 ABMIL slices

            # ── 7a. Attention bar chart + top-5 slices ─────────────────────
            n_rows = 3 if sp_attn is not None else 2
            fig = plt.figure(figsize=(18, 4 * n_rows))
            gs  = fig.add_gridspec(n_rows, 6, hspace=0.35, wspace=0.3)

            ax_bar = fig.add_subplot(gs[0, :])
            colors_bar = ['tomato' if z in tumor_slices else 'steelblue'
                          for z in range(args.n_slices)]
            ax_bar.bar(range(args.n_slices), attn_w, color=colors_bar, alpha=0.8)
            for k in top_k:
                ax_bar.annotate('★', (k, attn_w[k]), ha='center', va='bottom',
                                fontsize=9, color='gold')
            ax_bar.set_xlabel('Slice index'); ax_bar.set_ylabel('ABMIL weight')
            ax_bar.set_title(
                f'{pid}  GT:{true}  Pred:{pred} (p={prob:.3f})  '
                f'[red = tumor slice]',
                color=color, fontweight='bold')
            ax_bar.grid(True, alpha=0.3, axis='y')

            # Row 1: top-5 T2W slices
            for rank, zi in enumerate(top_k):
                ax = fig.add_subplot(gs[1, rank])
                ax.imshow(t2w_vol[:, :, zi], cmap='gray')
                if tumor_vol[:, :, zi].max() > 0:
                    ax.contour(tumor_vol[:, :, zi], levels=[0.5],
                               colors='red', linewidths=1.5)
                ax.set_title(f'z={zi}  w={attn_w[zi]:.4f}', fontsize=9)
                ax.axis('off')

            # Row 2 (spatial model only): same slices with spatial attention overlay
            if sp_attn is not None:
                for rank, zi in enumerate(top_k):
                    ax = fig.add_subplot(gs[2, rank])
                    sp_up = _upsample_spatial_attn(sp_attn[zi], H_img, W_img)
                    sp_up = sp_up * (gland_vol[:, :, zi] > 0.05)  # clip to gland
                    ax.imshow(t2w_vol[:, :, zi], cmap='gray')
                    ax.imshow(sp_up, cmap='hot', alpha=0.45, vmin=0, vmax=1)
                    if tumor_vol[:, :, zi].max() > 0:
                        ax.contour(tumor_vol[:, :, zi], levels=[0.5],
                                   colors='cyan', linewidths=1.2)
                    ax.set_title(f'z={zi}  spatial attn', fontsize=9)
                    ax.axis('off')

            plt.savefig(os.path.join(attn_dir, f'attn_{pid}.png'),
                        dpi=200, bbox_inches='tight')
            plt.close()

            # ── 7b. GIF for csPCa patients ────────────────────────────────
            if lbl == 1:
                from PIL import Image as _PIL_Image
                import io as _io
                gif_frames = []
                max_w = float(attn_w.max())
                n_cols = 3 if sp_attn is not None else 2
                col_ratios = [2, 2, 1] if sp_attn is not None else [2.5, 1]
                for si in range(args.n_slices):
                    t2w_i   = t2w_vol[:, :, si]
                    tumor_i = tumor_vol[:, :, si]
                    fig_g, axes_g = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5),
                                                  gridspec_kw={'width_ratios': col_ratios})

                    # Col 0: T2W + tumor contour
                    axes_g[0].imshow(t2w_i, cmap='gray')
                    if tumor_i.max() > 0:
                        axes_g[0].contour(tumor_i, levels=[0.5], colors='red', linewidths=1.5)
                    axes_g[0].set_title(
                        f'T2W  z={si}  ({"tumor" if si in tumor_slices else "no tumor"})',
                        color='red' if si in tumor_slices else 'black')
                    axes_g[0].axis('off')

                    # Col 1 (spatial only): T2W + spatial attention overlay
                    if sp_attn is not None:
                        sp_up = _upsample_spatial_attn(sp_attn[si], H_img, W_img)
                        sp_up = sp_up * (gland_vol[:, :, si] > 0.05)  # clip to gland
                        axes_g[1].imshow(t2w_i, cmap='gray')
                        axes_g[1].imshow(sp_up, cmap='hot', alpha=0.45, vmin=0, vmax=1)
                        if tumor_i.max() > 0:
                            axes_g[1].contour(tumor_i, levels=[0.5],
                                              colors='cyan', linewidths=1.2)
                        axes_g[1].set_title(f'Spatial attn  z={si}', fontsize=10)
                        axes_g[1].axis('off')

                    # Last col: ABMIL slice weight bar
                    bar_ax = axes_g[-1]
                    bar_colors = ['tomato' if z in tumor_slices else 'steelblue'
                                  for z in range(args.n_slices)]
                    bar_colors[si] = 'gold'
                    bar_ax.bar(range(args.n_slices), attn_w, color=bar_colors, alpha=0.85)
                    bar_ax.axvline(si, color='yellow', lw=2, alpha=0.9)
                    bar_ax.set_xlim(-0.5, args.n_slices - 0.5)
                    bar_ax.set_ylim(0, max_w * 1.15)
                    bar_ax.set_xlabel('Slice'); bar_ax.set_ylabel('ABMIL weight')
                    bar_ax.set_title(f'w={attn_w[si]:.4f}', fontsize=10)
                    bar_ax.grid(True, alpha=0.3, axis='y')

                    fig_g.suptitle(f'{pid}  GT:{true}  Pred:{pred} (p={prob:.3f})',
                                   fontsize=11, color=color, fontweight='bold')
                    plt.tight_layout()
                    buf = _io.BytesIO()
                    fig_g.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    gif_frames.append(_PIL_Image.open(buf).copy())
                    buf.close()
                    plt.close(fig_g)

                gif_frames[0].save(
                    os.path.join(attn_dir, f'attn_{pid}.gif'),
                    save_all=True, append_images=gif_frames[1:],
                    duration=250, loop=0)

        print(f"Attention visualization saved to {attn_dir}/")

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log',          type=str,   default='')
    parser.add_argument('--ckpt',         type=str,   required=True)
    parser.add_argument('--output-dir',   type=str,   required=True)
    parser.add_argument('--n-slices',     type=int,   default=32)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--val-size',     type=float, default=0.15)
    parser.add_argument('--test-size',    type=float, default=0.15)
    parser.add_argument('--backbone',     type=str,   default='small',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--abmil-hidden', type=int,   default=256)
    parser.add_argument('--abmil-dropout', type=float, default=0.25)
    parser.add_argument('--add-gland-ch', action='store_true')
    parser.add_argument('--spatial-attn', action='store_true',
                        help='Use SpatialMILModel (per-slice spatial attention)')
    parser.add_argument('--no-attn-vis',  dest='attn_vis', action='store_false')
    parser.set_defaults(attn_vis=True)
    main(parser.parse_args())
