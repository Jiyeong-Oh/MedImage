"""
Training script for CNN-head PI-CAI csPCa classification.
Backbone: MedViT (depth-as-channel 96ch) → [B, 1024, 7, 7] spatial features
Head: CNN attention block (se/cbam/gem/ms) instead of flat MLP
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset import PatientVolumeDataset, load_labels
from model import build_model


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_p   = F.log_softmax(logits, dim=1)
        p_t     = torch.exp(log_p).gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w = (1 - p_t) ** self.gamma
        loss_ce = F.cross_entropy(logits, targets, weight=self.weight,
                                  label_smoothing=self.label_smoothing, reduction='none')
        return (focal_w * loss_ce).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_splits(records, val_size, test_size, seed):
    labels = np.array([r[1] for r in records])
    train_val, test, lbl_tv, _ = train_test_split(
        records, labels, test_size=test_size, stratify=labels, random_state=seed)
    relative_val = val_size / (1.0 - test_size)
    train, val, _, _ = train_test_split(
        train_val, lbl_tv, test_size=relative_val, stratify=lbl_tv, random_state=seed)
    return train, val, test


def compute_class_weights(labels, device, cspca_weight=0.0):
    counts = np.bincount(labels)
    total  = len(labels)
    w = [total / (2 * c) for c in counts]
    if cspca_weight > 0:
        w[1] = cspca_weight
    print(f"  Class weights: ciPCa={w[0]:.3f}  csPCa={w[1]:.3f}  (1:{w[1]/w[0]:.1f})")
    return torch.tensor(w, dtype=torch.float32).to(device)


def make_val_loader(records, args, nw):
    n_ch = 3 if args.add_gland_ch else 2
    ds = PatientVolumeDataset(records, augment=False, aug_strong=False, n_slices=args.n_slices,
                              soft_mask_factor=args.soft_mask_factor, n_ch_per_slice=n_ch)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=nw, pin_memory=True)


class BalancedBatchSampler(torch.utils.data.Sampler):
    """Yields batches with equal pos/neg samples; oversamples minority via replacement."""
    def __init__(self, labels, batch_size):
        self.pos = [i for i, l in enumerate(labels) if l == 1]
        self.neg = [i for i, l in enumerate(labels) if l == 0]
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.n_batches = (len(self.neg) + self.half - 1) // self.half

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        half, total = self.half, self.n_batches * self.half
        pos_idx, neg_idx = [], []
        while len(pos_idx) < total:
            pos_idx += random.sample(self.pos, min(len(self.pos), total - len(pos_idx)))
        while len(neg_idx) < total:
            neg_idx += random.sample(self.neg, min(len(self.neg), total - len(neg_idx)))
        for b in range(self.n_batches):
            batch = pos_idx[b*half:(b+1)*half] + neg_idx[b*half:(b+1)*half]
            random.shuffle(batch)
            yield batch


def eval_auc(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, tumor_2d, labels, _ in loader:
            probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    if len(set(all_labels)) < 2:
        return float('nan'), np.array(all_labels), np.array(all_probs)
    return roc_auc_score(all_labels, all_probs), np.array(all_labels), np.array(all_probs)


def make_optimizer(model, lr_backbone, lr_head, weight_decay):
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if (name.startswith('backbone.stem.0.conv') or name.startswith('cnn_head')
                or name.startswith('tumor_probe')):
            head_params.append(param)
        else:
            backbone_params.append(param)
    print(f"  Optimizer: backbone LR={lr_backbone}, cnn_head LR={lr_head}")
    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params,     'lr': lr_head},
    ], weight_decay=weight_decay)


def train_model(train_records, val_records, args, device, nw):
    print(f"\n{'='*50}")
    cs_tr  = sum(r[1] for r in train_records)
    cs_val = sum(r[1] for r in val_records)
    print(f"Train: {len(train_records)} (csPCa={cs_tr}, ciPCa={len(train_records)-cs_tr})")
    print(f"Val:   {len(val_records)}   (csPCa={cs_val}, ciPCa={len(val_records)-cs_val})")

    n_ch = 3 if args.add_gland_ch else 2
    train_ds = PatientVolumeDataset(
        train_records,
        augment=not args.aug_strong and not args.aug_scale,
        aug_strong=args.aug_strong,
        aug_scale=args.aug_scale,
        aug_t2w_only=args.aug_t2w_only,
        no_hflip=not args.hflip,
        gland_z_center=args.gland_z_center,
        n_slices=args.n_slices,
        soft_mask_factor=args.soft_mask_factor,
        n_ch_per_slice=n_ch,
        intensity_aug=args.intensity_aug,
        cs_oversample=args.cs_oversample,
    )

    train_labels_arr = np.array([s[1] for s in train_ds.samples], dtype=np.float64)
    counts  = np.bincount(train_labels_arr.astype(int))
    cs_eff  = counts[1] if len(counts) > 1 else 0
    print(f"  Effective train size: {len(train_ds)} (csPCa={cs_eff}, ciPCa={counts[0]})")
    weights = 1.0 / counts[train_labels_arr.astype(int)]

    if args.cs_oversample > 1:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=nw, pin_memory=True)
    elif args.balanced_batch:
        bbs = BalancedBatchSampler(train_labels_arr.astype(int).tolist(), args.batch_size)
        train_loader = DataLoader(train_ds, batch_sampler=bbs, num_workers=nw, pin_memory=True)
    else:
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=nw, pin_memory=True)
    val_loader   = make_val_loader(val_records, args, nw)

    n_ch  = 3 if args.add_gland_ch else 2
    model = build_model(num_classes=2, pretrained=not args.scratch, n_slices=args.n_slices,
                        head_type=args.cnn_head, dropout=args.head_dropout,
                        backbone=args.backbone, n_ch_per_slice=n_ch)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_labels_np = np.array([r[1] for r in train_records])
    class_weights = compute_class_weights(train_labels_np, device, args.cspca_weight)
    if args.focal_gamma > 0:
        criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma,
                              label_smoothing=args.label_smoothing)
        print(f"  Loss: FocalLoss(gamma={args.focal_gamma}, ls={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                        label_smoothing=args.label_smoothing)
        print(f"  Loss: CrossEntropyLoss(ls={args.label_smoothing})")

    ckpt_path = os.path.join(args.output_dir, 'best.pth')
    best_auc, best_epoch, patience_count = 0.0, 0, 0

    core = model.module if isinstance(model, nn.DataParallel) else model
    if args.scratch:
        optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
        print(f"  Optimizer: uniform LR={args.lr_head} (scratch mode)")
    else:
        optimizer = make_optimizer(core, args.lr_backbone, args.lr_head, args.weight_decay)
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.t_max, eta_min=1e-7)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.lr_factor,
            patience=args.lr_patience, min_lr=1e-7)

    target_lrs = [pg['lr'] for pg in optimizer.param_groups]

    if args.freeze_epochs > 0:
        for param in optimizer.param_groups[0]['params']:
            param.requires_grad = False
        print(f"  Backbone frozen for first {args.freeze_epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            for param in optimizer.param_groups[0]['params']:
                param.requires_grad = True
            patience_count = 0
            print(f"  Backbone unfrozen at epoch {epoch}  (patience reset)")

        if args.stage2_epoch > 0 and epoch == args.stage2_epoch:
            nat_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=nat_sampler,
                                      num_workers=nw, pin_memory=True)
            patience_count = 0
            print(f"  Stage 2 at epoch {epoch}: switched to natural distribution sampling")

        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            scale = epoch / args.warmup_epochs
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = target_lrs[i] * scale

        model.train()
        if args.freeze_bn:
            bn_src = model.backbone if hasattr(model, 'backbone') else model
            for m in bn_src.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        running_loss = attn_l = 0.0
        for imgs, tumor_2d, lbls, _ in train_loader:
            imgs, lbls = imgs.to(device).contiguous(), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            if args.attn_lambda > 0:
                tumor_7   = (F.adaptive_max_pool2d(tumor_2d.to(device), 7) > 0.5).float()
                loss_attn = F.binary_cross_entropy(core.last_attn_map, tumor_7)
                loss = loss + args.attn_lambda * loss_attn
                attn_l += loss_attn.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        val_auc, _, _ = eval_auc(core, val_loader, device)
        old_lr_hd = optimizer.param_groups[-1]['lr']
        if epoch > args.warmup_epochs:
            if args.scheduler == 'cosine':
                scheduler.step()
            else:
                scheduler.step(val_auc)
        new_lr_hd = optimizer.param_groups[-1]['lr']

        lr_str = (f"LR bb={optimizer.param_groups[0]['lr']:.1e} hd={new_lr_hd:.1e}"
                  if len(optimizer.param_groups) > 1 else f"LR={new_lr_hd:.1e}")
        attn_str = f" attn:{attn_l/len(train_loader):.4f}" if args.attn_lambda > 0 else ""
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"loss: {running_loss/len(train_records):.4f}{attn_str} | "
              f"val AUC: {val_auc:.4f} | {lr_str}")

        if args.scheduler != 'cosine' and epoch > args.warmup_epochs and new_lr_hd < old_lr_hd:
            patience_count = 0
            print(f"  LR reduced: {old_lr_hd:.1e} → {new_lr_hd:.1e}  (patience reset)")

        if val_auc > best_auc:
            best_auc, best_epoch, patience_count = val_auc, epoch, 0
            torch.save(core.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stop at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    print(f"\nBest val AUC: {best_auc:.4f} at epoch {best_epoch}")
    return best_auc, ckpt_path


def evaluate_test(ckpt_path, test_records, val_records, args, device, nw):
    test_loader = make_val_loader(test_records, args, nw)
    n_ch  = 3 if args.add_gland_ch else 2
    model = build_model(num_classes=2, pretrained=False, n_slices=args.n_slices,
                        head_type=args.cnn_head, dropout=args.head_dropout,
                        backbone=args.backbone, n_ch_per_slice=n_ch)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False), strict=False)
    model = model.to(device)

    val_loader = make_val_loader(val_records, args, nw)
    _, val_labels, val_probs = eval_auc(model, val_loader, device)
    _fpr, _tpr, _thr = roc_curve(val_labels, val_probs)
    youden_thr = float(_thr[np.argmax(_tpr - _fpr)])

    test_auc, test_labels, test_probs = eval_auc(model, test_loader, device)

    cs_test = sum(r[1] for r in test_records)
    print(f"\n=== TEST SET ({len(test_records)} patients: "
          f"csPCa={cs_test}, ciPCa={len(test_records)-cs_test}) ===")
    print(f"AUC-ROC     : {test_auc:.4f}  |  Youden thr: {youden_thr:.4f}")
    for label, thr in [("0.500 ", 0.5), (f"{youden_thr:.3f}*", youden_thr)]:
        preds = (test_probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_labels, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1   = f1_score(test_labels, preds, pos_label=1, zero_division=0)
        print(f"  thr={label} → sens={sens:.4f}  spec={spec:.4f}  F1={f1:.4f}"
              f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")


def save_config(args, output_dir):
    config = {
        "run_name":     os.path.basename(output_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training":     vars(args),
        "model": {
            "type":      "ProstateCNNModel",
            "backbone":  f"MedViT_{args.backbone}",
            "cnn_head":  args.cnn_head,
            "input_ch":  args.n_slices * 3,
            "feat_size": "7×7",
            "dropout":   args.head_dropout,
        },
    }
    path = os.path.join(output_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {path}")


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    save_config(args, args.output_dir)
    nw = min(4, os.cpu_count() or 1)

    records = load_labels()
    labels  = np.array([r[1] for r in records])
    print(f"Total: {len(records)}  csPCa: {labels.sum()}  ciPCa: {(labels==0).sum()}")

    train_records, val_records, test_records = make_splits(
        records, val_size=args.val_size, test_size=args.test_size, seed=args.seed)

    cs_tr  = sum(r[1] for r in train_records)
    cs_val = sum(r[1] for r in val_records)
    cs_te  = sum(r[1] for r in test_records)
    print(f"Split → train: {len(train_records)} (csPCa={cs_tr}) | "
          f"val: {len(val_records)} (csPCa={cs_val}) | "
          f"test: {len(test_records)} (csPCa={cs_te})")

    best_auc, ckpt_path = train_model(train_records, val_records, args, device, nw)
    evaluate_test(ckpt_path, test_records, val_records, args, device, nw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',       type=int,   default=150)
    parser.add_argument('--lr-backbone',  type=float, default=1e-5)
    parser.add_argument('--lr-head',      type=float, default=3e-4)
    parser.add_argument('--lr-factor',    type=float, default=0.5)
    parser.add_argument('--lr-patience',  type=int,   default=10)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience',     type=int,   default=30)
    parser.add_argument('--batch-size',   type=int,   default=8)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--n-slices',     type=int,   default=32)
    parser.add_argument('--cspca-weight', type=float, default=0.0)
    parser.add_argument('--focal-gamma',  type=float, default=0.0)
    parser.add_argument('--backbone',     type=str,   default='small',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--cnn-head',     type=str,   default='cbam',
                        choices=['se', 'cbam', 'gem', 'ms'],
                        help='CNN head type: se | cbam | gem | ms')
    parser.add_argument('--head-dropout', type=float, default=0.3)
    parser.add_argument('--aug-t2w-only', action='store_true')
    parser.add_argument('--aug-strong',   action='store_true',
                        help='Use stronger augmentation (larger ranges, higher probs)')
    parser.add_argument('--aug-scale',    action='store_true',
                        help='Scale-only augmentation: zoom in/out 0.85-1.15, no rotation/flip/intensity')
    parser.add_argument('--hflip',       action='store_true',
                        help='Enable horizontal flip augmentation')
    parser.add_argument('--gland-z-center', action='store_true')
    parser.add_argument('--val-size',     type=float, default=0.15)
    parser.add_argument('--test-size',    type=float, default=0.15)
    parser.add_argument('--warmup-epochs', type=int,  default=0)
    parser.add_argument('--freeze-epochs', type=int,  default=0)
    parser.add_argument('--freeze-bn',     action='store_true',
                        help='Keep backbone BN layers in eval mode throughout training')
    parser.add_argument('--scheduler',    type=str,   default='rlrop',
                        choices=['rlrop', 'cosine'])
    parser.add_argument('--t-max',        type=int,   default=150)
    parser.add_argument('--attn-lambda',    type=float, default=0.0,
                        help='Weight of tumor attention probe loss (0=off; train only)')
    parser.add_argument('--output-dir',   type=str,   default='./output/run')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing epsilon (0=disabled, e.g. 0.1)')
    parser.add_argument('--scratch',          action='store_true',
                        help='Train from scratch — random init, uniform LR=lr-head')
    parser.add_argument('--soft-mask-factor', type=float, default=0.0,
                        help='Outside-gland intensity factor (0=hard zero, 0.1=soft)')
    parser.add_argument('--add-gland-ch',   action='store_true',
                        help='Use 3ch per slice (add explicit gland channel; default: 2ch T2W+ADC)')
    parser.add_argument('--cs-oversample',   type=int, default=1,
                        help='Repeat csPCa samples N times with different aug (1=off, 5≈balance)')
    parser.add_argument('--intensity-aug',    action='store_true',
                        help='Enable intensity augmentation (noise, gamma, scale/shift; default: off)')
    parser.add_argument('--balanced-batch',  action='store_true',
                        help='Strict 50/50 pos/neg batch sampler (oversamples minority)')
    parser.add_argument('--stage2-epoch',    type=int, default=0,
                        help='Epoch to switch from balanced to natural distribution sampling')
    main(parser.parse_args())
