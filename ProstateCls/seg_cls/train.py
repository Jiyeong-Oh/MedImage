"""
Multi-task training: segmentation (tumor) + classification (csPCa).

Key design choices:
  - Seg loss applied only to patients with non-empty tumor masks (has_tumor flag)
  - Seg loss = Dice + BCE, computed WITHIN the gland mask region
  - Cls loss = CrossEntropy with class weights
  - Label smoothing supported for cls loss
  - Backbone LR=1e-5 (pretrained), decoder+head LR=3e-4 (new)
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
from dataset import load_labels
from seg_cls.dataset import SegClsDataset
from seg_cls.model import build_model


# ── Losses ────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class focal loss with optional label smoothing."""
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


def dice_loss(pred, target, eps=1e-6):
    """Soft Dice loss. pred and target: [B, 1, H, W], pred already sigmoid-ed."""
    p = pred.flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return 1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)


def seg_loss_fn(tumor_pred, tumor_target, gland_2d):
    """
    Dice + BCE within gland region.
    tumor_pred:   [B, 1, H, W] sigmoid output
    tumor_target: [B, 1, H, W] binary ground truth
    gland_2d:     [B, 1, H, W] binary gland mask
    """
    # weight BCE by gland mask: focus loss inside prostate
    bce = F.binary_cross_entropy(tumor_pred, tumor_target,
                                  weight=gland_2d.float() + 0.1,  # small bg weight
                                  reduction='mean')
    dice = dice_loss(tumor_pred * gland_2d, tumor_target * gland_2d).mean()
    return 0.5 * bce + 0.5 * dice


# ── Utilities ─────────────────────────────────────────────────────────────────

def compute_class_weights(labels, device, cspca_weight=0.0):
    counts = np.bincount(labels)
    total  = len(labels)
    w = [total / (2 * c) for c in counts]
    if cspca_weight > 0:
        w[1] = cspca_weight
    print(f"  Class weights: ciPCa={w[0]:.3f}  csPCa={w[1]:.3f}  (1:{w[1]/w[0]:.1f})")
    return torch.tensor(w, dtype=torch.float32).to(device)


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


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_splits(records, val_size, test_size, seed):
    labels = np.array([r[1] for r in records])
    train_val, test, lv, _ = train_test_split(records, labels,
                                               test_size=test_size, stratify=labels, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val, _, _ = train_test_split(train_val, lv,
                                         test_size=rel_val, stratify=lv, random_state=seed)
    return train, val, test


def make_optimizer(model, lr_backbone, lr_head, weight_decay):
    backbone_params, new_params = [], []
    for name, p in model.named_parameters():
        if name.startswith('backbone.stem.0.conv') or not name.startswith('backbone.'):
            new_params.append(p)
        else:
            backbone_params.append(p)
    print(f"  backbone params={len(backbone_params)}  new params={len(new_params)}")
    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': new_params,      'lr': lr_head},
    ], weight_decay=weight_decay)


def eval_auc(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, gland_2d, tumor_2d, has_tumor, labels, _ in loader:
            logits, _ = model(imgs.to(device), gland_2d.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs); all_labels.extend(labels.numpy())
    if len(set(all_labels)) < 2:
        return float('nan'), np.array(all_labels), np.array(all_probs)
    return roc_auc_score(all_labels, all_probs), np.array(all_labels), np.array(all_probs)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(train_records, val_records, args, device, nw):
    print(f"\n{'='*50}")
    cs_tr = sum(r[1] for r in train_records)
    print(f"Train: {len(train_records)} (csPCa={cs_tr})  Val: {len(val_records)}")

    n_ch = 3 if args.add_gland_ch else 2
    train_ds = SegClsDataset(train_records, augment=True, aug_scale=args.aug_scale,
                              n_slices=args.n_slices, soft_mask_factor=args.soft_mask_factor,
                              n_ch_per_slice=n_ch, intensity_aug=args.intensity_aug,
                              cs_oversample=args.cs_oversample)
    val_ds   = SegClsDataset(val_records,   augment=False, n_slices=args.n_slices,
                              soft_mask_factor=args.soft_mask_factor, n_ch_per_slice=n_ch)

    train_labels = np.array([s[1] for s in train_ds.samples])
    counts = np.bincount(train_labels)
    cs_eff = counts[1] if len(counts) > 1 else 0
    print(f"  Effective train size: {len(train_ds)} (csPCa={cs_eff}, ciPCa={counts[0]})")

    if args.cs_oversample > 1:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=nw, pin_memory=True)
    elif args.balanced_batch:
        bbs = BalancedBatchSampler(train_labels.astype(int).tolist(), args.batch_size)
        train_loader = DataLoader(train_ds, batch_sampler=bbs, num_workers=nw, pin_memory=True)
    else:
        weights = 1.0 / counts[train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)

    n_ch  = 3 if args.add_gland_ch else 2
    model = build_model(num_classes=2, pretrained=not args.scratch, n_slices=args.n_slices,
                        dropout=args.dropout, backbone=args.backbone, n_ch_per_slice=n_ch).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    core = model.module if isinstance(model, nn.DataParallel) else model

    class_weights = compute_class_weights(train_labels, device, args.cspca_weight)
    if args.focal_gamma > 0:
        cls_criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma,
                                  label_smoothing=args.label_smoothing)
        print(f"  Loss: FocalLoss(gamma={args.focal_gamma}, ls={args.label_smoothing})")
    else:
        cls_criterion = nn.CrossEntropyLoss(weight=class_weights,
                                            label_smoothing=args.label_smoothing)
        print(f"  Loss: CrossEntropyLoss(ls={args.label_smoothing})")

    if args.scratch:
        optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
        print(f"  Optimizer: uniform LR={args.lr_head} (scratch mode)")
    else:
        optimizer = make_optimizer(core, args.lr_backbone, args.lr_head, args.weight_decay)
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-7)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.lr_factor,
            patience=args.lr_patience, min_lr=1e-7)

    ckpt_path = os.path.join(args.output_dir, 'best.pth')
    best_auc, patience_count = 0.0, 0

    if args.freeze_epochs > 0:
        for p in optimizer.param_groups[0]['params']:
            p.requires_grad = False
        print(f"  Backbone frozen for first {args.freeze_epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            for p in optimizer.param_groups[0]['params']:
                p.requires_grad = True
            patience_count = 0
            print(f"  Backbone unfrozen at epoch {epoch}")

        if args.stage2_epoch > 0 and epoch == args.stage2_epoch:
            nat_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=nat_sampler,
                                      num_workers=nw, pin_memory=True)
            patience_count = 0
            print(f"  Stage 2 at epoch {epoch}: switched to natural distribution sampling")

        model.train()
        if args.freeze_bn:
            bn_src = core.backbone if hasattr(core, 'backbone') else core
            for m in bn_src.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        total_loss = cls_l = seg_l = attn_l = 0.0
        for imgs, gland_2d, tumor_2d, has_tumor, labels, _ in train_loader:
            imgs     = imgs.to(device)
            gland_2d = gland_2d.to(device)
            tumor_2d = tumor_2d.to(device)
            labels   = labels.to(device)

            optimizer.zero_grad()
            cls_logits, tumor_pred = model(imgs, gland_2d)

            loss_cls = cls_criterion(cls_logits, labels)

            # seg loss only for patients with non-empty tumor masks
            mask = has_tumor.to(device)
            if mask.any():
                loss_seg = seg_loss_fn(tumor_pred[mask],
                                       tumor_2d[mask],
                                       gland_2d[mask]).mean()
            else:
                loss_seg = torch.tensor(0.0, device=device)

            # attention supervision: activate on tumor (csPCa), suppress elsewhere (ciPCa→zeros)
            if args.attn_lambda > 0:
                attn_map  = core.sp_attn.last_map
                tumor_7   = (F.adaptive_max_pool2d(tumor_2d, 7) > 0.5).float()
                loss_attn = F.binary_cross_entropy(attn_map, tumor_7)
            else:
                loss_attn = torch.tensor(0.0, device=device)

            loss = loss_cls + args.seg_lambda * loss_seg + args.attn_lambda * loss_attn
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            cls_l      += loss_cls.item()
            seg_l      += loss_seg.item()
            attn_l     += loss_attn.item()

        val_auc, _, _ = eval_auc(model, val_loader, device)
        if args.scheduler == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_auc)
        lr_hd = optimizer.param_groups[-1]['lr']
        lr_str = (f"LR bb={optimizer.param_groups[0]['lr']:.1e} hd={lr_hd:.1e}"
                  if len(optimizer.param_groups) > 1 else f"LR={lr_hd:.1e}")
        n = len(train_loader)
        attn_str = f" attn:{attn_l/n:.4f}" if args.attn_lambda > 0 else ""
        print(f"  Epoch {epoch:3d}/{args.epochs} | loss:{total_loss/n:.4f}"
              f" cls:{cls_l/n:.4f} seg:{seg_l/n:.4f}{attn_str}"
              f" | val AUC:{val_auc:.4f} | {lr_str}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(core.state_dict(), ckpt_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"Best val AUC: {best_auc:.4f}")
    return best_auc, ckpt_path


# ── Test evaluation ───────────────────────────────────────────────────────────

def evaluate_test(ckpt_path, test_records, val_records, args, device, nw):
    n_ch  = 3 if args.add_gland_ch else 2
    model = build_model(num_classes=2, pretrained=False, n_slices=args.n_slices,
                        dropout=args.dropout, backbone=args.backbone, n_ch_per_slice=n_ch).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    val_ds = SegClsDataset(val_records, augment=False, n_slices=args.n_slices,
                           soft_mask_factor=args.soft_mask_factor, n_ch_per_slice=n_ch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True)
    _, val_labels, val_probs = eval_auc(model, val_loader, device)
    _fpr, _tpr, _thr = roc_curve(val_labels, val_probs)
    youden_thr = float(_thr[np.argmax(_tpr - _fpr)])

    test_ds     = SegClsDataset(test_records, augment=False, n_slices=args.n_slices,
                                soft_mask_factor=args.soft_mask_factor, n_ch_per_slice=n_ch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True)

    test_auc, test_labels, test_probs = eval_auc(model, test_loader, device)

    cs_test = sum(r[1] for r in test_records)
    print(f"\n=== TEST SET ({len(test_records)} patients: csPCa={cs_test}, ciPCa={len(test_records)-cs_test}) ===")
    print(f"AUC-ROC     : {test_auc:.4f}  |  Youden thr: {youden_thr:.4f}")
    for label, thr in [("0.500 ", 0.5), (f"{youden_thr:.3f}*", youden_thr)]:
        preds = (test_probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_labels, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1   = f1_score(test_labels, preds, pos_label=1, zero_division=0)
        print(f"  thr={label} → sens={sens:.4f}  spec={spec:.4f}  F1={f1:.4f}"
              f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nw = min(4, os.cpu_count())
    set_seed(args.seed)

    cfg = {
        'run_name': os.path.basename(args.output_dir),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    records = load_labels()
    train_records, val_records, test_records = make_splits(
        records, val_size=args.val_size, test_size=args.test_size, seed=args.seed)

    cs_tr = sum(r[1] for r in train_records)
    cs_val = sum(r[1] for r in val_records)
    cs_te  = sum(r[1] for r in test_records)
    print(f"Split → train:{len(train_records)}(csPCa={cs_tr}) "
          f"val:{len(val_records)}(csPCa={cs_val}) "
          f"test:{len(test_records)}(csPCa={cs_te})")

    best_auc, ckpt_path = train_model(train_records, val_records, args, device, nw)
    evaluate_test(ckpt_path, test_records, val_records, args, device, nw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',         type=int,   default=150)
    parser.add_argument('--lr-backbone',    type=float, default=1e-5)
    parser.add_argument('--lr-head',        type=float, default=3e-4)
    parser.add_argument('--lr-factor',      type=float, default=0.5)
    parser.add_argument('--lr-patience',    type=int,   default=10)
    parser.add_argument('--weight-decay',   type=float, default=1e-4)
    parser.add_argument('--patience',       type=int,   default=30)
    parser.add_argument('--batch-size',     type=int,   default=4)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--n-slices',       type=int,   default=32)
    parser.add_argument('--dropout',        type=float, default=0.3)
    parser.add_argument('--seg-lambda',     type=float, default=0.5,
                        help='Weight of segmentation loss (0=cls only)')
    parser.add_argument('--attn-lambda',    type=float, default=0.0,
                        help='Weight of spatial attention supervision loss (0=off; csPCa only, train only)')
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--aug-scale',      action='store_true')
    parser.add_argument('--freeze-epochs',  type=int,   default=0)
    parser.add_argument('--freeze-bn',      action='store_true',
                        help='Keep backbone BN layers in eval mode throughout training')
    parser.add_argument('--backbone',       type=str,   default='small',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--val-size',       type=float, default=0.15)
    parser.add_argument('--test-size',      type=float, default=0.15)
    parser.add_argument('--scratch',          action='store_true',
                        help='Train from scratch — random init, uniform LR=lr-head')
    parser.add_argument('--soft-mask-factor', type=float, default=0.0,
                        help='Outside-gland intensity factor (0=hard zero, 0.1=soft)')
    parser.add_argument('--add-gland-ch',   action='store_true',
                        help='Use 3ch per slice (add explicit gland channel; default: 2ch T2W+ADC)')
    parser.add_argument('--intensity-aug',    action='store_true',
                        help='Enable intensity augmentation (noise, gamma, scale/shift; default: off)')
    parser.add_argument('--cs-oversample',   type=int, default=1,
                        help='Repeat csPCa samples N times with different aug (1=off, 5≈balance)')
    parser.add_argument('--cspca-weight',    type=float, default=0.0,
                        help='csPCa loss weight. 0=auto (inverse freq ≈3.5)')
    parser.add_argument('--focal-gamma',     type=float, default=0.0,
                        help='Focal loss gamma. 0=CrossEntropyLoss, >0=FocalLoss (e.g. 2.0)')
    parser.add_argument('--balanced-batch',  action='store_true',
                        help='Strict 50/50 pos/neg batch sampler (oversamples minority)')
    parser.add_argument('--stage2-epoch',    type=int, default=0,
                        help='Epoch to switch from balanced to natural distribution sampling')
    parser.add_argument('--scheduler',      type=str,   default='rlrop',
                        choices=['rlrop', 'cosine'],
                        help='LR scheduler: rlrop=ReduceLROnPlateau, cosine=CosineAnnealingLR')
    parser.add_argument('--output-dir',     type=str,   default='./output/seg_cls_base')
    main(parser.parse_args())
