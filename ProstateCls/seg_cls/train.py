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
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset import load_labels
from seg_cls.dataset import SegClsDataset
from seg_cls.model import build_model


# ── Losses ────────────────────────────────────────────────────────────────────

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

    train_ds = SegClsDataset(train_records, augment=True, aug_scale=args.aug_scale,
                              n_slices=args.n_slices)
    val_ds   = SegClsDataset(val_records,   augment=False, n_slices=args.n_slices)

    train_labels = np.array([r[1] for r in train_records])
    weights = 1.0 / np.bincount(train_labels)[train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)

    model = build_model(num_classes=2, pretrained=True, n_slices=args.n_slices,
                        dropout=args.dropout, backbone=args.backbone).to(device)

    counts = np.bincount(train_labels); total = len(train_labels)
    w = torch.tensor([total/(2*c) for c in counts], dtype=torch.float32).to(device)
    print(f"  Class weights: ciPCa={w[0]:.3f}  csPCa={w[1]:.3f}")

    cls_criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)

    optimizer = make_optimizer(model, args.lr_backbone, args.lr_head, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience, min_lr=1e-7)

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

        model.train()
        total_loss = cls_l = seg_l = 0.0
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

            loss = loss_cls + args.seg_lambda * loss_seg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            cls_l      += loss_cls.item()
            seg_l      += loss_seg.item()

        val_auc, _, _ = eval_auc(model, val_loader, device)
        scheduler.step(val_auc)
        lr_bb = optimizer.param_groups[0]['lr']
        lr_hd = optimizer.param_groups[1]['lr']
        print(f"  Epoch {epoch:3d}/{args.epochs} | loss:{total_loss/len(train_loader):.4f}"
              f" cls:{cls_l/len(train_loader):.4f} seg:{seg_l/len(train_loader):.4f}"
              f" | val AUC:{val_auc:.4f} | LR bb={lr_bb:.1e} hd={lr_hd:.1e}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), ckpt_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"Best val AUC: {best_auc:.4f}")
    return best_auc, ckpt_path


# ── Test evaluation ───────────────────────────────────────────────────────────

def evaluate_test(ckpt_path, test_records, args, device, nw):
    model = build_model(num_classes=2, pretrained=False, n_slices=args.n_slices,
                        dropout=args.dropout, backbone=args.backbone).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    test_ds     = SegClsDataset(test_records, augment=False, n_slices=args.n_slices)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True)

    test_auc, labels, probs = eval_auc(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    cs_test = sum(r[1] for r in test_records)
    print(f"\n=== TEST SET ({len(test_records)} patients: csPCa={cs_test}, ciPCa={len(test_records)-cs_test}) ===")
    print(f"AUC-ROC     : {test_auc:.4f}")
    print(f"Sensitivity : {tp/(tp+fn):.4f}")
    print(f"Specificity : {tn/(tn+fp):.4f}")
    print(f"F1 (csPCa)  : {f1_score(labels, preds, pos_label=1):.4f}")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")


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
    evaluate_test(ckpt_path, test_records, args, device, nw)


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
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--aug-scale',      action='store_true')
    parser.add_argument('--freeze-epochs',  type=int,   default=0)
    parser.add_argument('--backbone',       type=str,   default='small',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--val-size',       type=float, default=0.15)
    parser.add_argument('--test-size',      type=float, default=0.15)
    parser.add_argument('--output-dir',     type=str,   default='./output/seg_cls_base')
    main(parser.parse_args())
