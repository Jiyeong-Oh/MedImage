"""
Training script for PI-CAI csPCa classification — channel adapter variant.
Uses ChannelAdaptedConv: learnable 1×1 Conv(96→3) + pretrained Conv(3→64).

Differential LR:
  - stem.0.conv.adapter  → lr_head  (new, randomly init)
  - rest of backbone     → lr_backbone (pretrained)
  - proj_head            → lr_head
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

# dataset from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset import PatientVolumeDataset, load_labels

# local channel-adapter model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model


class FocalLoss(nn.Module):
    """Multi-class focal loss: FL = -α_t * (1 - p_t)^γ * log(p_t)"""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        p_t   = torch.exp(log_p).gather(1, targets.unsqueeze(1)).squeeze(1)
        loss  = -(1 - p_t) ** self.gamma * log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.weight is not None:
            loss = self.weight[targets] * loss
        return loss.mean()


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
    ds = PatientVolumeDataset(records, augment=False, aug_strong=False, n_slices=args.n_slices)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=nw, pin_memory=True)


def eval_auc(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    if len(set(all_labels)) < 2:
        return float('nan'), np.array(all_labels), np.array(all_probs)
    return roc_auc_score(all_labels, all_probs), np.array(all_labels), np.array(all_probs)


def make_optimizer(model, lr_backbone, lr_head, lr_adapter, weight_decay):
    adapter_params, proj_params, backbone_params = [], [], []
    for name, param in model.named_parameters():
        if 'stem.0.conv.adapter' in name:
            adapter_params.append(param)
        elif name.startswith('proj_head'):
            proj_params.append(param)
        else:
            backbone_params.append(param)
    print(f"  Optimizer: backbone LR={lr_backbone}, adapter LR={lr_adapter}, proj_head LR={lr_head}")
    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': adapter_params,  'lr': lr_adapter},
        {'params': proj_params,     'lr': lr_head},
    ], weight_decay=weight_decay)


def train_model(train_records, val_records, args, device, nw):
    print(f"\n{'='*50}")
    cs_tr  = sum(r[1] for r in train_records)
    cs_val = sum(r[1] for r in val_records)
    print(f"Train: {len(train_records)} (csPCa={cs_tr}, ciPCa={len(train_records)-cs_tr})")
    print(f"Val:   {len(val_records)}   (csPCa={cs_val}, ciPCa={len(val_records)-cs_val})")

    train_ds = PatientVolumeDataset(train_records,
                                    augment=not args.aug_strong,
                                    aug_strong=args.aug_strong,
                                    n_slices=args.n_slices)

    train_labels_arr = np.array([r[1] for r in train_records], dtype=np.float64)
    counts  = np.bincount(train_labels_arr.astype(int))
    weights = 1.0 / counts[train_labels_arr.astype(int)]

    sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=nw, pin_memory=True)
    val_loader   = make_val_loader(val_records, args, nw)

    model = build_model(num_classes=2, pretrained=True, n_slices=args.n_slices,
                        head_depth=args.head_depth, backbone=args.backbone,
                        adapter_mid_ch=args.adapter_mid_ch)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_labels_np = np.array([r[1] for r in train_records])
    class_weights = compute_class_weights(train_labels_np, device, args.cspca_weight)
    if args.focal_gamma > 0:
        criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma)
        print(f"  Loss: FocalLoss(gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Loss: CrossEntropyLoss")
    ckpt_path       = os.path.join(args.output_dir, 'best.pth')
    best_auc, best_epoch, patience_count = 0.0, 0, 0

    core      = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = make_optimizer(core, args.lr_backbone, args.lr_head, args.lr_adapter, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.lr_factor,
        patience=args.lr_patience, min_lr=1e-7)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, lbls, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        val_auc, _, _ = eval_auc(core, val_loader, device)
        old_lr_hd = optimizer.param_groups[2]['lr']
        scheduler.step(val_auc)
        new_lr_hd = optimizer.param_groups[2]['lr']

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"loss: {running_loss/len(train_records):.4f} | "
              f"val AUC: {val_auc:.4f} | "
              f"LR bb={optimizer.param_groups[0]['lr']:.1e} "
              f"adapter={optimizer.param_groups[1]['lr']:.1e} "
              f"hd={new_lr_hd:.1e}")

        if new_lr_hd < old_lr_hd:
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


def evaluate_test(ckpt_path, test_records, args, device, nw):
    test_loader = make_val_loader(test_records, args, nw)
    model = build_model(num_classes=2, pretrained=False, n_slices=args.n_slices,
                        head_depth=args.head_depth, backbone=args.backbone,
                        adapter_mid_ch=args.adapter_mid_ch)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model = model.to(device)
    test_auc, test_labels, test_probs = eval_auc(model, test_loader, device)

    preds = (test_probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1   = f1_score(test_labels, preds, pos_label=1, zero_division=0)

    cs_test = sum(r[1] for r in test_records)
    print(f"\n=== TEST SET ({len(test_records)} patients: "
          f"csPCa={cs_test}, ciPCa={len(test_records)-cs_test}) ===")
    print(f"AUC-ROC     : {test_auc:.4f}")
    print(f"Sensitivity : {sens:.4f}")
    print(f"Specificity : {spec:.4f}")
    print(f"F1 (csPCa)  : {f1:.4f}")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")


def save_config(args, output_dir):
    tmp = build_model(num_classes=2, pretrained=False, n_slices=args.n_slices,
                      head_depth=args.head_depth, backbone=args.backbone,
                      adapter_mid_ch=args.adapter_mid_ch)
    head_layers = []
    for m in tmp.proj_head:
        if isinstance(m, nn.Linear):
            head_layers.append(f"Linear({m.in_features}→{m.out_features})")
        elif isinstance(m, nn.Dropout):
            head_layers.append(f"Dropout(p={m.p})")
        else:
            head_layers.append(type(m).__name__)
    del tmp

    config = {
        "run_name":     os.path.basename(output_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training":     vars(args),
        "model": {
            "backbone":        f"MedViT_{args.backbone}",
            "stem":            "ChannelAdaptedConv (1×1 adapter + pretrained 3×3)",
            "input_ch":        args.n_slices * 3,
            "input_size":      224,
            "head":            head_layers,
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
    evaluate_test(ckpt_path, test_records, args, device, nw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',          type=int,   default=150)
    parser.add_argument('--lr-backbone',     type=float, default=1e-5)
    parser.add_argument('--lr-head',         type=float, default=3e-4)
    parser.add_argument('--lr-adapter',      type=float, default=3e-4,
                        help='Adapter-specific LR (separate from proj_head). Default=lr_head')
    parser.add_argument('--lr-factor',       type=float, default=0.5,
                        help='ReduceLROnPlateau factor')
    parser.add_argument('--lr-patience',     type=int,   default=10,
                        help='Epochs without val AUC improvement before LR reduction')
    parser.add_argument('--adapter-mid-ch',  type=int,   default=0,
                        help='Intermediate channels in 2-layer adapter. 0=single 1x1 layer')
    parser.add_argument('--weight-decay',    type=float, default=1e-4)
    parser.add_argument('--patience',     type=int,   default=30,
                        help='Early stopping patience (increased to allow recovery after LR decay)')
    parser.add_argument('--batch-size',   type=int,   default=8)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--n-slices',     type=int,   default=32)
    parser.add_argument('--cspca-weight', type=float, default=0.0,
                        help='csPCa loss weight. 0=auto (inverse freq ≈3.5)')
    parser.add_argument('--focal-gamma',  type=float, default=0.0,
                        help='Focal loss gamma. 0=CrossEntropyLoss, >0=FocalLoss (e.g. 2.0)')
    parser.add_argument('--head-depth',   type=int,   default=2,
                        help='Head MLP layers: 2=1024→512→256→2, 3=→128→2')
    parser.add_argument('--backbone',     type=str,   default='small',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--aug-strong',   action='store_true')
    parser.add_argument('--val-size',     type=float, default=0.15)
    parser.add_argument('--test-size',    type=float, default=0.15)
    parser.add_argument('--t-max',        type=int,   default=150)
    parser.add_argument('--output-dir',   type=str,   default='./output/adapter')
    main(parser.parse_args())
