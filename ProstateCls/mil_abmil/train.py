"""
MIL-ABMIL training for PI-CAI csPCa classification.
Input per patient: [n_slices, nch, H, W] (each slice processed independently).
Aggregation: Gated Attention-Based MIL → patient-level logits + per-slice attn weights.
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

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)  # mil_abmil/ — for local dataset.py and model.py
from dataset import MILDataset, load_labels  # mil_abmil/dataset.py (re-exports load_labels)
from model import build_model                # mil_abmil/model.py


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight          = weight
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_p   = F.log_softmax(logits, dim=1)
        p_t     = torch.exp(log_p).gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w = (1 - p_t) ** self.gamma
        loss_ce = F.cross_entropy(logits, targets, weight=self.weight,
                                  label_smoothing=self.label_smoothing, reduction='none')
        return (focal_w * loss_ce).mean()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_splits(records, val_size, test_size, seed):
    labels = np.array([r[1] for r in records])
    train_val, test, lbl_tv, _ = train_test_split(
        records, labels, test_size=test_size, stratify=labels, random_state=seed)
    rv = val_size / (1.0 - test_size)
    train, val, _, _ = train_test_split(
        train_val, lbl_tv, test_size=rv, stratify=lbl_tv, random_state=seed)
    return train, val, test


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


def make_dataset(records, args, augment=False):
    nch = 3 if args.add_gland_ch else 2
    os  = args.cs_oversample if augment else 1
    return MILDataset(records, augment=augment, n_slices=args.n_slices,
                      n_ch_per_slice=nch, input_size=224, cs_oversample=os)


def eval_auc(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for bags, _t2d, _ts, labels, _ in loader:
            logits, _ = model(bags.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    if len(set(all_labels)) < 2:
        return float('nan'), np.array(all_labels), np.array(all_probs)
    return roc_auc_score(all_labels, all_probs), np.array(all_labels), np.array(all_probs)


def make_optimizer(model, lr_backbone, lr_head, weight_decay):
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if (name.startswith('backbone.stem.0.conv')
                or name.startswith('aggregator.')):
            head_params.append(param)
        else:
            backbone_params.append(param)
    print(f"  Optimizer: backbone LR={lr_backbone}, head(stem+aggregator) LR={lr_head}")
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

    nch      = 3 if args.add_gland_ch else 2
    train_ds = make_dataset(train_records, args, augment=True)
    val_ds   = make_dataset(val_records,   args, augment=False)

    train_labels_arr = np.array([s[1] for s in train_ds.samples], dtype=np.float64)
    counts  = np.bincount(train_labels_arr.astype(int))
    cs_eff  = counts[1] if len(counts) > 1 else 0
    print(f"  Effective train size: {len(train_ds)} (csPCa={cs_eff}, ciPCa={counts[0]})")

    if args.cs_oversample > 1:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=nw, pin_memory=True)
    elif args.balanced_batch:
        bbs = BalancedBatchSampler(train_labels_arr.astype(int).tolist(), args.batch_size)
        train_loader = DataLoader(train_ds, batch_sampler=bbs,
                                  num_workers=nw, pin_memory=True)
    else:
        weights = 1.0 / counts[train_labels_arr.astype(int)]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)

    model = build_model(nch=nch, backbone=args.backbone,
                        abmil_hidden=args.abmil_hidden,
                        abmil_dropout=args.abmil_dropout,
                        pretrained=not args.scratch,
                        spatial_attn=args.spatial_attn)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_labels_np = np.array([r[1] for r in train_records])
    class_weights   = compute_class_weights(train_labels_np, device, args.cspca_weight)
    if args.focal_gamma > 0:
        criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma,
                              label_smoothing=args.label_smoothing)
        print(f"  Loss: FocalLoss(gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                        label_smoothing=args.label_smoothing)
        print(f"  Loss: CrossEntropyLoss")

    ckpt_path                  = os.path.join(args.output_dir, 'best.pth')
    best_auc, best_epoch, patience_count = 0.0, 0, 0

    core = model.module if isinstance(model, nn.DataParallel) else model
    if args.scratch:
        optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr_head,
                                       weight_decay=args.weight_decay)
    else:
        optimizer = make_optimizer(core, args.lr_backbone, args.lr_head, args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            for m in core.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        running_loss = attn_l = 0.0
        for bags, tumor_2d, tumor_slices, lbls, _ in train_loader:
            bags, lbls = bags.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits, attn_w = model(bags)
            loss = criterion(logits, lbls)

            if args.attn_lambda > 0:
                # Supervise attention weights with per-slice tumor presence
                # csPCa: soft target = normalized tumor_slices (sum=1)
                # ciPCa: soft target = uniform 1/S
                S       = attn_w.shape[1]
                ts      = tumor_slices.to(device)               # [B, S]
                ts_sum  = ts.sum(dim=1, keepdim=True).clamp(min=1.0)
                target  = torch.where(lbls.unsqueeze(1).bool(),
                                      ts / ts_sum,
                                      torch.full_like(ts, 1.0 / S))  # [B, S]
                loss_attn = F.kl_div(attn_w.log().clamp(min=-10), target, reduction='batchmean')
                loss = loss + args.attn_lambda * loss_attn
                attn_l += loss_attn.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * bags.size(0)

        val_auc, _, _ = eval_auc(core, val_loader, device)
        scheduler.step()
        lr_str = (f"LR bb={optimizer.param_groups[0]['lr']:.1e} "
                  f"hd={optimizer.param_groups[-1]['lr']:.1e}"
                  if len(optimizer.param_groups) > 1
                  else f"LR={optimizer.param_groups[0]['lr']:.1e}")
        attn_str = f" attn:{attn_l/len(train_loader):.4f}" if args.attn_lambda > 0 else ""
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"loss: {running_loss/len(train_records):.4f}{attn_str} | "
              f"val AUC: {val_auc:.4f} | {lr_str}")

        if val_auc > best_auc:
            best_auc, best_epoch, patience_count = val_auc, epoch, 0
            torch.save(core.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"\nBest val AUC: {best_auc:.4f} at epoch {best_epoch}")
    return best_auc, ckpt_path


def evaluate_test(ckpt_path, test_records, val_records, args, device, nw):
    nch  = 3 if args.add_gland_ch else 2
    test_ds = make_dataset(test_records, args, augment=False)
    val_ds  = make_dataset(val_records,  args, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True)

    model = build_model(nch=nch, backbone=args.backbone,
                        abmil_hidden=args.abmil_hidden,
                        abmil_dropout=args.abmil_dropout,
                        pretrained=False,
                        spatial_attn=args.spatial_attn)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False),
                          strict=False)
    model = model.to(device)

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

    return test_auc, test_labels, test_probs, val_labels, val_probs


def save_config(args, output_dir):
    cfg = vars(args).copy()
    cfg['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',        type=int,   default=150)
    p.add_argument('--batch-size',    type=int,   default=4)
    p.add_argument('--lr-backbone',   type=float, default=1e-5)
    p.add_argument('--lr-head',       type=float, default=3e-4)
    p.add_argument('--weight-decay',  type=float, default=1e-4)
    p.add_argument('--patience',      type=int,   default=30)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--n-slices',      type=int,   default=32)
    p.add_argument('--val-size',      type=float, default=0.15)
    p.add_argument('--test-size',     type=float, default=0.15)
    p.add_argument('--backbone',      type=str,   default='small',
                   choices=['small', 'base', 'large'])
    p.add_argument('--abmil-hidden',  type=int,   default=256)
    p.add_argument('--abmil-dropout', type=float, default=0.25)
    p.add_argument('--add-gland-ch',  action='store_true')
    p.add_argument('--focal-gamma',   type=float, default=0.0)
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--cspca-weight',  type=float, default=0.0)
    p.add_argument('--freeze-bn',     action='store_true', default=True)
    p.add_argument('--no-freeze-bn',  dest='freeze_bn', action='store_false')
    p.add_argument('--scratch',       action='store_true')
    p.add_argument('--attn-lambda',   type=float, default=0.0)
    p.add_argument('--cs-oversample', type=int,   default=1,
                   help='Repeat csPCa bags N times with different aug (1=off, 5≈balance)')
    p.add_argument('--balanced-batch', action='store_true',
                   help='Use BalancedBatchSampler (50/50 pos/neg per batch)')
    p.add_argument('--spatial-attn',  action='store_true',
                   help='2.5D: per-slice spatial attention (7x7) + ABMIL over slices')
    p.add_argument('--output-dir',    type=str,   default='output/run')
    args = p.parse_args()

    if args.spatial_attn:
        args.add_gland_ch = True  # gland channel (ch2) required for spatial mask

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nw     = min(4, os.cpu_count() or 1)
    print(f"Device: {device}  workers: {nw}")

    records          = load_labels()
    train_r, val_r, test_r = make_splits(records, args.val_size, args.test_size, args.seed)
    save_config(args, args.output_dir)

    best_auc, ckpt_path = train_model(train_r, val_r, args, device, nw)

    nch  = 3 if args.add_gland_ch else 2
    result = evaluate_test(ckpt_path, test_r, val_r, args, device, nw)
    test_auc = result[0]

    summary = {'best_val_auc': best_auc, 'test_auc': test_auc}
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary}")


if __name__ == '__main__':
    main()
