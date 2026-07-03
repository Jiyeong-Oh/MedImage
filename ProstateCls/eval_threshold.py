"""
Optimal-threshold evaluator.
Loads a saved checkpoint, finds the Youden-J-optimal threshold on the val set,
then reports full metrics at both 0.5 and the optimal threshold on the test set.
Also supports simple ensemble (average probabilities) across multiple checkpoints.

Usage (single model):
    python eval_threshold.py --method weight_tiling --run wt_lrbal

Usage (ensemble):
    python eval_threshold.py --ensemble \
        weight_tiling/wt_lrbal weight_tiling/wt_warmup slice_wise/slice_wise_freeze
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from dataset import load_labels, PatientVolumeDataset

METHOD_ROOTS = {
    'weight_tiling':   os.path.join(BASE, 'weight_tiling'),
    'channel_adapter': os.path.join(BASE, 'channel_adapter'),
    'mask_guided':     os.path.join(BASE, 'mask_guided'),
    'slice_transformer': os.path.join(BASE, 'slice_transformer'),
    'slice_wise':      os.path.join(BASE, 'slice_wise'),
}


def load_method_build_fn(method):
    model_path = os.path.join(METHOD_ROOTS[method], 'model.py')
    spec = importlib.util.spec_from_file_location(f'{method}_model', model_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model


def make_splits(records, val_size, test_size, seed=42):
    labels = np.array([r[1] for r in records])
    tv, test, lbl_tv, _ = train_test_split(
        records, labels, test_size=test_size, stratify=labels, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val, _, _ = train_test_split(
        tv, lbl_tv, test_size=rel_val, stratify=lbl_tv, random_state=seed)
    return train, val, test


def get_probs(model, records, cfg, device, method):
    if method == 'mask_guided':
        spec = importlib.util.spec_from_file_location(
            'mask_dataset', os.path.join(METHOD_ROOTS[method], 'dataset.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ds = mod.MaskGuidedDataset(records, augment=False)
    elif method == 'slice_wise':
        spec = importlib.util.spec_from_file_location(
            'sw_dataset', os.path.join(METHOD_ROOTS[method], 'dataset.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ds = mod.SliceWiseDataset(records, augment=False)
    else:
        ds = PatientVolumeDataset(records, augment=False,
                                  n_slices=cfg['n_slices'])

    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if method == 'mask_guided':
                imgs, masks, lbls, _ = batch
                logits = model(imgs.to(device), masks.to(device))
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            elif method == 'slice_wise':
                imgs, lbls, _ = batch
                logits = model(imgs.to(device))           # [B, N, 2]
                probs  = torch.sigmoid(logits[:, :, 1].max(dim=1)[0]).cpu().numpy()
            else:
                imgs, lbls, _ = batch
                probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(lbls.numpy())
    return np.array(all_labels), np.array(all_probs)


def report(labels, probs, threshold, tag):
    auc  = roc_auc_score(labels, probs)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = f1_score(labels, pred, pos_label=1, zero_division=0)
    print(f"  [{tag}]  thr={threshold:.3f}  AUC={auc:.4f}  "
          f"Sens={sens:.4f}  Spec={spec:.4f}  F1={f1:.4f}  "
          f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return {'auc': auc, 'sens': sens, 'spec': spec, 'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'threshold': threshold}


def youden_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr + (1 - fpr) - 1
    return float(thresholds[np.argmax(j)])


def f1_threshold(labels, probs):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        f1 = f1_score(labels, (probs >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def eval_one(method, run, device, records=None):
    if records is None:
        records = load_labels()
    cfg_path = os.path.join(METHOD_ROOTS[method], 'output', run, 'config.json')
    ckpt     = os.path.join(METHOD_ROOTS[method], 'output', run, 'best.pth')
    if not os.path.exists(ckpt):
        print(f"  No checkpoint: {ckpt}")
        return None, None

    with open(cfg_path) as f:
        cfg = json.load(f)['training']

    build_model = load_method_build_fn(method)
    if method == 'channel_adapter':
        model = build_model(num_classes=2, pretrained=False,
                            n_slices=cfg['n_slices'],
                            head_depth=cfg['head_depth'],
                            backbone=cfg['backbone'],
                            adapter_mid_ch=cfg.get('adapter_mid_ch', 0))
    elif method == 'slice_transformer':
        model = build_model(num_classes=2, pretrained=False,
                            n_slices=cfg['n_slices'],
                            head_depth=cfg['head_depth'],
                            backbone=cfg['backbone'],
                            nhead=cfg.get('nhead', 8),
                            num_layers=cfg.get('num_layers', 2),
                            dim_feedforward=cfg.get('dim_feedforward', 2048),
                            tf_dropout=cfg.get('tf_dropout', 0.1),
                            pooling=cfg.get('pooling', 'cls'))
    else:
        model = build_model(num_classes=2, pretrained=False,
                            n_slices=cfg['n_slices'],
                            head_depth=cfg['head_depth'],
                            backbone=cfg['backbone'])

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model = model.to(device)

    _, val, test = make_splits(records,
                               val_size=cfg['val_size'],
                               test_size=cfg['test_size'],
                               seed=cfg['seed'])

    val_labels,  val_probs  = get_probs(model, val,  cfg, device, method)
    test_labels, test_probs = get_probs(model, test, cfg, device, method)

    t_youden = youden_threshold(val_labels, val_probs)
    t_f1     = f1_threshold(val_labels, val_probs)

    print(f"\n{'='*60}")
    print(f"  {method}/{run}")
    print(f"  val AUC={roc_auc_score(val_labels, val_probs):.4f}  "
          f"→ optimal thr (Youden)={t_youden:.3f}  (F1)={t_f1:.3f}")
    print(f"  --- TEST ---")
    r_05     = report(test_labels, test_probs, 0.5,      '@0.5')
    r_youden = report(test_labels, test_probs, t_youden, '@Youden')
    r_f1     = report(test_labels, test_probs, t_f1,     '@F1opt')
    return test_labels, test_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',   type=str, default='weight_tiling')
    parser.add_argument('--run',      type=str, default='wt_lrbal')
    parser.add_argument('--ensemble', nargs='+', default=None,
                        metavar='METHOD/RUN',
                        help='e.g. weight_tiling/wt_lrbal weight_tiling/wt_warmup')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records = load_labels()

    if args.ensemble:
        print(f"\nENSEMBLE of {len(args.ensemble)} models")
        all_test_labels = None
        all_test_probs  = []
        for item in args.ensemble:
            method, run = item.split('/')
            lbl, probs = eval_one(method, run, device, records)
            if probs is not None:
                all_test_probs.append(probs)
                all_test_labels = lbl

        if all_test_probs:
            ens_probs = np.mean(all_test_probs, axis=0)
            # need val probs for ensemble threshold — use the first model's val split
            method0, run0 = args.ensemble[0].split('/')
            cfg_path = os.path.join(METHOD_ROOTS[method0], 'output', run0, 'config.json')
            with open(cfg_path) as f:
                cfg = json.load(f)['training']
            _, val, test = make_splits(records, cfg['val_size'], cfg['test_size'], cfg['seed'])

            # rebuild all models to get val probs for ensemble threshold
            val_probs_list = []
            for item in args.ensemble:
                method, run = item.split('/')
                build_model = load_method_build_fn(method)
                ckpt = os.path.join(METHOD_ROOTS[method], 'output', run, 'best.pth')
                with open(os.path.join(METHOD_ROOTS[method], 'output', run, 'config.json')) as f:
                    cfg2 = json.load(f)['training']
                if method == 'channel_adapter':
                    m = build_model(num_classes=2, pretrained=False, n_slices=cfg2['n_slices'],
                                    head_depth=cfg2['head_depth'], backbone=cfg2['backbone'],
                                    adapter_mid_ch=cfg2.get('adapter_mid_ch', 0))
                elif method == 'slice_transformer':
                    m = build_model(num_classes=2, pretrained=False, n_slices=cfg2['n_slices'],
                                    head_depth=cfg2['head_depth'], backbone=cfg2['backbone'],
                                    nhead=cfg2.get('nhead', 8), num_layers=cfg2.get('num_layers', 2),
                                    dim_feedforward=cfg2.get('dim_feedforward', 2048),
                                    tf_dropout=cfg2.get('tf_dropout', 0.1),
                                    pooling=cfg2.get('pooling', 'cls'))
                else:
                    m = build_model(num_classes=2, pretrained=False, n_slices=cfg2['n_slices'],
                                    head_depth=cfg2['head_depth'], backbone=cfg2['backbone'])
                m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
                m = m.to(device)
                vl, vp = get_probs(m, val, cfg2, device, method)
                val_probs_list.append(vp)

            ens_val_probs  = np.mean(val_probs_list, axis=0)
            t_youden = youden_threshold(vl, ens_val_probs)
            t_f1     = f1_threshold(vl, ens_val_probs)

            print(f"\n{'='*60}")
            print(f"  ENSEMBLE RESULTS ({len(args.ensemble)} models)")
            print(f"  val optimal thr (Youden)={t_youden:.3f}  (F1)={t_f1:.3f}")
            print(f"  --- TEST ---")
            report(all_test_labels, ens_probs, 0.5,      'ens@0.5')
            report(all_test_labels, ens_probs, t_youden, 'ens@Youden')
            report(all_test_labels, ens_probs, t_f1,     'ens@F1opt')
    else:
        eval_one(args.method, args.run, device, records)


if __name__ == '__main__':
    main()
