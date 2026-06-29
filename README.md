# PI-CAI Prostate Cancer Classification

Binary classification of prostate MRI studies: **csPCa** (clinically significant, label 1) vs. **ciPCa** (clinically insignificant, label 0).

MedViT is pretrained on ImageNet and expects a 3-channel RGB input. This project compares three strategies for adapting it to 96-channel prostate MRI input (32 axial slices × 3 modalities: T2W, ADC, gland mask).

---

## Dataset

| | |
|---|---|
| Source | PI-CAI challenge, registered & preprocessed |
| Patients | 451 total — csPCa: 65 (14.4%), ciPCa: 386 (85.6%) |
| Input | `[B, 96, 224, 224]` — channel order: `[T2W₀, ADC₀, gland₀, T2W₁, ...]` |
| Split (seed=42) | Train 315 / Val 68 / Test 68, stratified |
| Class imbalance | WeightedRandomSampler + weighted CE loss (ciPCa 0.583 : csPCa 3.500) |

---

## Methods

### Method 1 · Weight Tiling

Pretrained first conv weights `[64, 3, 3, 3]` are tiled 32× along the channel dimension and divided by 32 to initialize a new `Conv2d(96→64, 3×3)`. The backbone otherwise remains intact.

```
[B, 96, 224, 224]
    ↓  Conv(96→64, 3×3)  ← pretrained weights tiled 32×, fine-tuned at lr=3e-4
    ↓  MedViT backbone   ← pretrained, fine-tuned at lr=1e-5
[B, 1024] → MLP head → [B, 2]
```

The 3×3 kernel learns cross-channel spatial patterns from the first epoch, with pretrained spatial priors intact. No new parameters beyond the inflated conv.

---

### Method 2 · Channel Adapter

A learnable `1×1 Conv(96→3)` is inserted before the original pretrained first conv, which is left completely unchanged.

```
[B, 96, 224, 224]
    ↓  Conv(96→3, 1×1)   ← randomly initialized, lr=1e-4
    ↓  Conv(3→64, 3×3)   ← pretrained, untouched, lr=1e-5
    ↓  MedViT backbone   ← pretrained, lr=1e-5
[B, 1024] → MLP head → [B, 2]
```

The adapter learns a linear channel projection; the pretrained conv processes the output as if it were a 3-channel image. Three separate LR groups (backbone / adapter / head) prevent the randomly initialized adapter from destabilizing pretrained weights.

> Small backbone (MedViT_small) consistently collapsed with a random adapter — noisy initial activations overwhelmed the limited capacity. MedViT_base (3× more parameters) proved robust.

---

### Method 3 · Slice Transformer

Each of the 32 slices is processed independently through the pretrained MedViT backbone (3-channel input unchanged). The 32 resulting feature vectors are aggregated by a Transformer encoder.

```
[B, 96, 224, 224]
    ↓  reshape → [B, 32, 3, 224, 224]
    ↓  MedViT backbone (shared, 3ch, lr=1e-5)   ← processed per patient to avoid OOM
[B, 32, 1024]
    ↓  + CLS token + positional embedding
    ↓  Transformer Encoder (2 layers, 8 heads)   ← lr=3e-4
    ↓  CLS output
[B, 1024] → MLP head → [B, 2]
```

The pretrained first conv is used exactly as designed — no modification. The Transformer captures inter-slice relationships. **Memory note**: processing `B×32` images simultaneously causes OOM; fixed by a per-patient loop (peak memory = 32 images regardless of batch size).

---

## Training

| Hyperparameter | Value |
|---|---|
| Backbone LR | 1e-5 |
| Head / new layers LR | 3e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=10 epochs) |
| Early stopping | patience=30 (resets after each LR drop) |
| Loss | CrossEntropyLoss (weighted) or FocalLoss (γ=2) |
| Optimizer | AdamW, weight_decay=1e-4 |
| Grad clip | max_norm=1.0 |

**ReduceLROnPlateau**: LR halves whenever val AUC fails to improve for 10 consecutive epochs. Early stopping patience resets after each LR reduction, giving the model a fresh chance at each new LR level.

---

## Results

Test set: 68 patients (10 csPCa, 58 ciPCa). Metrics at threshold=0.5.

| Method | Run | Test AUC | Sensitivity | Specificity | F1 | TP | FP | TN | FN |
|--------|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Weight Tiling | `focal_deep` | **0.919** | **0.90** | 0.78 | 0.563 | 9 | 13 | 45 | 1 |
| Weight Tiling | `base_ce` | 0.907 | 0.80 | **0.90** | **0.667** | 8 | 6 | 52 | 2 |
| Weight Tiling | `deeper_head` | 0.898 | 0.90 | 0.83 | 0.621 | 9 | 10 | 48 | 1 |
| Weight Tiling | `baseline` | 0.898 | 0.90 | 0.76 | 0.545 | 9 | 14 | 44 | 1 |
| Weight Tiling | `focal_base` | 0.822 | 0.80 | 0.83 | 0.571 | 8 | 10 | 48 | 2 |
| Channel Adapter | `adapter_base` | 0.881 | 0.80 | 0.84 | 0.593 | 8 | 9 | 49 | 2 |
| Slice Transformer | `slice_tf_small` | 0.764 | 0.20 | 0.95 | 0.267 | 2 | 3 | 55 | 8 |

**Weight Tiling dominates.** The inflated 3×3 conv retains spatial priors and requires no additional parameters beyond the channel dimension change. `focal_deep` achieves the highest AUC (0.919) and sensitivity (0.90, misses 1 of 10 cancers), preferred clinically. `base_ce` produces the fewest false positives (6 FP) for the best F1.

**Channel Adapter** is viable but requires a larger backbone for stability, and the 1×1 adapter cannot capture spatial cross-channel patterns the way a 3×3 inflated conv can.

**Slice Transformer** overfits despite the elegant design — the Transformer encoder (~6M new params) cannot be reliably trained on 45 positive examples. Val AUC peaked at 0.824 but test AUC was only 0.764, with sensitivity collapsing to 0.20.

---

## Grad-CAM

Grad-CAM highlights spatial regions driving the csPCa prediction (T2W center slice). Each row: **T2W** | **heatmap** | **overlay**.

### focal_deep — Correctly predicted csPCa patients (TP: 9/10)

| Patient | Grad-CAM |
|---------|----------|
| 10043_1000043 | ![](ProstateCls/weight_tiling/figures/focal_deep/gradcam/gradcam_10043_1000043.png) |
| 10257_1000261 | ![](ProstateCls/weight_tiling/figures/focal_deep/gradcam/gradcam_10257_1000261.png) |
| 10398_1000404 | ![](ProstateCls/weight_tiling/figures/focal_deep/gradcam/gradcam_10398_1000404.png) |
| 10463_1000471 | ![](ProstateCls/weight_tiling/figures/focal_deep/gradcam/gradcam_10463_1000471.png) |
| 10486_1000494 | ![](ProstateCls/weight_tiling/figures/focal_deep/gradcam/gradcam_10486_1000494.png) |
| 10549_1000561 | ![](ProstateCls/weight_tiling/figures/focal_deep/gradcam/gradcam_10549_1000561.png) |
| 10558_1000570 | ![](ProstateCls/weight_tiling/figures/focal_deep/gradcam/gradcam_10558_1000570.png) |
| 10568_1000580 | ![](ProstateCls/weight_tiling/figures/focal_deep/gradcam/gradcam_10568_1000580.png) |
| 10589_1000603 | ![](ProstateCls/weight_tiling/figures/focal_deep/gradcam/gradcam_10589_1000603.png) |

### Method comparison — Patient 10043_1000043 (TP in all methods)

| Method | Grad-CAM |
|--------|----------|
| Weight Tiling (`focal_deep`) | ![](ProstateCls/weight_tiling/figures/focal_deep/gradcam/gradcam_10043_1000043.png) |
| Channel Adapter (`adapter_base`) | ![](ProstateCls/channel_adapter/figures/adapter_base/gradcam/gradcam_10043_1000043.png) |

---

## Reproducing

```bash
cd ProstateCls/

# Weight Tiling
cd weight_tiling
bash 0_submit.sh focal_deep "--focal-gamma 2.0 --head-depth 2"
bash 0_submit.sh base_ce    "--backbone base"
bash 1_submit_vis.sh focal_deep

# Channel Adapter
cd ../channel_adapter
bash 0_submit.sh adapter_base   # default: backbone=base

# Slice Transformer
cd ../slice_transformer
bash 0_submit.sh slice_tf_small
bash 0_submit.sh slice_tf_mean "--pooling mean"
```

Python: `/N/slate/ohjiye/envs/medvit/bin/python3`
