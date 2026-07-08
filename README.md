# MedImage

Medical image deep-learning research workspace. Three independent projects:

| Project | Task | Dataset |
|---------|------|---------|
| [ProstateCls/](ProstateCls/) | Prostate cancer binary classification | PI-CAI (1,131 patients) |
| [MedViT/](MedViT/) | Medical image classification | MedMNIST-2D / ImageNet |
| [SwinSegNet/](SwinSegNet/) | Colon gland segmentation | GlaS |

---

## PI-CAI Prostate Cancer Classification

Binary classification of prostate MRI studies: **csPCa** (clinically significant, ISUP ≥ 2) vs. **ciPCa** (ISUP 0–1).

**Input:** T2W MRI · ADC map · Gland mask — 32 axial slices per patient  
**Class imbalance:** csPCa 15.8% (179/1,131) · Train/Val/Test = 791/170/170

### Methods

| # | Method | Architecture | Key idea |
|---|--------|--------------|----------|
| 1 | [Weight Tiling](ProstateCls/weight_tiling/) | MedViT_small + MLP | 32 slices × 3 ch = 96-ch input; first conv tiled ÷32 |
| 2 | [Channel Adapter](ProstateCls/channel_adapter/) | MedViT + adapter | Learnable 96-ch recalibration before backbone |
| 3 | [CNN Head](ProstateCls/cnn_head/) | MedViT + CBAM | SE + spatial attention on 7×7 feature map |
| 4 | [Mask Guided](ProstateCls/mask_guided/) | MedViT + gland gate | Gland mask downsampled to 7×7 gates backbone features |
| 5 | [Seg+Cls](ProstateCls/seg_cls/) | MedViT + U-Net | Tumor segmentation decoder; tumor gate on cls features |
| 6 | [MIL-ABMIL](ProstateCls/mil_abmil/) | MedViT per-slice + gated attention | Per-slice bags; ABMIL aggregation |

### Best Results (Test set, n=170)

| Method | Run | Test AUC | Sensitivity | Specificity |
|--------|-----|:--------:|:-----------:|:-----------:|
| Weight Tiling | wt_fbn_fbal | **0.800** | 55.6% | 78.3% |
| Channel Adapter | ca_os5 | 0.762 | 37.0% | 84.6% |
| MIL-ABMIL | baseline | 0.759 | 92.6% | 42.0% |
| CNN Head | cnn_fbn_focal | 0.739 | 40.7% | 86.7% |
| Mask Guided | mg_fbn_bal | 0.710 | 48.1% | 80.4% |
| Seg+Cls | — | *pending* | — | — |

→ See [ProstateCls/README.md](ProstateCls/README.md) for full results and method details.

### Best Model Visualizations (`wt_fbn_fbal`, Test AUC 0.800)

| ROC & PR Curve | Learning Curve | Confusion Matrix |
|:-:|:-:|:-:|
| ![ROC](ProstateCls/weight_tiling/figures/wt_fbn_fbal/roc_pr_curve.png) | ![LR](ProstateCls/weight_tiling/figures/wt_fbn_fbal/learning_curve.png) | ![CM](ProstateCls/weight_tiling/figures/wt_fbn_fbal/confusion_matrix.png) |

**Grad-CAM — Patient 10085 (csPCa True Positive)**

| Slice 11 | Slice 12 |
|:-:|:-:|
| ![z11](ProstateCls/weight_tiling/figures/wt_fbn_fbal/gradcam/gradcam_10085_1000085_z11.png) | ![z12](ProstateCls/weight_tiling/figures/wt_fbn_fbal/gradcam/gradcam_10085_1000085_z12.png) |

---

## MedViT

CNN-Transformer hybrid backbone for medical image classification. See [MedViT/](MedViT/) for architecture details and training scripts.

## SwinSegNet

Swin Transformer U-Net for colon gland segmentation (GlaS dataset). See [SwinSegNet/](SwinSegNet/) for notebooks and model code.
