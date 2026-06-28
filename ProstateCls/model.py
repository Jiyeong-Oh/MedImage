"""
MedViT_small adapted for binary prostate cancer classification.
Input:  [B, n_slices*3, 224, 224]  (depth-as-channel, default 32 slices → 96ch)
Output: [B, 2]  (logits for ciPCa=0, csPCa=1)

Weight inflation: first conv 3→96ch via tiling pretrained weights (÷n_slices).
All other backbone layers use pretrained weights unchanged.
"""
import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/geode3/home/u070/ohjiye/Quartz/MedImage/MedViT')
import MedViT as _medvit


CKPT_PATH = '/N/slate/ohjiye/medvit_ckpt/MedViT_small.pth'


def build_model(num_classes=2, pretrained=True, ckpt_path=CKPT_PATH, n_slices=32,
                head_dropout=0.2):
    model = _medvit.MedViT_small(num_classes=num_classes)

    if pretrained:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model', ckpt)
        state = {k: v for k, v in state.items() if 'proj_head' not in k}
        model.load_state_dict(state, strict=False)
        print(f"[model] Pretrained backbone loaded from {ckpt_path}")

    # ── 1. Inflate first conv: 3ch → n_slices*3 ch ───────────────────────────
    if n_slices > 1:
        old_conv = model.stem[0].conv                   # Conv2d(3, 64, 3, stride=2, pad=1)
        old_w    = old_conv.weight.data                 # [64, 3, 3, 3]
        new_conv = nn.Conv2d(
            n_slices * 3, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        new_conv.weight.data = old_w.repeat(1, n_slices, 1, 1) / n_slices
        model.stem[0].conv = new_conv
        print(f"[model] First conv inflated: 3 → {n_slices*3} channels (weight tiling ÷{n_slices})")

    # ── 2. Replace proj_head: 1024 → 512 → 256 → num_classes ────────────────
    # Gradual 2× reduction per step; dropout 0.2 to avoid over-regularizing
    # small dataset (315 train patients)
    in_features = model.proj_head[0].in_features  # 1024
    model.proj_head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(head_dropout),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(head_dropout),
        nn.Linear(256, num_classes),
    )
    print(f"[model] proj_head: {in_features} → 512 → 256 → {num_classes}  "
          f"(dropout={head_dropout})")

    return model
