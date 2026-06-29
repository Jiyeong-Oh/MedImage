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


CKPT_PATHS = {
    'small': '/N/slate/ohjiye/medvit_ckpt/MedViT_small.pth',
    'base':  '/N/slate/ohjiye/medvit_ckpt/MedViT_base.pth',
    'large': '/N/slate/ohjiye/medvit_ckpt/MedViT_large.pth',
}
CKPT_PATH = CKPT_PATHS['small']  # backward compat

_BUILDERS = {
    'small': _medvit.MedViT_small,
    'base':  _medvit.MedViT_base,
    'large': _medvit.MedViT_large,
}


def build_model(num_classes=2, pretrained=True, ckpt_path=None, n_slices=32,
                head_dropout=0.2, head_depth=2, backbone='small'):
    ckpt_path = ckpt_path or CKPT_PATHS[backbone]
    model = _BUILDERS[backbone](num_classes=num_classes)

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

    # ── 2. Replace proj_head ─────────────────────────────────────────────────
    in_features = model.proj_head[0].in_features  # 1024
    # head_depth = number of hidden layers
    # 2 (default): 1024→512→256→2  (current deeper_head)
    # 3:           1024→512→256→128→2
    # 4:           1024→512→256→128→64→2
    hidden = [512, 256, 128, 64][:head_depth]
    dims   = [in_features] + hidden + [num_classes]
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i+1]), nn.GELU(), nn.Dropout(head_dropout)]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    model.proj_head = nn.Sequential(*layers)
    arch = " → ".join(str(d) for d in dims)
    print(f"[model] backbone: MedViT_{backbone}  proj_head: {arch}  (dropout={head_dropout})")

    return model
