"""
MedViT with learnable channel adapter for binary prostate cancer classification.
Input:  [B, n_slices*3, 224, 224]
Output: [B, 2]

Channel adapter: learnable Conv2d(96→3, 1×1) inserted before the original pretrained
first conv (3→64). The original first conv keeps its pretrained weights unchanged and
is trained at backbone LR; the adapter is randomly initialized and trained at head LR.

Contrast with ../model.py (weight tiling): pretrained 3ch weights are tiled 32× and
divided by 32 to initialize a new Conv2d(96→64, 3×3).
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

_BUILDERS = {
    'small': _medvit.MedViT_small,
    'base':  _medvit.MedViT_base,
    'large': _medvit.MedViT_large,
}


class ChannelAdaptedConv(nn.Module):
    """Learnable adapter + original pretrained conv (3→64).
    mid_ch=0: single 1×1 (in_ch→3)
    mid_ch>0: two-layer (in_ch→mid_ch→3) with BN+GELU
    """
    def __init__(self, in_ch, orig_conv, mid_ch=0):
        super().__init__()
        out_ch = orig_conv.in_channels
        if mid_ch > 0:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.GELU(),
                nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            )
            nn.init.kaiming_normal_(self.adapter[0].weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.adapter[3].weight, mode='fan_out', nonlinearity='relu')
        else:
            self.adapter = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out', nonlinearity='relu')
        self.orig_conv = orig_conv  # pretrained weights, trained at backbone LR

    def forward(self, x):
        return self.orig_conv(self.adapter(x))


def build_model(num_classes=2, pretrained=True, ckpt_path=None, n_slices=32,
                head_dropout=0.2, head_depth=2, backbone='small', adapter_mid_ch=0):
    ckpt_path = ckpt_path or CKPT_PATHS[backbone]
    model = _BUILDERS[backbone](num_classes=num_classes)

    if pretrained:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model', ckpt)
        state = {k: v for k, v in state.items() if 'proj_head' not in k}
        model.load_state_dict(state, strict=False)
        print(f"[model] Pretrained backbone loaded from {ckpt_path}")

    # ── 1. Replace first conv with ChannelAdaptedConv ─────────────────────────
    if n_slices > 1:
        old_conv = model.stem[0].conv           # Conv2d(3, 64, 3, stride=2, pad=1)
        model.stem[0].conv = ChannelAdaptedConv(n_slices * 3, old_conv, mid_ch=adapter_mid_ch)
        in_ch = n_slices * 3
        out_ch = old_conv.in_channels
        if adapter_mid_ch > 0:
            print(f"[model] Channel adapter: {in_ch} → {adapter_mid_ch} → {out_ch} → {old_conv.out_channels} "
                  f"(2-layer 1×1 + pretrained 3×3)")
        else:
            print(f"[model] Channel adapter: {in_ch} → {out_ch} → {old_conv.out_channels} "
                  f"(1-layer 1×1 + pretrained 3×3)")

    # ── 2. Replace proj_head ─────────────────────────────────────────────────
    in_features = model.proj_head[0].in_features  # 1024
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
