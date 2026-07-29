"""
MedViT_small adapted for binary prostate cancer classification.
Input:  [B, n_slices*3, 224, 224]  (depth-as-channel, default 32 slices → 96ch)
Output: [B, 2]  (logits for ciPCa=0, csPCa=1)

Weight inflation: first conv 3→96ch via tiling pretrained weights (÷n_slices).
All other backbone layers use pretrained weights unchanged.

Head: AttentionPoolingHead — learns [B, 1, H, H] spatial weights.
  Standard (--high-res off): f4 [B, 1024, 7, 7]   — coarse but full backbone
  High-res  (--high-res on):  f3 [B,  512, 14, 14] — finer grid, skips last stage
  The attention map IS the classifier's end-to-end decision, not a post-hoc approximation.
"""
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class AttentionPoolingHead(nn.Module):
    """Spatial attention pooling head for end-to-end interpretable classification.

    Replaces GAP + MLP. A 1×1 conv produces a soft spatial weight map over the
    backbone's 7×7 feature grid; the weighted sum feeds a single linear classifier.
    The [B, 1, 7, 7] attention map is the classifier's own decision mechanism —
    no post-hoc gradient approximation required.

    forward() returns (logits [B, num_classes], attn [B, 1, H, W]).
    """
    def __init__(self, in_features=1024, num_classes=2):
        super().__init__()
        self.attn_conv  = nn.Conv2d(in_features, 1, 1, bias=False)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        attn = F.softmax(self.attn_conv(x).view(B, 1, -1), dim=2).view(B, 1, H, W)
        feat = (attn * x).sum(dim=[2, 3])   # [B, C]
        return self.classifier(feat), attn


F3_CH = 512   # MedViT stage-3 output channels (all variants: small/base/large)


def build_model(num_classes=2, pretrained=True, ckpt_path=None, n_slices=32,
                head_dropout=0.2, head_depth=2, backbone='small', n_ch_per_slice=3,
                high_res=False):
    """high_res: use f3 [B, 512, 14, 14] for attention instead of f4 [B, 1024, 7, 7]."""
    ckpt_path = ckpt_path or CKPT_PATHS[backbone]
    model = _BUILDERS[backbone](num_classes=num_classes)

    if pretrained:
        ckpt  = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model', ckpt)
        state = {k: v for k, v in state.items() if 'proj_head' not in k}
        model.load_state_dict(state, strict=False)
        print(f"[model] Pretrained backbone loaded from {ckpt_path}")

    # ── 1. Inflate first conv: 3ch → n_slices*n_ch_per_slice ch ──────────────
    if n_slices > 1:
        in_ch    = n_slices * n_ch_per_slice
        old_conv = model.stem[0].conv
        old_w    = old_conv.weight.data             # [64, 3, 3, 3]
        new_conv = nn.Conv2d(
            in_ch, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        new_conv.weight.data = old_w[:, :n_ch_per_slice, :, :].repeat(1, n_slices, 1, 1) / n_slices
        model.stem[0].conv = new_conv
        print(f"[model] First conv inflated: {n_ch_per_slice} → {in_ch} channels (weight tiling ÷{n_slices})")

    # ── 2. Replace proj_head with AttentionPoolingHead ────────────────────────
    model.last_attn_map = None

    if high_res:
        # Use f3 [B, 512, 14, 14] — finer 14×14 spatial grid, skips last backbone stage
        model.f3_norm   = nn.BatchNorm2d(F3_CH)         # new BN for f3 (no norm applied at f3 in MedViT)
        model.proj_head = AttentionPoolingHead(F3_CH, num_classes)
        f3_idx          = model.stage_out_idx[-2]        # e.g. 16 for MedViT_small

        def _forward_hires(self, x):
            x = self.stem(x)
            f3 = None
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == self.stage_out_idx[-2]:
                    f3 = x                              # [B, 512, 14, 14]
            f3 = self.f3_norm(f3)
            logits, attn = self.proj_head(f3)
            self.last_attn_map = attn                   # [B, 1, 14, 14]
            return logits

        model.forward = types.MethodType(_forward_hires, model)
        print(f"[model] backbone: MedViT_{backbone}  "
              f"head: AttentionPoolingHead({F3_CH} → attn[14×14] + Linear → {num_classes})  [HIGH-RES f3]")
    else:
        in_features     = model.proj_head[0].in_features   # 1024
        model.proj_head = AttentionPoolingHead(in_features, num_classes)

        def _forward(self, x):
            x = self.stem(x)
            for layer in self.features:
                x = layer(x)
            x = self.norm(x)                    # [B, 1024, 7, 7]
            logits, attn = self.proj_head(x)
            self.last_attn_map = attn           # [B, 1, 7, 7]
            return logits

        model.forward = types.MethodType(_forward, model)
        print(f"[model] backbone: MedViT_{backbone}  "
              f"head: AttentionPoolingHead(1024 → attn[7×7] + Linear → {num_classes})")

    return model
