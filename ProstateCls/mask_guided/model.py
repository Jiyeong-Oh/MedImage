"""
Mask-guided MedViT for PI-CAI prostate cancer classification.

Two-stream architecture:
  Stream 1 (backbone): [B, 96, 224, 224] → MedViT (96ch inflated) → [B, 1024]
  Stream 2 (mask):     [B, 32, 224, 224] → spatial avg → [B, 32] → Linear → [B, 64]
  Combined:            [B, 1088] → MLP head → [B, 2]

Stream 1 processes the ROI-cropped, mask-normalized MRI.
Stream 2 encodes per-slice prostate coverage (which slices contain the most gland tissue),
giving the model explicit spatial context about the 3D extent of the prostate.
"""
import sys
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

_BUILDERS = {
    'small': _medvit.MedViT_small,
    'base':  _medvit.MedViT_base,
    'large': _medvit.MedViT_large,
}

FEAT_DIM = 1024
MASK_DIM = 64


class AttentionMaskHead(nn.Module):
    """Attention pooling over f4 + mask context combination + linear classifier.
    The attention map [B, 1, 7, 7] IS the classifier's spatial decision (end-to-end).
    forward() returns (logits, attn).
    """
    def __init__(self, feat_dim=1024, mask_dim=64, num_classes=2):
        super().__init__()
        self.attn_conv  = nn.Conv2d(feat_dim, 1, 1, bias=False)
        self.classifier = nn.Linear(feat_dim + mask_dim, num_classes)

    def forward(self, f4, mask_ctx):
        B, C, H, W = f4.shape
        attn = F.softmax(self.attn_conv(f4).view(B, 1, -1), dim=2).view(B, 1, H, W)
        feat = (attn * f4).sum(dim=[2, 3])                  # [B, C]
        combined = torch.cat([feat, mask_ctx], dim=1)
        return self.classifier(combined), attn


class MaskGuidedModel(nn.Module):
    """
    Receives a fully-configured MedViT backbone (pretrained weights loaded,
    first conv already inflated to 96ch, proj_head replaced with Identity).
    """
    def __init__(self, backbone_medvit, n_slices=32, num_classes=2,
                 head_dropout=0.2, head_depth=2):
        super().__init__()
        self.n_slices = n_slices
        self.backbone = backbone_medvit
        self.norm     = backbone_medvit.norm
        self.last_attn_map = None

        # Mask context branch: per-slice prostate coverage → [B, 64]
        self.mask_branch = nn.Sequential(
            nn.Linear(n_slices, MASK_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Attention pooling + combined classification (stored as proj_head for optimizer compat)
        self.proj_head = AttentionMaskHead(FEAT_DIM, MASK_DIM, num_classes)
        print(f"[model] backbone: MedViT  "
              f"head: AttentionMaskHead(1024+64 → attn[7×7] + Linear → {num_classes})")

    def forward(self, x, mask):
        h = self.backbone.stem(x)
        for layer in self.backbone.features:
            h = layer(h)
        f4 = self.backbone.norm(h)                          # [B, 1024, 7, 7]
        mask_cov = mask.mean(dim=[2, 3])                    # [B, 32] per-slice coverage
        mask_ctx = self.mask_branch(mask_cov)               # [B, 64]
        logits, attn = self.proj_head(f4, mask_ctx)
        self.last_attn_map = attn                           # [B, 1, 7, 7]
        return logits


def build_model(num_classes=2, pretrained=True, ckpt_path=None, n_slices=32,
                head_dropout=0.2, head_depth=2, backbone='small', n_ch_per_slice=3):
    path   = ckpt_path or CKPT_PATHS[backbone]
    medvit = _BUILDERS[backbone](num_classes=num_classes)

    # 1. Load pretrained weights while first conv is still 3ch
    if pretrained:
        ckpt  = torch.load(path, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt)
        state = {k: v for k, v in state.items() if 'proj_head' not in k}
        medvit.load_state_dict(state, strict=False)
        print(f"[model] Pretrained backbone loaded from {path}")

    # 2. Inflate first conv → n_slices*n_ch_per_slice ch using pretrained weights
    in_ch    = n_slices * n_ch_per_slice
    old_conv = medvit.stem[0].conv              # Conv2d(3, 64, 3×3) with pretrained weights
    old_w    = old_conv.weight.data             # [64, 3, 3, 3]
    new_conv = nn.Conv2d(
        in_ch, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    new_conv.weight.data = old_w[:, :n_ch_per_slice, :, :].repeat(1, n_slices, 1, 1) / n_slices
    medvit.stem[0].conv = new_conv
    print(f"[model] First conv inflated: {n_ch_per_slice} → {in_ch} channels (weight tiling ÷{n_slices})")

    # 3. Replace proj_head with Identity → backbone outputs [B, 1024]
    medvit.proj_head = nn.Identity()

    # 4. Build two-stream model
    model = MaskGuidedModel(
        backbone_medvit=medvit,
        n_slices=n_slices,
        num_classes=num_classes,
        head_dropout=head_dropout,
        head_depth=head_depth,
    )
    return model
