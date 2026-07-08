"""
MedViT + U-Net segmentation decoder + gland-gated classification head.

Architecture:
  Input [B, 96, 224, 224]
    → MedViT backbone (feature hooks at 4 stages)
        f1 [B,  96, 56, 56]
        f2 [B, 256, 28, 28]
        f3 [B, 512, 14, 14]
        f4 [B,1024,  7,  7]
    → Segmentation decoder (U-Net, skip connections from f1-f3)
        Output: tumor_pred [B, 1, 224, 224]  (sigmoid * gland_2d)
    → Classification head (CBAM on gland-gated f4)
        f4 * downsampled_gland → CBAM → GAP → cat(pooled seg) → MLP → [B, 2]
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
_BUILDERS = {'small': _medvit.MedViT_small, 'base': _medvit.MedViT_base, 'large': _medvit.MedViT_large}

# MedViT_small: depths=[3,4,10,3] → stage_out_idx=[2,6,16,19]
# channels: stem→64, s1→96, s2→256, s3→512, s4→1024
_STAGE_CH = {
    'small': [96, 256, 512, 1024],
    'base':  [96, 256, 512, 1024],
    'large': [96, 256, 512, 1024],
}


# ── Building blocks ───────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch, out_ch, k=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k//2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DecodeBlock(nn.Module):
    """Upsample 2× → concat skip → double conv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            _conv_bn_relu(in_ch + skip_ch, out_ch),
            _conv_bn_relu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 8)
        self.fc = nn.Sequential(nn.Linear(ch, mid), nn.ReLU(inplace=True), nn.Linear(mid, ch))

    def forward(self, x):
        s = x.mean([2, 3])
        return x * torch.sigmoid(self.fc(s)).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.last_map = None  # [B, 1, H, W] — set each forward pass

    def forward(self, x):
        a = torch.sigmoid(self.conv(
            torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        ))  # [B, 1, H, W]
        self.last_map = a
        return x * a


# ── Main model ────────────────────────────────────────────────────────────────

class SegClsModel(nn.Module):
    def __init__(self, backbone, stage_out_idx, stage_ch, num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone      = backbone
        self.stage_out_idx = stage_out_idx  # [idx_s1, idx_s2, idx_s3, idx_s4]
        self._feats        = {}

        # hook intermediate features
        for i, idx in enumerate(stage_out_idx):
            backbone.features[idx].register_forward_hook(self._make_hook(i))

        c1, c2, c3, c4 = stage_ch   # 96, 256, 512, 1024

        # ── Segmentation decoder ─────────────────────────────
        self.dec3 = DecodeBlock(c4, c3, 256)   # 7→14
        self.dec2 = DecodeBlock(256, c2, 128)   # 14→28
        self.dec1 = DecodeBlock(128, c1, 64)    # 28→56
        self.seg_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            _conv_bn_relu(64, 32),
            nn.Conv2d(32, 1, 1),                # [B, 1, 224, 224]
        )

        # ── Classification head ──────────────────────────────
        self.ch_attn  = SEBlock(c4)
        self.sp_attn  = SpatialAttention()
        self.gap      = nn.AdaptiveAvgPool2d(1)

        # seg projection: pool tumor_pred to a compact vector for cls fusion
        self.seg_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),            # [B, 1, 7, 7]
            nn.Flatten(),                        # [B, 49]
            nn.Linear(49, 64), nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4 + 64, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),     nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _make_hook(self, i):
        def hook(module, input, output):
            self._feats[i] = output
        return hook

    def forward(self, x, gland_2d):
        """
        x        : [B, 96, 224, 224]
        gland_2d : [B,  1, 224, 224]  — 2D max-projection of gland mask
        Returns  : (cls_logits [B,2], tumor_pred [B,1,224,224])
        """
        self._feats.clear()

        # backbone forward (hooks populate self._feats)
        h = self.backbone.stem(x)
        for layer in self.backbone.features:
            h = layer(h)
        h = self.backbone.norm(h)   # [B, 1024, 7, 7]

        f1 = self._feats[0]   # [B,  96, 56, 56]
        f2 = self._feats[1]   # [B, 256, 28, 28]
        f3 = self._feats[2]   # [B, 512, 14, 14]
        f4 = h                # [B,1024,  7,  7]

        # ── Segmentation decoder ─────────────────────────────
        d3  = self.dec3(f4, f3)                     # [B, 256, 14, 14]
        d2  = self.dec2(d3, f2)                     # [B, 128, 28, 28]
        d1  = self.dec1(d2, f1)                     # [B,  64, 56, 56]
        seg_logits = self.seg_up(d1)                # [B, 1, 224, 224]

        # constrain predictions to within-gland region
        tumor_pred = torch.sigmoid(seg_logits) * gland_2d

        # ── Gland + tumor gated classification ───────────────────
        gland_7 = F.adaptive_avg_pool2d(gland_2d, 7)        # [B, 1, 7, 7]
        tumor_7 = F.adaptive_avg_pool2d(tumor_pred, 7)      # [B, 1, 7, 7]  no detach → cls grad → decoder
        # tumor gate: amplifies tumor region; (1+tumor_7) ensures features never zeroed for ciPCa
        f4_gated = f4 * (gland_7 + 1e-3) * (1.0 + tumor_7)

        f4_att = self.sp_attn(self.ch_attn(f4_gated))       # CBAM
        feat_cls = self.gap(f4_att)                          # [B, 1024, 1, 1]

        seg_feat = self.seg_proj(tumor_pred.detach())        # [B, 64]  detach: avoid double-gradient path

        cls_input = torch.cat([feat_cls.flatten(1), seg_feat], dim=1)  # [B, 1088]
        cls_logits = self.classifier(cls_input)

        return cls_logits, tumor_pred


# ── Builder ───────────────────────────────────────────────────────────────────

def build_model(num_classes=2, pretrained=True, n_slices=32,
                dropout=0.3, backbone='small', n_ch_per_slice=3):
    ckpt_path = CKPT_PATHS[backbone]
    base = _BUILDERS[backbone](num_classes=num_classes)

    if pretrained:
        ckpt  = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model', ckpt)
        state = {k: v for k, v in state.items() if 'proj_head' not in k}
        base.load_state_dict(state, strict=False)
        print(f"[model] Pretrained loaded: {ckpt_path}")

    # inflate first conv → n_slices*n_ch_per_slice ch
    in_ch    = n_slices * n_ch_per_slice
    old_conv = base.stem[0].conv
    new_conv  = nn.Conv2d(
        in_ch, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride, padding=old_conv.padding, bias=False,
    )
    new_conv.weight.data = old_conv.weight.data[:, :n_ch_per_slice, :, :].repeat(1, n_slices, 1, 1) / n_slices
    base.stem[0].conv = new_conv
    print(f"[model] First conv inflated: {n_ch_per_slice} → {in_ch} ch")

    # stage_out_idx depends on depths
    depths_map = {'small': [3,4,10,3], 'base': [3,4,20,3], 'large': [3,4,30,3]}
    depths = depths_map[backbone]
    stage_out_idx = [sum(depths[:i+1]) - 1 for i in range(4)]
    print(f"[model] Stage hook indices: {stage_out_idx}")

    stage_ch = _STAGE_CH[backbone]
    model = SegClsModel(base, stage_out_idx, stage_ch,
                        num_classes=num_classes, dropout=dropout)
    print(f"[model] SegClsModel built  backbone=MedViT_{backbone}")
    return model
