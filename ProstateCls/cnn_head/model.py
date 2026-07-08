"""
MedViT backbone + CNN spatial head for PI-CAI classification.

Backbone output before avgpool: [B, 1024, 7, 7]
CNN head processes spatial features instead of flat MLP on pooled vector.

Head variants (--cnn-head):
  se    — SE channel attention + GAP + FC(1024→512→2)
  cbam  — CBAM (SE + spatial attention) + GAP + FC
  gem   — CBAM + GeM pooling (learnable p) + FC
  ms    — Multi-scale DW conv + CBAM + GAP + FC
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
_FEAT_CH = {'small': 1024, 'base': 1024, 'large': 1024}


# ── Attention building blocks ─────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        s = x.mean(dim=[2, 3])          # [B, C]
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(x.size(0), -1, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class GeMPool(nn.Module):
    """Generalized Mean Pooling. p=1 → GAP, p→∞ → GMP."""
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p_init)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


class MultiScaleConv(nn.Module):
    """1×1, 3×3-DW, 5×5-DW, 7×7-DW parallel branches → concat → project."""
    def __init__(self, channels):
        super().__init__()
        mid = channels // 4

        def _branch(k):
            layers = [nn.Conv2d(channels, mid, 1, bias=False),
                      nn.BatchNorm2d(mid), nn.GELU()]
            if k > 1:
                layers += [nn.Conv2d(mid, mid, k, padding=k // 2, groups=mid, bias=False),
                           nn.BatchNorm2d(mid), nn.GELU()]
            return nn.Sequential(*layers)

        self.b1 = _branch(1)
        self.b3 = _branch(3)
        self.b5 = _branch(5)
        self.b7 = _branch(7)
        self.proj = nn.Sequential(
            nn.Conv2d(mid * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(torch.cat([self.b1(x), self.b3(x), self.b5(x), self.b7(x)], dim=1))


# ── CNN Head variants ─────────────────────────────────────────────────────────

class SEHead(nn.Module):
    """SE channel attention → GAP → FC"""
    def __init__(self, in_ch, num_classes=2, dropout=0.3):
        super().__init__()
        self.attn = SEBlock(in_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.attn(x)))


class CBAMHead(nn.Module):
    """CBAM (SE + spatial attention) → GAP → FC"""
    def __init__(self, in_ch, num_classes=2, dropout=0.3):
        super().__init__()
        self.ch_attn  = SEBlock(in_ch)
        self.sp_attn  = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.ch_attn(x)
        x = self.sp_attn(x)
        return self.classifier(self.pool(x))


class GeMHead(nn.Module):
    """CBAM + GeM pooling → FC"""
    def __init__(self, in_ch, num_classes=2, dropout=0.3):
        super().__init__()
        self.ch_attn  = SEBlock(in_ch)
        self.sp_attn  = SpatialAttention()
        self.gem  = GeMPool()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.ch_attn(x)
        x = self.sp_attn(x)
        return self.classifier(self.gem(x))


class MSHead(nn.Module):
    """Multi-scale DW conv (residual) + CBAM + GAP → FC"""
    def __init__(self, in_ch, num_classes=2, dropout=0.3):
        super().__init__()
        self.ms_conv = MultiScaleConv(in_ch)
        self.ch_attn = SEBlock(in_ch)
        self.sp_attn = SpatialAttention()
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = x + self.ms_conv(x)   # residual multi-scale refinement
        x = self.ch_attn(x)
        x = self.sp_attn(x)
        return self.classifier(self.pool(x))


_HEAD_CLASSES = {'se': SEHead, 'cbam': CBAMHead, 'gem': GeMHead, 'ms': MSHead}


# ── Wrapper model ─────────────────────────────────────────────────────────────

class ProstateCNNModel(nn.Module):
    def __init__(self, backbone, cnn_head):
        super().__init__()
        self.backbone     = backbone
        self.cnn_head     = cnn_head
        self.tumor_probe  = nn.Sequential(nn.Conv2d(1024, 1, 1, bias=False), nn.Sigmoid())
        self.last_attn_map = None

    def forward(self, x):
        x = self.backbone.stem(x)
        for layer in self.backbone.features:
            x = layer(x)
        f4 = self.backbone.norm(x)              # [B, 1024, 7, 7]
        self.last_attn_map = self.tumor_probe(f4)
        return self.cnn_head(f4)


# ── Builder ───────────────────────────────────────────────────────────────────

def build_model(num_classes=2, pretrained=True, ckpt_path=None,
                n_slices=32, head_type='cbam', dropout=0.3, backbone='small', n_ch_per_slice=3):
    ckpt_path = ckpt_path or CKPT_PATHS[backbone]

    # Build backbone (MedViT keeps its proj_head but we never call it)
    base = _BUILDERS[backbone](num_classes=num_classes)

    if pretrained:
        ckpt  = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model', ckpt)
        state = {k: v for k, v in state.items() if 'proj_head' not in k}
        base.load_state_dict(state, strict=False)
        print(f"[model] Pretrained backbone loaded from {ckpt_path}")

    # Inflate first conv 3 → n_slices*n_ch_per_slice channels
    if n_slices > 1:
        in_ch    = n_slices * n_ch_per_slice
        old_conv = base.stem[0].conv
        old_w    = old_conv.weight.data
        new_conv = nn.Conv2d(
            in_ch, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        new_conv.weight.data = old_w[:, :n_ch_per_slice, :, :].repeat(1, n_slices, 1, 1) / n_slices
        base.stem[0].conv = new_conv
        print(f"[model] First conv inflated: {n_ch_per_slice} → {in_ch} channels (÷{n_slices})")

    in_ch = _FEAT_CH[backbone]
    HeadCls = _HEAD_CLASSES[head_type]
    head = HeadCls(in_ch, num_classes=num_classes, dropout=dropout)
    print(f"[model] CNN head: {head_type.upper()}  backbone: MedViT_{backbone}")

    return ProstateCNNModel(base, head)
