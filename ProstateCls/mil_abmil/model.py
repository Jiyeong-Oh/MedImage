"""
MIL-ABMIL model for PI-CAI classification.
Architecture:
  Encoder:    MedViT_small/base/large, first conv inflated 3ch → nch
  Aggregator: Gated Attention-Based MIL (Ilse et al. 2018)
              OR Spatial MIL: per-slice spatial attention (7×7) + ABMIL over slices

  Standard forward:  x [B, S, nch, H, W]
    → encoder([B*S, nch, H, W]) → GAP+flatten → [B*S, 1024]
    → view [B, S, 1024] → GatedABMIL → logits [B, 2], slice_attn [B, S]

  Spatial forward:   x [B, S, nch, H, W]
    → encoder([B*S, nch, H, W]) → f4 [B*S, 1024, 7, 7]  (no GAP)
    → SpatialGatedABMIL:
        sp_attn [B*S, 1, 7, 7] per-slice spatial softmax
        slice_feat [B*S, 1024] = weighted sum over spatial dim
        [B, S, 1024] → ABMIL → logits [B, 2]  +  slice_attn [B, S]
"""
import os
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
FEAT_DIMS = {'small': 1024, 'base': 1024, 'large': 1024}


class GatedABMIL(nn.Module):
    """Gated Attention-Based MIL aggregator (Ilse et al. 2018)."""
    def __init__(self, feat_dim=1024, hidden=256, dropout=0.25, n_classes=2):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(feat_dim, hidden), nn.Tanh(), nn.Dropout(dropout))
        self.U = nn.Sequential(nn.Linear(feat_dim, hidden), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(hidden, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, feats):
        # feats: [B, S, feat_dim]
        scores  = self.w(self.V(feats) * self.U(feats))  # [B, S, 1]
        weights = torch.softmax(scores, dim=1)            # [B, S, 1]
        bag     = (weights * feats).sum(dim=1)            # [B, feat_dim]
        return self.classifier(bag), weights.squeeze(-1)  # [B, 2], [B, S]


class SpatialGatedABMIL(nn.Module):
    """Per-slice spatial attention (7x7) + Gated ABMIL over slices.

    Stage 1 — spatial: 1x1 conv + softmax over each slice's f4 [B*S, 1024, 7, 7]
               → per-slice feature vector [B*S, 1024]
    Stage 2 — slice:   standard gated ABMIL over [B, S, 1024]
               → patient bag vector [B, 1024] + slice weights [B, S]

    Stores last_spatial_attn [B, S, 1, 7, 7] and last_slice_attn [B, S] for visualization.
    forward(feat_maps, n_slices) → (logits [B,2], slice_attn [B,S], spatial_attn [B,S,1,H,W])
    """
    def __init__(self, feat_dim=1024, hidden=256, dropout=0.25, n_classes=2):
        super().__init__()
        self.sp_attn_conv = nn.Conv2d(feat_dim, 1, 1, bias=False)
        self.V = nn.Sequential(nn.Linear(feat_dim, hidden), nn.Tanh(), nn.Dropout(dropout))
        self.U = nn.Sequential(nn.Linear(feat_dim, hidden), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(hidden, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )
        self.last_spatial_attn = None  # [B, S, 1, H, W]
        self.last_slice_attn   = None  # [B, S]

    def forward(self, feat_maps, n_slices):
        # feat_maps: [B*S, C, H, W]
        BS, C, H, W = feat_maps.shape
        B = BS // n_slices
        S = n_slices

        # 1. Spatial attention per slice: softmax over H*W locations
        sp_logit = self.sp_attn_conv(feat_maps)                         # [B*S, 1, H, W]
        sp_attn  = F.softmax(sp_logit.view(BS, 1, -1), dim=2).view(BS, 1, H, W)

        # 2. Spatially-weighted per-slice feature
        slice_feats = (sp_attn * feat_maps).sum(dim=[2, 3])             # [B*S, C]
        slice_feats = slice_feats.view(B, S, C)                         # [B, S, C]

        # 3. ABMIL over slices
        scores  = self.w(self.V(slice_feats) * self.U(slice_feats))     # [B, S, 1]
        slice_w = torch.softmax(scores, dim=1)                          # [B, S, 1]
        bag     = (slice_w * slice_feats).sum(dim=1)                    # [B, C]
        logits  = self.classifier(bag)                                  # [B, 2]

        sp_attn_4d = sp_attn.view(B, S, 1, H, W)
        self.last_spatial_attn = sp_attn_4d.detach()
        self.last_slice_attn   = slice_w.squeeze(-1).detach()           # [B, S]

        return logits, slice_w.squeeze(-1), sp_attn_4d


class MILModel(nn.Module):
    def __init__(self, backbone, aggregator):
        super().__init__()
        self.backbone   = backbone
        self.aggregator = aggregator

    def forward(self, x):
        # x: [B, S, C, H, W]
        B, S, C, H, W = x.shape
        feats = self.backbone(x.reshape(B * S, C, H, W))  # [B*S, feat_dim]
        feats = feats.view(B, S, -1)                       # [B, S, feat_dim]
        return self.aggregator(feats)                      # logits, attn_weights


class SpatialMILModel(nn.Module):
    """MIL model: extracts f4 [B*S, 1024, 7, 7] per slice (no GAP)
    then applies SpatialGatedABMIL: per-slice spatial attention + ABMIL."""
    def __init__(self, backbone, aggregator):
        super().__init__()
        self.backbone   = backbone
        self.aggregator = aggregator

    def _extract_features(self, x):
        """Return f4 [N, C, H, W] — bypass avgpool+flatten in backbone.forward."""
        x = self.backbone.stem(x)
        for layer in self.backbone.features:
            x = layer(x)
        return self.backbone.norm(x)  # [N, 1024, 7, 7]

    def forward(self, x):
        # x: [B, S, C, H, W]
        B, S, C, H, W = x.shape
        feat_maps = self._extract_features(x.reshape(B * S, C, H, W))  # [B*S, 1024, 7, 7]
        logits, slice_attn, _ = self.aggregator(feat_maps, n_slices=S)
        return logits, slice_attn


def build_model(nch=2, backbone='small', abmil_hidden=256, abmil_dropout=0.25,
                pretrained=True, ckpt_path=None, spatial_attn=False):
    ckpt_path = ckpt_path or CKPT_PATHS[backbone]
    bb = _BUILDERS[backbone](num_classes=2)

    if pretrained and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location='cpu')
        sd = sd.get('model', sd)
        sd = {k: v for k, v in sd.items() if 'proj_head' not in k}
        bb.load_state_dict(sd, strict=False)
        print(f"[model] Pretrained loaded from {ckpt_path}")

    # Inflate first conv 3ch → nch
    orig_conv = bb.stem[0].conv           # Conv2d(3, 64, 3, stride=2, pad=1)
    orig_w    = orig_conv.weight.data     # [64, 3, 3, 3]
    if nch <= 3:
        new_w = orig_w[:, :nch, :, :] * (3.0 / nch)
    else:
        repeats = nch // 3 + 1
        new_w = orig_w.repeat(1, repeats, 1, 1)[:, :nch, :, :] / (nch / 3.0)
    new_conv = nn.Conv2d(nch, orig_conv.out_channels,
                         kernel_size=orig_conv.kernel_size,
                         stride=orig_conv.stride,
                         padding=orig_conv.padding,
                         bias=False)
    new_conv.weight.data.copy_(new_w)
    bb.stem[0].conv = new_conv
    print(f"[model] First conv inflated: 3 → {nch} channels")

    feat_dim = FEAT_DIMS[backbone]

    if spatial_attn:
        bb.proj_head = nn.Identity()  # keep for state dict compat
        aggregator = SpatialGatedABMIL(feat_dim=feat_dim, hidden=abmil_hidden,
                                        dropout=abmil_dropout)
        model = SpatialMILModel(bb, aggregator)
        print(f"[model] SpatialMILModel: MedViT_{backbone}  "
              f"sp_attn[7x7/slice] + GatedABMIL(hidden={abmil_hidden}  drop={abmil_dropout})")
    else:
        bb.proj_head = nn.Identity()
        aggregator = GatedABMIL(feat_dim=feat_dim, hidden=abmil_hidden,
                                 dropout=abmil_dropout)
        model = MILModel(bb, aggregator)
        print(f"[model] MILModel: MedViT_{backbone}  "
              f"GatedABMIL(feat={feat_dim}  hidden={abmil_hidden}  drop={abmil_dropout})")

    return model
