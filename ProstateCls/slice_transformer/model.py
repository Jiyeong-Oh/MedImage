"""
MedViT slice-by-slice + Transformer encoder for prostate cancer classification.

Input:  [B, n_slices*3, 224, 224]  (same format as dataset.py)
Output: [B, 2]

Architecture:
  1. Reshape → [B*n_slices, 3, 224, 224]
  2. MedViT backbone (pretrained, 3ch unchanged) → [B*n_slices, 1024]
  3. Reshape → [B, n_slices, 1024]
  4. Prepend CLS token + learnable positional embedding → [B, n_slices+1, 1024]
  5. Transformer Encoder (num_layers, nhead)
  6. Extract CLS token → [B, 1024]
  7. Classification head → [B, num_classes]

Key advantage: pretrained first conv is used as-is (3ch), no weight inflation or adapter.
Memory note: backbone processes B*n_slices images in parallel → use smaller batch_size (4).
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


class MedViTFeatureExtractor(nn.Module):
    """MedViT backbone up to (and including) global avg pool — outputs [B, d_model]."""
    def __init__(self, backbone):
        super().__init__()
        self.stem     = backbone.stem
        self.features = backbone.features
        self.norm     = backbone.norm
        self.avgpool  = backbone.avgpool

    def forward(self, x):
        x = self.stem(x)
        for layer in self.features:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)  # [B, d_model]


class SliceTransformerModel(nn.Module):
    def __init__(self, backbone, n_slices=32, num_classes=2,
                 nhead=8, num_layers=2, dim_feedforward=2048,
                 tf_dropout=0.1, head_depth=2, head_dropout=0.2,
                 pooling='cls'):
        super().__init__()
        self.n_slices = n_slices
        self.pooling  = pooling
        self.extractor = MedViTFeatureExtractor(backbone)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            d_model = self.extractor(dummy).shape[1]
        print(f"[model] d_model={d_model}")

        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_slices + 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_slices, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=tf_dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(d_model)

        hidden = [512, 256, 128, 64][:head_depth]
        dims   = [d_model] + hidden + [num_classes]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.GELU(), nn.Dropout(head_dropout)]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.head = nn.Sequential(*layers)

        arch = " → ".join(str(d) for d in dims)
        print(f"[model] Transformer: {num_layers} layers, {nhead} heads, ff={dim_feedforward}  pooling={pooling}")
        print(f"[model] head: {arch}  (dropout={head_dropout})")

    def forward(self, x):
        B, C, H, W = x.shape                                      # [B, 96, 224, 224]
        x = x.view(B, self.n_slices, C // self.n_slices, H, W)    # [B, 32, 3, H, W]

        feats = torch.stack(
            [self.extractor(x[b]) for b in range(B)], dim=0
        )                                                          # [B, 32, d_model]

        if self.pooling == 'cls':
            cls = self.cls_token.expand(B, -1, -1)                 # [B, 1, d_model]
            seq = torch.cat([cls, feats], dim=1) + self.pos_embed  # [B, 33, d_model]
            seq = self.transformer(seq)
            out = self.norm_out(seq[:, 0])                         # CLS token
        else:
            seq = feats + self.pos_embed                           # [B, 32, d_model]
            seq = self.transformer(seq)
            out = self.norm_out(seq.mean(dim=1))                   # mean over slices

        return self.head(out)


def build_model(num_classes=2, pretrained=True, ckpt_path=None, n_slices=32,
                nhead=8, num_layers=2, dim_feedforward=2048, tf_dropout=0.1,
                head_depth=2, head_dropout=0.2, backbone='small', pooling='cls'):
    ckpt_path = ckpt_path or CKPT_PATHS[backbone]
    base = _BUILDERS[backbone](num_classes=num_classes)

    if pretrained:
        ckpt  = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model', ckpt)
        state = {k: v for k, v in state.items() if 'proj_head' not in k}
        base.load_state_dict(state, strict=False)
        print(f"[model] Pretrained backbone loaded from {ckpt_path}")

    model = SliceTransformerModel(
        backbone=base, n_slices=n_slices, num_classes=num_classes,
        nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward,
        tf_dropout=tf_dropout, head_depth=head_depth, head_dropout=head_dropout,
        pooling=pooling,
    )
    print(f"[model] backbone: MedViT_{backbone}  stem: 3ch pretrained (unchanged)")
    return model
