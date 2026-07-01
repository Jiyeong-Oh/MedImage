"""
SliceWiseModel for MIL-max prostate cancer classification.
Input:  [B, N, 3, H, W]  (N slices, 3 modalities, standard 3-channel backbone)
Output: [B, N, 2]         (per-slice logits; MIL-max applied in loss/eval)

No weight inflation — pretrained MedViT 3-channel weights used as-is.
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


class SliceWiseModel(nn.Module):
    """Wraps MedViT backbone: processes each slice independently, returns per-slice logits."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        """x: [B, N, 3, H, W] → [B, N, 2]"""
        B, N, C, H, W = x.shape
        logits = self.backbone(x.view(B * N, C, H, W))  # [B*N, 2]
        return logits.view(B, N, 2)


def build_model(num_classes=2, pretrained=True, ckpt_path=None, n_slices=32,
                head_dropout=0.2, head_depth=2, backbone='small'):
    ckpt_path = ckpt_path or CKPT_PATHS[backbone]
    medvit = _BUILDERS[backbone](num_classes=num_classes)

    if pretrained:
        ckpt  = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model', ckpt)
        state = {k: v for k, v in state.items() if 'proj_head' not in k}
        medvit.load_state_dict(state, strict=False)
        print(f"[model] Pretrained backbone loaded from {ckpt_path}")

    # Replace proj_head with deeper MLP
    in_features = medvit.proj_head[0].in_features  # 1024
    hidden = [512, 256, 128, 64][:head_depth]
    dims   = [in_features] + hidden + [num_classes]
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i+1]), nn.GELU(), nn.Dropout(head_dropout)]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    medvit.proj_head = nn.Sequential(*layers)
    arch = " → ".join(str(d) for d in dims)
    print(f"[model] backbone: MedViT_{backbone}  proj_head: {arch}  (dropout={head_dropout})")

    return SliceWiseModel(medvit)
