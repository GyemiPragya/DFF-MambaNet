"""
models/encoder.py
==================
CNN encoder branch: a pretrained ResNet34 backbone used to extract
multi-scale **local** spatial features.

The encoder exposes the four standard ResNet stages (``layer1``..``layer4``)
as a feature pyramid:

    Stage 1 -> stride  4, channels  64
    Stage 2 -> stride  8, channels 128
    Stage 3 -> stride 16, channels 256
    Stage 4 -> stride 32, channels 512

These four feature maps are later paired, scale-by-scale, with the matching
outputs of the Vision Mamba branch (see ``models/vision_mamba.py``) inside
the Adaptive Attention-Based Dynamic Feature Fusion module
(``models/fusion.py``).
"""

from typing import List

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNetEncoder(nn.Module):
    """Multi-scale CNN feature extractor built on top of torchvision's ResNet34.

    Parameters
    ----------
    pretrained : bool
        Whether to initialize the backbone with ImageNet-pretrained weights.
    freeze_stem : bool
        If True, freezes the stem (conv1 + bn1) — occasionally useful for
        very small fine-tuning datasets, disabled by default.
    """

    # Channel width of each pyramid stage — exposed as a class attribute so
    # other modules (fusion, decoder, network) can read it without
    # instantiating the encoder.
    OUT_CHANNELS: List[int] = [64, 128, 256, 512]
    OUT_STRIDES: List[int] = [4, 8, 16, 32]

    def __init__(self, pretrained: bool = True, freeze_stem: bool = False):
        super().__init__()

        weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet34(weights=weights)

        # Stem: 7x7 conv (stride 2) + BN + ReLU + 3x3 max-pool (stride 2)
        # -> output stride 4, 64 channels. We keep it as a single block since
        # it is shared context for stage 1.
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )

        # Each of these is a stack of BasicBlocks; we keep them as separate
        # modules so we can grab the intermediate activations for the
        # feature pyramid.
        self.layer1 = backbone.layer1  # stride 4,  64 ch
        self.layer2 = backbone.layer2  # stride 8,  128 ch
        self.layer3 = backbone.layer3  # stride 16, 256 ch
        self.layer4 = backbone.layer4  # stride 32, 512 ch

        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run the backbone and return a 4-level feature pyramid.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch, shape (B, 3, H, W).

        Returns
        -------
        List[torch.Tensor]
            ``[f1, f2, f3, f4]`` with strides 4, 8, 16, 32 and channel
            widths 64, 128, 256, 512 respectively.
        """
        x = self.stem(x)
        f1 = self.layer1(x)     # (B,  64, H/4,  W/4)
        f2 = self.layer2(f1)    # (B, 128, H/8,  W/8)
        f3 = self.layer3(f2)    # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)    # (B, 512, H/32, W/32)
        return [f1, f2, f3, f4]


if __name__ == "__main__":
    model = ResNetEncoder(pretrained=False)
    dummy = torch.randn(2, 3, 256, 256)
    feats = model(dummy)
    for i, f in enumerate(feats, start=1):
        print(f"Stage {i}: {f.shape}")
