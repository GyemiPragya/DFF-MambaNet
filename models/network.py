"""
models/network.py
==================
``DFFMambaNet`` — the complete hybrid CNN-Mamba segmentation network:

    Input Image
        |
        +----------------------+----------------------+
        |                                              |
    CNN Encoder (ResNet34)                  Vision Mamba Encoder
    [f1, f2, f3, f4]                          [m1, m2, m3, m4]
        |                                              |
        +-------------------+--------------------------+
                            |
              AADFF (one instance per pyramid stage)
              [fused1, fused2, fused3, fused4]
                            |
                     U-Net Decoder
                            |
                  Per-pixel class logits

This file is intentionally thin: it owns no novel logic itself, it simply
wires together the CNN branch (``models/encoder.py``), the Vision Mamba
branch (``models/vision_mamba.py``), four instances of the project's
research contribution (``models/fusion.py``), and the decoder
(``models/decoder.py``).
"""

from typing import List, Optional

import torch
import torch.nn as nn

from models.encoder import ResNetEncoder
from models.vision_mamba import VisionMambaEncoder
from models.fusion import AdaptiveAttentionDynamicFusion
from models.decoder import UNetDecoder


class DFFMambaNet(nn.Module):
    """Hybrid CNN-Mamba segmentation network with Adaptive Attention-Based
    Dynamic Feature Fusion.

    Parameters
    ----------
    num_classes : int
        Number of output segmentation classes.
    encoder_pretrained : bool
        Whether the ResNet34 CNN branch is initialized from ImageNet
        weights.
    stage_channels : tuple
        Channel widths shared by both encoder branches at each of the 4
        pyramid stages. Defaults to ResNet34's natural widths.
    mamba_d_state, mamba_expand, mamba_depths, mamba_use_official_kernel,
    mamba_max_tokens_fallback :
        Forwarded to :class:`~models.vision_mamba.VisionMambaEncoder`.
    fusion_reduction : int
        Channel-attention squeeze ratio used inside every AADFF instance.
    decoder_channels : tuple
        Internal channel widths of the 4 U-Net decoder stages.
    """

    def __init__(
        self,
        num_classes: int = 6,
        encoder_pretrained: bool = True,
        stage_channels=(64, 128, 256, 512),
        mamba_d_state: int = 2,
        mamba_expand: int = 1,
        mamba_depths=(1, 0, 0, 0),
        mamba_use_official_kernel: bool = True,
        mamba_max_tokens_fallback: int = 256,
        fusion_reduction: int = 8,
        decoder_channels=(256, 128, 64, 32),
    ):
        super().__init__()

        self.num_classes = num_classes
        self.stage_channels = tuple(stage_channels)

        # ------------------------------------------------------------------
        # Two parallel encoder branches.
        # ------------------------------------------------------------------
        self.cnn_encoder = ResNetEncoder(pretrained=encoder_pretrained)
        self.mamba_encoder = VisionMambaEncoder(
            in_channels=3,
            stage_channels=stage_channels,
            depths=mamba_depths,
            d_state=mamba_d_state,
            expand=mamba_expand,
            use_official_kernel=mamba_use_official_kernel,
            max_tokens_fallback=mamba_max_tokens_fallback,
        )

        # ------------------------------------------------------------------
        # One AADFF fusion module per pyramid stage (the research contribution).
        # ------------------------------------------------------------------
        self.fusion_modules = nn.ModuleList(
            [
                AdaptiveAttentionDynamicFusion(in_channels=ch, reduction=fusion_reduction)
                for ch in stage_channels
            ]
        )

        # ------------------------------------------------------------------
        # U-Net decoder consuming the fused pyramid.
        # ------------------------------------------------------------------
        self.decoder = UNetDecoder(
            encoder_channels=stage_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor, return_alpha_maps: bool = False):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch, shape (B, 3, H, W).
        return_alpha_maps : bool
            If True, also returns the list of AADFF mixing-coefficient maps
            (one per pyramid stage) for visualization/analysis.

        Returns
        -------
        torch.Tensor, or (torch.Tensor, List[torch.Tensor])
            Per-pixel class logits, shape (B, num_classes, H, W), and
            optionally the per-stage alpha maps.
        """
        input_size = x.shape[-2:]

        cnn_feats = self.cnn_encoder(x)        # [f1, f2, f3, f4] local features
        mamba_feats = self.mamba_encoder(x)    # [m1, m2, m3, m4] global-context features

        fused_feats: List[torch.Tensor] = []
        alpha_maps: List[torch.Tensor] = []
        for fusion_module, cnn_f, mamba_f in zip(self.fusion_modules, cnn_feats, mamba_feats):
            if return_alpha_maps:
                fused, alpha = fusion_module(cnn_f, mamba_f, return_alpha=True)
                alpha_maps.append(alpha)
            else:
                fused = fusion_module(cnn_f, mamba_f)
            fused_feats.append(fused)

        logits = self.decoder(fused_feats, output_size=input_size)

        if return_alpha_maps:
            return logits, alpha_maps
        return logits

    def get_trainable_parameter_groups(self, base_lr: float, backbone_lr_mult: float = 0.1):
        """Convenience helper for differential learning rates.

        It is common practice to fine-tune a pretrained backbone (the CNN
        encoder here) with a smaller learning rate than the rest of the
        (randomly initialized) network. Returns a list of parameter-group
        dicts ready to be passed straight to a PyTorch optimizer.
        """
        backbone_params = list(self.cnn_encoder.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.parameters() if id(p) not in backbone_ids]

        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
            {"params": other_params, "lr": base_lr},
        ]

    def count_parameters(self) -> dict:
        """Return a breakdown of trainable parameter counts per sub-module."""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "cnn_encoder": _count(self.cnn_encoder),
            "mamba_encoder": _count(self.mamba_encoder),
            "fusion_modules": _count(self.fusion_modules),
            "decoder": _count(self.decoder),
            "total": _count(self),
        }


def build_model(config) -> DFFMambaNet:
    """Factory that builds a :class:`DFFMambaNet` directly from a
    ``configs.config.Config``-like object, so ``train.py``/``test.py``/
    ``inference.py`` only need a single call site.
    """
    return DFFMambaNet(
        num_classes=config.NUM_CLASSES,
        encoder_pretrained=config.ENCODER_PRETRAINED,
        stage_channels=config.STAGE_CHANNELS,
        mamba_d_state=config.MAMBA_D_STATE,
        mamba_expand=config.MAMBA_EXPAND,
        mamba_depths=config.MAMBA_BLOCKS_PER_STAGE,
        mamba_use_official_kernel=config.MAMBA_USE_OFFICIAL_KERNEL,
        mamba_max_tokens_fallback=config.MAMBA_MAX_TOKENS_FALLBACK,
        fusion_reduction=config.FUSION_REDUCTION,
        decoder_channels=config.DECODER_CHANNELS,
    )


if __name__ == "__main__":
    model = DFFMambaNet(num_classes=6, encoder_pretrained=False, mamba_use_official_kernel=False)
    dummy = torch.randn(2, 3, 256, 256)
    logits, alphas = model(dummy, return_alpha_maps=True)
    print(f"Output logits shape: {logits.shape}")
    for i, a in enumerate(alphas, start=1):
        print(f"  Stage {i} alpha map: {a.shape}, mean={a.mean().item():.3f}")
    print("Parameter counts:", model.count_parameters())
