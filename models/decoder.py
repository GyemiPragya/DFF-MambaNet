"""
models/decoder.py
==================
U-Net style decoder that progressively upsamples the fused multi-scale
feature pyramid (produced by stacking :class:`~models.fusion.AdaptiveAttentionDynamicFusion`
at every encoder stage) back up to a dense, per-pixel segmentation map.

Skip connections from shallower (higher-resolution) fused stages are
concatenated at each upsampling step, in the classic U-Net fashion, so that
fine spatial detail lost during encoder downsampling can be recovered.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU layers — the standard U-Net "double conv"."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """One decoder stage: upsample, concatenate the skip connection, then
    a double-conv block to mix the combined features.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # Learned (transposed-conv) upsampling tends to give crisper
        # boundaries than plain bilinear upsampling for this kind of dense
        # prediction task, at a modest parameter cost.
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Defensive resize in case of off-by-one spatial mismatches caused
        # by odd input resolutions (e.g. patch sizes not divisible by 32).
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """U-Net decoder consuming a 4-level fused feature pyramid.

    Parameters
    ----------
    encoder_channels : Sequence[int]
        Channel widths of the fused pyramid, shallow-to-deep, e.g.
        ``(64, 128, 256, 512)`` to match the CNN/Mamba encoder stages.
    decoder_channels : Sequence[int]
        Channel widths used internally by each decoder stage, deep-to-
        shallow, e.g. ``(256, 128, 64, 32)``.
    num_classes : int
        Number of output segmentation classes.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int] = (64, 128, 256, 512),
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        num_classes: int = 6,
    ):
        super().__init__()
        assert len(encoder_channels) == 4, "Decoder expects a 4-level feature pyramid."
        assert len(decoder_channels) == 4, "Decoder expects 4 decoder stages."

        c1, c2, c3, c4 = encoder_channels  # shallow -> deep (strides 4, 8, 16, 32)
        d1, d2, d3, d4 = decoder_channels  # used deep -> shallow

        # Stage operating on the deepest features (stride 32 -> 16),
        # skip-connecting with stage-3 fused features (c3 channels).
        self.up1 = UpBlock(in_channels=c4, skip_channels=c3, out_channels=d1)
        # stride 16 -> 8, skip with stage-2 fused features (c2 channels)
        self.up2 = UpBlock(in_channels=d1, skip_channels=c2, out_channels=d2)
        # stride 8 -> 4, skip with stage-1 fused features (c1 channels)
        self.up3 = UpBlock(in_channels=d2, skip_channels=c1, out_channels=d3)
        # stride 4 -> 1 (back to input resolution): no further skip connection
        # is available, so we use a plain upsample + conv "head".
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(d3, d4, kernel_size=4, stride=4),
            ConvBlock(d4, d4),
        )

        self.classifier = nn.Conv2d(d4, num_classes, kernel_size=1)

    def forward(self, fused_features: List[torch.Tensor], output_size=None) -> torch.Tensor:
        """
        Parameters
        ----------
        fused_features : List[torch.Tensor]
            ``[f1, f2, f3, f4]`` fused features, shallow-to-deep (strides
            4, 8, 16, 32), as produced by the network's 4 AADFF modules.
        output_size : Tuple[int, int], optional
            If given, the final logits are resized (bilinear) to exactly
            this (H, W) — used to guarantee the output matches the
            original input resolution regardless of small rounding effects
            from strided convolutions.

        Returns
        -------
        torch.Tensor
            Per-pixel class logits, shape (B, num_classes, H, W).
        """
        f1, f2, f3, f4 = fused_features

        x = self.up1(f4, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)
        x = self.up4(x)

        logits = self.classifier(x)

        if output_size is not None and logits.shape[-2:] != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

        return logits


if __name__ == "__main__":
    decoder = UNetDecoder()
    f1 = torch.randn(2, 64, 64, 64)
    f2 = torch.randn(2, 128, 32, 32)
    f3 = torch.randn(2, 256, 16, 16)
    f4 = torch.randn(2, 512, 8, 8)
    out = decoder([f1, f2, f3, f4], output_size=(256, 256))
    print(f"Decoder output shape: {out.shape}")
