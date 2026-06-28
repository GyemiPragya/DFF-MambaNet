"""
models/fusion.py
=================
**Adaptive Attention-Based Dynamic Feature Fusion (AADFF)** — the central
research contribution of this project.

Motivation
----------
The CNN branch (ResNet34) is excellent at capturing fine-grained local
texture and sharp object boundaries, but its limited receptive field means
it struggles with long-range context (e.g. telling apart a large flat
rooftop from a parking lot purely from local texture). The Vision Mamba
branch captures exactly that long-range, global context via its
selective-scan SSM, but is comparatively weaker at precise local detail.

Most existing hybrid CNN-SSM segmentation works combine the two branches
with a **static** rule — typically a fixed concatenation followed by a
plain convolution, or a fixed-weight sum. This throws away useful
information: the *ideal* mixing ratio between "trust the local CNN
evidence" and "trust the global Mamba evidence" is not a constant — it
varies from image to image, from spatial location to spatial location, and
from channel to channel (e.g. a "Car" pixel near a sharp boundary should
lean on the CNN branch, while a large, texturally ambiguous "Low
Vegetation" region should lean on the Mamba branch's global context).

AADFF addresses this directly. Conceptually:

    Fusion = alpha ⊙ CNN_refined + (1 - alpha) ⊙ Mamba_refined

where ``alpha`` is **not** a fixed scalar but a full spatial map predicted
*dynamically*, per-sample, from the two input feature maps themselves, via a
combination of:

    1. **Channel attention** — re-weights each branch's feature channels
       using global context (squeeze-and-excite style, with both average-
       and max-pooled descriptors), so that the most informative channels
       from each branch dominate before fusion.
    2. **Spatial attention** — re-weights each branch's features at every
       spatial location, so that, e.g., the CNN branch can be emphasized
       precisely along object boundaries while the Mamba branch is
       emphasized in broad, homogeneous regions.
    3. **Adaptive dynamic gating** — a lightweight gating network predicts
       the per-location, per-channel mixing coefficient ``alpha`` from the
       two *attention-refined* feature maps, and the final fused
       representation is the convex combination above.

Every step is implemented as a small, clearly separated sub-module so the
mechanism stays transparent and easy to inspect/ablate.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-excite style channel attention.

    Produces a per-channel gate in (0, 1) from both global average-pooled
    and global max-pooled descriptors of the input, following the design
    of CBAM (Woo et al., 2018), which empirically improves on using
    average-pooling alone.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> channel gate of shape (B, C, 1, 1)."""
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention.

    Produces a single-channel, per-pixel gate in (0, 1) from the channel-
    wise average and max statistics at every spatial location, processed
    by a small convolution to incorporate local context into the gate.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> spatial gate of shape (B, 1, H, W)."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        gate_input = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(gate_input))


class BranchAttentionRefiner(nn.Module):
    """Applies channel attention then spatial attention to a single branch's
    feature map, refining it before the dynamic fusion gate is computed.
    """

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> attention-refined (B, C, H, W)."""
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class AdaptiveAttentionDynamicFusion(nn.Module):
    """**AADFF**: Adaptive Attention-Based Dynamic Feature Fusion.

    Fuses a CNN (local) feature map and a Vision Mamba (global) feature
    map of identical shape into a single representation, using a learned,
    input-dependent mixing coefficient rather than a fixed combination
    rule.

    Pipeline
    --------
    1. Independently refine each branch with channel + spatial attention
       (:class:`BranchAttentionRefiner`).
    2. Concatenate the two refined branches and predict a per-location,
       per-channel gate ``alpha`` in (0, 1) via a lightweight 1x1-conv
       gating network.
    3. Fuse: ``out = alpha * cnn_refined + (1 - alpha) * mamba_refined``.
    4. Pass the fused tensor through a final 3x3 convolution + BatchNorm +
       ReLU to smooth the combination and project back to the target
       channel width (in case the caller requests a different output width
       than the input branches).

    Parameters
    ----------
    in_channels : int
        Channel width of *both* input branches (they must match, since one
        of the central design choices here is that CNN and Mamba branches
        share channel widths stage-by-stage — see ``models/encoder.py`` and
        ``models/vision_mamba.py``).
    out_channels : int, optional
        Output channel width after the final fusion conv. Defaults to
        ``in_channels`` (no projection).
    reduction : int
        Channel-attention squeeze ratio (see :class:`ChannelAttention`).
    spatial_kernel : int
        Kernel size of the spatial-attention convolution.
    """

    def __init__(self, in_channels: int, out_channels: int = None,
                 reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        out_channels = out_channels or in_channels

        # --- Step 1: per-branch attention refinement ---------------------
        self.cnn_refiner = BranchAttentionRefiner(in_channels, reduction, spatial_kernel)
        self.mamba_refiner = BranchAttentionRefiner(in_channels, reduction, spatial_kernel)

        # --- Step 2: adaptive dynamic gating network ----------------------
        # Takes the concatenation of both *refined* branches (2*C channels)
        # and predicts a per-pixel, per-channel mixing coefficient alpha
        # with the SAME shape as a single branch (C channels), so that the
        # mixing ratio can vary not just spatially but per-channel too.
        gate_hidden = max(in_channels // reduction, 8)
        self.gate_network = nn.Sequential(
            nn.Conv2d(in_channels * 2, gate_hidden, kernel_size=1, bias=True),
            nn.BatchNorm2d(gate_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),   # alpha in (0, 1), elementwise
        )

        # --- Step 3/4: post-fusion smoothing + output projection ---------
        self.output_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, cnn_feat: torch.Tensor, mamba_feat: torch.Tensor,
                return_alpha: bool = False):
        """Fuse a CNN feature map with a Vision Mamba feature map.

        Parameters
        ----------
        cnn_feat : torch.Tensor
            Local features from the CNN branch, shape (B, C, H, W).
        mamba_feat : torch.Tensor
            Global-context features from the Vision Mamba branch, shape
            (B, C, H, W) — must match ``cnn_feat`` in shape.
        return_alpha : bool
            If True, also returns the predicted mixing-coefficient map
            (useful for visualizing *where* the network is leaning on the
            CNN branch vs. the Mamba branch).

        Returns
        -------
        torch.Tensor, or (torch.Tensor, torch.Tensor) if `return_alpha`
            The fused feature map of shape (B, out_channels, H, W), and
            optionally the alpha map of shape (B, C, H, W).
        """
        if cnn_feat.shape != mamba_feat.shape:
            raise ValueError(
                f"AADFF expects matching shapes for both branches, got "
                f"cnn_feat={tuple(cnn_feat.shape)} vs mamba_feat={tuple(mamba_feat.shape)}. "
                "Make sure the CNN and Vision Mamba encoders use matching stage "
                "channel widths/strides."
            )

        # Step 1: refine each branch independently with channel + spatial attention.
        cnn_refined = self.cnn_refiner(cnn_feat)
        mamba_refined = self.mamba_refiner(mamba_feat)

        # Step 2: predict the dynamic, input-dependent mixing coefficient alpha
        # from the concatenation of both refined branches.
        concat_feat = torch.cat([cnn_refined, mamba_refined], dim=1)
        alpha = self.gate_network(concat_feat)

        # Step 3: convex combination — Fusion = alpha * CNN + (1 - alpha) * Mamba.
        fused = alpha * cnn_refined + (1.0 - alpha) * mamba_refined

        # Step 4: smooth the combination and project to the desired output width.
        fused = self.output_proj(fused)

        if return_alpha:
            return fused, alpha
        return fused


if __name__ == "__main__":
    fusion = AdaptiveAttentionDynamicFusion(in_channels=128)
    cnn_feat = torch.randn(2, 128, 32, 32)
    mamba_feat = torch.randn(2, 128, 32, 32)
    out, alpha = fusion(cnn_feat, mamba_feat, return_alpha=True)
    print(f"Fused output shape: {out.shape}")
    print(f"Alpha map shape:    {alpha.shape}  (mean={alpha.mean().item():.3f})")
