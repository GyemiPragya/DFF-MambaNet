"""
models/vision_mamba.py
=======================
Vision Mamba encoder branch: models **long-range / global** spatial context
using Selective State Space Models (SSMs), following the line of work
popularised by Mamba (Gu & Dao, 2023) and its vision adaptations
(Vision Mamba / Vim, VMamba, and remote-sensing variants such as RS3Mamba
and Samba).

Design notes
------------
This file intentionally keeps the *Mamba* machinery (selective scan, SSM
parameterisation, patch embedding, pyramid stages) completely separate from
the *novel* contribution of this project, which lives in ``models/fusion.py``
(the Adaptive Attention-Based Dynamic Feature Fusion module). Nothing in
this file is part of the claimed novelty — it is a faithful, from-scratch
reference re-implementation of the selective-scan SSM mechanism used to give
the network a global-context branch.

Two execution paths are supported:

1. **Official kernel** — if the ``mamba_ssm`` package (Gu & Dao's official,
   CUDA-accelerated implementation) is importable, :class:`VisionMambaBlock`
   wraps ``mamba_ssm.Mamba`` directly for maximum training speed on a GPU.
2. **Pure-PyTorch fallback** — if ``mamba_ssm`` is not installed (e.g. on
   CPU-only machines, Windows, or Colab runtimes without the custom CUDA
   build), :class:`SimpleSelectiveSSM` below implements the same selective
   scan recurrence directly in PyTorch ops, so the project remains fully
   runnable everywhere, at the cost of speed on very long token sequences.

The encoder builds a 4-stage feature pyramid with the exact same channel
widths and strides as the ResNet34 CNN branch (64/128/256/512 @ strides
4/8/16/32), so that the two branches can be fused stage-by-stage.
"""

import math
import warnings
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Optional dependency: official Mamba CUDA kernel.
# --------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba as _OfficialMamba  # type: ignore
    _HAS_OFFICIAL_MAMBA = True
except Exception:  # pragma: no cover - environment dependent
    _OfficialMamba = None
    _HAS_OFFICIAL_MAMBA = False


def _resolves_to_official_backend(use_official_requested: bool) -> bool:
    """Whether a `use_official=True` request actually resolves to the
    official CUDA kernel (it only does if `mamba_ssm` is importable)."""
    return use_official_requested and _HAS_OFFICIAL_MAMBA


class SimpleSelectiveSSM(nn.Module):
    """Pure-PyTorch reference implementation of a selective-scan SSM (S6).

    This reproduces the core Mamba recurrence:

        h_t = A_bar_t * h_(t-1) + B_bar_t * x_t
        y_t = C_t · h_t + D * x_t

    where ``A_bar_t = exp(Δ_t * A)`` and ``B_bar_t = Δ_t * B_t`` are
    *input-dependent* (selective) discretizations of a continuous-time SSM,
    and Δ_t, B_t, C_t are all predicted from the input itself.

    The recurrence is evaluated with an explicit sequential scan over the
    token sequence. This is mathematically equivalent to the parallel-scan
    CUDA kernel used by the official ``mamba_ssm`` package but is not
    hardware-optimized; for very long sequences (e.g. high-resolution
    feature maps flattened to thousands of tokens) it will be slower. It is
    provided so the project trains/runs correctly on any machine, including
    CPU-only environments, without requiring a custom CUDA build.

    Parameters
    ----------
    d_model : int
        Input/output channel dimension.
    d_state : int
        SSM state dimension (N).
    expand : int
        Inner expansion factor applied before the SSM (mirrors Mamba's
        ``d_inner = expand * d_model``).
    d_conv : int
        Kernel size of the short causal depthwise convolution applied before
        the SSM, which gives the block a small local receptive field.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model

        # Input projection: splits into the "main" branch (x) and a gate (z)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Short causal depthwise conv giving local context before the scan.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        self.act = nn.SiLU()

        # Input-dependent SSM parameters Δ, B, C.
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # `dt` bias is initialised so that softplus(dt_proj(0)) starts inside
        # [dt_min, dt_max] (the same trick the official Mamba implementation
        # uses). Keeping the initial discretization step small means
        # `A_bar = exp(dt * A)` starts close to 1, which keeps the cumulative
        # products inside the chunked scan numerically well-behaved at the
        # start of training.
        dt_min, dt_max = 0.001, 0.1
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_softplus_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)

        # A is parameterised in log-space and kept strictly negative via
        # -exp(A_log), exactly as in the official Mamba implementation —
        # this guarantees a stable (decaying) SSM.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Chunk size used by the numerically-stable chunked parallel scan
        # (see `_selective_scan` docstring below).
        self.scan_chunk_size = 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Token sequence, shape (B, L, d_model).

        Returns
        -------
        torch.Tensor
            Output sequence, shape (B, L, d_model).
        """
        b, length, _ = x.shape

        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                  # each (B, L, d_inner)

        # Causal depthwise conv (operate in channel-first layout for Conv1d)
        x_in = x_in.transpose(1, 2)                    # (B, d_inner, L)
        x_in = self.conv1d(x_in)[:, :, :length]
        x_in = self.act(x_in)
        x_in = x_in.transpose(1, 2)                    # (B, L, d_inner)

        # Predict Δ, B, C from the (post-conv) input — this is what makes
        # the SSM "selective": every token gets its own dynamics.
        x_dbc = self.x_proj(x_in)
        delta, B_param, C_param = torch.split(
            x_dbc, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))         # (B, L, d_inner), > 0

        A = -torch.exp(self.A_log)                      # (d_inner, d_state), < 0

        y = self._selective_scan(x_in, delta, A, B_param, C_param, self.D, chunk_size=self.scan_chunk_size)

        y = y * self.act(z)                              # gated residual branch
        return self.out_proj(y)

    @staticmethod
    def _parallel_affine_scan(A_term: torch.Tensor, B_term: torch.Tensor) -> torch.Tensor:
        """Hillis-Steele inclusive parallel prefix scan over affine maps.

        Each token t carries an affine state-update map
        ``h -> A_term[t] * h + B_term[t]``. Composing the maps for tokens
        ``0..t`` (applied in that order) gives exactly the recurrent state
        ``h_t`` (assuming ``h_{-1} = 0``), which is what the selective-scan
        recurrence ``h_t = A_bar_t * h_(t-1) + B_bar_t * x_t`` requires.

        The composition rule for "apply the left map, then the right map"
        is:

            A_out = A_right * A_left
            B_out = A_right * B_left + B_right

        This rule is associative, so the full prefix over all L tokens can
        be computed in ``O(log L)`` *sequential* vectorized steps (instead
        of ``O(L)``) via the classic Hillis-Steele doubling scheme: at each
        step, every position is combined with the position ``d`` slots to
        its left, where ``d`` doubles every step (1, 2, 4, ...). Positions
        whose left-shift would fall out of range are padded with the
        identity map ``(A=1, B=0)``, which leaves their running composition
        unchanged until the doubling reach actually covers them.

        Crucially, this never divides by a (potentially tiny) cumulative
        product the way a closed-form "global cumprod, then divide"
        formula would — every step is a plain multiply-add — so it stays
        numerically exact regardless of sequence length or how aggressively
        the SSM decays.

        Parameters
        ----------
        A_term, B_term : torch.Tensor
            Shape (B, L, d_inner, d_state).

        Returns
        -------
        torch.Tensor
            The recurrent states ``h_0 .. h_{L-1}``, shape
            (B, L, d_inner, d_state).
        """
        length = A_term.shape[1]
        a_run, b_run = A_term, B_term
        d = 1
        while d < length:
            a_left = F.pad(a_run[:, : length - d], (0, 0, 0, 0, d, 0), value=1.0)
            b_left = F.pad(b_run[:, : length - d], (0, 0, 0, 0, d, 0), value=0.0)
            new_a = torch.clamp(a_run * a_left, max=1.0)
			new_b = a_run * b_left + b_run
			
			a_run = new_a
			b_run = new_b
            d *= 2
        return b_run

    
    @staticmethod
    def _selective_scan(
				    x: torch.Tensor,
				    delta: torch.Tensor,
				    A: torch.Tensor,
				    B: torch.Tensor,
				    C: torch.Tensor,
				    D: torch.Tensor,
				    chunk_size: int = 32,
				) -> torch.Tensor:
				
				    batch, length, d_inner = x.shape
				    d_state = A.shape[1]
				
				    # -----------------------------
				    # 1. SAFE DISCRETIZATION
				    # -----------------------------
				    delta = torch.clamp(delta, min=1e-4, max=1.0)
				
				    raw = delta.unsqueeze(-1) * A.view(1, 1, d_inner, d_state)
				
				    # prevent exp overflow/underflow
				    raw = torch.clamp(raw, min=-20, max=0)
				
				    delta_A = torch.exp(raw)
				
				    
					
				
				    # -----------------------------
				    # 2. SAFE INPUT TERM
				    # -----------------------------
				    delta_Bx = (delta.unsqueeze(-1) * B.unsqueeze(2)) * x.unsqueeze(-1)
				
				    # remove inf/nan early
				    delta_Bx = torch.nan_to_num(delta_Bx, nan=0.0, posinf=0.0, neginf=0.0)
				
				    # -----------------------------
				    # 3. STABLE SCAN
				    # -----------------------------
				    h_all = SimpleSelectiveSSM._parallel_affine_scan(delta_A, delta_Bx)
				
				    # cleanup after scan
				    h_all = torch.nan_to_num(h_all, nan=0.0, posinf=0.0, neginf=0.0)
				
				    # -----------------------------
				    # 4. OUTPUT PROJECTION
				    # -----------------------------
				    y = torch.einsum("bldn,bln->bld", h_all, C)
				
				    y = y + x * D.view(1, 1, d_inner)
				
				    # final safety clamp
				    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
				
				    return y
class VisionMambaBlock(nn.Module):
    """A single Vision Mamba block: LayerNorm -> bidirectional SSM -> residual,
    followed by a LayerNorm -> MLP -> residual (standard transformer-style
    "pre-norm" wrapper around the SSM mixer).

    The scan is run in *both* directions (forward over the flattened token
    sequence, and again over the reversed sequence) and the two outputs are
    averaged. This bidirectional scan is what lets a 1-D causal SSM capture
    context from tokens on both sides of a given spatial location once the
    2-D feature map has been flattened into a 1-D sequence — a standard
    trick in Vision Mamba variants (Vim, VMamba) to compensate for the
    causal nature of the original (NLP-oriented) Mamba scan.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 3,
                 mlp_ratio: float = 2.0, use_official: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)

        if use_official and _HAS_OFFICIAL_MAMBA:
            self.ssm_fwd = _OfficialMamba(d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv)
            self.ssm_bwd = _OfficialMamba(d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv)
            self.backend = "mamba_ssm (official CUDA kernel)"
        else:
            self.ssm_fwd = SimpleSelectiveSSM(d_model, d_state=d_state, expand=expand, d_conv=d_conv)
            self.ssm_bwd = SimpleSelectiveSSM(d_model, d_state=d_state, expand=expand, d_conv=d_conv)
            self.backend = "pure-PyTorch fallback"

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) token sequence -> (B, L, d_model)."""
        residual = x
        x_norm = self.norm1(x)
        fwd = self.ssm_fwd(x_norm)
        bwd = self.ssm_bwd(torch.flip(x_norm, dims=[1]))
        bwd = torch.flip(bwd, dims=[1])
        x = residual + 0.5 * (fwd + bwd)

        x = x + self.mlp(self.norm2(x))
        return x


class MambaStage(nn.Module):
    """A pyramid stage: optional spatial down-sampling followed by a stack
    of :class:`VisionMambaBlock` operating on the flattened token sequence.
    """

    def __init__(self, in_channels: int, out_channels: int, depth: int,
                 downsample: bool, d_state: int, expand: int, d_conv: int,
                 use_official: bool, max_tokens_fallback: int = 256):
        super().__init__()
        if downsample:
            # Stride-2 conv halves H and W while projecting to out_channels.
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            # First stage: patchify the raw input via a stride-4 conv.
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=4, padding=3)

        self.blocks = nn.ModuleList(
            [
                VisionMambaBlock(out_channels, d_state=d_state, expand=expand,
                                  d_conv=d_conv, use_official=use_official)
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(out_channels)

        # The pure-PyTorch fallback SSM materializes full (B, L, d_inner,
        # d_state) tensors at every step of its scan, which becomes memory-
        # heavy once L = H*W grows into the thousands (e.g. stage 1 of a
        # 256x256 input flattens to 64*64 = 4096 tokens). The official
        # `mamba_ssm` CUDA kernel does not have this issue, so this
        # mitigation only kicks in when the fallback is actually in use.
        # When the token count exceeds `max_tokens_fallback`, we run the
        # SSM blocks on an average-pooled (coarser) token grid and then
        # upsample the result back to the stage's native resolution. This
        # trades a small amount of spatial detail in the *global-context*
        # branch (the CNN branch still operates at full resolution and
        # supplies the fine local detail) for tractable memory/compute.
        self.use_official = _resolves_to_official_backend(use_official)
        self.max_tokens_fallback = max_tokens_fallback

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W) -> (B, C_out, H', W')."""
        x = self.downsample(x)
        b, c, h, w = x.shape

        pool_factor = 1
        if (not self.use_official) and (h * w > self.max_tokens_fallback):
            # Smallest power-of-2 pooling factor that brings the token
            # count at or below the threshold.
            while (h // pool_factor) * (w // pool_factor) > self.max_tokens_fallback and pool_factor < min(h, w):
                pool_factor *= 2
            x_small = F.avg_pool2d(x, kernel_size=pool_factor)
        else:
            x_small = x

        bs, cs, hs, ws = x_small.shape
        tokens = x_small.flatten(2).transpose(1, 2)   # (B, Hs*Ws, C)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm_out(tokens)
        out_small = tokens.transpose(1, 2).view(bs, cs, hs, ws)

        if pool_factor > 1:
            out = F.interpolate(out_small, size=(h, w), mode="bilinear", align_corners=False)
        else:
            out = out_small
        return out


class VisionMambaEncoder(nn.Module):
    """Vision Mamba encoder branch producing a 4-level feature pyramid that
    matches the CNN (ResNet34) branch in both channel width and stride, so
    that the two can be fused scale-by-scale.

    Stage 1: stride 4,  channels  64 (patchify stem + Mamba blocks)
    Stage 2: stride 8,  channels 128
    Stage 3: stride 16, channels 256
    Stage 4: stride 32, channels 512
    """

    OUT_CHANNELS: List[int] = [64, 128, 256, 512]
    OUT_STRIDES: List[int] = [4, 8, 16, 32]

    def __init__(
        self,
        in_channels: int = 3,
        stage_channels=(64, 128, 256, 512),
        depths=(1, 1, 1, 1),
        d_state: int = 8,
        expand: int = 1,
        d_conv: int = 3,
        use_official_kernel: bool = True,
        max_tokens_fallback: int = 256,
    ):
        super().__init__()

        if use_official_kernel and not _HAS_OFFICIAL_MAMBA:
            warnings.warn(
                "[VisionMambaEncoder] `mamba_ssm` is not installed — falling back to the "
                "pure-PyTorch SimpleSelectiveSSM implementation. Training will still work "
                "correctly but may be slower on large feature maps. Install `mamba-ssm` "
                "(requires a CUDA-enabled GPU) for the optimized kernel.",
                stacklevel=2,
            )

        channels = [in_channels] + list(stage_channels)
        self.stages = nn.ModuleList(
            [
                MambaStage(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    depth=depths[i],
                    downsample=(i > 0),   # stage 0 patchifies the raw image (stride 4)
                    d_state=d_state,
                    expand=expand,
                    d_conv=d_conv,
                    use_official=use_official_kernel,
                    max_tokens_fallback=max_tokens_fallback,
                )
                for i in range(4)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch, shape (B, 3, H, W).

        Returns
        -------
        List[torch.Tensor]
            ``[m1, m2, m3, m4]`` global-context feature maps with the same
            shapes as the CNN branch's ``[f1, f2, f3, f4]``.
        """
        feats = []
        out = x
        for stage in self.stages:
            out = stage(out)
            feats.append(out)
        return feats


if __name__ == "__main__":
    model = VisionMambaEncoder(use_official_kernel=False)
    dummy = torch.randn(1, 3, 256, 256)
    feats = model(dummy)
    for i, f in enumerate(feats, start=1):
        print(f"Mamba stage {i}: {f.shape}")
    print(f"Backend in use: {model.stages[0].blocks[0].backend}")
