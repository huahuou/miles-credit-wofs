# credit/models/wxformer/aurora_crossformer_wrf.py
#
# AuroraCrossFormerWRF — Aurora-inspired front-end + CrossFormer UNet backbone
#
# Key difference from CrossFormerWRFEnsemble:
#
#   1. VerticalLevelEmbedder  — replaces the flat VariableTokenEmbedder.
#      Each (upper-air variable, level) pair gets its own independent 1×1 Conv2d
#      projector (inspired by Aurora's per-variable Conv3d in LevelPatchEmbed).
#      A learnable level encoding (nn.Embedding) is added before summation so the
#      backbone can distinguish eta level 3 from eta level 14 of the same variable.
#      2-D surface/forcing/static variables each get a single per-variable projector.
#
#   2. Enhanced FiLM — 4-quadrant boundary pooling retains coarse spatial structure
#      of the boundary context rather than discarding it via global average pooling.
#
#   3. Everything else unchanged: CrossFormer encoder, AnticheckboardUpBlock decoder,
#      StochasticDecompositionLayer noise injection, TensorPadding, VariableTokenDecoder.
#
# Trainer compatibility
# ─────────────────────
# forward(x, x_boundary, x_time_encode, forecast_step=0, ensemble_size=1)
# Identical signature to CrossFormerWRFEnsemble. Compatible with
# TrainerWRFMulti ('multi-step-wrf') and TrainerWRFMultiEnsemble
# ('multi-step-wrf-ensemble') with tendency_boundary: True.
#
# Config key: model.type: 'aurora_crossformer_wrf'

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from credit.boundary_padding import TensorPadding
from credit.models.base_model import BaseModel
from credit.models.crossformer import CrossEmbedLayer, Transformer, apply_spectral_norm
from credit.models.wxformer.crossformer_wrf_ensemble import (
    AnticheckboardUpBlock,
    VariableTokenDecoder,
    _num_groups,
)
from credit.models.wxformer.stochastic_decomposition_layer import StochasticDecompositionLayer
from credit.postblock import PostBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VerticalLevelEmbedder
# ---------------------------------------------------------------------------

class VerticalLevelEmbedder(nn.Module):
    """
    Aurora-inspired per-(variable, level) independent 1×1 projector.

    For each upper-air variable V with L levels and F input frames per level:
        - Registers L separate Conv2d(F, D, 1) projectors keyed 'V_L0' … 'V_L{L-1}'.
        - Adds a learnable level bias (nn.Embedding of shape (L, D)) to each
          projection so the backbone can distinguish vertical levels.
        - Applies an independent learnable scalar sigmoid gate per (var, level).

    For each 2-D variable (surface, dyn_forcing, forcing, static):
        - Registers a single Conv2d(F, D, 1) projector keyed by variable name.
        - Applies a learnable scalar sigmoid gate.

    All projections are *summed* into the shared embed_dim and LayerNorm-ed,
    preserving backbone width D regardless of how many (var, level) pairs exist.

    Variable flexibility
    ~~~~~~~~~~~~~~~~~~~~
    * Drop a variable at inference: exclude its channels from the input tensor.
      The registered projectors still exist in the state dict but are never called.
    * Add a new upper-air variable: call add_upper_air_variable(name, start_ch).
      Registers L new Conv2d projectors (zero-init) without touching the backbone.
    * Add a new 2-D variable: call add_surface_variable(name, start_ch).
    * Load old checkpoints with strict=False when extending the variable set.

    Args:
        upper_air_spec (list): [(name, start_ch), ...] — upper-air variables.
                               Channels start_ch … start_ch + L*F - 1 in flat x.
        surface_spec   (list): [(name, start_ch), ...] — 2-D variables.
                               Channels start_ch … start_ch + F - 1 in flat x.
        embed_dim      (int):  output embedding dimension D.
        frames         (int):  number of input time frames F (default 1).
        levels         (int):  number of vertical levels L (default 17).
    """

    def __init__(
        self,
        upper_air_spec: List[Tuple[str, int]],
        surface_spec: List[Tuple[str, int]],
        embed_dim: int,
        frames: int = 1,
        levels: int = 17,
    ):
        super().__init__()
        self.upper_air_spec: List[Tuple[str, int]] = list(upper_air_spec)
        self.surface_spec: List[Tuple[str, int]] = list(surface_spec)
        self.embed_dim = embed_dim
        self.frames = frames
        self.levels = levels

        self.projectors = nn.ModuleDict()
        self.gate_logits = nn.ParameterDict()

        # Per-(var, level) projectors for upper-air variables
        for name, _ in upper_air_spec:
            for lev in range(levels):
                key = f"{name}_L{lev}"
                self.projectors[key] = nn.Conv2d(frames, embed_dim, kernel_size=1)
                self.gate_logits[key] = nn.Parameter(torch.zeros(1))

        # Per-variable projectors for 2-D variables
        for name, _ in surface_spec:
            self.projectors[name] = nn.Conv2d(frames, embed_dim, kernel_size=1)
            self.gate_logits[name] = nn.Parameter(torch.zeros(1))

        # Learnable vertical level encoding: shape (L, D)
        self.level_embed = nn.Embedding(levels, embed_dim)
        nn.init.normal_(self.level_embed.weight, std=0.02)

        # Channel-wise LayerNorm applied after summing projections
        self.norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_total, H, W) — flat concatenation of all variable channels.
        Returns:
            (B, embed_dim, H, W)
        """
        out: Optional[torch.Tensor] = None

        # Precompute all level encodings in one Embedding lookup
        lev_indices = torch.arange(self.levels, device=x.device)
        level_encs = self.level_embed(lev_indices)  # (L, D)

        # ── upper-air variables (per level) ───────────────────────────────
        for name, start_ch in self.upper_air_spec:
            for lev in range(self.levels):
                key = f"{name}_L{lev}"
                lev_start = start_ch + lev * self.frames
                x_lev = x[:, lev_start:lev_start + self.frames]     # (B, F, H, W)
                proj = self.projectors[key](x_lev)                   # (B, D, H, W)
                # Add level positional encoding (broadcast over spatial dims)
                proj = proj + level_encs[lev].view(1, -1, 1, 1)
                gate = torch.sigmoid(self.gate_logits[key])          # scalar
                out = gate * proj if out is None else out + gate * proj

        # ── 2-D variables (surface, dyn_forcing, forcing, static) ─────────
        for name, start_ch in self.surface_spec:
            x_var = x[:, start_ch:start_ch + self.frames]            # (B, F, H, W)
            proj = self.projectors[name](x_var)                      # (B, D, H, W)
            gate = torch.sigmoid(self.gate_logits[name])
            out = gate * proj if out is None else out + gate * proj

        if out is None:
            raise RuntimeError("VerticalLevelEmbedder: both specs are empty")

        # LayerNorm over the channel axis (treat H, W as batch)
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(-1, C)                # (B*H*W, D)
        out = self.norm(out)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)         # (B, D, H, W)

    # ------------------------------------------------------------------
    def add_upper_air_variable(self, name: str, start_ch: int) -> None:
        """Register a new upper-air variable (L new projectors, zero-init)."""
        if any(n == name for n, _ in self.upper_air_spec):
            return
        for lev in range(self.levels):
            key = f"{name}_L{lev}"
            proj = nn.Conv2d(self.frames, self.embed_dim, kernel_size=1)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.projectors[key] = proj
            self.gate_logits[key] = nn.Parameter(torch.zeros(1))
        self.upper_air_spec.append((name, start_ch))

    def add_surface_variable(self, name: str, start_ch: int) -> None:
        """Register a new 2-D variable (1 new projector, zero-init)."""
        if name in self.projectors:
            return
        proj = nn.Conv2d(self.frames, self.embed_dim, kernel_size=1)
        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)
        self.projectors[name] = proj
        self.gate_logits[name] = nn.Parameter(torch.zeros(1))
        self.surface_spec.append((name, start_ch))


# ---------------------------------------------------------------------------
# AuroraCrossFormerWRF
# ---------------------------------------------------------------------------

class AuroraCrossFormerWRF(BaseModel):
    """
    Aurora-inspired front-end + CrossFormer UNet backbone for WRF emulation.

    Replaces the flat VariableTokenEmbedder in CrossFormerWRFEnsemble with a
    VerticalLevelEmbedder that gives each (variable, vertical level) pair its
    own independent 1×1 Conv2d projector and adds a learnable level encoding.

    The FiLM boundary conditioning is enhanced with 4-quadrant spatial pooling
    to preserve coarse spatial structure of the boundary/tendency signal.

    All other components are identical to CrossFormerWRFEnsemble:
      • CrossFormer 4-stage UNet encoder (CrossEmbedLayer + Transformer)
      • AnticheckboardUpBlock UNet decoder
      • StochasticDecompositionLayer multi-scale noise injection
      • TensorPadding mirror boundary padding
      • VariableTokenDecoder per-variable output heads

    Args  (identical to CrossFormerWRFEnsemble)
    ────
    varname_upper_air   (list[str]): 3-D prognostic variables.
    varname_surface     (list[str]): 2-D prognostic variables.
    varname_dyn_forcing (list[str]): 2-D dynamic forcing (input-only).
    varname_forcing     (list[str]): 2-D static forcing (input-only).
    varname_static      (list[str]): 2-D static fields (input-only).
    varname_diagnostic  (list[str]): 2-D diagnostic / output-only variables.
    varname_boundary_upper   (list[str]): boundary 3-D variables.
    varname_boundary_surface (list[str]): boundary 2-D variables.
    levels              (int):  vertical levels for interior upper-air vars.
    boundary_levels     (int):  vertical levels for boundary upper-air vars.
    frames              (int):  number of input time steps (default 1).
    embed_dim           (int):  interior token embedding dimension.
    boundary_embed_dim  (int):  boundary token embedding dimension.
    time_encode_dim     (int):  size of the time-encoding input vector.
    dim, depth, dim_head, global_window_size, local_window_size,
    cross_embed_kernel_sizes, cross_embed_strides,
    attn_dropout, ff_dropout, use_spectral_norm: CrossFormer backbone params.
    noise_injection     (dict): stochastic noise configuration.
    padding_conf        (dict): TensorPadding boundary-padding configuration.
    post_conf           (dict): PostBlock post-processing configuration.
    """

    def __init__(
        self,
        varname_upper_air: List[str],
        varname_surface: List[str],
        varname_dyn_forcing: List[str],
        varname_forcing: List[str],
        varname_static: List[str],
        varname_diagnostic: List[str],
        varname_boundary_upper: List[str],
        varname_boundary_surface: List[str],
        levels: int = 17,
        boundary_levels: int = 17,
        frames: int = 1,
        embed_dim: int = 192,
        boundary_embed_dim: int = 64,
        time_encode_dim: int = 12,
        dim=(64, 128, 256, 512),
        depth=(2, 2, 8, 2),
        dim_head: int = 32,
        global_window_size=(8, 4, 2, 1),
        local_window_size: int = 8,
        cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides=(2, 2, 2, 2),
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        use_spectral_norm: bool = True,
        noise_injection: Optional[dict] = None,
        padding_conf: Optional[dict] = None,
        post_conf: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()

        # ── coerce sequence args ───────────────────────────────────────────
        dim = tuple(dim)
        depth = tuple(depth)
        global_window_size = tuple(global_window_size)
        cross_embed_kernel_sizes = tuple(tuple(k) for k in cross_embed_kernel_sizes)
        cross_embed_strides = tuple(cross_embed_strides)
        local_window_size = int(local_window_size)

        assert len(dim) == 4, "dim must have exactly 4 elements"
        assert len(depth) == 4, "depth must have exactly 4 elements"
        assert all(s == 2 for s in cross_embed_strides), (
            "AuroraCrossFormerWRF requires cross_embed_strides=(2,2,2,2)."
        )

        self.frames = frames
        self.local_window_size = local_window_size
        self.num_stages = len(dim)

        # ── save variable lists ────────────────────────────────────────────
        self.varname_upper_air = list(varname_upper_air)
        self.varname_surface = list(varname_surface)
        self.varname_dyn_forcing = list(varname_dyn_forcing)
        self.varname_forcing = list(varname_forcing)
        self.varname_static = list(varname_static)
        self.varname_diagnostic = list(varname_diagnostic)

        # ── interior VerticalLevelEmbedder specs ───────────────────────────
        # Channel order in flat x delivered by the trainer:
        #   upper_air × (levels × frames), surface × frames,
        #   dyn_forcing × frames, forcing × frames, static × frames
        int_upper_spec: List[Tuple[str, int]] = []
        int_surf_spec: List[Tuple[str, int]] = []
        cursor = 0

        for name in varname_upper_air:
            int_upper_spec.append((name, cursor))
            cursor += levels * frames
        for name in varname_surface:
            int_surf_spec.append((name, cursor))
            cursor += 1 * frames
        for name in varname_dyn_forcing:
            int_surf_spec.append((name, cursor))
            cursor += 1 * frames
        for name in varname_forcing:
            int_surf_spec.append((name, cursor))
            cursor += 1 * frames
        for name in varname_static:
            int_surf_spec.append((name, cursor))
            cursor += 1 * frames

        # ── boundary VerticalLevelEmbedder specs ───────────────────────────
        bnd_upper_spec: List[Tuple[str, int]] = []
        bnd_surf_spec: List[Tuple[str, int]] = []
        cursor = 0

        for name in varname_boundary_upper:
            bnd_upper_spec.append((f"bnd_{name}", cursor))
            cursor += boundary_levels * frames
        for name in varname_boundary_surface:
            bnd_surf_spec.append((f"bnd_{name}", cursor))
            cursor += 1 * frames

        # ── output spec ───────────────────────────────────────────────────
        out_spec: List[Tuple[str, int]] = []
        for name in varname_upper_air:
            out_spec.append((name, levels))
        for name in varname_surface:
            out_spec.append((name, 1))
        for name in varname_diagnostic:
            out_spec.append((name, 1))
        self._n_out_channels: int = sum(n for _, n in out_spec)

        # ── variable embedders ─────────────────────────────────────────────
        self.interior_embedder = VerticalLevelEmbedder(
            upper_air_spec=int_upper_spec,
            surface_spec=int_surf_spec,
            embed_dim=embed_dim,
            frames=frames,
            levels=levels,
        )
        self.boundary_embedder = VerticalLevelEmbedder(
            upper_air_spec=bnd_upper_spec,
            surface_spec=bnd_surf_spec,
            embed_dim=boundary_embed_dim,
            frames=frames,
            levels=boundary_levels,
        )

        # ── enhanced FiLM: 4-quadrant boundary pool + time_encode → (γ, β) ──
        # 4-quadrant pooling produces 4 × boundary_embed_dim spatial descriptors;
        # global pool adds 1 × boundary_embed_dim; together they retain coarse
        # spatial structure of the boundary signal.
        film_in = 5 * boundary_embed_dim + time_encode_dim
        self.film_mlp = nn.Sequential(
            nn.Linear(film_in, film_in * 2),
            nn.SiLU(),
            nn.Linear(film_in * 2, 2 * embed_dim),
        )

        # ── CrossFormer encoder ───────────────────────────────────────────
        last_dim = dim[-1]
        dims = [embed_dim, *dim]
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.encoder_layers = nn.ModuleList()
        for (d_in, d_out), n_layers, g_wsize, kern, stride in zip(
            dim_pairs,
            depth,
            global_window_size,
            cross_embed_kernel_sizes,
            cross_embed_strides,
        ):
            self.encoder_layers.append(
                nn.ModuleList([
                    CrossEmbedLayer(d_in, d_out, kern, stride),
                    Transformer(
                        d_out,
                        local_window_size=local_window_size,
                        global_window_size=g_wsize,
                        depth=n_layers,
                        dim_head=dim_head,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                    ),
                ])
            )

        # ── Anti-checkerboard UNet decoder ───────────────────────────────
        self.up_block1 = AnticheckboardUpBlock(last_dim, last_dim // 2)
        self.up_block2 = AnticheckboardUpBlock(last_dim // 2 + dim[2], last_dim // 4)
        self.up_block3 = AnticheckboardUpBlock(last_dim // 4 + dim[1], last_dim // 8)
        self.up_block4 = AnticheckboardUpBlock(last_dim // 8 + dim[0], last_dim // 8)

        # ── Output projection ─────────────────────────────────────────────
        self.out_decoder = VariableTokenDecoder(out_spec, last_dim // 8)

        # ── Multi-scale stochastic noise injection ─────────────────────────
        noise_conf = noise_injection or {}
        self.use_noise = bool(noise_conf.get("activate", False))
        self.sample_latent_if_none = bool(noise_conf.get("sample_latent_if_none", True))
        self.freeze_base_model_weights = bool(noise_conf.get("freeze_base_model_weights", False))
        self.encoder_noise = bool(noise_conf.get("encoder_noise", True))
        self.correlated = bool(noise_conf.get("correlated", False))

        if self.use_noise:
            latent_dim = int(noise_conf.get("latent_dim", 128))
            self.latent_dim = latent_dim
            noise_scales = list(noise_conf.get("noise_scales", [0.30, 0.15, 0.05, 0.01]))
            assert len(noise_scales) == 4, "noise_scales must have 4 values"

            self.dec_noise1 = StochasticDecompositionLayer(latent_dim, last_dim // 2,  noise_scales[0])
            self.dec_noise2 = StochasticDecompositionLayer(latent_dim, last_dim // 4,  noise_scales[1])
            self.dec_noise3 = StochasticDecompositionLayer(latent_dim, last_dim // 8,  noise_scales[2])
            self.dec_noise4 = StochasticDecompositionLayer(latent_dim, last_dim // 8,  noise_scales[3])

            if self.encoder_noise:
                enc_scale = noise_scales[-1]
                self.enc_noise_layers = nn.ModuleList([
                    StochasticDecompositionLayer(latent_dim, dim[0], enc_scale),
                    StochasticDecompositionLayer(latent_dim, dim[1], enc_scale),
                    StochasticDecompositionLayer(latent_dim, dim[2], enc_scale),
                ])

            if self.freeze_base_model_weights:
                for param in self.parameters():
                    param.requires_grad = False
                noise_mods = [self.dec_noise1, self.dec_noise2, self.dec_noise3, self.dec_noise4]
                if self.encoder_noise:
                    noise_mods += list(self.enc_noise_layers)
                for m in noise_mods:
                    for p in m.parameters():
                        p.requires_grad = True
                logger.info(
                    "AuroraCrossFormerWRF: base weights frozen; "
                    "only noise-injection layers are trainable."
                )
        else:
            self.latent_dim = int(noise_conf.get("latent_dim", 128))

        # ── Optional boundary padding ─────────────────────────────────────
        if padding_conf is None:
            padding_conf = {"activate": False}
        self.use_padding = bool(padding_conf["activate"])
        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)

        # ── Optional spectral norm ────────────────────────────────────────
        if use_spectral_norm:
            logger.info("AuroraCrossFormerWRF: applying spectral norm")
            apply_spectral_norm(self)

        # ── Optional post-processing block ────────────────────────────────
        if post_conf is None:
            post_conf = {"activate": False}
        self.use_post_block = bool(post_conf["activate"])
        if self.use_post_block:
            self.postblock = PostBlock(post_conf)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_window_padding(self, H: int, W: int) -> Tuple[int, int]:
        required = self.local_window_size * (2 ** self.num_stages)
        return (-H) % required, (-W) % required

    def _sample_latent(self, B: int, device, dtype) -> torch.Tensor:
        return torch.randn(B, self.latent_dim, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,              # (B, C,   T, H, W)  – interior state
        x_boundary: torch.Tensor,     # (B, C_b, T, H, W)  – boundary / tendency
        x_time_encode: torch.Tensor,  # (B, time_encode_dim)
        forecast_step: int = 0,
        ensemble_size: int = 1,
    ) -> torch.Tensor:
        # ── optional post-block bookkeeping ───────────────────────────────
        x_copy = x.clone().detach() if self.use_post_block else None

        # ── optional boundary padding (TensorPadding) ─────────────────────
        if self.use_padding:
            x = self.padding_opt.pad(x)
            x_boundary = self.padding_opt.pad(x_boundary)

        B = x.shape[0]
        orig_H, orig_W = x.shape[-2], x.shape[-1]

        # ── flatten time into channel dim ─────────────────────────────────
        if x.shape[2] == 1:
            x_2d = x.squeeze(2)                                      # (B, C,   H, W)
            xb_2d = x_boundary.squeeze(2)                            # (B, C_b, H, W)
        else:
            B2, C, T, H, W = x.shape
            x_2d = x.reshape(B2, C * T, H, W)
            Bb, Cb, Tb, Hb, Wb = x_boundary.shape
            xb_2d = x_boundary.reshape(Bb, Cb * Tb, Hb, Wb)

        # ── Aurora-style per-(var,level) variable token embedding ─────────
        x_emb = self.interior_embedder(x_2d)    # (B, embed_dim, H, W)
        xb_emb = self.boundary_embedder(xb_2d)  # (B, boundary_embed_dim, H, W)

        # ── Enhanced FiLM: 4-quadrant + global boundary pool ─────────────
        # 4-quadrant spatial pooling using avg_pool2d (deterministic on CUDA).
        # Kernel/stride = H//2, W//2 so the output is always (B, bnd_D, 2, 2).
        _Hb, _Wb = xb_emb.shape[-2], xb_emb.shape[-1]
        _kH, _kW = max(1, _Hb // 2), max(1, _Wb // 2)
        xb_quad = F.avg_pool2d(xb_emb, kernel_size=(_kH, _kW), stride=(_kH, _kW)).flatten(1)  # (B, 4*bnd_D)
        xb_global = xb_emb.mean(dim=(-2, -1))                        # (B, bnd_D)
        film_cond = torch.cat([xb_quad, xb_global, x_time_encode], dim=1)
        gamma_beta = self.film_mlp(film_cond)                         # (B, 2*embed_dim)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        x_emb = gamma.view(B, -1, 1, 1) * x_emb + beta.view(B, -1, 1, 1)

        # ── window-divisibility padding ───────────────────────────────────
        _, _, H, W = x_emb.shape
        pad_H, pad_W = self._compute_window_padding(H, W)
        if pad_H or pad_W:
            x_emb = F.pad(x_emb, (0, pad_W, 0, pad_H))

        # ── latent noise sampling ──────────────────────────────────────────
        use_noise_now = self.use_noise and ensemble_size > 1

        if use_noise_now and self.correlated:
            _z_shared = self._sample_latent(B, x_emb.device, x_emb.dtype)
        else:
            _z_shared = None

        def _z() -> torch.Tensor:
            if _z_shared is not None:
                return _z_shared
            return self._sample_latent(B, x_emb.device, x_emb.dtype)

        # ── CrossFormer encoder ───────────────────────────────────────────
        feat = x_emb
        encodings: List[torch.Tensor] = []
        for k, (cel, transformer) in enumerate(self.encoder_layers):
            feat = cel(feat)
            feat = transformer(feat)
            if use_noise_now and self.encoder_noise and k < 3:
                feat = self.enc_noise_layers[k](feat, _z())
            encodings.append(feat)

        # ── Anti-checkerboard UNet decoder ───────────────────────────────

        # Stage 1: H/16 → H/8
        feat = self.up_block1(feat)
        if use_noise_now:
            feat = self.dec_noise1(feat, _z())
        feat = torch.cat([feat, encodings[2]], dim=1)

        # Stage 2: H/8 → H/4
        feat = self.up_block2(feat)
        if use_noise_now:
            feat = self.dec_noise2(feat, _z())
        feat = torch.cat([feat, encodings[1]], dim=1)

        # Stage 3: H/4 → H/2
        feat = self.up_block3(feat)
        if use_noise_now:
            feat = self.dec_noise3(feat, _z())
        feat = torch.cat([feat, encodings[0]], dim=1)

        # Stage 4: H/2 → H
        feat = self.up_block4(feat)
        if use_noise_now:
            feat = self.dec_noise4(feat, _z())

        # ── crop to pre-window-pad spatial size ───────────────────────────
        feat = feat[:, :, :orig_H, :orig_W].contiguous()

        # ── per-variable output projection ────────────────────────────────
        out = self.out_decoder(feat)                                   # (B, C_out, H, W)
        out = out.unsqueeze(2)                                         # (B, C_out, 1, H, W)

        # ── optional boundary unpadding ───────────────────────────────────
        if self.use_padding:
            out = self.padding_opt.unpad(out)

        # ── optional post-processing block ────────────────────────────────
        if self.use_post_block:
            out = self.postblock({"y_pred": out, "x": x_copy})

        return out

    # ------------------------------------------------------------------
    # Fine-tuning helpers
    # ------------------------------------------------------------------

    def _next_interior_ch(self) -> int:
        """Return the next available channel index in the flat interior input."""
        n = len(self.interior_embedder.upper_air_spec) * self.interior_embedder.levels * self.frames
        n += len(self.interior_embedder.surface_spec) * self.frames
        return n

    def _next_boundary_ch(self) -> int:
        """Return the next available channel index in the flat boundary input."""
        n = len(self.boundary_embedder.upper_air_spec) * self.boundary_embedder.levels * self.frames
        n += len(self.boundary_embedder.surface_spec) * self.frames
        return n

    def add_upper_air_variable(self, name: str) -> None:
        """Add a new prognostic upper-air variable (input projectors + output head, zero-init)."""
        self.interior_embedder.add_upper_air_variable(name, self._next_interior_ch())
        self.out_decoder.add_output_variable(name, self.interior_embedder.levels)
        self.varname_upper_air.append(name)
        self._n_out_channels += self.interior_embedder.levels

    def add_surface_variable(self, name: str) -> None:
        """Add a new prognostic 2-D variable (input projector + output head, zero-init)."""
        self.interior_embedder.add_surface_variable(name, self._next_interior_ch())
        self.out_decoder.add_output_variable(name, 1)
        self.varname_surface.append(name)
        self._n_out_channels += 1

    def add_diagnostic_variable(self, name: str) -> None:
        """Add a new diagnostic output-only variable (output head, zero-init)."""
        self.out_decoder.add_output_variable(name, 1)
        self.varname_diagnostic.append(name)
        self._n_out_channels += 1

    def add_boundary_upper_variable(self, name: str) -> None:
        """Add a new boundary upper-air input variable (projectors, zero-init)."""
        self.boundary_embedder.add_upper_air_variable(f"bnd_{name}", self._next_boundary_ch())

    def add_boundary_surface_variable(self, name: str) -> None:
        """Add a new boundary 2-D input variable (projector, zero-init)."""
        self.boundary_embedder.add_surface_variable(f"bnd_{name}", self._next_boundary_ch())
