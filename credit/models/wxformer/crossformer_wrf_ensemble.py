# credit/models/wxformer/crossformer_wrf_ensemble.py
#
# CrossFormerWRFEnsemble – a variable-flexible, resolution-flexible ensemble
# NWP model that combines:
#
#   1. Anti-checkerboard decoder  – nearest-exact + Conv2d upsampling replaces
#      both PixelShuffle (swin_wrf_v2) and ConvTranspose2d (crossformer).
#
#   2. VariableTokenEmbedder      – per-variable 1×1 Conv2d projectors whose
#      outputs are *summed* into a fixed embed_dim.  The backbone channel
#      dimension never changes when variables are added or removed; only a
#      new Conv2d leaf is needed for a new variable.  Gate logits initialised
#      to 0 (sigmoid = 0.5) ensure new variables start with minimal influence.
#
#   3. Multi-scale noise injection – StochasticDecompositionLayer applied at
#      four decoder scales with scale-appropriate initial noise factors.  A
#      single latent vector z is shared (or freshly sampled, controlled by
#      `correlated`) across injection sites so cross-scale coherence is
#      configurable.
#
#   4. Dynamic resolution          – pad to the next multiple of
#      local_window_size × 2^num_stages before the encoder, crop back to
#      original size after the decoder; identical strategy to swin_wrf_v2.
#
#   5. FiLM conditioning           – boundary embedding (global-average-pooled)
#      concatenated with time encoding feeds a 2-layer MLP to produce
#      per-channel (gamma, beta) that modulates the interior embedding before
#      encoding.
#
# Trainer compatibility
# ─────────────────────
# forward(x, x_boundary, x_time_encode, forecast_step=0, ensemble_size=1)
# This is identical to WRFTransformer and is consumed unchanged by
# TrainerWRFMulti ('multi-step-wrf') and TrainerWRFMultiEnsemble
# ('multi-step-wrf-ensemble').

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from credit.boundary_padding import TensorPadding
from credit.models.base_model import BaseModel
from credit.models.crossformer import CrossEmbedLayer, Transformer, apply_spectral_norm
from credit.models.wxformer.stochastic_decomposition_layer import StochasticDecompositionLayer
from credit.postblock import PostBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num_groups(channels: int, target: int = 32) -> int:
    """Largest divisor of *channels* that is <= *target*."""
    for g in [target, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# AnticheckboardUpBlock
# ---------------------------------------------------------------------------

class AnticheckboardUpBlock(nn.Module):
    """
    2× upsampling that is free of checkerboard artefacts.

    Replaces both:
      * PixelShuffle  (swin_wrf_v2.UpBlock)     – creates a periodic 2×2 tile
      * ConvTranspose2d (crossformer.UpBlock)     – produces spectral checkerboard

    Design:
        nearest-exact Upsample  →  3×3 Conv2d  →  residual GroupNorm+SiLU block

    The ``nearest-exact`` mode duplicates grid values exactly at integer
    positions with no interpolation artefacts.  The subsequent 3×3 conv
    blends neighbours uniformly, breaking any residual periodicity.

    Args:
        in_chans    (int): input channels.
        out_chans   (int): output channels.
        num_groups  (int): GroupNorm groups (default auto).
        num_residuals (int): number of Conv-GN-SiLU layers in residual branch.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        num_groups: Optional[int] = None,
        num_residuals: int = 2,
    ):
        super().__init__()
        if num_groups is None:
            num_groups = _num_groups(out_chans)

        # The upsampling path: interpolate → blend
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),   # checkerboard-free
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
        )
        self.output_channels = out_chans

        # Residual path operating on the upsampled features
        blk: List[nn.Module] = []
        for _ in range(num_residuals):
            blk += [
                nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups, out_chans),
                nn.SiLU(),
            ]
        self.residual = nn.Sequential(*blk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x + self.residual(x)


# ---------------------------------------------------------------------------
# VariableTokenEmbedder
# ---------------------------------------------------------------------------

class VariableTokenEmbedder(nn.Module):
    """
    Projects a flat channel tensor containing N variables into a fixed-width
    embedding by summing per-variable projections.

    Shape contract
    ~~~~~~~~~~~~~~
    input : (B, C_total, H, W)   – flat concatenation of all variable channels
    output: (B, embed_dim, H, W)

    Variable flexibility
    ~~~~~~~~~~~~~~~~~~~~
    * Drop a variable at inference: just exclude its channels from the input
      slice_spec.  The projector still exists in the state dict but is never
      called.  Checkpoint keys are unaffected.
    * Add a variable at fine-tuning: call add_variable(name, n_chans, start).
      The new projector is zero-initialised so the backbone output is initially
      unchanged.  Load the old checkpoint with strict=False.

    Args:
        slice_spec (list): [(name, start_ch, n_chans), ...]  one entry per
                           variable (or variable group, e.g. all levels of T).
        embed_dim  (int):  output embedding dimension.
    """

    def __init__(self, slice_spec: List[Tuple[str, int, int]], embed_dim: int):
        super().__init__()
        self.slice_spec: List[Tuple[str, int, int]] = list(slice_spec)
        self.embed_dim = embed_dim

        self.projectors = nn.ModuleDict(
            {name: nn.Conv2d(n_chans, embed_dim, kernel_size=1)
             for name, _, n_chans in slice_spec}
        )
        # Learnable gate per variable; sigmoid(0) = 0.5 at init.
        self.gate_logits = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name, _, _ in slice_spec}
        )
        # Channel-wise layer norm applied after summing projections.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_total, H, W)
        out: Optional[torch.Tensor] = None
        for name, start, n_chans in self.slice_spec:
            proj = self.projectors[name](x[:, start:start + n_chans])  # (B, E, H, W)
            gate = torch.sigmoid(self.gate_logits[name])                # scalar
            out = gate * proj if out is None else out + gate * proj

        if out is None:
            raise RuntimeError("VariableTokenEmbedder: slice_spec is empty")

        # Apply LayerNorm over the channel axis (treat spatial dims as batch).
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        out = self.norm(out)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)            # (B, C, H, W)

    def add_variable(self, name: str, n_chans: int, start_channel: int) -> None:
        """Register a new variable projector for fine-tuning (zero-initialised)."""
        if name in self.projectors:
            return
        proj = nn.Conv2d(n_chans, self.embed_dim, kernel_size=1)
        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)
        self.projectors[name] = proj
        self.gate_logits[name] = nn.Parameter(torch.zeros(1))
        self.slice_spec.append((name, start_channel, n_chans))


# ---------------------------------------------------------------------------
# VariableTokenDecoder
# ---------------------------------------------------------------------------

class VariableTokenDecoder(nn.Module):
    """
    Reconstructs per-variable output tensors from a shared feature map by
    applying per-variable 1×1 Conv2d projectors and concatenating the results.

    Shape contract
    ~~~~~~~~~~~~~~
    input : (B, feat_dim, H, W)
    output: (B, sum(n_chans), H, W)  – variables concatenated in order

    Args:
        output_spec (list): [(name, n_chans), ...]  one entry per output variable.
        feat_dim    (int):  input feature dimension.
    """

    def __init__(self, output_spec: List[Tuple[str, int]], feat_dim: int):
        super().__init__()
        self.output_spec: List[Tuple[str, int]] = list(output_spec)
        self.de_projectors = nn.ModuleDict(
            {name: nn.Conv2d(feat_dim, n_chans, kernel_size=1)
             for name, n_chans in output_spec}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.de_projectors[name](x) for name, _ in self.output_spec],
            dim=1,
        )  # (B, C_total_out, H, W)


# ---------------------------------------------------------------------------
# CrossFormerWRFEnsemble
# ---------------------------------------------------------------------------

class CrossFormerWRFEnsemble(BaseModel):
    """
    Variable-flexible, resolution-flexible WRF ensemble forecast model.

    See module docstring for architecture overview.

    Args
    ────
    varname_upper_air   (list[str]): 3-D prognostic variables (each occupies
                                    *levels* channels in the flat input).
    varname_surface     (list[str]): 2-D prognostic variables (1 channel each).
    varname_dyn_forcing (list[str]): 2-D dynamic forcing, input-only (1 ch).
    varname_forcing     (list[str]): 2-D static forcing, input-only.
    varname_static      (list[str]): 2-D static fields, input-only.
    varname_diagnostic  (list[str]): 2-D diagnostic/output-only variables.
    varname_boundary_upper   (list[str]): boundary 3-D variables.
    varname_boundary_surface (list[str]): boundary 2-D variables.
    levels              (int):  vertical levels for interior upper-air vars.
    boundary_levels     (int):  vertical levels for boundary upper-air vars.
    frames              (int):  number of input time steps (default 1).
    embed_dim           (int):  interior variable embedding dimension.
    boundary_embed_dim  (int):  boundary embedding dimension.
    time_encode_dim     (int):  dimension of the time-encoding input vector.
    dim                 (tuple[int]):  encoder channel dims, 4 stages.
    depth               (tuple[int]):  transformer depth per encoder stage.
    dim_head            (int):  attention head dimension.
    global_window_size  (tuple[int]):  global attention window per stage.
    local_window_size   (int):  local attention window (all stages).
    cross_embed_kernel_sizes: kernel sizes for CrossEmbedLayer per stage.
    cross_embed_strides (tuple[int]):  stride for CrossEmbedLayer per stage.
    attn_dropout, ff_dropout (float): dropout rates.
    use_spectral_norm   (bool): apply spectral norm to Conv/Linear layers.
    noise_injection     (dict): stochastic noise configuration (see below).
    padding_conf        (dict): TensorPadding boundary-padding config.
    post_conf           (dict): PostBlock post-processing config.

    noise_injection keys
    ~~~~~~~~~~~~~~~~~~~~
    activate             (bool)   default False
    latent_dim           (int)    default 128
    noise_scales         (list)   4 floats, coarsest→finest (default [0.3,0.15,0.05,0.01])
    encoder_noise        (bool)   default True
    correlated           (bool)   default False – share z across all sites
    sample_latent_if_none (bool)  default True
    freeze_base_model_weights (bool) default False
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
            "CrossFormerWRFEnsemble requires cross_embed_strides=(2,2,2,2); "
            "other strides are not supported in the symmetric U-Net decoder."
        )

        self.frames = frames
        self.local_window_size = local_window_size
        self.num_stages = len(dim)

        # ── save variable lists for rollout / diagnostics ─────────────────
        self.varname_upper_air = list(varname_upper_air)
        self.varname_surface = list(varname_surface)
        self.varname_dyn_forcing = list(varname_dyn_forcing)
        self.varname_forcing = list(varname_forcing)
        self.varname_static = list(varname_static)
        self.varname_diagnostic = list(varname_diagnostic)

        # ── interior slice spec ───────────────────────────────────────────
        # Channel order in x (from TrainerWRFMulti*):
        #   upper_air × levels, surface × 1, dyn_forcing × 1,
        #   forcing × 1, static × 1   (all × frames)
        int_spec: List[Tuple[str, int, int]] = []
        cursor = 0
        for name in varname_upper_air:
            n = levels * frames
            int_spec.append((name, cursor, n))
            cursor += n
        for name in varname_surface:
            n = 1 * frames
            int_spec.append((name, cursor, n))
            cursor += n
        for name in varname_dyn_forcing:
            n = 1 * frames
            int_spec.append((name, cursor, n))
            cursor += n
        for name in varname_forcing:
            n = 1 * frames
            int_spec.append((name, cursor, n))
            cursor += n
        for name in varname_static:
            n = 1 * frames
            int_spec.append((name, cursor, n))
            cursor += n

        # ── boundary slice spec ───────────────────────────────────────────
        bnd_spec: List[Tuple[str, int, int]] = []
        cursor = 0
        for name in varname_boundary_upper:
            n = boundary_levels * frames
            bnd_spec.append((f"bnd_{name}", cursor, n))
            cursor += n
        for name in varname_boundary_surface:
            n = 1 * frames
            bnd_spec.append((f"bnd_{name}", cursor, n))
            cursor += n

        # ── output spec ───────────────────────────────────────────────────
        # Target channel order (from trainer): upper_air*levels, surface, diagnostic
        out_spec: List[Tuple[str, int]] = []
        for name in varname_upper_air:
            out_spec.append((name, levels))
        for name in varname_surface:
            out_spec.append((name, 1))
        for name in varname_diagnostic:
            out_spec.append((name, 1))
        self._n_out_channels: int = sum(n for _, n in out_spec)

        # ── variable embedders ────────────────────────────────────────────
        self.interior_embedder = VariableTokenEmbedder(int_spec, embed_dim)
        self.boundary_embedder = VariableTokenEmbedder(bnd_spec, boundary_embed_dim)

        # ── FiLM: boundary pool + time_encode → (gamma, beta) ────────────
        film_in = boundary_embed_dim + time_encode_dim
        self.film_mlp = nn.Sequential(
            nn.Linear(film_in, film_in * 2),
            nn.SiLU(),
            nn.Linear(film_in * 2, 2 * embed_dim),
        )

        # ── CrossFormer encoder ───────────────────────────────────────────
        # First CrossEmbedLayer takes embed_dim (not raw input_channels)
        last_dim = dim[-1]
        dims = [embed_dim, *dim]
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.encoder_layers = nn.ModuleList()
        for (d_in, d_out), n_layers, g_wsize, l_wsize, kern, stride in zip(
            dim_pairs,
            depth,
            global_window_size,
            [local_window_size] * 4,
            cross_embed_kernel_sizes,
            cross_embed_strides,
        ):
            self.encoder_layers.append(
                nn.ModuleList([
                    CrossEmbedLayer(d_in, d_out, kern, stride),
                    Transformer(
                        d_out,
                        local_window_size=l_wsize,
                        global_window_size=g_wsize,
                        depth=n_layers,
                        dim_head=dim_head,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                    ),
                ])
            )

        # ── Anti-checkerboard decoder ─────────────────────────────────────
        # Skip connections from encoder stages 0-2 are concatenated into
        # the decoder at the matching spatial scale.
        # Stage layout (with dim=(64,128,256,512)):
        #   up1: last_dim          → last_dim//2  (H/16 → H/8),  cat enc[2]=dim[2]
        #   up2: last_dim//2+dim[2] → last_dim//4  (H/8  → H/4),  cat enc[1]=dim[1]
        #   up3: last_dim//4+dim[1] → last_dim//8  (H/4  → H/2),  cat enc[0]=dim[0]
        #   up4: last_dim//8+dim[0] → last_dim//8  (H/2  → H)
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

            # noise_scales[i] corresponds to decoder stage i (coarsest first).
            # Larger values at coarser scales model mesoscale uncertainty.
            noise_scales = list(noise_conf.get("noise_scales", [0.30, 0.15, 0.05, 0.01]))
            assert len(noise_scales) == 4, "noise_scales must have 4 values"

            self.dec_noise1 = StochasticDecompositionLayer(latent_dim, last_dim // 2,  noise_scales[0])
            self.dec_noise2 = StochasticDecompositionLayer(latent_dim, last_dim // 4,  noise_scales[1])
            self.dec_noise3 = StochasticDecompositionLayer(latent_dim, last_dim // 8,  noise_scales[2])
            self.dec_noise4 = StochasticDecompositionLayer(latent_dim, last_dim // 8,  noise_scales[3])

            if self.encoder_noise:
                enc_scale = noise_scales[-1]  # smallest scale for encoder
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
                    "CrossFormerWRFEnsemble: base weights frozen; "
                    "only noise-injection layers are trainable."
                )
        else:
            self.latent_dim = int(noise_conf.get("latent_dim", 128))  # for API compat

        # ── Optional boundary padding ─────────────────────────────────────
        if padding_conf is None:
            padding_conf = {"activate": False}
        self.use_padding = bool(padding_conf["activate"])
        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)

        # ── Optional spectral norm ────────────────────────────────────────
        if use_spectral_norm:
            logger.info("CrossFormerWRFEnsemble: applying spectral norm")
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
        """Smallest padding so H and W are divisible by
        ``local_window_size × 2^num_stages``.

        This satisfies the short- and long-distance attention divisibility
        constraints in CrossEmbedLayer + Transformer for all 4 stages.
        """
        required = self.local_window_size * (2 ** self.num_stages)
        return (-H) % required, (-W) % required

    def _sample_latent(self, B: int, device, dtype) -> torch.Tensor:
        return torch.randn(B, self.latent_dim, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,              # (B, C,   T, H, W)  – interior
        x_boundary: torch.Tensor,     # (B, C_b, T, H, W)  – boundary
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
        # Record spatial size *after* TensorPadding so we can crop back to it
        # before calling unpad (which removes the TensorPadding boundary ring).
        orig_H, orig_W = x.shape[-2], x.shape[-1]

        # ── flatten time into channel dim ─────────────────────────────────
        # Trainer delivers (B, C, T, H, W); CrossFormer operates on 2-D fields.
        # For T=1: squeeze; for T>1: reshape C*T into the channel axis.
        if x.shape[2] == 1:
            x_2d = x.squeeze(2)                                      # (B, C, H, W)
            xb_2d = x_boundary.squeeze(2)
        else:
            B2, C, T, H, W = x.shape
            x_2d = x.reshape(B2, C * T, H, W)
            Bb, Cb, Tb, Hb, Wb = x_boundary.shape
            xb_2d = x_boundary.reshape(Bb, Cb * Tb, Hb, Wb)

        # ── variable token embedding ──────────────────────────────────────
        x_emb = self.interior_embedder(x_2d)    # (B, embed_dim, H, W)
        xb_emb = self.boundary_embedder(xb_2d)  # (B, boundary_embed_dim, H, W)

        # ── FiLM: boundary global pool + time encode → (γ, β) ────────────
        xb_pooled = xb_emb.mean(dim=(-2, -1))                         # (B, B_dim)
        film_cond = torch.cat([xb_pooled, x_time_encode], dim=1)      # (B, B_dim+T_dim)
        gamma_beta = self.film_mlp(film_cond)                          # (B, 2*embed_dim)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        x_emb = gamma.view(B, -1, 1, 1) * x_emb + beta.view(B, -1, 1, 1)

        # ── window-divisibility padding ───────────────────────────────────
        _, _, H, W = x_emb.shape
        pad_H, pad_W = self._compute_window_padding(H, W)
        if pad_H or pad_W:
            x_emb = F.pad(x_emb, (0, pad_W, 0, pad_H))  # pad right & bottom

        # ── sample latent noise ───────────────────────────────────────────
        # correlated=True:  one z per forward; shared across all injection sites
        # correlated=False: fresh z per injection site (default)
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
            feat = cel(feat)           # CrossEmbedLayer  (stride-2 downsample)
            feat = transformer(feat)   # local + global attention
            if use_noise_now and self.encoder_noise and k < 3:
                feat = self.enc_noise_layers[k](feat, _z())
            encodings.append(feat)

        # ── Anti-checkerboard decoder ─────────────────────────────────────
        # feat is now at H/16 (4 stride-2 downsampling stages).

        # Stage 1: H/16 → H/8 ────────────────────────────────────────────
        feat = self.up_block1(feat)                    # (B, last_dim//2, H/8)
        if use_noise_now:
            feat = self.dec_noise1(feat, _z())
        feat = torch.cat([feat, encodings[2]], dim=1)  # skip from enc stage 2

        # Stage 2: H/8 → H/4 ─────────────────────────────────────────────
        feat = self.up_block2(feat)                    # (B, last_dim//4, H/4)
        if use_noise_now:
            feat = self.dec_noise2(feat, _z())
        feat = torch.cat([feat, encodings[1]], dim=1)  # skip from enc stage 1

        # Stage 3: H/4 → H/2 ─────────────────────────────────────────────
        feat = self.up_block3(feat)                    # (B, last_dim//8, H/2)
        if use_noise_now:
            feat = self.dec_noise3(feat, _z())
        feat = torch.cat([feat, encodings[0]], dim=1)  # skip from enc stage 0

        # Stage 4: H/2 → H ────────────────────────────────────────────────
        feat = self.up_block4(feat)                    # (B, last_dim//8, H)
        if use_noise_now:
            feat = self.dec_noise4(feat, _z())

        # ── lossless crop to pre-window-pad spatial size ──────────────────
        feat = feat[:, :, :orig_H, :orig_W].contiguous()

        # ── per-variable output projection ────────────────────────────────
        out = self.out_decoder(feat)                   # (B, C_out, orig_H, orig_W)
        out = out.unsqueeze(2)                         # (B, C_out, 1, H, W)

        # ── optional boundary unpadding ───────────────────────────────────
        if self.use_padding:
            out = self.padding_opt.unpad(out)          # removes TensorPadding ring

        # ── optional post-processing block ────────────────────────────────
        if self.use_post_block:
            out = self.postblock({"y_pred": out, "x": x_copy})

        return out
