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
)
from credit.models.wxformer.stochastic_decomposition_layer import StochasticDecompositionLayer
from credit.postblock import PostBlock

logger = logging.getLogger(__name__)


class DAProfileEmbedder(nn.Module):
    """Role-aware DA input embedder.

    DA inputs have many input-only upper-air context channels. This embedder
    compresses each variable profile as one slice instead of materializing one
    full-resolution activation per variable-level pair.
    """

    def __init__(
        self,
        prognostic_spec: List[Tuple[str, int, int]],
        context_spec: List[Tuple[str, int, int]],
        surface_spec: List[Tuple[str, int, int]],
        embed_dim: int,
    ):
        super().__init__()
        self.prognostic_spec = list(prognostic_spec)
        self.context_spec = list(context_spec)
        self.surface_spec = list(surface_spec)
        self.embed_dim = embed_dim

        self.prognostic_projectors = nn.ModuleDict(
            {name: nn.Conv2d(n_chans, embed_dim, kernel_size=1) for name, _, n_chans in self.prognostic_spec}
        )
        self.context_projectors = nn.ModuleDict(
            {name: nn.Conv2d(n_chans, embed_dim, kernel_size=1) for name, _, n_chans in self.context_spec}
        )
        self.surface_projectors = nn.ModuleDict(
            {name: nn.Conv2d(n_chans, embed_dim, kernel_size=1) for name, _, n_chans in self.surface_spec}
        )

        self.gate_logits = nn.ParameterDict()
        for name, _, _ in self.prognostic_spec + self.context_spec + self.surface_spec:
            self.gate_logits[name] = nn.Parameter(torch.zeros(1))

        self.role_gates = nn.ParameterDict(
            {
                "prognostic": nn.Parameter(torch.zeros(1)),
                "context": nn.Parameter(torch.zeros(1)),
                "surface": nn.Parameter(torch.zeros(1)),
            }
        )
        self.norm = nn.LayerNorm(embed_dim)

    def _sum_role(self, x: torch.Tensor, spec: List[Tuple[str, int, int]], projectors: nn.ModuleDict) -> Optional[torch.Tensor]:
        out: Optional[torch.Tensor] = None
        for name, start, n_chans in spec:
            proj = projectors[name](x[:, start:start + n_chans])
            gate = torch.sigmoid(self.gate_logits[name]).to(dtype=proj.dtype)
            out = gate * proj if out is None else out + gate * proj
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prog = self._sum_role(x, self.prognostic_spec, self.prognostic_projectors)
        context = self._sum_role(x, self.context_spec, self.context_projectors)
        surface = self._sum_role(x, self.surface_spec, self.surface_projectors)

        out: Optional[torch.Tensor] = None
        for role_name, role_value in (
            ("prognostic", prog),
            ("context", context),
            ("surface", surface),
        ):
            if role_value is None:
                continue
            role_gate = torch.sigmoid(self.role_gates[role_name]).to(dtype=role_value.dtype)
            out = role_gate * role_value if out is None else out + role_gate * role_value

        if out is None:
            raise RuntimeError("DAProfileEmbedder: all input specs are empty")

        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(-1, C)
        out = self.norm(out)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class DABoundaryEmbedder(nn.Module):
    def __init__(self, slice_spec: List[Tuple[str, int, int]], embed_dim: int):
        super().__init__()
        self.slice_spec = list(slice_spec)
        self.embed_dim = embed_dim
        self.projectors = nn.ModuleDict(
            {name: nn.Conv2d(n_chans, embed_dim, kernel_size=1) for name, _, n_chans in self.slice_spec}
        )
        self.gate_logits = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name, _, _ in self.slice_spec}
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: Optional[torch.Tensor] = None
        for name, start, n_chans in self.slice_spec:
            proj = self.projectors[name](x[:, start:start + n_chans])
            gate = torch.sigmoid(self.gate_logits[name]).to(dtype=proj.dtype)
            out = gate * proj if out is None else out + gate * proj
        if out is None:
            raise RuntimeError("DABoundaryEmbedder: slice_spec is empty")

        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(-1, C)
        out = self.norm(out)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class AuroraCrossFormerWRFDA(BaseModel):
    """DA-specific Aurora/CrossFormer model.

    This model keeps the trainer-facing signature of ``aurora_crossformer_wrf``
    but replaces the expensive per-level DA input tokenization with role-aware
    profile compression.
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
        varname_input_upper_air: Optional[List[str]] = None,
        levels: int = 17,
        boundary_levels: int = 17,
        frames: int = 1,
        embed_dim: int = 128,
        boundary_embed_dim: int = 64,
        time_encode_dim: int = 12,
        dim=(64, 128, 192, 256),
        depth=(2, 2, 6, 2),
        dim_head: int = 32,
        global_window_size=(8, 6, 4, 1),
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

        dim = tuple(dim)
        depth = tuple(depth)
        global_window_size = tuple(global_window_size)
        cross_embed_kernel_sizes = tuple(tuple(k) for k in cross_embed_kernel_sizes)
        cross_embed_strides = tuple(cross_embed_strides)
        local_window_size = int(local_window_size)

        assert len(dim) == 4, "dim must have exactly 4 elements"
        assert len(depth) == 4, "depth must have exactly 4 elements"
        assert all(s == 2 for s in cross_embed_strides), (
            "AuroraCrossFormerWRFDA requires cross_embed_strides=(2,2,2,2)."
        )

        self.frames = frames
        self.local_window_size = local_window_size
        self.num_stages = len(dim)

        self.varname_upper_air = list(varname_upper_air)
        self.varname_input_upper_air = list(varname_input_upper_air or [])
        self.varname_surface = list(varname_surface)
        self.varname_dyn_forcing = list(varname_dyn_forcing)
        self.varname_forcing = list(varname_forcing)
        self.varname_static = list(varname_static)
        self.varname_diagnostic = list(varname_diagnostic)

        prog_spec: List[Tuple[str, int, int]] = []
        context_spec: List[Tuple[str, int, int]] = []
        surface_spec: List[Tuple[str, int, int]] = []
        cursor = 0
        for name in varname_upper_air:
            n = levels * frames
            prog_spec.append((name, cursor, n))
            cursor += n
        for name in self.varname_input_upper_air:
            n = levels * frames
            context_spec.append((name, cursor, n))
            cursor += n
        for name in varname_surface:
            n = frames
            surface_spec.append((name, cursor, n))
            cursor += n
        for name in varname_dyn_forcing:
            n = frames
            surface_spec.append((name, cursor, n))
            cursor += n
        for name in varname_forcing:
            n = frames
            surface_spec.append((name, cursor, n))
            cursor += n
        for name in varname_static:
            n = frames
            surface_spec.append((name, cursor, n))
            cursor += n

        bnd_spec: List[Tuple[str, int, int]] = []
        cursor = 0
        for name in varname_boundary_upper:
            n = boundary_levels * frames
            bnd_spec.append((f"bnd_{name}", cursor, n))
            cursor += n
        for name in varname_boundary_surface:
            n = frames
            bnd_spec.append((f"bnd_{name}", cursor, n))
            cursor += n

        out_spec: List[Tuple[str, int]] = []
        for name in varname_upper_air:
            out_spec.append((name, levels))
        for name in varname_surface:
            out_spec.append((name, 1))
        for name in varname_diagnostic:
            out_spec.append((name, 1))
        self._n_out_channels = sum(n for _, n in out_spec)

        self.interior_embedder = DAProfileEmbedder(prog_spec, context_spec, surface_spec, embed_dim)
        self.boundary_embedder = DABoundaryEmbedder(bnd_spec, boundary_embed_dim)

        film_in = 5 * boundary_embed_dim + time_encode_dim
        self.film_mlp = nn.Sequential(
            nn.Linear(film_in, film_in * 2),
            nn.SiLU(),
            nn.Linear(film_in * 2, 2 * embed_dim),
        )

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
                nn.ModuleList(
                    [
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
                    ]
                )
            )

        self.up_block1 = AnticheckboardUpBlock(last_dim, last_dim // 2)
        self.up_block2 = AnticheckboardUpBlock(last_dim // 2 + dim[2], last_dim // 4)
        self.up_block3 = AnticheckboardUpBlock(last_dim // 4 + dim[1], last_dim // 8)
        self.up_block4 = AnticheckboardUpBlock(last_dim // 8 + dim[0], last_dim // 8)
        self.out_decoder = VariableTokenDecoder(out_spec, last_dim // 8)

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

            self.dec_noise1 = StochasticDecompositionLayer(latent_dim, last_dim // 2, noise_scales[0])
            self.dec_noise2 = StochasticDecompositionLayer(latent_dim, last_dim // 4, noise_scales[1])
            self.dec_noise3 = StochasticDecompositionLayer(latent_dim, last_dim // 8, noise_scales[2])
            self.dec_noise4 = StochasticDecompositionLayer(latent_dim, last_dim // 8, noise_scales[3])

            if self.encoder_noise:
                enc_scale = noise_scales[-1]
                self.enc_noise_layers = nn.ModuleList(
                    [
                        StochasticDecompositionLayer(latent_dim, dim[0], enc_scale),
                        StochasticDecompositionLayer(latent_dim, dim[1], enc_scale),
                        StochasticDecompositionLayer(latent_dim, dim[2], enc_scale),
                    ]
                )

            if self.freeze_base_model_weights:
                for param in self.parameters():
                    param.requires_grad = False
                noise_mods = [self.dec_noise1, self.dec_noise2, self.dec_noise3, self.dec_noise4]
                if self.encoder_noise:
                    noise_mods += list(self.enc_noise_layers)
                for module in noise_mods:
                    for param in module.parameters():
                        param.requires_grad = True
                logger.info(
                    "AuroraCrossFormerWRFDA: base weights frozen; only noise-injection layers are trainable."
                )
        else:
            self.latent_dim = int(noise_conf.get("latent_dim", 128))

        if padding_conf is None:
            padding_conf = {"activate": False}
        self.use_padding = bool(padding_conf["activate"])
        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)

        if use_spectral_norm:
            logger.info("AuroraCrossFormerWRFDA: applying spectral norm")
            apply_spectral_norm(self)

        if post_conf is None:
            post_conf = {"activate": False}
        self.use_post_block = bool(post_conf["activate"])
        if self.use_post_block:
            self.postblock = PostBlock(post_conf)

    def _compute_window_padding(self, H: int, W: int) -> Tuple[int, int]:
        required = self.local_window_size * (2 ** self.num_stages)
        return (-H) % required, (-W) % required

    def _sample_latent(self, B: int, device, dtype) -> torch.Tensor:
        return torch.randn(B, self.latent_dim, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        x_boundary: torch.Tensor,
        x_time_encode: torch.Tensor,
        forecast_step: int = 0,
        ensemble_size: int = 1,
    ) -> torch.Tensor:
        x_copy = x.clone().detach() if self.use_post_block else None

        if self.use_padding:
            x = self.padding_opt.pad(x)
            x_boundary = self.padding_opt.pad(x_boundary)

        B = x.shape[0]
        orig_H, orig_W = x.shape[-2], x.shape[-1]

        if x.shape[2] == 1:
            x_2d = x.squeeze(2)
            xb_2d = x_boundary.squeeze(2)
        else:
            B2, C, T, H, W = x.shape
            x_2d = x.reshape(B2, C * T, H, W)
            Bb, Cb, Tb, Hb, Wb = x_boundary.shape
            xb_2d = x_boundary.reshape(Bb, Cb * Tb, Hb, Wb)

        x_emb = self.interior_embedder(x_2d)
        xb_emb = self.boundary_embedder(xb_2d)

        _Hb, _Wb = xb_emb.shape[-2], xb_emb.shape[-1]
        _kH, _kW = max(1, _Hb // 2), max(1, _Wb // 2)
        xb_quad = F.avg_pool2d(xb_emb, kernel_size=(_kH, _kW), stride=(_kH, _kW)).flatten(1)
        xb_global = xb_emb.mean(dim=(-2, -1))
        film_cond = torch.cat([xb_quad, xb_global, x_time_encode], dim=1)
        gamma_beta = self.film_mlp(film_cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        x_emb = gamma.view(B, -1, 1, 1) * x_emb + beta.view(B, -1, 1, 1)

        _, _, H, W = x_emb.shape
        pad_H, pad_W = self._compute_window_padding(H, W)
        if pad_H or pad_W:
            x_emb = F.pad(x_emb, (0, pad_W, 0, pad_H))

        use_noise_now = self.use_noise and ensemble_size > 1
        if use_noise_now and self.correlated:
            z_shared = self._sample_latent(B, x_emb.device, x_emb.dtype)
        else:
            z_shared = None

        def _z() -> torch.Tensor:
            if z_shared is not None:
                return z_shared
            return self._sample_latent(B, x_emb.device, x_emb.dtype)

        feat = x_emb
        encodings: List[torch.Tensor] = []
        for k, (cel, transformer) in enumerate(self.encoder_layers):
            feat = cel(feat)
            feat = transformer(feat)
            if use_noise_now and self.encoder_noise and k < 3:
                feat = self.enc_noise_layers[k](feat, _z())
            encodings.append(feat)

        feat = self.up_block1(feat)
        if use_noise_now:
            feat = self.dec_noise1(feat, _z())
        feat = torch.cat([feat, encodings[2]], dim=1)

        feat = self.up_block2(feat)
        if use_noise_now:
            feat = self.dec_noise2(feat, _z())
        feat = torch.cat([feat, encodings[1]], dim=1)

        feat = self.up_block3(feat)
        if use_noise_now:
            feat = self.dec_noise3(feat, _z())
        feat = torch.cat([feat, encodings[0]], dim=1)

        feat = self.up_block4(feat)
        if use_noise_now:
            feat = self.dec_noise4(feat, _z())

        feat = feat[:, :, :orig_H, :orig_W].contiguous()
        out = self.out_decoder(feat).unsqueeze(2)

        if self.use_padding:
            out = self.padding_opt.unpad(out)

        if self.use_post_block:
            out = self.postblock({"y_pred": out, "x": x_copy})

        return out
