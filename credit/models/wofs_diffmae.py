"""Conditional DiffMAE for WoFS precip random-mask inpainting."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from credit.models.base_model import BaseModel
from credit.models.wofs_mae_adapters import (
    Attention,
    Block,
    CrossAttention,
    Mlp,
    WoFSInputAdapter,
    build_2d_sincos_posemb,
    trunc_normal_,
)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Iterable[int]) -> torch.Tensor:
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(tuple(x_shape)) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(
    timesteps: int,
    start: float = -3,
    end: float = 3,
    tau: float = 1,
    clamp_min: float = 1e-5,
) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((x * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, clamp_min, 0.999)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


class CrossSelfDecoderBlock(nn.Module):
    """DiffMAE decoder block: target tokens cross-attend context, then self-attend."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        target_grid_size: Optional[Tuple[int, int]] = None,
        target_window_size: int = 0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm_cross_q = norm_layer(dim)
        self.norm_cross_ctx = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm_self = norm_layer(dim)
        if target_window_size and target_window_size > 0:
            if target_grid_size is None:
                raise ValueError("target_grid_size is required when target_window_size > 0")
            self.self_attn = WindowAttention2D(
                dim,
                num_heads=num_heads,
                grid_size=target_grid_size,
                window_size=int(target_window_size),
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            self.self_attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path_prob = float(drop_path)

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_prob == 0.0:
            return x
        keep = 1.0 - self.drop_path_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep)
        return x * random_tensor / keep

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.shape[1] > 0:
            x = x + self._drop_path(self.cross_attn(self.norm_cross_q(x), self.norm_cross_ctx(context)))
        x = x + self._drop_path(self.self_attn(self.norm_self(x)))
        x = x + self._drop_path(self.mlp(self.norm_mlp(x)))
        return x


class WindowAttention2D(nn.Module):
    """Local self-attention over a 2-D target-token grid."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size: Tuple[int, int],
        window_size: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.window_size = int(window_size)
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        grid_h, grid_w = self.grid_size
        if n != grid_h * grid_w:
            raise ValueError(f"WindowAttention2D expected {grid_h * grid_w} tokens, got {n}")
        w = self.window_size
        x_grid = x.reshape(b, grid_h, grid_w, d)
        pad_h = (w - grid_h % w) % w
        pad_w = (w - grid_w % w) % w
        if pad_h or pad_w:
            x_grid = F.pad(x_grid, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = x_grid.shape[1], x_grid.shape[2]
        x_windows = x_grid.reshape(b, padded_h // w, w, padded_w // w, w, d)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, d)
        x_windows = self.attn(x_windows)
        x_grid = x_windows.reshape(b, padded_h // w, padded_w // w, w, w, d)
        x_grid = x_grid.permute(0, 1, 3, 2, 4, 5).reshape(b, padded_h, padded_w, d)
        return x_grid[:, :grid_h, :grid_w].reshape(b, n, d)


class ResidualConvRefiner(nn.Module):
    """Small zero-initialized residual CNN for reducing patch-edge artifacts."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
        depth: int = 2,
        kernel_size: int = 3,
        groups: int = 1,
    ):
        super().__init__()
        depth = max(1, int(depth))
        hidden_channels = int(hidden_channels)
        kernel_size = int(kernel_size)
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_channels = int(channels)
        for _ in range(depth - 1):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=int(groups) if in_channels == hidden_channels else 1,
                )
            )
            layers.append(nn.GELU())
            in_channels = hidden_channels
        final = nn.Conv2d(in_channels, int(channels), kernel_size=kernel_size, padding=padding)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class WoFSDiffMAE(BaseModel):
    """Diffusion model that inpaints normalized precip conditioned on unmasked WoFS context."""

    def __init__(
        self,
        modality_channels: Optional[Dict[str, int]] = None,
        conditioned_modalities: Optional[list[str]] = None,
        target_modality: str = "precip",
        precip_grouping: str = "legacy",
        decoder_type: str = "concat_self",
        grouped_decoder_scope: str = "per_group",
        precip_group_names: Optional[list[str]] = None,
        precip_group_channels: Optional[list[int]] = None,
        patch_size: int = 8,
        surface_forcing_stride: int = 16,
        image_size: Tuple[int, int] = (304, 304),
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        condition_encoder_depth: int = 0,
        condition_encoder_mlp_ratio: Optional[float] = None,
        target_attention_window_size: int = 0,
        anti_patch_refiner: Optional[dict] = None,
        diffusion: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.modality_channels = modality_channels or {
            "background": 102,
            "precip": 136,
            "reflectivity": 17,
            "surface": 2,
            "forcing": 10,
        }
        self.conditioned_modalities = conditioned_modalities or ["background", "surface", "forcing", "reflectivity"]
        self.target_modality = target_modality
        self.precip_grouping = str(precip_grouping).strip().lower()
        self.grouped_precip = self.precip_grouping in {"grouped", "by_variable", "factorized"}
        self.decoder_type = str(decoder_type).strip().lower()
        if self.decoder_type not in {"concat_self", "cross_self"}:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")
        self.grouped_decoder_scope = str(grouped_decoder_scope).strip().lower()
        if self.grouped_decoder_scope not in {"per_group", "joint"}:
            raise ValueError(f"Unsupported grouped_decoder_scope: {grouped_decoder_scope}")
        self.precip_group_names = list(precip_group_names or [])
        self.precip_group_channels = list(precip_group_channels or [])
        self.patch_size = int(patch_size)
        self.surface_forcing_stride = int(surface_forcing_stride)
        self.image_size = tuple(image_size)
        self.embed_dim = int(embed_dim)
        self.channels = int(self.modality_channels[target_modality])
        self.output_channels = self.channels
        self.input_channels = self.channels
        self.frames = 1
        self.self_condition = False
        self.condition = True
        self.target_grid_size = (self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size)
        self.target_attention_window_size = int(target_attention_window_size)
        self.condition_encoder_depth = int(condition_encoder_depth)
        self.condition_encoder = nn.ModuleList()
        self.condition_encoder_norm = nn.Identity()
        if self.condition_encoder_depth > 0:
            condition_mlp_ratio = float(condition_encoder_mlp_ratio or mlp_ratio)
            condition_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.condition_encoder_depth)]
            self.condition_encoder = nn.ModuleList(
                [
                    Block(
                        self.embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=condition_mlp_ratio,
                        drop_path=condition_dpr[i],
                    )
                    for i in range(self.condition_encoder_depth)
                ]
            )
            self.condition_encoder_norm = nn.LayerNorm(self.embed_dim)
        refiner_conf = anti_patch_refiner or {}
        self.anti_patch_refiner_enabled = bool(refiner_conf.get("enabled", False))
        self.anti_patch_refiner = nn.Identity()
        if self.anti_patch_refiner_enabled:
            self.anti_patch_refiner = ResidualConvRefiner(
                channels=self.channels,
                hidden_channels=int(refiner_conf.get("hidden_channels", 64)),
                depth=int(refiner_conf.get("depth", 2)),
                kernel_size=int(refiner_conf.get("kernel_size", 3)),
                groups=int(refiner_conf.get("groups", 1)),
            )

        sf_h = math.ceil(self.image_size[0] / self.surface_forcing_stride) * self.surface_forcing_stride
        sf_w = math.ceil(self.image_size[1] / self.surface_forcing_stride) * self.surface_forcing_stride
        sf_image_size = (sf_h, sf_w)

        self.condition_adapters = nn.ModuleDict()
        for mod in self.conditioned_modalities:
            stride = self.surface_forcing_stride if mod in {"surface", "forcing"} else self.patch_size
            adapter_image_size = sf_image_size if mod in {"surface", "forcing"} else self.image_size
            self.condition_adapters[mod] = WoFSInputAdapter(
                num_channels=self.modality_channels[mod],
                patch_size=stride,
                embed_dim=self.embed_dim,
                image_size=adapter_image_size,
            )

        self.visible_precip_adapter = WoFSInputAdapter(
            num_channels=self.channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            image_size=self.image_size,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

        time_dim = self.embed_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(self.embed_dim),
            nn.Linear(self.embed_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, self.embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if self.decoder_type == "cross_self":
            self.blocks = nn.ModuleList(
                [
                    CrossSelfDecoderBlock(
                        self.embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[i],
                        target_grid_size=self.target_grid_size,
                        target_window_size=self.target_attention_window_size,
                    )
                    for i in range(depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [Block(self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i]) for i in range(depth)]
            )
        self.norm = nn.LayerNorm(self.embed_dim)

        self.legacy_precip_adapter = None
        self.legacy_out_proj = None
        self.precip_group_slices: list[Tuple[int, int]] = []
        self.precip_group_adapters = nn.ModuleDict()
        self.precip_group_out_proj = nn.ModuleDict()
        self.precip_group_embeds = nn.ParameterDict()
        if self.grouped_precip:
            self._init_grouped_precip_tokenization()
        else:
            self.legacy_precip_adapter = WoFSInputAdapter(
                num_channels=self.channels,
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                image_size=self.image_size,
            )
            self.legacy_out_proj = nn.Linear(self.embed_dim, self.channels * self.patch_size * self.patch_size)

        n_h, n_w = self.target_grid_size
        pos_emb = build_2d_sincos_posemb(n_h, n_w, self.embed_dim)
        self.register_buffer("target_pos_emb", pos_emb)
        self.target_modality_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.visible_precip_modality_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.target_modality_emb, std=0.02)
        trunc_normal_(self.visible_precip_modality_emb, std=0.02)

        diffusion_conf = diffusion or {}
        self.objective = diffusion_conf.get("objective", "pred_v")
        if self.objective not in {"pred_noise", "pred_x0", "pred_v"}:
            raise ValueError(f"Unsupported diffusion objective: {self.objective}")
        self.num_timesteps = int(diffusion_conf.get("timesteps", 1000))
        self.sampling_timesteps = int(diffusion_conf.get("sampling_timesteps", self.num_timesteps))
        self.ddim_sampling_eta = float(diffusion_conf.get("ddim_sampling_eta", 0.0))
        beta_schedule = diffusion_conf.get("beta_schedule", "sigmoid")
        if beta_schedule == "linear":
            betas = linear_beta_schedule(self.num_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.num_timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(self.num_timesteps)
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")
        self._register_diffusion_buffers(betas)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _register_diffusion_buffers(self, betas: torch.Tensor) -> None:
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float())
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod).float())
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1).float())
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)).float())
        self.register_buffer("posterior_mean_coef1", (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float())
        self.register_buffer("posterior_mean_coef2", ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).float())

    def _init_grouped_precip_tokenization(self) -> None:
        if not self.precip_group_names:
            raise ValueError("precip_group_names must be provided when precip_grouping is grouped")
        if not self.precip_group_channels:
            if self.channels % len(self.precip_group_names) != 0:
                raise ValueError(
                    "precip_group_channels must be provided unless precip channels divide evenly "
                    "across precip_group_names"
                )
            group_size = self.channels // len(self.precip_group_names)
            self.precip_group_channels = [group_size] * len(self.precip_group_names)
        if len(self.precip_group_names) != len(self.precip_group_channels):
            raise ValueError("precip_group_names and precip_group_channels must have the same length")
        if sum(self.precip_group_channels) != self.channels:
            raise ValueError(
                f"Grouped precip channels sum to {sum(self.precip_group_channels)} but target has {self.channels}"
            )

        start = 0
        for name, n_ch in zip(self.precip_group_names, self.precip_group_channels):
            stop = start + int(n_ch)
            self.precip_group_slices.append((start, stop))
            self.precip_group_adapters[name] = WoFSInputAdapter(
                num_channels=int(n_ch),
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                image_size=self.image_size,
            )
            self.precip_group_out_proj[name] = nn.Linear(self.embed_dim, int(n_ch) * self.patch_size * self.patch_size)
            emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(emb, std=0.02)
            self.precip_group_embeds[name] = emb
            start = stop

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def patchify_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 3:
            mask = mask[:, None]
        pooled = F.max_pool2d(mask.float(), kernel_size=self.patch_size, stride=self.patch_size)
        return rearrange(pooled, "b 1 h w -> b (h w)")

    def expand_patch_mask(self, patch_mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
        n_h = self.image_size[0] // self.patch_size
        n_w = self.image_size[1] // self.patch_size
        if patch_mask.dim() == 4:
            if not self.grouped_precip:
                raise ValueError("4-D height masks require grouped precip tokenization")
            b, n_groups, _, _ = patch_mask.shape
            if n_groups != len(self.precip_group_slices):
                raise ValueError(
                    f"Height mask has {n_groups} groups but model has {len(self.precip_group_slices)} precip groups"
                )
            expanded = []
            for group_idx, (start, stop) in enumerate(self.precip_group_slices):
                n_ch = stop - start
                group_mask = patch_mask[:, group_idx, :n_ch, :]
                if group_mask.shape[1] != n_ch:
                    raise ValueError(
                        f"Height mask group {group_idx} has {group_mask.shape[1]} levels, expected {n_ch}"
                    )
                expanded.append(group_mask.reshape(b, n_ch, n_h, n_w))
            mask = torch.cat(expanded, dim=1)
        elif patch_mask.dim() == 3:
            mask = patch_mask.reshape(patch_mask.shape[0], patch_mask.shape[1], n_h, n_w)
            if self.grouped_precip and patch_mask.shape[1] == len(self.precip_group_slices):
                mask = torch.cat(
                    [
                        mask[:, group_idx: group_idx + 1].expand(-1, stop - start, -1, -1)
                        for group_idx, (start, stop) in enumerate(self.precip_group_slices)
                    ],
                    dim=1,
                )
        else:
            mask = patch_mask.reshape(patch_mask.shape[0], 1, n_h, n_w)
        mask = mask.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)
        return mask[:, :, :height, :width]

    def precip_group_index_for_channel(self, channel_idx: int) -> int:
        if not self.grouped_precip:
            return 0
        for group_idx, (start, stop) in enumerate(self.precip_group_slices):
            if start <= channel_idx < stop:
                return group_idx
        raise IndexError(f"channel index {channel_idx} outside grouped precip slices")

    def random_precip_mask(self, batch_size: int, mask_ratio: float | tuple[float, float], device: torch.device) -> torch.Tensor:
        n_tokens = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        if isinstance(mask_ratio, (list, tuple)):
            lo, hi = float(mask_ratio[0]), float(mask_ratio[1])
            ratios = torch.empty(batch_size, device=device).uniform_(lo, hi)
        else:
            ratios = torch.full((batch_size,), float(mask_ratio), device=device)
        ratios = ratios.clamp(0.0, 1.0)
        noise = torch.rand(batch_size, n_tokens, device=device)
        ids = torch.argsort(noise, dim=1)
        ranks = torch.argsort(ids, dim=1)
        n_mask = torch.round(ratios * n_tokens).long().clamp(0, n_tokens)
        return (ranks < n_mask[:, None]).float()

    def random_channel_precip_mask(self, batch_size: int, mask_ratio: float | tuple[float, float], device: torch.device) -> torch.Tensor:
        if self.grouped_precip:
            return self.random_group_precip_mask(batch_size, mask_ratio, device)
        n_tokens = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        n_elements = self.channels * n_tokens
        if isinstance(mask_ratio, (list, tuple)):
            lo, hi = float(mask_ratio[0]), float(mask_ratio[1])
            ratios = torch.empty(batch_size, device=device).uniform_(lo, hi)
        else:
            ratios = torch.full((batch_size,), float(mask_ratio), device=device)
        ratios = ratios.clamp(0.0, 1.0)
        noise = torch.rand(batch_size, n_elements, device=device)
        ids = torch.argsort(noise, dim=1)
        ranks = torch.argsort(ids, dim=1)
        n_mask = torch.round(ratios * n_elements).long().clamp(0, n_elements)
        return (ranks < n_mask[:, None]).float().reshape(batch_size, self.channels, n_tokens)

    def random_group_precip_mask(self, batch_size: int, mask_ratio: float | tuple[float, float], device: torch.device) -> torch.Tensor:
        n_tokens = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        n_groups = len(self.precip_group_slices) if self.grouped_precip else self.channels
        if isinstance(mask_ratio, (list, tuple)):
            lo, hi = float(mask_ratio[0]), float(mask_ratio[1])
            ratios = torch.empty(batch_size, device=device).uniform_(lo, hi)
        else:
            ratios = torch.full((batch_size,), float(mask_ratio), device=device)
        ratios = ratios.clamp(0.0, 1.0)
        noise = torch.rand(batch_size, n_groups * n_tokens, device=device)
        ids = torch.argsort(noise, dim=1)
        ranks = torch.argsort(ids, dim=1)
        n_mask = torch.round(ratios * (n_groups * n_tokens)).long().clamp(0, n_groups * n_tokens)
        return (ranks < n_mask[:, None]).float().reshape(batch_size, n_groups, n_tokens)

    def random_height_precip_mask(
        self,
        batch_size: int,
        mask_ratio: float | tuple[float, float],
        device: torch.device,
        masked_levels: Optional[list[int]] = None,
        visible_levels: Optional[list[int]] = None,
    ) -> torch.Tensor:
        n_tokens = (self.image_size[0] // self.patch_size) * (self.image_size[1] // self.patch_size)
        if not self.grouped_precip:
            return self.random_channel_precip_mask(batch_size, mask_ratio, device)
        group_channels = [stop - start for start, stop in self.precip_group_slices]
        if len(set(group_channels)) != 1:
            raise ValueError("Height masks require all precip groups to have the same number of levels")
        n_groups = len(group_channels)
        n_levels = group_channels[0]
        if isinstance(mask_ratio, (list, tuple)):
            lo, hi = float(mask_ratio[0]), float(mask_ratio[1])
            ratios = torch.empty(batch_size, device=device).uniform_(lo, hi)
        else:
            ratios = torch.full((batch_size,), float(mask_ratio), device=device)
        ratios = ratios.clamp(0.0, 1.0)

        eligible = torch.ones(n_levels, device=device, dtype=torch.bool)
        if masked_levels is not None:
            eligible = torch.zeros(n_levels, device=device, dtype=torch.bool)
            for level in masked_levels:
                if 0 <= int(level) < n_levels:
                    eligible[int(level)] = True
        if visible_levels is not None:
            for level in visible_levels:
                if 0 <= int(level) < n_levels:
                    eligible[int(level)] = False
        n_eligible = int(eligible.sum().item())
        mask = torch.zeros(batch_size, n_groups, n_levels, n_tokens, device=device)
        if n_eligible == 0:
            return mask

        n_elements = n_groups * n_eligible * n_tokens
        noise = torch.rand(batch_size, n_elements, device=device)
        ids = torch.argsort(noise, dim=1)
        ranks = torch.argsort(ids, dim=1)
        n_mask = torch.round(ratios * n_elements).long().clamp(0, n_elements)
        sampled = (ranks < n_mask[:, None]).float().reshape(batch_size, n_groups, n_eligible, n_tokens)
        mask[:, :, eligible, :] = sampled
        return mask

    def token_mask_from_precip_mask(self, precip_mask: torch.Tensor) -> torch.Tensor:
        if precip_mask.dim() == 4:
            return precip_mask.amax(dim=(1, 2))
        if precip_mask.dim() == 3:
            return precip_mask.amax(dim=1)
        return precip_mask

    def _visible_precip_condition_image(
        self,
        precip_visible: Optional[torch.Tensor],
        precip_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if precip_visible is None:
            return None
        pixel_mask = self.expand_patch_mask(precip_mask, precip_visible.shape[-2], precip_visible.shape[-1]).to(
            precip_visible.dtype
        )
        return precip_visible * (1.0 - pixel_mask)

    def _masked_diffusion_state(self, x: torch.Tensor, precip_mask: torch.Tensor) -> torch.Tensor:
        """Keep the diffusion Markov state on masked target pixels only."""
        pixel_mask = self.expand_patch_mask(precip_mask, x.shape[-2], x.shape[-1]).to(device=x.device, dtype=x.dtype)
        return x * pixel_mask

    def _legacy_forward(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond_tokens = self._condition_tokens_once(cond, precip_visible=precip_visible, precip_mask=precip_mask)
        return self._legacy_forward_with_condition_tokens(noisy_precip, t, cond_tokens, precip_mask)

    def _legacy_forward_with_condition_tokens(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: list[torch.Tensor],
        precip_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, _, orig_h, orig_w = noisy_precip.shape
        noisy_tokens = self.legacy_precip_adapter(noisy_precip)
        token_mask = self.token_mask_from_precip_mask(precip_mask).to(noisy_tokens.dtype).unsqueeze(-1)
        target_tokens = noisy_tokens + self.mask_token * token_mask
        pos = rearrange(self.target_pos_emb.to(noisy_precip.dtype), "b d h w -> b (h w) d")
        target_tokens = target_tokens + pos + self.target_modality_emb

        tokens = self._decode_target_tokens(target_tokens, cond_tokens, t)
        patches = self.legacy_out_proj(tokens)
        n_h = self.image_size[0] // self.patch_size
        n_w = self.image_size[1] // self.patch_size
        x = patches.reshape(b, n_h, n_w, self.channels, self.patch_size, self.patch_size)
        x = torch.einsum("bhwcpq->bchpwq", x).reshape(b, self.channels, n_h * self.patch_size, n_w * self.patch_size)
        return self.anti_patch_refiner(x[:, :, :orig_h, :orig_w])

    def _decode_target_tokens(
        self,
        target_tokens: torch.Tensor,
        cond_tokens: list[torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        time_tokens = self.time_mlp(t).unsqueeze(1)
        if self.decoder_type == "cross_self":
            target_tokens = target_tokens + time_tokens
            empty_context = target_tokens.new_zeros(target_tokens.shape[0], 0, target_tokens.shape[-1])
            for block_idx, block in enumerate(self.blocks):
                context = self._decoder_context_for_block(cond_tokens, block_idx, empty_context)
                target_tokens = block(target_tokens, context)
            return self.norm(target_tokens)

        tokens = torch.cat([target_tokens] + cond_tokens, dim=1)
        tokens = tokens + time_tokens
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens[:, : target_tokens.shape[1]])

    def _condition_tokens(
        self,
        cond: Dict[str, torch.Tensor],
        precip_visible: Optional[torch.Tensor] = None,
        precip_mask: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        cond_tokens = []
        for mod in self.conditioned_modalities:
            if mod in cond:
                cond_tokens.append(self.condition_adapters[mod](cond[mod]))
        if precip_visible is not None:
            if precip_mask is None:
                raise ValueError("precip_mask is required when precip_visible is used for conditioning")
            visible_img = self._visible_precip_condition_image(precip_visible, precip_mask)
            visible_tokens = self.visible_precip_adapter(visible_img)
            pos = rearrange(self.target_pos_emb.to(visible_img.dtype), "b d h w -> b (h w) d")
            cond_tokens.append(visible_tokens + pos + self.visible_precip_modality_emb)
        return cond_tokens

    def _encode_condition_tokens(self, cond_tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        if not cond_tokens or self.condition_encoder_depth <= 0:
            return cond_tokens
        encoded = torch.cat(cond_tokens, dim=1)
        encoded_by_layer = []
        for block in self.condition_encoder:
            encoded = block(encoded)
            encoded_by_layer.append(encoded)
        encoded_by_layer[-1] = self.condition_encoder_norm(encoded_by_layer[-1])
        return encoded_by_layer

    def _decoder_context_for_block(
        self,
        cond_tokens: list[torch.Tensor],
        decoder_block_idx: int,
        empty_context: torch.Tensor,
    ) -> torch.Tensor:
        if not cond_tokens:
            return empty_context
        if self.condition_encoder_depth <= 0:
            return torch.cat(cond_tokens, dim=1)
        if len(cond_tokens) == 1:
            return cond_tokens[0]
        if len(cond_tokens) == len(self.blocks):
            return cond_tokens[-1 - decoder_block_idx]
        encoder_idx = round((len(cond_tokens) - 1) * (len(self.blocks) - 1 - decoder_block_idx) / max(len(self.blocks) - 1, 1))
        return cond_tokens[int(encoder_idx)]

    def _condition_tokens_once(
        self,
        cond: Dict[str, torch.Tensor],
        precip_visible: Optional[torch.Tensor] = None,
        precip_mask: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        return self._encode_condition_tokens(
            self._condition_tokens(cond, precip_visible=precip_visible, precip_mask=precip_mask)
        )

    def _grouped_forward(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond_tokens = self._condition_tokens_once(cond, precip_visible=precip_visible, precip_mask=precip_mask)
        return self._grouped_forward_with_condition_tokens(noisy_precip, t, cond_tokens, precip_mask)

    def _grouped_forward_with_condition_tokens(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: list[torch.Tensor],
        precip_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, _, orig_h, orig_w = noisy_precip.shape
        if self.grouped_decoder_scope == "joint":
            return self._grouped_forward_joint(noisy_precip, t, cond_tokens, precip_mask)

        outs = []
        for group_idx, (name, (start, stop)) in enumerate(zip(self.precip_group_names, self.precip_group_slices)):
            group_tokens = self._group_tokens(noisy_precip, precip_mask, group_idx, name, start, stop)
            tokens = self._decode_target_tokens(group_tokens, cond_tokens, t)
            patches = self.precip_group_out_proj[name](tokens)
            n_ch = stop - start
            n_h = self.image_size[0] // self.patch_size
            n_w = self.image_size[1] // self.patch_size
            group_out = patches.reshape(b, n_h, n_w, n_ch, self.patch_size, self.patch_size)
            group_out = torch.einsum("bhwcpq->bchpwq", group_out).reshape(b, n_ch, n_h * self.patch_size, n_w * self.patch_size)
            outs.append(group_out[:, :, :orig_h, :orig_w])
        return self.anti_patch_refiner(torch.cat(outs, dim=1))

    def _group_tokens(
        self,
        noisy_precip: torch.Tensor,
        precip_mask: torch.Tensor,
        group_idx: int,
        name: str,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        group_x = noisy_precip[:, start:stop]
        group_tokens = self.precip_group_adapters[name](group_x)
        if precip_mask.dim() == 2:
            group_mask = precip_mask
        elif precip_mask.dim() == 3:
            group_mask = precip_mask[:, group_idx, :]
        elif precip_mask.dim() == 4:
            group_mask = precip_mask[:, group_idx].amax(dim=1)
        else:
            raise ValueError(f"Unsupported precip_mask shape {tuple(precip_mask.shape)}")
        token_mask = group_mask.to(group_tokens.dtype).unsqueeze(-1)
        group_tokens = group_tokens + self.mask_token * token_mask
        pos = rearrange(self.target_pos_emb.to(noisy_precip.dtype), "b d h w -> b (h w) d")
        return group_tokens + pos + self.target_modality_emb + self.precip_group_embeds[name]

    def _grouped_forward_joint(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: list[torch.Tensor],
        precip_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, _, orig_h, orig_w = noisy_precip.shape
        group_tokens = [
            self._group_tokens(noisy_precip, precip_mask, group_idx, name, start, stop)
            for group_idx, (name, (start, stop)) in enumerate(zip(self.precip_group_names, self.precip_group_slices))
        ]
        n_tokens = group_tokens[0].shape[1]
        decoded = self._decode_target_tokens(torch.cat(group_tokens, dim=1), cond_tokens, t)
        decoded_groups = decoded.split(n_tokens, dim=1)

        outs = []
        for tokens, name, (start, stop) in zip(decoded_groups, self.precip_group_names, self.precip_group_slices):
            patches = self.precip_group_out_proj[name](tokens)
            n_ch = stop - start
            n_h = self.image_size[0] // self.patch_size
            n_w = self.image_size[1] // self.patch_size
            group_out = patches.reshape(b, n_h, n_w, n_ch, self.patch_size, self.patch_size)
            group_out = torch.einsum("bhwcpq->bchpwq", group_out).reshape(b, n_ch, n_h * self.patch_size, n_w * self.patch_size)
            outs.append(group_out[:, :, :orig_h, :orig_w])
        return self.anti_patch_refiner(torch.cat(outs, dim=1))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_predictions(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
        cond_tokens: Optional[list[torch.Tensor]] = None,
    ):
        x_t = self._masked_diffusion_state(x_t, precip_mask)
        if cond_tokens is None:
            model_out = self.forward(x_t, t, cond, precip_mask, precip_visible=precip_visible)
        else:
            model_out = self._forward_with_condition_tokens(x_t, t, cond_tokens, precip_mask)
        if self.objective == "pred_noise":
            pred_noise = model_out
            pred_x_start = self.predict_start_from_noise(x_t, t, pred_noise)
        elif self.objective == "pred_x0":
            pred_x_start = model_out
            pred_noise = self.predict_noise_from_start(x_t, t, pred_x_start)
        else:
            pred_x_start = self.predict_start_from_v(x_t, t, model_out)
            pred_noise = self.predict_noise_from_start(x_t, t, pred_x_start)
        return pred_noise, pred_x_start

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
        cond_tokens: Optional[list[torch.Tensor]] = None,
    ):
        _, x_start = self.model_predictions(
            x_t,
            t,
            cond,
            precip_mask,
            precip_visible=precip_visible,
            cond_tokens=cond_tokens,
        )
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def _repaint_times(self, start_time: int, jump_length: int, jump_n_sample: int) -> list[int]:
        jump_length = max(1, int(jump_length))
        jump_n_sample = max(1, int(jump_n_sample))
        t_t = int(start_time) + 1
        jumps = {
            time: jump_n_sample - 1
            for time in range(0, t_t - jump_length, jump_length)
        }
        times = []
        time = t_t
        while time >= 1:
            time -= 1
            times.append(time)
            if jumps.get(time, 0) > 0:
                jumps[time] -= 1
                for _ in range(jump_length):
                    time += 1
                    times.append(time)
        times.append(-1)
        return times

    def _repaint_forward_step(self, img: torch.Tensor, from_time: int, to_time: int) -> torch.Tensor:
        if to_time <= from_time:
            return img
        for time in range(from_time + 1, to_time + 1):
            beta = self.betas[time]
            img = (1.0 - beta).sqrt() * img + beta.sqrt() * torch.randn_like(img)
        return img

    def _ddim_timesteps(self, total_steps: int) -> list[int]:
        total_steps = max(1, min(int(total_steps), self.num_timesteps))
        if total_steps == self.num_timesteps:
            return list(reversed(range(self.num_timesteps)))

        grid = torch.linspace(0, self.num_timesteps - 1, steps=total_steps, device=self.device)
        rounded = torch.round(grid).long().tolist()

        deduped: list[int] = []
        seen: set[int] = set()
        for time in rounded:
            time = int(time)
            if time not in seen:
                deduped.append(time)
                seen.add(time)

        return list(reversed(deduped))

    def _ddim_step(
        self,
        img: torch.Tensor,
        time: int,
        time_next: int,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        eta: float,
        precip_visible: Optional[torch.Tensor] = None,
        cond_tokens: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_cond = torch.full((img.shape[0],), time, device=img.device, dtype=torch.long)
        pred_noise, x_start = self.model_predictions(
            img,
            time_cond,
            cond,
            precip_mask,
            precip_visible=precip_visible,
            cond_tokens=cond_tokens,
        )

        if time_next < 0:
            x_start = self._masked_diffusion_state(x_start, precip_mask)
            return x_start, x_start, pred_noise

        alpha_t = self.alphas_cumprod[time].to(device=img.device, dtype=img.dtype)
        alpha_prev = self.alphas_cumprod[time_next].to(device=img.device, dtype=img.dtype)
        eps = torch.finfo(img.dtype).eps if img.dtype.is_floating_point else 1.0e-8
        alpha_t = alpha_t.clamp(min=eps, max=1.0 - eps)
        alpha_prev = alpha_prev.clamp(min=eps, max=1.0 - eps)

        one_minus_alpha_t = (1.0 - alpha_t).clamp_min(0.0)
        one_minus_alpha_prev = (1.0 - alpha_prev).clamp_min(0.0)
        sigma_sq = (eta**2) * (one_minus_alpha_prev / one_minus_alpha_t) * (1.0 - alpha_t / alpha_prev)
        sigma_sq = sigma_sq.clamp_min(0.0)
        sigma = torch.sqrt(sigma_sq)
        c = torch.sqrt((one_minus_alpha_prev - sigma_sq).clamp_min(0.0))
        noise = torch.randn_like(img) if eta > 0 else torch.zeros_like(img)
        img_prev = torch.sqrt(alpha_prev) * x_start + c * pred_noise + sigma * noise
        img_prev = self._masked_diffusion_state(img_prev, precip_mask)
        return img_prev, x_start, pred_noise

    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        precip_visible: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b = x_start.shape[0]
        if t is None:
            t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        if noise is None:
            noise = torch.randn_like(x_start)
        if precip_visible is None:
            precip_visible = x_start
        x_t = self._masked_diffusion_state(self.q_sample(x_start, t, noise), precip_mask)
        model_out = self.forward(x_t, t, cond, precip_mask, precip_visible=precip_visible)
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            target = self.predict_v(x_start, t, noise)
        pixel_mask = self.expand_patch_mask(precip_mask, x_start.shape[-2], x_start.shape[-1]).to(x_start.dtype)
        loss_raw = (model_out - target) ** 2
        denom = pixel_mask.sum()
        if pixel_mask.shape[1] == 1:
            denom = denom * x_start.shape[1]
        loss = (loss_raw * pixel_mask).sum() / denom.clamp_min(1.0)
        return {"loss": loss, "model_out": model_out, "target": target, "x_t": x_t, "t": t}

    def forward(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.grouped_precip:
            return self._grouped_forward(noisy_precip, t, cond, precip_mask, precip_visible=precip_visible)
        return self._legacy_forward(noisy_precip, t, cond, precip_mask, precip_visible=precip_visible)

    def _forward_with_condition_tokens(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: list[torch.Tensor],
        precip_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.grouped_precip:
            return self._grouped_forward_with_condition_tokens(noisy_precip, t, cond_tokens, precip_mask)
        return self._legacy_forward_with_condition_tokens(noisy_precip, t, cond_tokens, precip_mask)

    @torch.no_grad()
    def sample_precip(
        self,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
        sampling_timesteps: Optional[int] = None,
        eta: Optional[float] = None,
        sampler: str = "ddim",
        repaint_jump_length: int = 10,
        repaint_jump_n_sample: int = 10,
        return_all_timesteps: bool = False,
        clamp_final_visible: bool = True,
    ) -> torch.Tensor:
        del clamp_final_visible
        first = next(iter(cond.values()))
        b, _, h, w = first.shape
        shape = (b, self.channels, h, w)
        img = torch.randn(shape, device=first.device, dtype=first.dtype)

        sampler = str(sampler).strip().lower()
        if sampler not in {"ddim", "ddpm", "repaint", "repaint_ddim"}:
            raise ValueError(f"Unsupported DiffMAE sampler: {sampler!r}")
        total_steps = self.num_timesteps if sampler in {"ddpm", "repaint"} else (sampling_timesteps or self.sampling_timesteps)
        eta = self.ddim_sampling_eta if eta is None else float(eta)
        start_time = self.num_timesteps - 1
        if sampler == "repaint" and sampling_timesteps is not None:
            start_time = min(start_time, max(0, int(sampling_timesteps) - 1))
        cond_tokens = self._condition_tokens_once(cond, precip_visible=precip_visible, precip_mask=precip_mask)
        imgs = [img] if return_all_timesteps else None

        if sampler == "ddpm":
            img = self._masked_diffusion_state(img, precip_mask)
            for time in reversed(range(self.num_timesteps)):
                time_cond = torch.full((b,), time, device=first.device, dtype=torch.long)
                model_mean, _, model_log_variance, _ = self.p_mean_variance(
                    img,
                    time_cond,
                    cond,
                    precip_mask,
                    precip_visible=precip_visible,
                    cond_tokens=cond_tokens,
                )
                noise = torch.randn_like(img) if time > 0 else torch.zeros_like(img)
                img = model_mean + (0.5 * model_log_variance).exp() * noise
                img = self._masked_diffusion_state(img, precip_mask)
                if imgs is not None:
                    imgs.append(img)
            return torch.stack(imgs, dim=1) if imgs is not None else img

        if sampler == "repaint":
            img = self._masked_diffusion_state(img, precip_mask)
            times = self._repaint_times(start_time, repaint_jump_length, repaint_jump_n_sample)
            if not times or times[0] != start_time:
                raise RuntimeError(f"Invalid RePaint schedule starts with {times[0] if times else None}, expected {start_time}")
            for time, time_next in zip(times[:-1], times[1:]):
                if time_next > time:
                    img = self._repaint_forward_step(img, time, time_next)
                    img = self._masked_diffusion_state(img, precip_mask)
                    if imgs is not None:
                        imgs.append(img)
                    continue
                time_cond = torch.full((b,), time, device=first.device, dtype=torch.long)
                model_mean, _, model_log_variance, _ = self.p_mean_variance(
                    img,
                    time_cond,
                    cond,
                    precip_mask,
                    precip_visible=precip_visible,
                    cond_tokens=cond_tokens,
                )
                noise = torch.randn_like(img) if time > 0 else torch.zeros_like(img)
                img = model_mean + (0.5 * model_log_variance).exp() * noise
                img = self._masked_diffusion_state(img, precip_mask)
                if imgs is not None:
                    imgs.append(img)
            return torch.stack(imgs, dim=1) if imgs is not None else img

        if sampler == "repaint_ddim":
            img = self._masked_diffusion_state(img, precip_mask)
            times = self._ddim_timesteps(total_steps)
            repaint_n_sample = max(1, int(repaint_jump_n_sample))
            for idx, time in enumerate(times):
                time_next = times[idx + 1] if idx + 1 < len(times) else -1
                pred_noise = None
                x_start = None
                for resample_idx in range(repaint_n_sample):
                    img, x_start, pred_noise = self._ddim_step(
                        img,
                        time,
                        time_next,
                        cond,
                        precip_mask,
                        eta,
                        precip_visible=precip_visible,
                        cond_tokens=cond_tokens,
                    )
                    if resample_idx < repaint_n_sample - 1:
                        alpha = self.alphas_cumprod[time].to(device=img.device, dtype=img.dtype)
                        img = alpha.sqrt() * x_start + (1.0 - alpha).sqrt() * torch.randn_like(img)
                        img = self._masked_diffusion_state(img, precip_mask)
                if imgs is not None:
                    imgs.append(img)
            return torch.stack(imgs, dim=1) if imgs is not None else img

        img = self._masked_diffusion_state(img, precip_mask)
        times = self._ddim_timesteps(total_steps)
        for idx, time in enumerate(times):
            time_next = times[idx + 1] if idx + 1 < len(times) else -1
            img, x_start, _ = self._ddim_step(
                img,
                time,
                time_next,
                cond,
                precip_mask,
                eta,
                precip_visible=precip_visible,
                cond_tokens=cond_tokens,
            )
            if imgs is not None:
                imgs.append(img)
        return torch.stack(imgs, dim=1) if imgs is not None else img


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb
