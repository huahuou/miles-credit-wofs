"""Conditional DiffMAE for WoFS precip random-mask inpainting."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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


def build_3d_sincos_posemb(levels: int, h: int, w: int, embed_dim: int = 768, temperature: float = 10000.0) -> torch.Tensor:
    """Return fixed 3-D sin-cos embeddings shaped (1, levels * h * w, embed_dim)."""
    device = torch.device("cpu")
    z, y, x = torch.meshgrid(
        torch.arange(levels, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    fourier_dim = max(1, embed_dim // 6)
    omega = torch.arange(fourier_dim, device=device, dtype=torch.float32)
    omega = 1.0 / (temperature ** (omega / max(fourier_dim - 1, 1)))
    z = z.flatten()[:, None].float() * omega[None, :]
    y = y.flatten()[:, None].float() * omega[None, :]
    x = x.flatten()[:, None].float() * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim=1)
    pe = F.pad(pe, (0, embed_dim - pe.shape[1]))
    return pe[None, :, :]


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
        target_grid_size: Optional[Tuple[int, ...]] = None,
        target_window_size: int | Tuple[int, int, int] = 0,
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
        use_window = any(int(v) > 0 for v in target_window_size) if isinstance(target_window_size, (list, tuple)) else int(target_window_size) > 0
        if use_window:
            if target_grid_size is None:
                raise ValueError("target_grid_size is required when target_window_size > 0")
            if len(target_grid_size) == 3:
                self.self_attn = WindowAttention3D(
                    dim,
                    num_heads=num_heads,
                    grid_size=target_grid_size,
                    window_size=target_window_size,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                )
            else:
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


class WindowAttention3D(nn.Module):
    """Local self-attention over a 3-D target-token grid."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size: Tuple[int, int, int],
        window_size: int | Tuple[int, int, int],
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.grid_size = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
        if isinstance(window_size, (list, tuple)):
            self.window_size = tuple(max(1, int(v)) for v in window_size)
        else:
            w = max(1, int(window_size))
            self.window_size = (w, w, w)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        grid_z, grid_h, grid_w = self.grid_size
        if n != grid_z * grid_h * grid_w:
            raise ValueError(f"WindowAttention3D expected {grid_z * grid_h * grid_w} tokens, got {n}")
        wz, wh, ww = self.window_size
        x_grid = x.reshape(b, grid_z, grid_h, grid_w, d)
        pad_z = (wz - grid_z % wz) % wz
        pad_h = (wh - grid_h % wh) % wh
        pad_w = (ww - grid_w % ww) % ww
        if pad_z or pad_h or pad_w:
            x_grid = F.pad(x_grid, (0, 0, 0, pad_w, 0, pad_h, 0, pad_z))
        padded_z, padded_h, padded_w = x_grid.shape[1], x_grid.shape[2], x_grid.shape[3]
        x_windows = x_grid.reshape(
            b,
            padded_z // wz,
            wz,
            padded_h // wh,
            wh,
            padded_w // ww,
            ww,
            d,
        )
        x_windows = x_windows.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, wz * wh * ww, d)
        x_windows = self.attn(x_windows)
        x_grid = x_windows.reshape(b, padded_z // wz, padded_h // wh, padded_w // ww, wz, wh, ww, d)
        x_grid = x_grid.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, padded_z, padded_h, padded_w, d)
        return x_grid[:, :grid_z, :grid_h, :grid_w].reshape(b, n, d)


class Precip3DPatchAdapter(nn.Module):
    """Patch precip as variables over a vertical-lat-lon cube grid."""

    def __init__(
        self,
        variable_count: int,
        level_count: int,
        patch_size: Tuple[int, int, int],
        embed_dim: int,
        image_size: Tuple[int, int],
    ):
        super().__init__()
        self.variable_count = int(variable_count)
        self.level_count = int(level_count)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.embed_dim = int(embed_dim)
        self.image_size = tuple(int(v) for v in image_size)
        self.grid_size = (
            math.ceil(self.level_count / self.patch_size[0]),
            math.ceil(self.image_size[0] / self.patch_size[1]),
            math.ceil(self.image_size[1] / self.patch_size[2]),
        )
        self.padded_size = (
            self.grid_size[0] * self.patch_size[0],
            self.grid_size[1] * self.patch_size[1],
            self.grid_size[2] * self.patch_size[2],
        )
        self.proj = nn.Conv3d(
            self.variable_count,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        expected_channels = self.variable_count * self.level_count
        if c != expected_channels:
            raise ValueError(f"Expected {expected_channels} precip channels, got {c}")
        x = x.reshape(b, self.variable_count, self.level_count, h, w)
        pad_d = self.padded_size[0] - self.level_count
        pad_h = self.padded_size[1] - h
        pad_w = self.padded_size[2] - w
        if pad_d < 0 or pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"Input shape {(self.level_count, h, w)} exceeds configured padded size {self.padded_size}"
            )
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        x = self.proj(x)
        x = rearrange(x, "b d z h w -> b (z h w) d")
        return self.norm(x)


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


class WoFSDiffMAE(nn.Module):
    """Diffusion model that inpaints normalized precip conditioned on unmasked WoFS context."""

    def __init__(
        self,
        modality_channels: Optional[Dict[str, int]] = None,
        conditioned_modalities: Optional[list[str]] = None,
        target_modality: str = "precip",
        precip_grouping: str = "level",
        decoder_type: str = "concat_self",
        precip_group_names: Optional[list[str]] = None,
        precip_group_channels: Optional[list[int]] = None,
        level_group_size: int = 1,
        precip_patch_size: Optional[Tuple[int, int, int] | list[int]] = None,
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
        condition_encoder_inputs: str = "all",
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
        if self.precip_grouping != "level":
            raise ValueError(f"Unsupported precip_grouping: {precip_grouping!r}. This model only supports 'level'.")
        self.decoder_type = str(decoder_type).strip().lower()
        if self.decoder_type not in {"concat_self", "cross_self"}:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")
        self.precip_group_names = list(precip_group_names or [])
        self.precip_group_channels = list(precip_group_channels or [])
        self.level_group_size = max(1, int(level_group_size))
        self.patch_size = int(patch_size)
        if precip_patch_size is None:
            self.precip_patch_size = (self.level_group_size, self.patch_size, self.patch_size)
        else:
            if len(precip_patch_size) != 3:
                raise ValueError(f"precip_patch_size must have 3 entries, got {precip_patch_size}")
            self.precip_patch_size = tuple(int(v) for v in precip_patch_size)
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
        self.precip_grid_size = (1, self.target_grid_size[0], self.target_grid_size[1])
        if isinstance(target_attention_window_size, (list, tuple)):
            self.target_attention_window_size = tuple(int(v) for v in target_attention_window_size)
        else:
            self.target_attention_window_size = int(target_attention_window_size)
        self.condition_encoder_depth = int(condition_encoder_depth)
        self.condition_encoder_inputs = str(condition_encoder_inputs).strip().lower()
        if self.condition_encoder_inputs not in {"all", "visible_precip"}:
            raise ValueError(
                "condition_encoder_inputs must be 'all' or 'visible_precip', "
                f"got {condition_encoder_inputs!r}"
            )
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

        self.level_count = 0
        self.level_token_count = 0
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

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

        time_dim = self.embed_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(self.embed_dim),
            nn.Linear(self.embed_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, self.embed_dim),
        )

        self.precip_group_slices: list[Tuple[int, int]] = []
        self.precip_token_adapter = None
        self.precip_token_out_proj = None
        self.level_token_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.level_variable_count = 0
        trunc_normal_(self.level_token_emb, std=0.02)
        self._init_precip_3d_tokenization()

        n_z, n_h, n_w = self.precip_grid_size
        level_pos_emb = build_3d_sincos_posemb(n_z, n_h, n_w, self.embed_dim)
        self.register_buffer("target_level_pos_emb", level_pos_emb)
        self.register_buffer("visible_level_pos_emb", level_pos_emb.clone())
        self.target_modality_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.visible_precip_modality_emb = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.target_modality_emb, std=0.02)
        trunc_normal_(self.visible_precip_modality_emb, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if self.decoder_type == "cross_self":
            self.blocks = nn.ModuleList(
                [
                    CrossSelfDecoderBlock(
                        self.embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[i],
                        target_grid_size=self.precip_grid_size,
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

    def _init_precip_3d_tokenization(self) -> None:
        if not self.precip_group_names:
            raise ValueError("precip_group_names must be provided when precip_grouping is level")
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
                f"Level precip channels sum to {sum(self.precip_group_channels)} but target has {self.channels}"
            )
        if len(set(self.precip_group_channels)) != 1:
            raise ValueError("Level precip grouping requires all variable groups to have the same number of levels")

        self.level_count = int(self.precip_group_channels[0])
        self.level_variable_count = len(self.precip_group_names)
        if self.level_count <= 0:
            raise ValueError("Level precip grouping requires a positive level count")
        self.level_group_size = int(self.precip_patch_size[0])
        self.level_token_count = math.ceil(self.level_count / self.level_group_size)
        start = 0
        for n_ch in self.precip_group_channels:
            stop = start + int(n_ch)
            self.precip_group_slices.append((start, stop))
            start = stop
        self.precip_token_adapter = Precip3DPatchAdapter(
            variable_count=self.level_variable_count,
            level_count=self.level_count,
            patch_size=self.precip_patch_size,
            embed_dim=self.embed_dim,
            image_size=self.image_size,
        )
        self.precip_grid_size = self.precip_token_adapter.grid_size
        self.precip_token_out_proj = nn.Linear(
            self.embed_dim,
            self.level_variable_count
            * self.precip_patch_size[0]
            * self.precip_patch_size[1]
            * self.precip_patch_size[2],
        )

    def _variable_index_for_channel(self, channel_idx: int) -> int:
        for group_idx, (start, stop) in enumerate(self.precip_group_slices):
            if start <= channel_idx < stop:
                return group_idx
        raise IndexError(f"channel index {channel_idx} outside level slices")

    def _precip_3d_tokens(
        self,
        x: torch.Tensor,
        precip_mask: torch.Tensor,
        use_visible: bool = False,
    ) -> torch.Tensor:
        if self.precip_token_adapter is None:
            raise ValueError("3D precip tokenization is not enabled")
        tokens = self.precip_token_adapter(x)
        pos = self.visible_level_pos_emb if use_visible else self.target_level_pos_emb
        tokens = tokens + pos.to(device=x.device, dtype=x.dtype) + self.level_token_emb

        if not use_visible:
            token_mask = self.token_mask_from_precip_mask(precip_mask)
            if token_mask.dim() == 3:
                token_mask = token_mask.reshape(token_mask.shape[0], -1)
            if token_mask.dim() != 2:
                raise ValueError(f"Unsupported token_mask shape {tuple(token_mask.shape)} for 3D tokenization")
            tokens = tokens + self.mask_token * token_mask[:, :, None].to(tokens.dtype)
        return tokens

    def _level_forward_with_condition_tokens(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: list[torch.Tensor],
        precip_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, _, orig_h, orig_w = noisy_precip.shape
        target_tokens = self._precip_3d_tokens(noisy_precip, precip_mask, use_visible=False)
        tokens = self._decode_target_tokens(target_tokens, cond_tokens, t)
        patches = self.precip_token_out_proj(tokens)
        p_d, p_h, p_w = self.precip_patch_size
        n_z, n_h, n_w = self.precip_grid_size
        patches = patches.reshape(
            b,
            n_z,
            n_h,
            n_w,
            self.level_variable_count,
            p_d,
            p_h,
            p_w,
        )
        x = patches.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            b,
            self.level_variable_count,
            n_z * p_d,
            n_h * p_h,
            n_w * p_w,
        )
        x = x[:, :, : self.level_count, :orig_h, :orig_w].reshape(
            b,
            self.level_variable_count * self.level_count,
            orig_h,
            orig_w,
        )
        return self.anti_patch_refiner(x)

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def patchify_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 4:
            mask = mask[:, None]
        if mask.dim() != 5:
            raise ValueError(f"Expected 5-D precip mask, got shape {tuple(mask.shape)}")
        pad_d = (self.precip_patch_size[0] - mask.shape[2] % self.precip_patch_size[0]) % self.precip_patch_size[0]
        pad_h = (self.precip_patch_size[1] - mask.shape[3] % self.precip_patch_size[1]) % self.precip_patch_size[1]
        pad_w = (self.precip_patch_size[2] - mask.shape[4] % self.precip_patch_size[2]) % self.precip_patch_size[2]
        if pad_d or pad_h or pad_w:
            mask = F.pad(mask.float(), (0, pad_w, 0, pad_h, 0, pad_d))
        pooled = F.max_pool3d(mask.float(), kernel_size=self.precip_patch_size, stride=self.precip_patch_size)
        return rearrange(pooled, "b 1 z h w -> b (z h w)")

    def expand_patch_mask(self, patch_mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if patch_mask.dim() != 2:
            raise ValueError(f"Unsupported precip mask shape {tuple(patch_mask.shape)} for 3D expansion")
        n_z, n_h, n_w = self.precip_grid_size
        mask = patch_mask.reshape(patch_mask.shape[0], n_z, n_h, n_w)
        mask = mask.repeat_interleave(self.precip_patch_size[0], dim=1)
        mask = mask.repeat_interleave(self.precip_patch_size[1], dim=2)
        mask = mask.repeat_interleave(self.precip_patch_size[2], dim=3)
        return mask[:, : self.level_count, :height, :width]

    def precip_group_index_for_channel(self, channel_idx: int) -> int:
        return self._variable_index_for_channel(channel_idx)

    def random_precip_mask(self, batch_size: int, mask_ratio: float | tuple[float, float], device: torch.device) -> torch.Tensor:
        n_tokens = self.precip_grid_size[0] * self.precip_grid_size[1] * self.precip_grid_size[2]
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
        raise ValueError("channel masks are not supported in the 3D precip-token model")

    def random_height_precip_mask(self, *args, **kwargs) -> torch.Tensor:
        raise ValueError("height-specific masks are not supported in the 3D precip-token model")

    def token_mask_from_precip_mask(self, precip_mask: torch.Tensor) -> torch.Tensor:
        if precip_mask.dim() == 2:
            return precip_mask
        raise ValueError(f"Unsupported precip mask shape {tuple(precip_mask.shape)} for 3D tokenization")

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
        pixel_mask = pixel_mask.repeat_interleave(self.level_variable_count, dim=1)
        return precip_visible * (1.0 - pixel_mask)

    def _masked_diffusion_state(self, x: torch.Tensor, precip_mask: torch.Tensor) -> torch.Tensor:
        """Keep the diffusion Markov state on masked target pixels only."""
        pixel_mask = self.expand_patch_mask(precip_mask, x.shape[-2], x.shape[-1]).to(device=x.device, dtype=x.dtype)
        pixel_mask = pixel_mask.repeat_interleave(self.level_variable_count, dim=1)
        return x * pixel_mask

    def _precip_pixel_mask(self, precip_mask: torch.Tensor, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        pixel_mask = self.expand_patch_mask(precip_mask, height, width).to(device=precip_mask.device, dtype=dtype)
        return pixel_mask.repeat_interleave(self.level_variable_count, dim=1)

    def _compose_inpaint_state(
        self,
        reverse_state: torch.Tensor,
        t: torch.Tensor,
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Use noisy known pixels plus the current reverse state on masked pixels."""
        if precip_visible is None:
            return reverse_state
        pixel_mask = self._precip_pixel_mask(
            precip_mask,
            reverse_state.shape[-2],
            reverse_state.shape[-1],
            reverse_state.dtype,
        )
        known_noise = torch.randn_like(precip_visible)
        known_state = self.q_sample(precip_visible, t, known_noise).to(reverse_state.dtype)
        return known_state * (1.0 - pixel_mask) + reverse_state * pixel_mask

    def _prepare_reverse_state(
        self,
        reverse_state: torch.Tensor,
        t: torch.Tensor,
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor],
        inpaint_mode: str = "masked_only",
    ) -> torch.Tensor:
        mode = str(inpaint_mode).strip().lower()
        if mode in {"masked_only", "masked", "no_visible_noise"}:
            return reverse_state
        if mode in {"compose_visible", "visible_noise", "inpaint", "noisy_visible"}:
            return self._compose_inpaint_state(reverse_state, t, precip_mask, precip_visible)
        raise ValueError(f"Unsupported inpaint_mode: {inpaint_mode!r}")

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
    ) -> tuple[list[torch.Tensor], list[bool]]:
        cond_tokens = []
        encode_flags = []
        for mod in self.conditioned_modalities:
            if mod in cond:
                cond_tokens.append(self.condition_adapters[mod](cond[mod]))
                encode_flags.append(self.condition_encoder_inputs == "all")
        if precip_visible is not None:
            if precip_mask is None:
                raise ValueError("precip_mask is required when precip_visible is used for conditioning")
            visible_img = self._visible_precip_condition_image(precip_visible, precip_mask)
            visible_tokens = self._precip_3d_tokens(visible_img, precip_mask, use_visible=True)
            cond_tokens.append(visible_tokens + self.visible_precip_modality_emb)
            encode_flags.append(True)
        return cond_tokens, encode_flags

    def _encode_condition_tokens(self, cond_tokens: list[torch.Tensor], encode_flags: Optional[list[bool]] = None) -> list[torch.Tensor]:
        if not cond_tokens or self.condition_encoder_depth <= 0:
            return cond_tokens
        if encode_flags is None:
            encode_flags = [True] * len(cond_tokens)
        if len(encode_flags) != len(cond_tokens):
            raise ValueError("encode_flags length must match cond_tokens length")
        encoder_inputs = [tokens for tokens, should_encode in zip(cond_tokens, encode_flags) if should_encode]
        decoder_only = [tokens for tokens, should_encode in zip(cond_tokens, encode_flags) if not should_encode]
        if not encoder_inputs:
            return decoder_only
        encoded = torch.cat(encoder_inputs, dim=1)
        encoded_by_layer = []
        for block in self.condition_encoder:
            encoded = block(encoded)
            encoded_by_layer.append(encoded)
        encoded_by_layer[-1] = self.condition_encoder_norm(encoded_by_layer[-1])
        if decoder_only:
            decoder_context = torch.cat(decoder_only, dim=1)
            encoded_by_layer = [torch.cat([layer_tokens, decoder_context], dim=1) for layer_tokens in encoded_by_layer]
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
        cond_tokens, encode_flags = self._condition_tokens(cond, precip_visible=precip_visible, precip_mask=precip_mask)
        return self._encode_condition_tokens(cond_tokens, encode_flags)

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
        compose_inpaint: bool = True,
    ):
        if compose_inpaint:
            x_t = self._compose_inpaint_state(x_t, t, precip_mask, precip_visible)
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
        inpaint_mode: str = "masked_only",
    ):
        x_t = self._prepare_reverse_state(x_t, t, precip_mask, precip_visible, inpaint_mode=inpaint_mode)
        _, x_start = self.model_predictions(
            x_t,
            t,
            cond,
            precip_mask,
            precip_visible=precip_visible,
            cond_tokens=cond_tokens,
            compose_inpaint=False,
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
        inpaint_mode: str = "masked_only",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_cond = torch.full((img.shape[0],), time, device=img.device, dtype=torch.long)
        img = self._prepare_reverse_state(img, time_cond, precip_mask, precip_visible, inpaint_mode=inpaint_mode)
        pred_noise, x_start = self.model_predictions(
            img,
            time_cond,
            cond,
            precip_mask,
            precip_visible=precip_visible,
            cond_tokens=cond_tokens,
            compose_inpaint=False,
        )

        if time_next < 0:
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
        x_t = self.q_sample(x_start, t, noise)
        model_out = self.forward(x_t, t, cond, precip_mask, precip_visible=precip_visible)
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            target = self.predict_v(x_start, t, noise)
        loss_raw = (model_out - target) ** 2
        loss = loss_raw.mean()
        return {"loss": loss, "model_out": model_out, "target": target, "x_t": x_t, "t": t}

    def forward(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond_tokens = self._condition_tokens_once(cond, precip_visible=precip_visible, precip_mask=precip_mask)
        return self._level_forward_with_condition_tokens(noisy_precip, t, cond_tokens, precip_mask)

    def _forward_with_condition_tokens(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: list[torch.Tensor],
        precip_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._level_forward_with_condition_tokens(noisy_precip, t, cond_tokens, precip_mask)

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
        inpaint_mode: str = "masked_only",
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
                    inpaint_mode=inpaint_mode,
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
                    inpaint_mode=inpaint_mode,
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
                        inpaint_mode=inpaint_mode,
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
                inpaint_mode=inpaint_mode,
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
