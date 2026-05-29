"""Native 3D FCDM-style diffusion model for WoFS precip inpainting."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from credit.models.wofs_diffmae import (
    cosine_beta_schedule,
    extract,
    linear_beta_schedule,
    sigmoid_beta_schedule,
)


def _modulate_3d(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]


class LayerNorm3d(nn.LayerNorm):
    def __init__(self, num_channels: int, eps: float = 1.0e-6, affine: bool = True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 4, 1, 2, 3)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half, 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class GRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        gx = torch.norm(x_fp32, p=2, dim=(2, 3, 4), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1.0e-6)
        y = self.gamma.float() * (x_fp32 * nx) + self.beta.float() + x_fp32
        return y.to(dtype=x.dtype)


class ConvNeXtBlock(nn.Module):
    """FCDM ConvNeXt block with native height axis and factorized kernels.

    The expensive spatial operator is depthwise ``1 x K x K``. A lightweight
    grouped ``3 x 1 x 1`` operator can mix neighboring heights without turning
    every block into a full 3D spatial convolution.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 2.0,
        spatial_kernel_size: int = 7,
        vertical_kernel_size: int = 3,
        vertical_mixing: bool = True,
    ):
        super().__init__()
        spatial_pad = spatial_kernel_size // 2
        vertical_pad = vertical_kernel_size // 2
        self.spatial_dwconv = nn.Conv3d(
            dim,
            dim,
            kernel_size=(1, spatial_kernel_size, spatial_kernel_size),
            padding=(0, spatial_pad, spatial_pad),
            groups=dim,
        )
        self.vertical_dwconv = (
            nn.Conv3d(
                dim,
                dim,
                kernel_size=(vertical_kernel_size, 1, 1),
                padding=(vertical_pad, 0, 0),
                groups=dim,
            )
            if vertical_mixing and vertical_kernel_size > 1
            else None
        )
        self.norm = LayerNorm3d(dim, affine=False, eps=1.0e-6)
        hidden = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv3d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(hidden)
        self.pwconv2 = nn.Conv3d(hidden, dim, kernel_size=1)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.spatial_dwconv(x)
        if self.vertical_dwconv is not None:
            h = h + self.vertical_dwconv(x)
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=1)
        h = _modulate_3d(self.norm(h), shift, scale)
        h = self.pwconv2(self.grn(self.act(self.pwconv1(h))))
        return x + gate[:, :, None, None, None] * h


class Downsample3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        return self.conv(x)


class ConvFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = LayerNorm3d(hidden_size, affine=False, eps=1.0e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.conv = nn.Conv3d(hidden_size, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.conv(_modulate_3d(self.norm(x), shift, scale))


class FCDMBackbone3D(nn.Module):
    """FCDM U-Net layout using native ``(B, C, Z, H, W)`` tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int,
        hidden_size: int = 128,
        depth: Optional[Iterable[int]] = None,
        mlp_ratio: float = 2.0,
        spatial_kernel_size: int = 7,
        vertical_kernel_size: int = 3,
        vertical_mixing_interval: int = 2,
    ):
        super().__init__()
        depth = list(depth or [1, 2, 4, 2, 1])
        if len(depth) != 5:
            raise ValueError(f"FCDM depth must have 5 entries, got {depth}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_levels = int(num_levels)
        vertical_mixing_interval = max(1, int(vertical_mixing_interval))

        self.t_embedder_1 = TimestepEmbedder(hidden_size)
        self.t_embedder_2 = TimestepEmbedder(hidden_size * 2)
        self.t_embedder_3 = TimestepEmbedder(hidden_size * 4)

        self.x_embedder = nn.Conv3d(self.in_channels, hidden_size, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.level_embed = nn.Parameter(torch.zeros(1, hidden_size, self.num_levels, 1, 1))
        self.encoder_level_1 = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden_size,
                    mlp_ratio,
                    spatial_kernel_size,
                    vertical_kernel_size,
                    vertical_mixing=(i % vertical_mixing_interval == 0),
                )
                for i in range(depth[0])
            ]
        )
        self.down1_2 = Downsample3d(hidden_size, hidden_size * 2)
        self.encoder_level_2 = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden_size * 2,
                    mlp_ratio,
                    spatial_kernel_size,
                    vertical_kernel_size,
                    vertical_mixing=(i % vertical_mixing_interval == 0),
                )
                for i in range(depth[1])
            ]
        )
        self.down2_3 = Downsample3d(hidden_size * 2, hidden_size * 4)
        self.latent = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden_size * 4,
                    mlp_ratio,
                    spatial_kernel_size,
                    vertical_kernel_size,
                    vertical_mixing=(i % vertical_mixing_interval == 0),
                )
                for i in range(depth[2])
            ]
        )
        self.up3_2 = Upsample3d(hidden_size * 4, hidden_size * 2)
        self.reduce_chans_2 = nn.Conv3d(hidden_size * 4, hidden_size * 2, kernel_size=1)
        self.decoder_level_2 = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden_size * 2,
                    mlp_ratio,
                    spatial_kernel_size,
                    vertical_kernel_size,
                    vertical_mixing=(i % vertical_mixing_interval == 0),
                )
                for i in range(depth[3])
            ]
        )
        self.up2_1 = Upsample3d(hidden_size * 2, hidden_size)
        self.reduce_chans_1 = nn.Conv3d(hidden_size * 2, hidden_size, kernel_size=1)
        self.decoder_level_1 = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden_size,
                    mlp_ratio,
                    spatial_kernel_size,
                    vertical_kernel_size,
                    vertical_mixing=(i % vertical_mixing_interval == 0),
                )
                for i in range(depth[4])
            ]
        )
        self.output_layer = nn.Conv3d(hidden_size, hidden_size, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.final_layer = ConvFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)
        nn.init.normal_(self.level_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, ConvNeXtBlock):
                nn.init.zeros_(module.adaLN_modulation[-1].weight)
                nn.init.zeros_(module.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.conv.weight)
        nn.init.zeros_(self.final_layer.conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c1 = self.t_embedder_1(t)
        c2 = self.t_embedder_2(t)
        c3 = self.t_embedder_3(t)

        enc1 = self.x_embedder(x) + self.level_embed[:, :, : x.shape[2]]
        for block in self.encoder_level_1:
            enc1 = block(enc1, c1)
        enc2 = self.down1_2(enc1)
        for block in self.encoder_level_2:
            enc2 = block(enc2, c2)
        latent = self.down2_3(enc2)
        for block in self.latent:
            latent = block(latent, c3)
        dec2 = self.up3_2(latent)
        dec2 = dec2[:, :, :, : enc2.shape[-2], : enc2.shape[-1]]
        dec2 = self.reduce_chans_2(torch.cat([dec2, enc2], dim=1))
        for block in self.decoder_level_2:
            dec2 = block(dec2, c2)
        dec1 = self.up2_1(dec2)
        dec1 = dec1[:, :, :, : enc1.shape[-2], : enc1.shape[-1]]
        dec1 = self.reduce_chans_1(torch.cat([dec1, enc1], dim=1))
        for block in self.decoder_level_1:
            dec1 = block(dec1, c1)
        return self.final_layer(self.output_layer(dec1), c1)


class WoFSFCDMXL(nn.Module):
    """Native 3D FCDM-style wrapper for WoFS precip, background, and reflectivity."""

    def __init__(
        self,
        modality_channels: Optional[Dict[str, int]] = None,
        precip_group_names: Optional[list[str]] = None,
        precip_group_channels: Optional[list[int]] = None,
        level_count: Optional[int] = None,
        include_visible_precip: bool = False,
        include_mask_channel: bool = False,
        image_size: tuple[int, int] = (300, 300),
        fcdm_hidden_size: int = 128,
        fcdm_depth: Optional[list[int]] = None,
        fcdm_mlp_ratio: float = 2.0,
        fcdm_spatial_kernel_size: int = 7,
        fcdm_vertical_kernel_size: int = 3,
        fcdm_vertical_mixing_interval: int = 2,
        pad_to_multiple: int = 4,
        diffusion: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        kwargs.pop("vertical_window", None)
        kwargs.pop("class_dropout_prob", None)
        if kwargs:
            raise ValueError(f"Unsupported WoFSFCDMXL config keys: {sorted(kwargs)}")
        self.modality_channels = modality_channels or {
            "background": 102,
            "precip": 136,
            "reflectivity": 17,
            "surface": 0,
            "forcing": 12,
        }
        self.precip_group_names = list(precip_group_names or [])
        self.precip_group_channels = list(precip_group_channels or [])
        self.channels = int(self.modality_channels["precip"])
        self.output_channels = self.channels
        self.input_channels = self.channels
        self.frames = 1
        self.self_condition = False
        self.condition = True
        self.image_size = tuple(int(v) for v in image_size)
        self.include_visible_precip = bool(include_visible_precip)
        self.include_mask_channel = bool(include_mask_channel)
        self.pad_to_multiple = max(1, int(pad_to_multiple))

        if not self.precip_group_channels:
            if not self.precip_group_names:
                raise ValueError("precip_group_names or precip_group_channels must be provided")
            if self.channels % len(self.precip_group_names) != 0:
                raise ValueError("precip channels must divide evenly across precip_group_names")
            self.precip_group_channels = [self.channels // len(self.precip_group_names)] * len(self.precip_group_names)
        if len(set(self.precip_group_channels)) != 1:
            raise ValueError("WoFSFCDMXL requires equal precip levels per precip variable")
        if sum(self.precip_group_channels) != self.channels:
            raise ValueError("precip_group_channels must sum to modality_channels['precip']")
        self.level_count = int(level_count or self.precip_group_channels[0])
        self.level_variable_count = len(self.precip_group_channels)
        if self.level_count <= 0:
            raise ValueError("level_count must be positive")
        if self.channels % self.level_count != 0:
            raise ValueError("precip channels must divide evenly by level_count")

        bg_channels = int(self.modality_channels.get("background", 0))
        refl_channels = int(self.modality_channels.get("reflectivity", 0))
        if bg_channels % self.level_count != 0:
            raise ValueError("background channels must divide evenly by level_count")
        if refl_channels % self.level_count != 0:
            raise ValueError("reflectivity channels must divide evenly by level_count")
        self.background_variable_count = bg_channels // self.level_count
        self.reflectivity_variable_count = refl_channels // self.level_count
        self.forcing_channels = int(self.modality_channels.get("forcing", 0)) + int(self.modality_channels.get("surface", 0))

        in_channels = self.level_variable_count + self.background_variable_count + self.reflectivity_variable_count + self.forcing_channels
        if self.include_visible_precip:
            in_channels += self.level_variable_count
        if self.include_mask_channel:
            in_channels += 1

        self.backbone = FCDMBackbone3D(
            in_channels=in_channels,
            out_channels=self.level_variable_count,
            num_levels=self.level_count,
            hidden_size=int(fcdm_hidden_size),
            depth=fcdm_depth or [1, 2, 4, 2, 1],
            mlp_ratio=float(fcdm_mlp_ratio),
            spatial_kernel_size=int(fcdm_spatial_kernel_size),
            vertical_kernel_size=int(fcdm_vertical_kernel_size),
            vertical_mixing_interval=int(fcdm_vertical_mixing_interval),
        )

        diffusion_conf = diffusion or {}
        self.objective = diffusion_conf.get("objective", "pred_x0")
        if self.objective not in {"pred_noise", "pred_x0", "pred_v"}:
            raise ValueError(f"Unsupported diffusion objective: {self.objective}")
        self.default_inpaint_mode = str(diffusion_conf.get("inpaint_mode", "compose_visible")).strip().lower()
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

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def _as_5d_precip(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} precip channels, got {c}")
        return x.reshape(b, self.level_variable_count, self.level_count, h, w)

    def _from_5d_precip(self, x: torch.Tensor) -> torch.Tensor:
        b, v, z, h, w = x.shape
        if v != self.level_variable_count or z != self.level_count:
            raise ValueError(f"Unexpected precip volume shape {tuple(x.shape)}")
        return x.reshape(b, v * z, h, w)

    def _reshape_context_3d(self, x: torch.Tensor, levels: int, name: str) -> torch.Tensor:
        b, c, h, w = x.shape
        if c % levels != 0:
            raise ValueError(f"{name} channels {c} are not divisible by levels {levels}")
        return x.reshape(b, c // levels, levels, h, w)

    def _dense_mask(self, precip_mask: Optional[torch.Tensor], height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if precip_mask is None:
            return torch.ones(1, self.level_count, height, width, device=device, dtype=dtype)
        mask = precip_mask.to(device=device, dtype=dtype)
        if mask.dim() == 2:
            n_z = self.level_count
            n_hw = mask.shape[1] // n_z
            side = int(math.sqrt(n_hw))
            if side * side * n_z != mask.shape[1]:
                raise ValueError(f"Cannot infer 3D patch grid from mask shape {tuple(mask.shape)}")
            mask = mask.reshape(mask.shape[0], n_z, side, side)
            mask = mask.repeat_interleave(math.ceil(height / side), dim=2)
            mask = mask.repeat_interleave(math.ceil(width / side), dim=3)
            return mask[:, :, :height, :width]
        if mask.dim() == 4:
            if mask.shape[1] == 1:
                mask = mask.expand(-1, self.level_count, -1, -1)
            if mask.shape[1] != self.level_count:
                raise ValueError(f"Expected mask level axis {self.level_count}, got {mask.shape[1]}")
            return mask
        if mask.dim() == 5 and mask.shape[1] == 1:
            return mask[:, 0]
        raise ValueError(f"Unsupported precip mask shape {tuple(mask.shape)}")

    def expand_patch_mask(self, precip_mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return self._dense_mask(precip_mask, height, width, precip_mask.device, precip_mask.dtype)

    def random_precip_mask(self, batch_size: int, mask_ratio: float | tuple[float, float], device: torch.device) -> torch.Tensor:
        h, w = self.image_size
        if isinstance(mask_ratio, (list, tuple)):
            lo, hi = float(mask_ratio[0]), float(mask_ratio[1])
            ratios = torch.empty(batch_size, device=device).uniform_(lo, hi)
        else:
            ratios = torch.full((batch_size,), float(mask_ratio), device=device)
        ratios = ratios.clamp(0.0, 1.0)
        noise = torch.rand(batch_size, self.level_count, h, w, device=device)
        flat = noise.flatten(1)
        ids = torch.argsort(flat, dim=1)
        ranks = torch.argsort(ids, dim=1)
        n_mask = torch.round(ratios * flat.shape[1]).long().clamp(0, flat.shape[1])
        return (ranks < n_mask[:, None]).reshape(batch_size, self.level_count, h, w).float()

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

    def _pad_3d(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        pad_h = (self.pad_to_multiple - h % self.pad_to_multiple) % self.pad_to_multiple
        pad_w = (self.pad_to_multiple - w % self.pad_to_multiple) % self.pad_to_multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (h, w)

    def _build_fcdm_volume(
        self,
        noisy_precip: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: Optional[torch.Tensor],
        precip_visible: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        precip = self._as_5d_precip(noisy_precip)
        b, _, _, h, w = precip.shape
        device, dtype = noisy_precip.device, noisy_precip.dtype
        mask = self._dense_mask(precip_mask, h, w, device, dtype)
        if mask.shape[0] == 1 and b > 1:
            mask = mask.expand(b, -1, -1, -1)
        visible = self._as_5d_precip(precip_visible) if precip_visible is not None else torch.zeros_like(precip)

        parts = [precip]
        if self.include_visible_precip:
            parts.append(visible * (1.0 - mask[:, None]))
        if self.include_mask_channel:
            parts.append(mask[:, None])

        parts.append(self._reshape_context_3d(cond["background"], self.level_count, "background"))
        parts.append(self._reshape_context_3d(cond["reflectivity"], self.level_count, "reflectivity"))

        forcing_parts = []
        if "surface" in cond and cond["surface"].shape[1] > 0:
            forcing_parts.append(cond["surface"])
        if "forcing" in cond and cond["forcing"].shape[1] > 0:
            forcing_parts.append(cond["forcing"])
        if forcing_parts:
            forcing = torch.cat(forcing_parts, dim=1)[:, :, None].expand(-1, -1, self.level_count, -1, -1)
        else:
            forcing = noisy_precip.new_zeros(b, 0, self.level_count, h, w)
        parts.append(forcing)

        x = torch.cat(parts, dim=1)
        return self._pad_3d(x)

    def forward(
        self,
        noisy_precip: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: Optional[torch.Tensor] = None,
        precip_visible: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, orig_hw = self._build_fcdm_volume(noisy_precip, cond, precip_mask, precip_visible)
        out = self.backbone(x, t)
        h, w = orig_hw
        out = out[:, :, :, :h, :w]
        return self._from_5d_precip(out)

    def model_predictions(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: Optional[torch.Tensor],
        precip_visible: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        model_out = self.forward(x_t, t, cond, precip_mask, precip_visible=precip_visible)
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

    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        precip_mask: Optional[torch.Tensor] = None,
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
        if precip_mask is not None:
            dense = self._dense_mask(precip_mask, x_start.shape[-2], x_start.shape[-1], x_start.device, x_start.dtype)
            dense = dense.repeat_interleave(self.level_variable_count, dim=1)
            loss = (loss_raw * dense).sum() / dense.sum().mul(self.level_variable_count).clamp_min(1.0)
        else:
            loss = loss_raw.mean()
        return {"loss": loss, "model_out": model_out, "target": target, "x_t": x_t, "t": t}

    def _ddim_timesteps(self, total_steps: int) -> list[int]:
        total_steps = max(1, min(int(total_steps), self.num_timesteps))
        if total_steps == self.num_timesteps:
            return list(reversed(range(self.num_timesteps)))
        grid = torch.linspace(0, self.num_timesteps - 1, steps=total_steps, device=self.device)
        times = torch.round(grid).long().tolist()
        deduped = []
        for time in times:
            if int(time) not in deduped:
                deduped.append(int(time))
        return list(reversed(deduped))

    def _compose_visible(self, img: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, visible: Optional[torch.Tensor]) -> torch.Tensor:
        if visible is None:
            return img
        dense = self._dense_mask(mask, img.shape[-2], img.shape[-1], img.device, img.dtype)
        dense = dense.repeat_interleave(self.level_variable_count, dim=1)
        known = self.q_sample(visible, t, torch.randn_like(visible))
        return known * (1.0 - dense) + img * dense

    @torch.no_grad()
    def sample_precip(
        self,
        cond: Dict[str, torch.Tensor],
        precip_mask: torch.Tensor,
        precip_visible: Optional[torch.Tensor] = None,
        sampling_timesteps: Optional[int] = None,
        eta: Optional[float] = None,
        sampler: str = "ddim",
        return_all_timesteps: bool = False,
        inpaint_mode: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if sampler not in {"ddim"}:
            raise ValueError("WoFSFCDMXL currently supports sampler='ddim'")
        if inpaint_mode is None:
            inpaint_mode = self.default_inpaint_mode
        first = next(iter(cond.values()))
        b, _, h, w = first.shape
        img = torch.randn(b, self.channels, h, w, device=first.device, dtype=first.dtype)
        eta = self.ddim_sampling_eta if eta is None else float(eta)
        times = self._ddim_timesteps(sampling_timesteps or self.sampling_timesteps)
        imgs = [img] if return_all_timesteps else None
        for idx, time in enumerate(times):
            time_next = times[idx + 1] if idx + 1 < len(times) else -1
            t = torch.full((b,), time, device=first.device, dtype=torch.long)
            if inpaint_mode in {"compose_visible", "visible_noise", "inpaint", "noisy_visible"}:
                img = self._compose_visible(img, t, precip_mask, precip_visible)
            pred_noise, x_start = self.model_predictions(img, t, cond, precip_mask, precip_visible=precip_visible)
            if time_next < 0:
                img = x_start
            else:
                alpha_t = self.alphas_cumprod[time].to(device=img.device, dtype=img.dtype)
                alpha_next = self.alphas_cumprod[time_next].to(device=img.device, dtype=img.dtype)
                sigma_sq = (eta**2) * ((1.0 - alpha_next) / (1.0 - alpha_t)) * (1.0 - alpha_t / alpha_next)
                sigma = torch.sqrt(sigma_sq.clamp_min(0.0))
                c = torch.sqrt((1.0 - alpha_next - sigma_sq).clamp_min(0.0))
                noise = torch.randn_like(img) if eta > 0.0 else torch.zeros_like(img)
                img = alpha_next.sqrt() * x_start + c * pred_noise + sigma * noise
            if imgs is not None:
                imgs.append(img)
        return torch.stack(imgs, dim=1) if imgs is not None else img
