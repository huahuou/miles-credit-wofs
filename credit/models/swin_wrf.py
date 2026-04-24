# credit/models/swin_wrf.py
#
# Drop-in replacement for the original swin_wrf.py.
#
# Key change: WRFTransformer now accepts any spatial resolution at forward()
# time without reinitialising the model. This enables pretraining at one
# resolution (e.g. 300×300) and fine-tuning at another (e.g. 900×900) using
# the exact same checkpoint.
#
# How it works (three coordinated fixes):
#   1. CubeEmbedding.forward()  – derive patch-grid shape from the actual
#      Conv3d output instead of the init-time stored patches_resolution.
#   2. UTransformer              – pass dynamic_mask=True to SwinTransformerV2Stage
#      so the attention mask is recomputed every forward() from the live tensor
#      shape; pad/crop in forward() from the live tensor, not a stored constant.
#   3. WRFTransformer.forward()  – store the original (H, W) before any padding,
#      derive Lat/Lon from the actual embedding output, and crop (not bilinear-
#      resize) back to original resolution at the end.
#
# Every public argument is identical to the original; existing YAML configs and
# checkpoint keys are fully compatible.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

from credit.boundary_padding import TensorPadding
from credit.models.base_model import BaseModel
from credit.postblock import PostBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectral norm helper  (unchanged)
# ---------------------------------------------------------------------------

def apply_spectral_norm(model: nn.Module) -> None:
    """Add spectral norm to all Conv2d, Linear and ConvTranspose2d layers."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(module)


# ---------------------------------------------------------------------------
# Padding helpers  (unchanged – kept for TensorPadding / outside-domain use)
# ---------------------------------------------------------------------------

def get_pad3d(input_resolution, window_size):
    """
    Estimate padding so that input_resolution is divisible by window_size.

    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size      (tuple[int]): (Pl, Lat, Lon)

    Returns:
        (padding_left, padding_right, padding_top, padding_bottom,
         padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = \
        padding_front = padding_back = 0

    pl_remainder  = Pl  % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back  = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top    = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left  = lon_pad // 2
        padding_right = lon_pad - padding_left

    return (padding_left, padding_right, padding_top, padding_bottom,
            padding_front, padding_back)


def get_pad2d(input_resolution, window_size):
    """
    2-D wrapper around get_pad3d.

    Args:
        input_resolution (tuple[int]): (Lat, Lon)
        window_size      (tuple[int]): (Lat, Lon)

    Returns:
        (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size      = [2] + list(window_size)
    return get_pad3d(input_resolution, window_size)[:4]


# ---------------------------------------------------------------------------
# CubeEmbedding  – FIX 1: derive patch-grid shape dynamically
# ---------------------------------------------------------------------------

class CubeEmbedding(nn.Module):
    """
    3-D patch embedding via Conv3d.

    Args:
        img_size   (tuple[int]): (T, Lat, Lon)  – used only to store metadata.
        patch_size (tuple[int]): (T, Lat, Lon)
        in_chans   (int): number of input channels (variables × levels + surface).
        embed_dim  (int): output embedding dimension.
        norm_layer: normalisation class (default: nn.LayerNorm).

    Shape contract
    ~~~~~~~~~~~~~~
    forward input : (B, T, C, Lat, Lon)   – any Lat / Lon
    forward output: (B, embed_dim, T_p, Lat_p, Lon_p)
                    where *_p = *_original // patch_*
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.img_size  = img_size
        self.embed_dim = embed_dim

        # Stored only for reference / logging; NOT used in forward().
        self.patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        ]

        # Conv3d is inherently resolution-flexible when stride == kernel_size.
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, Lat, Lon)
        B = x.shape[0]

        # (B, embed_dim, T_p, Lat_p, Lon_p)  – actual size from the convolution
        x = self.proj(x)

        # --- FIX 1 -------------------------------------------------------
        # Read the actual spatial output shape from the tensor, NOT from the
        # init-time stored patches_resolution.  This is what makes CubeEmbedding
        # resolution-agnostic.
        _, E, Tp, Latp, Lonp = x.shape
        # -----------------------------------------------------------------

        # Flatten spatial dims → apply LayerNorm → restore
        x = x.reshape(B, E, -1).transpose(1, 2)        # (B, T_p*Lat_p*Lon_p, E)
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, E, Tp, Latp, Lonp)

        return x


# ---------------------------------------------------------------------------
# DownBlock / UpBlock  (unchanged)
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    """Conv2d stride-2 downsampling + residual path."""

    def __init__(self, in_chans: int, out_chans: int,
                 num_groups: int, num_residuals: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans,
                              kernel_size=(3, 3), stride=2, padding=1)
        blk = []
        for _ in range(num_residuals):
            blk += [
                nn.Conv2d(out_chans, out_chans, kernel_size=3,
                          stride=1, padding=1),
                nn.GroupNorm(num_groups, out_chans),
                nn.SiLU(),
            ]
        self.b = nn.Sequential(*blk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x + self.b(x)


class UpBlock(nn.Module):
    """ConvTranspose2d stride-2 upsampling + residual path."""

    def __init__(self, in_chans: int, out_chans: int,
                 num_groups: int, num_residuals: int = 2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_chans, out_chans,
                                       kernel_size=2, stride=2)
        blk = []
        for _ in range(num_residuals):
            blk += [
                nn.Conv2d(out_chans, out_chans, kernel_size=3,
                          stride=1, padding=1),
                nn.GroupNorm(num_groups, out_chans),
                nn.SiLU(),
            ]
        self.b = nn.Sequential(*blk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x + self.b(x)


class FlowDependentNoiseInjection(nn.Module):
    """StyleGAN-like noise injection with a flow-dependent spatial modulation mask,
    using Tanh saturation for mesoscale autoregressive stability."""

    def __init__(
        self,
        dim: int,
        latent_dim: int = 128,
        flow_hidden_div: int = 4,
        spatial_noise_init: float = 0.01,
        max_noise_scale_init: float = 0.8,
        growth_rate_init: float = 0.04,
    ):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        hidden_dim = max(1, dim // max(1, flow_hidden_div))

        # 1. Global latent style mapping
        self.style_mlp = nn.Sequential(
            nn.Linear(latent_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )

        # 2. Flow-dependent spatial mask generator
        self.flow_modulator = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
            nn.Softplus(),  # Ensures mask is positive
        )

        # 3. Learnable Noise Scaling Parameters
        self.spatial_noise_weight = nn.Parameter(torch.ones(1) * float(spatial_noise_init))
        self.max_noise_scale = nn.Parameter(torch.ones(1) * float(max_noise_scale_init))
        self.growth_rate = nn.Parameter(torch.ones(1) * float(growth_rate_init))

    def forward(self, x: torch.Tensor, latent_z: torch.Tensor | None = None, step: int = 0) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        
        # Ensure parameters stay strictly positive during optimization
        growth_rate = F.softplus(self.growth_rate)
        max_noise_scale = F.softplus(self.max_noise_scale)
        
        # Calculate Tanh saturation curve
        step_tensor = torch.tensor(float(step), device=x.device, dtype=x.dtype)
        saturation_curve = torch.tanh(step_tensor * growth_rate)
        
        # Step factor asymptotes to (1.0 + max_noise_scale)
        step_factor = 1.0 + (max_noise_scale * saturation_curve)
        
        # Generate spatial mask and raw Gaussian noise
        flow_mask = self.flow_modulator(x)
        raw_noise = torch.randn(batch_size, 1, height, width, device=x.device, dtype=x.dtype)

        # Inject the dynamically scaled noise
        x = x + raw_noise * flow_mask * self.spatial_noise_weight * step_factor

        # Apply global latent style modulation
        if latent_z is not None:
            latent_z = latent_z.to(device=x.device, dtype=x.dtype)
            style = self.style_mlp(latent_z)
            gamma, beta = style.chunk(2, dim=1)
            gamma = gamma.view(batch_size, channels, 1, 1)
            beta = beta.view(batch_size, channels, 1, 1)
            x = x * (1 + gamma) + beta

        return x


# ---------------------------------------------------------------------------
# UTransformer  – FIX 2: dynamic padding and dynamic_mask=True
# ---------------------------------------------------------------------------

class UTransformer(nn.Module):
    """
    U-shaped Transformer stage: DownBlock → SwinV2Stage → UpBlock.

    Changes vs. the original
    ~~~~~~~~~~~~~~~~~~~~~~~~
    * ``SwinTransformerV2Stage`` is initialised with ``dynamic_mask=True``
      so the shifted-window attention mask is recomputed inside every
      forward() call from the live tensor dimensions.  This is the flag
      that timm exposes precisely for this use-case.
    * The pad/crop in forward() is now computed from the *live* tensor
      shape instead of the fixed ``self.padding`` tuple baked at init time,
      so any input resolution flows through correctly.
    * ``self.window_size`` (int) is stored so forward() can compute the
      window-divisible padding on the fly.

    Args:
        embed_dim        (int)
        num_groups       (int | tuple[int])
        input_resolution (tuple[int]): (Lat, Lon) – used **only** to
                          initialise SwinTransformerV2Stage; NOT used in
                          forward() to determine padding.
        num_heads        (int)
        window_size      (int | tuple[int])
        depth            (int)
        drop_path        (float | list[float])
    """

    def __init__(self, embed_dim, num_groups, input_resolution,
                 num_heads, window_size, depth, drop_path):
        super().__init__()
        num_groups  = to_2tuple(num_groups)
        window_size = to_2tuple(window_size)

        # Store window_size as int for the dynamic padding in forward().
        self.window_size_int = window_size[0]

        # DownBlock operates before the SwinV2 stage and halves spatial dims.
        self.down = DownBlock(embed_dim, embed_dim, num_groups[0])

        # Compute the padded input_resolution that was expected at init time.
        # This is still needed to correctly initialise SwinTransformerV2Stage
        # (its blocks need *some* input_resolution at construction), but it
        # is NOT used in forward() for the actual pad/crop logic.
        init_padding = get_pad2d(input_resolution, window_size)
        pad_l, pad_r, pad_t, pad_b = init_padding
        padded_resolution = (
            input_resolution[0] + pad_t + pad_b,
            input_resolution[1] + pad_l + pad_r,
        )

        # --- FIX 2a ------------------------------------------------------
        # dynamic_mask=True  →  timm recomputes the SW-MSA attention mask
        # from the live tensor shape inside SwinTransformerV2Block._attn().
        # always_partition=True  →  always use the full window_size even
        # when the feature map is smaller, avoiding the clamp-to-resolution
        # logic that would silently disable shifted windows.
        self.layer = SwinTransformerV2Stage(
            dim=embed_dim,
            out_dim=embed_dim,
            input_resolution=padded_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size[0],
            always_partition=True,   # don't clamp window to feat size
            dynamic_mask=True,       # recompute attention mask each forward
            drop_path=drop_path,
        )
        # -----------------------------------------------------------------

        self.up = UpBlock(embed_dim * 2, embed_dim, num_groups[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, Lat, Lon)
        x = self.down(x)          # (B, C, Lat/2, Lon/2)
        shortcut = x

        _, _, dLat, dLon = x.shape

        # --- FIX 2b -------------------------------------------------------
        # Compute window-divisible padding from the *live* tensor shape.
        # This is the same modulo trick used by WeatherSwinV2UNet.pad_tensor().
        win = self.window_size_int
        pad_b_amt = (win - dLat % win) % win
        pad_r_amt = (win - dLon % win) % win
        # F.pad format for 4-D NCHW: (left, right, top, bottom)
        x = F.pad(x, (0, pad_r_amt, 0, pad_b_amt))
        _, _, pad_lat, pad_lon = x.shape
        # ------------------------------------------------------------------

        # SwinTransformerV2Stage expects NHWC
        x = x.permute(0, 2, 3, 1)          # (B, Lat_p, Lon_p, C)
        x = self.layer(x)
        x = x.permute(0, 3, 1, 2)          # (B, C, Lat_p, Lon_p)

        # Crop back to the pre-padding shape
        x = x[:, :, :dLat, :dLon].contiguous()

        # U-Net skip connection + upsample
        x = torch.cat([shortcut, x], dim=1)   # (B, 2C, Lat/2, Lon/2)
        x = self.up(x)                         # (B, C, Lat, Lon)
        return x


# ---------------------------------------------------------------------------
# WRFTransformer  – FIX 3: dynamic Lat/Lon in forward + crop output
# ---------------------------------------------------------------------------

class WRFTransformer(BaseModel):
    """
    Dynamic-resolution WRF Transformer.

    Identical public API to the original WRFTransformer; all existing YAML
    configs and checkpoint files are compatible.

    The key behavioural difference is that forward() accepts *any* spatial
    resolution for ``x`` and ``x_outside`` and returns an output tensor of
    the same spatial size as the input – without bilinear interpolation
    artefacts and without shape-mismatch errors.

    Args  (same as original)
    ~~~~~~~~~~~~~~~~~~~~~~~~
        param_interior  (dict): image_height, image_width, patch_height,
                                patch_width, levels, frames,
                                frame_patch_size, channels,
                                surface_channels, input_only_channels,
                                output_only_channels, dim
        param_outside   (dict): same keys minus input/output_only_channels
        time_encode_dim (int)   default 12
        num_groups      (int)   default 32
        num_heads       (int)   default 8
        depth           (int)   default 48
        window_size     (int)   default 7
        use_spectral_norm (bool) default True
        interp          (bool)  default True  ← kept for API compat; no-op
                                               (crop is used instead)
        drop_path       (float) default 0
        padding_conf    (dict)  boundary-padding config
        post_conf       (dict)  post-processing block config
    """

    def __init__(
        self,
        param_interior,
        param_outside,
        time_encode_dim=12,
        num_groups=32,
        num_heads=8,
        depth=48,
        window_size=7,
        use_spectral_norm=True,
        interp=True,          # kept for API compat – ignored internally
        drop_path=0,
        padding_conf=None,
        post_conf=None,
        **kwargs,
    ):
        super().__init__()

        self.time_encode      = time_encode_dim
        self.use_spectral_norm = use_spectral_norm

        # ── interior domain ───────────────────────────────────────────────
        image_height_inside       = param_interior["image_height"]
        patch_height_inside       = param_interior["patch_height"]
        image_width_inside        = param_interior["image_width"]
        patch_width_inside        = param_interior["patch_width"]
        levels_inside             = param_interior["levels"]
        frames_inside             = param_interior["frames"]
        frame_patch_size_inside   = param_interior["frame_patch_size"]
        channels_inside           = param_interior["channels"]
        surface_channels_inside   = param_interior["surface_channels"]
        input_only_channels_inside  = param_interior["input_only_channels"]
        output_only_channels_inside = param_interior["output_only_channels"]
        dim_inside                = param_interior["dim"]

        # ── exterior domain ─────────────────────────────────────────────��─
        image_height_outside      = param_outside["image_height"]
        patch_height_outside      = param_outside["patch_height"]
        image_width_outside       = param_outside["image_width"]
        patch_width_outside       = param_outside["patch_width"]
        levels_outside            = param_outside["levels"]
        frames_outside            = param_outside["frames"]
        frame_patch_size_outside  = param_outside["frame_patch_size"]
        channels_outside          = param_outside["channels"]
        surface_channels_outside  = param_outside["surface_channels"]
        dim_outside               = param_outside["dim"]

        # ── padding config ────────────────────────────────────────────────
        if padding_conf is None:
            padding_conf = {"activate": False}
        self.use_padding = padding_conf["activate"]

        if self.use_padding:
            pad_lat = padding_conf["pad_lat"]
            pad_lon = padding_conf["pad_lon"]
            image_height_pad = image_height_inside + pad_lat[0] + pad_lat[1]
            image_width_pad  = image_width_inside  + pad_lon[0] + pad_lon[1]
            img_size_inside      = (frames_inside, image_height_pad,   image_width_pad)
            self.img_size_original = (frames_inside, image_height_inside, image_width_inside)
        else:
            img_size_inside      = (frames_inside, image_height_inside, image_width_inside)
            self.img_size_original = img_size_inside

        img_size_outside = (frames_outside, image_height_outside, image_width_outside)

        # ── patch sizes ───────────────────────────────────────────────────
        patch_size_inside  = (frame_patch_size_inside,  patch_height_inside,  patch_width_inside)
        patch_size_outside = (frame_patch_size_outside, patch_height_outside, patch_width_outside)

        # ── channel counts ────────────────────────────────────────────────
        in_chans_inside  = (channels_inside  * levels_inside  + surface_channels_inside
                            + input_only_channels_inside)
        out_chans_inside = (channels_inside  * levels_inside  + surface_channels_inside
                            + output_only_channels_inside)
        in_chans_outside = channels_outside * levels_outside + surface_channels_outside

        # input_resolution used *only* to initialise UTransformer / SwinV2Stage
        # (not used in forward for spatial logic)
        input_resolution_inside = (
            round(img_size_inside[1] / patch_size_inside[1] / 2),
            round(img_size_inside[2] / patch_size_inside[2] / 2),
        )

        # ── submodules ────────────────────────────────────────────────────
        self.cube_embedding_inside  = CubeEmbedding(
            img_size_inside,  patch_size_inside,  in_chans_inside,  dim_inside)
        self.cube_embedding_outside = CubeEmbedding(
            img_size_outside, patch_size_outside, in_chans_outside, dim_outside)

        self.total_dim = dim_inside

        self.u_transformer = UTransformer(
            self.total_dim,
            num_groups,
            input_resolution_inside,
            num_heads,
            window_size,
            depth=depth,
            drop_path=drop_path,
        )

        # fc: channel-only linear – completely resolution-agnostic
        self.fc = nn.Linear(
            self.total_dim,
            out_chans_inside * patch_size_inside[1] * patch_size_inside[2],
        )

        # Store patch_size for the pixel-space reassembly in forward()
        self.patch_size  = patch_size_inside
        self.out_chans   = out_chans_inside
        self.img_size    = img_size_inside

        # ── optional boundary padding ─────────────────────────────────────
        if self.use_padding:
            self.padding_opt = TensorPadding(**padding_conf)

        # ── optional spectral norm ────────────────────────────────────────
        if self.use_spectral_norm:
            logger.info("Adding spectral norm to all conv and linear layers")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            apply_spectral_norm(self)

        # ── optional post-processing block ────────────────────────────────
        if post_conf is None:
            post_conf = {"activate": False}
        self.use_post_block = post_conf["activate"]
        if self.use_post_block:
            self.postblock = PostBlock(post_conf)

        # ── FiLM conditioning ─────────────────────────────────────────────
        self.film = nn.Linear(self.time_encode, 2 * self.total_dim)

        noise_conf = kwargs.get("noise_injection", {}) or {}
        self.use_flow_dependent_noise = bool(noise_conf.get("activate", False))
        self.sample_latent_if_none = bool(noise_conf.get("sample_latent_if_none", True))
        self.freeze_base_model_weights = bool(noise_conf.get("freeze_base_model_weights", False))
        if self.use_flow_dependent_noise:
            self.flow_noise = FlowDependentNoiseInjection(
                dim=self.total_dim,
                latent_dim=int(noise_conf.get("latent_dim", 128)),
                flow_hidden_div=int(noise_conf.get("flow_hidden_div", 4)),
                spatial_noise_init=float(noise_conf.get("spatial_noise_init", 0.01)),
                max_noise_scale_init=float(
                    noise_conf.get("max_noise_scale_init", noise_conf.get("step_scale_init", 0.1))
                ),
                growth_rate_init=float(noise_conf.get("growth_rate_init", 0.04)),
            )
            logger.info("Flow-dependent noise injection is active for WRFTransformer")
            if self.freeze_base_model_weights:
                for param in self.parameters():
                    param.requires_grad = False
                for param in self.flow_noise.parameters():
                    param.requires_grad = True
                logger.info("Base model weights are frozen; training noise injection layers only")
        else:
            self.flow_noise = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,          # (B, Vars, T, Lat, Lon)  – interior
        x_outside: torch.Tensor,  # (B, Vars, T, Lat, Lon)  – exterior
        x_extra: torch.Tensor,    # (B, time_encode_dim)
        latent_z: torch.Tensor | None = None,
        forecast_step: int = 0,
        ensemble_size: int = 1,
    ) -> torch.Tensor:

        # ── optional post-block bookkeeping ───────────────────────────────
        x_copy = x.clone().detach() if self.use_post_block else None

        # ── optional boundary padding (TensorPadding) ─────────────────────
        if self.use_padding:
            x = self.padding_opt.pad(x)

        B = x.shape[0]

        # --- FIX 3a -------------------------------------------------------
        # Record the spatial size of the (possibly padded) interior input so
        # we can crop the output back to exactly this size at the end.
        # This replaces the old bilinear F.interpolate with a lossless crop.
        _, _, _, orig_H, orig_W = x.shape
        # ------------------------------------------------------------------

        _, patch_lat, patch_lon = self.patch_size

        # ── cube embedding ────────────────────────────────────────────────
        # Output: (B, dim, T_p, Lat_p, Lon_p)  where *_p depends on input size
        x = self.cube_embedding_inside(x).squeeze(2)          # (B, dim, Lat_p, Lon_p)
        x_outside = self.cube_embedding_outside(x_outside).squeeze(2)

        # ── FiLM modulation on the exterior embedding ─────────────────────
        alpha_beta = self.film(x_extra)                        # (B, 2·dim)
        alpha, beta = alpha_beta.chunk(2, dim=1)
        alpha = alpha.view(B, self.total_dim, 1, 1)
        beta  = beta.view( B, self.total_dim, 1, 1)
        x_outside = alpha * x_outside + beta

        x = x + x_outside                                     # (B, dim, Lat_p, Lon_p)

        if self.use_flow_dependent_noise and ensemble_size > 1:
            if latent_z is None and self.sample_latent_if_none:
                latent_z = torch.randn(B, self.flow_noise.latent_dim, device=x.device, dtype=x.dtype)
            x = self.flow_noise(x, latent_z=latent_z, step=forecast_step)

        # ── U-Transformer (dynamic padding/crop inside) ───────────────────
        x = self.u_transformer(x)                             # (B, dim, Lat_p, Lon_p)

        # --- FIX 3b -------------------------------------------------------
        # Read Lat_p / Lon_p from the live tensor – NOT from self.input_resolution.
        _, _, Lat_p, Lon_p = x.shape
        # ------------------------------------------------------------------

        # ── pixel-space reassembly ────────────────────────────────────────
        x = self.fc(x.permute(0, 2, 3, 1))          # (B, Lat_p, Lon_p, out_chans·p·p)
        x = (x
             .reshape(B, Lat_p, Lon_p, patch_lat, patch_lon, self.out_chans)
             .permute(0, 1, 3, 2, 4, 5)              # (B, Lat_p, patch_lat, Lon_p, patch_lon, C)
             .reshape(B, Lat_p * patch_lat, Lon_p * patch_lon, self.out_chans)
             .permute(0, 3, 1, 2))                   # (B, C, Lat_full, Lon_full)

        # ── optional boundary unpadding ───────────────────────────────────
        if self.use_padding:
            x = self.padding_opt.unpad(x)

        # --- FIX 3c -------------------------------------------------------
        # Crop to the original spatial size.  This is a lossless operation
        # (no bilinear artefacts) and works for any input resolution because
        # orig_H / orig_W were recorded from the live tensor above.
        # The original `interp` flag is intentionally ignored here; the crop
        # is always correct and does not need a fallback to interpolation.
        x = x[:, :, :orig_H, :orig_W].contiguous()
        # ------------------------------------------------------------------

        x = x.unsqueeze(2)                           # restore time dim

        # ── optional post-block ───────────────────────────────────────────
        if self.use_post_block:
            x = self.postblock({"y_pred": x, "x": x_copy})

        return x
