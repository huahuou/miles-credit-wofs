"""
WoFS Multi-Modal Masked Autoencoder (WoFSMultiModalMAE)
-------------------------------------------------------
Single end-to-end training stage — no VAE pre-training required.

The model ingests five modalities from WoFS NWP output:
    background   : (B, 102, H, W)  — T, QVAPOR, U, V, W, GEOPOT (6 vars × 17 levels)
    precip       : (B, 136, H, W)  — 8 hydrometeors × 17 levels
    reflectivity : (B, 17, H, W)   — REFL_10CM × 17 levels
    surface      : (B, 2, H, W)    — XLAND, HGT  (no level dim)
    forcing      : (B, 10, H, W)   — cos/sin lat/lon, day, time, solar, insolation

Three modalities (background, precip, reflectivity) participate in random Dirichlet
token masking. Surface and forcing are always visible.

At DA inference time:
    - supply background tokens (unmasked)
    - replace reflectivity with observed REFL (blend_alpha controls how much)
    - mask all precip tokens
    - decoder reconstructs precip → Q_analysis
"""

import logging
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.distributions.dirichlet import Dirichlet
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from credit.models.wofs_mae_adapters import (
    WoFSInputAdapter,
    WoFSOutputAdapter,
    Block,
    trunc_normal_,
)
from credit.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# Default modality channel counts
DEFAULT_MODALITY_CHANNELS: Dict[str, int] = {
    "background": 102,    # 6 vars × 17 levels
    "precip": 136,        # 8 vars × 17 levels
    "reflectivity": 17,   # 1 var × 17 levels
    "surface": 2,         # XLAND, HGT (no levels)
    "forcing": 10,        # dynamic forcing channels
}

MASKABLE_MODALITIES = ["background", "precip", "reflectivity"]
NON_MASKED_MODALITIES = ["surface", "forcing"]


class WoFSMultiModalMAE(BaseModel):
    """
    Multi-modal MAE for WoFS data assimilation.

    Args:
        modality_channels       : dict mapping modality name → channel count
        non_masked_modalities   : list of always-visible modality keys
        patch_size              : spatial patch size for physics modalities (default 8)
        surface_forcing_stride  : pooling stride for surface/forcing (default 16 → 19×19 tokens)
        image_size              : padded input size (H, W) — default (304, 304)
        embed_dim               : shared backbone token dimension
        depth                   : ViT encoder depth
        num_heads               : ViT encoder heads
        mlp_ratio               : MLP expansion ratio
        decoder_dim             : output adapter internal dimension
        decoder_depth           : output adapter self-attention depth
        decoder_num_heads       : output adapter heads
        num_global_tokens       : number of learnable global CLS-like tokens
        num_encoded_tokens      : number of maskable tokens kept per forward pass
        drop_path_rate          : stochastic depth rate for encoder
    """

    def __init__(
        self,
        modality_channels: Optional[Dict[str, int]] = None,
        non_masked_modalities: Optional[List[str]] = None,
        patch_size: int = 8,
        surface_forcing_stride: int = 16,
        image_size: Tuple[int, int] = (304, 304),
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        decoder_dim: int = 384,
        decoder_depth: int = 6,
        decoder_num_heads: int = 8,
        num_global_tokens: int = 2,
        num_encoded_tokens: int = 512,
        drop_path_rate: float = 0.1,
        use_gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.modality_channels = modality_channels or dict(DEFAULT_MODALITY_CHANNELS)
        self.non_masked_modalities = non_masked_modalities or list(NON_MASKED_MODALITIES)
        self.maskable_modalities = [m for m in self.modality_channels if m not in self.non_masked_modalities]
        self.patch_size = patch_size
        self.surface_forcing_stride = surface_forcing_stride
        self.image_size = tuple(image_size)
        self.embed_dim = embed_dim
        self.num_global_tokens = num_global_tokens
        self.num_encoded_tokens = num_encoded_tokens
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # ------------------------------------------------------------------ #
        # Input adapters
        # ------------------------------------------------------------------ #
        # Physics modalities: 8×8 patch, padded image_size
        # Surface/forcing: coarser stride (16×16 effective patch) so tokens are fewer
        sf_h = math.ceil(image_size[0] / surface_forcing_stride) * surface_forcing_stride
        sf_w = math.ceil(image_size[1] / surface_forcing_stride) * surface_forcing_stride
        sf_image_size = (sf_h, sf_w)
        # The adapter for surface/forcing uses patch_size=surface_forcing_stride
        # so each token covers surface_forcing_stride × surface_forcing_stride pixels.

        self.input_adapters = nn.ModuleDict()
        for mod, c in self.modality_channels.items():
            if mod in self.non_masked_modalities:
                self.input_adapters[mod] = WoFSInputAdapter(
                    num_channels=c,
                    patch_size=surface_forcing_stride,
                    embed_dim=embed_dim,
                    image_size=sf_image_size,
                )
            else:
                self.input_adapters[mod] = WoFSInputAdapter(
                    num_channels=c,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    image_size=image_size,
                )

        # ------------------------------------------------------------------ #
        # Global tokens
        # ------------------------------------------------------------------ #
        if num_global_tokens > 0:
            self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, embed_dim))
            trunc_normal_(self.global_tokens, std=0.02)
        else:
            self.global_tokens = None

        # ------------------------------------------------------------------ #
        # Backbone ViT encoder
        # ------------------------------------------------------------------ #
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ------------------------------------------------------------------ #
        # Output adapters (one per maskable modality)
        # ------------------------------------------------------------------ #
        all_tasks = list(self.modality_channels.keys())
        self.output_adapters = nn.ModuleDict()
        for mod in self.maskable_modalities:
            self.output_adapters[mod] = WoFSOutputAdapter(
                num_channels=self.modality_channels[mod],
                patch_size=patch_size,
                embed_dim=embed_dim,
                decoder_dim=decoder_dim,
                decoder_depth=decoder_depth,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                image_size=image_size,
                context_tasks=all_tasks,
                task=mod,
                use_task_queries=True,
            )

        self._init_weights()

        logger.info(
            "WoFSMultiModalMAE: modalities=%s, maskable=%s, embed_dim=%d, depth=%d, "
            "num_encoded_tokens=%d",
            list(self.modality_channels.keys()),
            self.maskable_modalities,
            embed_dim,
            depth,
            num_encoded_tokens,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # Token count helpers
    # ------------------------------------------------------------------ #
    def _num_tokens(self, mod: str) -> int:
        return self.input_adapters[mod].num_patches

    def _maskable_token_counts(self) -> List[int]:
        return [self._num_tokens(m) for m in self.maskable_modalities]

    def _total_maskable_tokens(self) -> int:
        return sum(self._maskable_token_counts())

    # ------------------------------------------------------------------ #
    # Dirichlet masking
    # ------------------------------------------------------------------ #
    def _sample_token_mask(
        self,
        B: int,
        device: torch.device,
        num_encoded_tokens: int,
        alphas: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample which maskable tokens are kept via Dirichlet allocation.

        Returns:
            ids_keep    : (B, num_encoded_tokens)  — indices of kept tokens (in maskable pool)
            ids_restore : (B, total_maskable)      — inverse permutation
            task_masks  : dict[mod] → (B, N_mod)  — 0=keep, 1=masked
        """
        n_mods = len(self.maskable_modalities)
        n_per_mod = self._maskable_token_counts()
        total = sum(n_per_mod)

        # Dirichlet allocation
        alpha_vec = torch.full((n_mods,), alphas, dtype=torch.float32)
        samples_per_mod = Dirichlet(alpha_vec).sample((B,)).to(device)  # (B, n_mods)
        samples_per_mod = (samples_per_mod * num_encoded_tokens).round().long()
        # Fix rounding
        diff = num_encoded_tokens - samples_per_mod.sum(dim=1, keepdim=True)
        samples_per_mod[:, 0] = samples_per_mod[:, 0] + diff.squeeze(1)
        # Clamp to [0, n_per_mod]
        for i, n in enumerate(n_per_mod):
            samples_per_mod[:, i] = samples_per_mod[:, i].clamp(0, n)

        # Build per-modality masks and shuffle indices
        task_masks: Dict[str, torch.Tensor] = {}
        all_keep_local = []  # list of (B, keep_i) local indices per modality

        offset = 0
        for i, mod in enumerate(self.maskable_modalities):
            n = n_per_mod[i]
            keep_i = samples_per_mod[:, i]  # (B,)

            noise = torch.rand(B, n, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)  # random permutation

            # mask: 0=keep, 1=remove
            mask = torch.arange(n, device=device).unsqueeze(0).expand(B, -1)
            mask = (mask >= keep_i.unsqueeze(1)).long()
            # unshuffle mask to original token order
            ids_restore_local = torch.argsort(ids_shuffle, dim=1)
            mask = torch.gather(mask, 1, ids_restore_local)
            task_masks[mod] = mask

            # global keep indices (in maskable pool)
            ids_keep_local = ids_shuffle[:, :keep_i.max().item()]
            # Trim each row to actual keep_i
            for b in range(B):
                pass  # We use the shuffled indices and mask together below
            # Store (offset, ids_shuffle) for building global ids_keep later
            all_keep_local.append((offset, ids_shuffle, keep_i, n))
            offset += n

        # Build global ids_keep and ids_restore over full maskable pool
        global_keep_parts = []
        for (off, ids_sh, keep_i, n) in all_keep_local:
            global_ids = ids_sh + off  # (B, n) global indices
            # We need to select keep_i[b] tokens per sample — take max and zero-pad
            max_keep = keep_i.max().item()
            global_ids_keep = global_ids[:, :max_keep]  # (B, max_keep) – may include extras
            global_keep_parts.append(global_ids_keep)

        # Concatenate and sort so ids_keep is in ascending order per sample
        ids_keep_raw = torch.cat(global_keep_parts, dim=1)  # (B, sum_max_keep)
        ids_keep = torch.sort(ids_keep_raw, dim=1)[0][:, :num_encoded_tokens]

        # ids_restore: inverse permutation over total maskable tokens
        # Build a combined mask over all maskable tokens
        combined_mask = torch.cat([task_masks[m] for m in self.maskable_modalities], dim=1)  # (B, total)
        ids_shuffle_combined = torch.argsort(combined_mask.float() + torch.rand_like(combined_mask.float()) * 0.001, dim=1)
        ids_restore = torch.argsort(ids_shuffle_combined, dim=1)
        ids_keep = ids_shuffle_combined[:, :num_encoded_tokens]

        return ids_keep, ids_restore, task_masks

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        modality_dict: Dict[str, torch.Tensor],
        mask_inputs: bool = True,
        num_encoded_tokens: Optional[int] = None,
        alphas: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            modality_dict       : {mod_key: (B, C, H, W)} — all modalities
            mask_inputs         : if True apply Dirichlet masking (training);
                                  if False keep all tokens (validation full-field)
            num_encoded_tokens  : override default if needed
            alphas              : Dirichlet concentration parameter

        Returns:
            recon_dict  : {mod_key: (B, C, H, W)} — reconstructed maskable modalities
            task_masks  : {mod_key: (B, N_tok)} — 0=observed, 1=masked
        """
        num_encoded_tokens = num_encoded_tokens or self.num_encoded_tokens
        B = next(iter(modality_dict.values())).shape[0]
        device = next(iter(modality_dict.values())).device

        # Store original H, W for cropping after decode
        first_phys = modality_dict.get("background", modality_dict.get("precip",
                      modality_dict.get("reflectivity")))
        orig_h, orig_w = first_phys.shape[-2], first_phys.shape[-1]

        # 1. Tokenize all modalities
        tokens_per_mod: Dict[str, torch.Tensor] = {}
        for mod, adapter in self.input_adapters.items():
            if mod in modality_dict:
                tokens_per_mod[mod] = adapter(modality_dict[mod])

        # 2. Gather maskable tokens pool: (B, total_maskable, embed_dim)
        maskable_tokens = torch.cat(
            [tokens_per_mod[m] for m in self.maskable_modalities], dim=1
        )
        total_maskable = maskable_tokens.shape[1]

        # 3. Sample mask or keep all
        if mask_inputs:
            ids_keep, ids_restore, task_masks = self._sample_token_mask(
                B, device, num_encoded_tokens, alphas
            )
        else:
            # No masking: keep all tokens
            ids_keep = torch.arange(total_maskable, device=device).unsqueeze(0).expand(B, -1)
            ids_restore = torch.arange(total_maskable, device=device).unsqueeze(0).expand(B, -1)
            task_masks = {m: torch.zeros(B, self._num_tokens(m), device=device, dtype=torch.long)
                         for m in self.maskable_modalities}

        # 4. Apply mask: gather kept tokens
        visible_maskable = torch.gather(
            maskable_tokens, 1,
            ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )  # (B, num_encoded_tokens, embed_dim)

        # 5. Gather always-visible tokens (surface, forcing, global)
        always_visible = []
        for mod in self.non_masked_modalities:
            if mod in tokens_per_mod:
                always_visible.append(tokens_per_mod[mod])
        if self.global_tokens is not None:
            always_visible.append(self.global_tokens.expand(B, -1, -1))

        if always_visible:
            full_visible = torch.cat([visible_maskable] + always_visible, dim=1)
        else:
            full_visible = visible_maskable

        # 6. Encoder (with optional gradient checkpointing to save activation memory)
        enc_out = full_visible
        for blk in self.encoder_blocks:
            if self.use_gradient_checkpointing and enc_out.requires_grad:
                enc_out = grad_checkpoint(blk, enc_out, use_reentrant=False)
            else:
                enc_out = blk(enc_out)
        enc_out = self.encoder_norm(enc_out)

        # Build input_info for output adapters
        input_info: Dict = {"tasks": OrderedDict(), "orig_h": orig_h, "orig_w": orig_w}
        cursor = 0
        for mod in list(tokens_per_mod.keys()):
            n = tokens_per_mod[mod].shape[1]
            input_info["tasks"][mod] = {
                "num_tokens": n,
                "start_idx": cursor,
                "end_idx": cursor + n,
            }
            cursor += n
        input_info["num_global_tokens"] = self.num_global_tokens

        # 7. Decode each maskable modality
        recon_dict: Dict[str, torch.Tensor] = {}
        maskable_token_counts = self._maskable_token_counts()
        for i, mod in enumerate(self.maskable_modalities):
            task_start = sum(maskable_token_counts[:i])
            recon_dict[mod] = self.output_adapters[mod](
                encoder_tokens=enc_out,
                ids_keep=ids_keep,
                ids_restore=ids_restore,
                num_task_tokens=total_maskable,
                task_start_idx=task_start,
                input_info=input_info,
            )

        return recon_dict, task_masks

    # ------------------------------------------------------------------ #
    # DA inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def assimilate(
        self,
        background: torch.Tensor,
        obs_refl: torch.Tensor,
        surface: torch.Tensor,
        forcing: torch.Tensor,
        blend_alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Data assimilation inference: reconstruct precip given observed reflectivity.

        Args:
            background  : (B, 102, H, W) — normalized background atmospheric state
            obs_refl    : (B, 17, H, W)  — normalized observed REFL_10CM
            surface     : (B, 2, H, W)   — normalized surface fields
            forcing     : (B, 10, H, W)  — normalized dynamic forcing
            blend_alpha : 1.0 = pure obs reflectivity, 0.0 = pure model REFL

        Returns:
            precip_analysis : (B, 136, H, W) — reconstructed precip in normalized space
        """
        # Build modality dict with blended reflectivity
        # background must contain REFL_10CM; we blend it with obs
        # For simplicity: pass obs_refl as the reflectivity modality directly
        modality_dict = {
            "background": background,
            "precip": torch.zeros(
                background.shape[0], self.modality_channels["precip"],
                background.shape[-2], background.shape[-1],
                device=background.device, dtype=background.dtype
            ),
            "reflectivity": obs_refl,
            "surface": surface,
            "forcing": forcing,
        }

        if blend_alpha < 1.0 and "reflectivity" in modality_dict:
            # Blend: replace some of obs with model background REFL
            # The caller should pass model_refl separately if blend_alpha < 1.0;
            # for now we just scale obs towards zero (background REFL ≈ 0 in normalized space)
            modality_dict["reflectivity"] = blend_alpha * obs_refl

        # Run with all tokens visible except precip (fully masked)
        # We achieve this by setting mask_inputs=False (keep all) and zeroing precip
        recon_dict, _ = self.forward(modality_dict, mask_inputs=False)
        return recon_dict["precip"]
