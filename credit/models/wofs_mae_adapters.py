"""
WoFS Multi-Modal MAE — Input and Output Adapters
-------------------------------------------------
Two building blocks used by WoFSMultiModalMAE:

    WoFSInputAdapter   : (B, C, H, W) → (B, N_tok, embed_dim)
                          Direct patch-linear projection (no VAE, no KL).
                          Identical to ViT patch embedding, extended with a
                          per-modality learnable modality embedding.

    WoFSOutputAdapter  : (B, N_vis, embed_dim) → (B, C, H, W)
                          Cross-attention decoder: spatial query tokens attend
                          to all encoder output tokens, then an MLP project back
                          to pixel space via patch reshape.

No VAE stage is needed because WoFS fields are on a complete NaN-free NWP grid
and are normalized to N(0,1) before any model call.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position embedding helpers
# ---------------------------------------------------------------------------

def build_2d_sincos_posemb(h: int, w: int, embed_dim: int = 768,
                            temperature: float = 10000.0) -> torch.Tensor:
    """Return fixed 2-D sin-cos positional embeddings shaped (1, embed_dim, h, w)."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos pos emb"
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature ** omega)
    out_w = torch.einsum("m,d->md", grid_w.flatten(), omega)
    out_h = torch.einsum("m,d->md", grid_h.flatten(), omega)
    pos_emb = torch.cat(
        [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
    )[None, :, :]  # (1, h*w, embed_dim)
    pos_emb = rearrange(pos_emb, "b (h w) d -> b d h w", h=h, w=w)
    return pos_emb


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        return tensor.normal_(0.0, std).clamp_(-2 * std, 2 * std)


# ---------------------------------------------------------------------------
# Transformer block (standard pre-norm)
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, Nq, C = query.shape
        Nk = context.shape[1]
        q = self.q(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    """Standard pre-norm ViT block (self-attention)."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        # stochastic depth via drop-path (simple implementation)
        self.drop_path_prob = drop_path

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_prob == 0.0:
            return x
        keep = 1.0 - self.drop_path_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep)
        return x * random_tensor / keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._drop_path(self.attn(self.norm1(x)))
        x = x + self._drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# WoFSInputAdapter
# ---------------------------------------------------------------------------

class WoFSInputAdapter(nn.Module):
    """
    Per-modality patch-linear input adapter.

    Converts (B, C, H, W) → (B, N_tok, embed_dim) via:
        1. Pad H,W to nearest multiple of patch_size
        2. Conv2d(C, embed_dim, kernel=patch_size, stride=patch_size)
        3. Reshape to (B, N_H * N_W, embed_dim)
        4. Add fixed 2D sin-cos positional embedding
        5. Add learnable modality embedding (broadcast over spatial tokens)
        6. LayerNorm

    Args:
        num_channels : int   — number of input channels C (levels folded in)
        patch_size   : int   — spatial patch size P
        embed_dim    : int   — token dimension d
        image_size   : tuple — (H, W) expected *after* padding; determines pos emb grid
    """

    def __init__(
        self,
        num_channels: int,
        patch_size: int = 8,
        embed_dim: int = 768,
        image_size: Tuple[int, int] = (304, 304),
    ):
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.image_size = image_size  # (H_pad, W_pad)

        N_H = image_size[0] // patch_size
        N_W = image_size[1] // patch_size
        self.num_patches = N_H * N_W

        # Modality-specific patch projection
        self.proj = nn.Conv2d(
            num_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Fixed 2D sin-cos positional embedding (1, embed_dim, N_H, N_W)
        pos_emb = build_2d_sincos_posemb(N_H, N_W, embed_dim)
        self.register_buffer("pos_emb", pos_emb)

        # Learnable modality embedding (broadcast over all tokens)
        self.modality_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.modality_emb, std=0.02)

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # ViT-style: initialize proj like a linear layer
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.reshape(w.shape[0], -1))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    @property
    def device(self):
        return self.proj.weight.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)  — pre-normalized field

        Returns:
            tokens : (B, N_tok, embed_dim)
        """
        B, C, H, W = x.shape
        # Pad to image_size if needed
        H_pad, W_pad = self.image_size
        if H != H_pad or W != W_pad:
            pad_h = H_pad - H
            pad_w = W_pad - W
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # (B, embed_dim, N_H, N_W)
        tokens = self.proj(x)

        # Add positional embedding
        pos = F.interpolate(
            self.pos_emb,
            size=tokens.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).to(x.dtype)
        tokens = tokens + pos

        # Reshape to sequence
        tokens = rearrange(tokens, "b d nh nw -> b (nh nw) d")

        # Add modality embedding and normalize
        tokens = tokens + self.modality_emb
        tokens = self.norm(tokens)
        return tokens


# ---------------------------------------------------------------------------
# WoFSOutputAdapter
# ---------------------------------------------------------------------------

class WoFSOutputAdapter(nn.Module):
    """
    Per-modality cross-attention output adapter.

    Decodes (B, N_vis, embed_dim) encoder output → (B, C, H, W) reconstruction:
        1. Project encoder tokens: embed_dim → decoder_dim
        2. Build spatial query tokens (positional + modality + task embeddings)
           Fill unobserved query positions with mask_token
        3. One cross-attention block: queries attend to all encoder tokens
        4. depth self-attention blocks on query side
        5. Linear out_proj: decoder_dim → C * P_H * P_W
        6. Rearrange patches → (B, C, H_pad, W_pad); crop to original H, W

    Args:
        num_channels    : int          — output channels C
        patch_size      : int          — spatial patch size P
        embed_dim       : int          — encoder token dimension
        decoder_dim     : int          — decoder internal dimension
        decoder_depth   : int          — extra self-attention blocks after cross-attn
        num_heads       : int          — decoder attention heads
        image_size      : tuple        — (H_pad, W_pad) after padding
        context_tasks   : list[str]    — all modality keys for per-task context embeddings
        task            : str | None   — which task this adapter reconstructs
    """

    def __init__(
        self,
        num_channels: int,
        patch_size: int = 8,
        embed_dim: int = 768,
        decoder_dim: int = 384,
        decoder_depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        image_size: Tuple[int, int] = (304, 304),
        context_tasks: Optional[List[str]] = None,
        task: Optional[str] = None,
        use_task_queries: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.image_size = image_size
        self.task = task
        self.use_task_queries = use_task_queries

        N_H = image_size[0] // patch_size
        N_W = image_size[1] // patch_size
        self.N_H = N_H
        self.N_W = N_W
        self.num_patches = N_H * N_W

        # Project encoder tokens to decoder_dim
        self.proj_context = nn.Linear(embed_dim, decoder_dim)

        # Learnable mask token (used for unobserved/masked positions in query)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        trunc_normal_(self.mask_token, std=0.02)

        # Per-context-task learned embeddings (added to context before cross-attn)
        if context_tasks:
            self.task_embeddings = nn.ParameterDict({
                t: nn.Parameter(torch.zeros(1, 1, decoder_dim)) for t in context_tasks
            })
            for emb in self.task_embeddings.values():
                trunc_normal_(emb, std=0.02)
        else:
            self.task_embeddings = None

        # Fixed sin-cos positional embedding for query tokens
        pos_emb = build_2d_sincos_posemb(N_H, N_W, decoder_dim)
        self.register_buffer("pos_emb", pos_emb)

        # Learnable modality embedding for this adapter's queries
        self.modality_emb = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        trunc_normal_(self.modality_emb, std=0.02)

        # Cross-attention layer
        self.cross_norm_q = nn.LayerNorm(decoder_dim)
        self.cross_norm_ctx = nn.LayerNorm(decoder_dim)
        self.cross_attn = CrossAttention(decoder_dim, num_heads=num_heads)
        self.cross_out_norm = nn.LayerNorm(decoder_dim)
        self.cross_mlp = Mlp(decoder_dim, hidden_features=int(decoder_dim * mlp_ratio))

        # Self-attention blocks
        if decoder_depth > 0:
            dpr = [x.item() for x in torch.linspace(0, 0.0, decoder_depth)]
            self.decoder_blocks = nn.Sequential(*[
                Block(decoder_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i])
                for i in range(decoder_depth)
            ])
        else:
            self.decoder_blocks = nn.Identity()

        # Output projection: decoder_dim → C * P * P
        self.out_norm = nn.LayerNorm(decoder_dim)
        self.out_proj = nn.Linear(decoder_dim, num_channels * patch_size * patch_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        encoder_tokens: torch.Tensor,
        ids_keep: torch.Tensor,
        ids_restore: torch.Tensor,
        num_task_tokens: int,
        task_start_idx: int,
        input_info: Dict,
    ) -> torch.Tensor:
        """
        Args:
            encoder_tokens : (B, N_vis, embed_dim)  — visible encoder output
            ids_keep       : (B, num_encoded_tokens) — token indices that were kept
            ids_restore    : (B, total_tokens)       — inverse permutation
            num_task_tokens: int  — total maskable tokens (all 3 maskable modalities)
            task_start_idx : int  — where this modality's tokens start in the full sequence
            input_info     : dict — modality info (for context task embeddings)

        Returns:
            out : (B, C, H, W)  — reconstructed field (cropped from padded size)
        """
        B = encoder_tokens.shape[0]

        # 1. Project encoder tokens
        ctx = self.proj_context(encoder_tokens)

        # 2. Add per-task context embeddings (identify origin modality in context)
        if self.task_embeddings is not None and input_info.get("tasks"):
            ctx_parts = []
            cursor = 0
            for mod_key, info in input_info["tasks"].items():
                n = info["num_tokens"]
                part = ctx[:, cursor: cursor + n, :]
                if mod_key in self.task_embeddings:
                    part = part + self.task_embeddings[mod_key]
                ctx_parts.append(part)
                cursor += n
            ctx = torch.cat(ctx_parts, dim=1)

        # 3. Build decoder query tokens (one per output patch)
        pos = F.interpolate(
            self.pos_emb, size=(self.N_H, self.N_W), mode="bicubic", align_corners=False
        ).to(encoder_tokens.dtype)
        pos = rearrange(pos, "b d nh nw -> b (nh nw) d").expand(B, -1, -1)
        queries = pos + self.modality_emb  # (B, N_tok, decoder_dim)

        # 4. Fill mask tokens for positions that were not kept
        #    ids_restore is over num_task_tokens (the maskable token pool)
        #    We need to reconstruct this modality's patch from the decoder tokens
        mask_tokens = self.mask_token.expand(B, num_task_tokens - ids_keep.shape[1], -1)
        # embed all visible tokens (in encoder order) and pad with mask tokens
        ctx_no_global = ctx[:, : ids_keep.shape[1], :]  # strip global tokens if any
        ctx_with_mask = torch.cat([ctx_no_global, mask_tokens], dim=1)  # (B, num_task_tokens, decoder_dim)
        # unshuffle
        ctx_with_mask = torch.gather(
            ctx_with_mask, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        )
        # Take only this modality's slice
        this_ctx = ctx_with_mask[:, task_start_idx: task_start_idx + self.num_patches, :]

        # Override query positions where we have observed data with the context
        if self.use_task_queries:
            queries = queries + this_ctx

        # 5. Cross attention: queries attend to full encoder context
        x = queries + self.cross_mlp(
            self.cross_out_norm(
                self.cross_attn(self.cross_norm_q(queries), self.cross_norm_ctx(ctx))
            )
        )

        # 6. Self-attention refinement
        x = self.decoder_blocks(x)

        # 7. Project to pixel space
        x = self.out_proj(self.out_norm(x))  # (B, N_tok, C*P*P)

        # 8. Reshape to image
        x = rearrange(
            x, "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)",
            nh=self.N_H, nw=self.N_W, ph=self.patch_size, pw=self.patch_size,
            c=self.num_channels,
        )

        # 9. Crop padded size back to original
        orig_h = input_info.get("orig_h", self.image_size[0])
        orig_w = input_info.get("orig_w", self.image_size[1])
        x = x[:, :, :orig_h, :orig_w]
        return x
