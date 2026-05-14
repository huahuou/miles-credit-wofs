"""MaskDiT-style conditional diffusion model for WoFS precip inpainting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from credit.models.wofs_diffmae import WindowAttention2D, WoFSDiffMAE
from credit.models.wofs_mae_adapters import Attention, Mlp


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MaskDiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning, adapted from MaskDiT."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        target_grid_size: tuple[int, int] | None = None,
        target_window_size: int = 0,
        target_window_shift_size: int = 0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if target_window_size and target_window_size > 0:
            if target_grid_size is None:
                raise ValueError("target_grid_size is required when target_window_size > 0")
            if target_window_shift_size and target_window_shift_size > 0:
                self.attn = ShiftedWindowAttention2D(
                    dim,
                    num_heads=num_heads,
                    grid_size=target_grid_size,
                    window_size=int(target_window_size),
                    shift_size=int(target_window_shift_size),
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                )
            else:
                self.attn = WindowAttention2D(
                    dim,
                    num_heads=num_heads,
                    grid_size=target_grid_size,
                    window_size=int(target_window_size),
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        self.drop_path_prob = float(drop_path)

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_prob == 0.0:
            return x
        keep = 1.0 - self.drop_path_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep)
        return x * random_tensor / keep

    def zero_init_adaln(self) -> None:
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + self._drop_path(gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)))
        x = x + self._drop_path(gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class ShiftedWindowAttention2D(nn.Module):
    """Window attention on a spatial grid with non-cyclic shifted windows."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size: tuple[int, int],
        window_size: int,
        shift_size: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if not (0 < self.shift_size < self.window_size):
            raise ValueError(
                f"shift_size must be in [1, window_size - 1], got shift_size={shift_size}, "
                f"window_size={window_size}"
            )
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _partition_windows(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, d = x.shape
        ws = self.window_size
        return (
            x.reshape(b, h // ws, ws, w // ws, ws, d)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, ws * ws, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        grid_h, grid_w = self.grid_size
        if n != grid_h * grid_w:
            raise ValueError(f"ShiftedWindowAttention2D expected {grid_h * grid_w} tokens, got {n}")

        ws = self.window_size
        shift = self.shift_size
        x_grid = x.reshape(b, grid_h, grid_w, d)
        valid = torch.ones(1, grid_h, grid_w, 1, dtype=x.dtype, device=x.device)

        pad_h = (ws - (grid_h + shift) % ws) % ws
        pad_w = (ws - (grid_w + shift) % ws) % ws
        pad = (0, 0, shift, pad_w, shift, pad_h)
        x_grid = F.pad(x_grid, pad)
        valid = F.pad(valid, pad)

        padded_h, padded_w = x_grid.shape[1], x_grid.shape[2]
        x_windows = self._partition_windows(x_grid)
        valid_windows = self._partition_windows(valid).squeeze(-1)
        valid_windows = valid_windows.repeat(b, 1)

        qkv = self.qkv(x_windows).reshape(
            x_windows.shape[0], ws * ws, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_mask = (1.0 - valid_windows[:, None, None, :]) * torch.finfo(x.dtype).min
        dropout_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(x_windows.shape[0], ws * ws, d)
        out = self.proj_drop(self.proj(out))

        out = out.reshape(b, padded_h // ws, padded_w // ws, ws, ws, d)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(b, padded_h, padded_w, d)
        out = out[:, shift: shift + grid_h, shift: shift + grid_w]
        return out.reshape(b, n, d)


class WoFSMaskDiT(WoFSDiffMAE):
    """WoFS DiffMAE with a MaskDiT-style adaLN transformer decoder."""

    def __init__(self, **kwargs):
        self.maskdit_condition_token_mode = str(kwargs.pop("maskdit_condition_token_mode", "append")).strip().lower()
        if self.maskdit_condition_token_mode not in {"append", "pooled_only"}:
            raise ValueError(
                "maskdit_condition_token_mode must be one of {'append', 'pooled_only'}, "
                f"got {self.maskdit_condition_token_mode!r}"
            )
        self.maskdit_zero_init_adaln = bool(kwargs.pop("maskdit_zero_init_adaln", True))
        self.maskdit_shifted_window = bool(kwargs.pop("maskdit_shifted_window", False))
        self.maskdit_window_shift_size = int(kwargs.pop("maskdit_window_shift_size", 0))
        super().__init__(**kwargs)

        depth = len(self.blocks)
        dpr = torch.linspace(0, float(kwargs.get("drop_path_rate", 0.0)), depth).tolist()
        target_window_size = 0
        if self.maskdit_condition_token_mode == "pooled_only":
            target_window_size = self.target_attention_window_size
        shift_size = self.maskdit_window_shift_size
        if shift_size <= 0 and target_window_size > 1:
            shift_size = target_window_size // 2
        self.blocks = nn.ModuleList(
            [
                MaskDiTBlock(
                    self.embed_dim,
                    cond_dim=self.embed_dim,
                    num_heads=int(kwargs.get("num_heads", 8)),
                    mlp_ratio=float(kwargs.get("mlp_ratio", 4.0)),
                    drop_path=float(dpr[i]),
                    target_grid_size=self.target_grid_size,
                    target_window_size=target_window_size,
                    target_window_shift_size=(
                        shift_size
                        if self.maskdit_shifted_window
                        and self.maskdit_condition_token_mode == "pooled_only"
                        and target_window_size > 0
                        and i % 2 == 1
                        else 0
                    ),
                )
                for i in range(depth)
            ]
        )
        self.condition_summary_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.blocks.apply(self._init_maskdit_weights)
        self.condition_summary_proj.apply(self._init_maskdit_weights)
        if self.maskdit_zero_init_adaln:
            for block in self.blocks:
                block.zero_init_adaln()

    def _init_maskdit_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _decode_target_tokens(
        self,
        target_tokens: torch.Tensor,
        cond_tokens: list[torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        if cond_tokens:
            context = torch.cat(cond_tokens, dim=1)
            cond_summary = context.mean(dim=1)
        else:
            context = target_tokens.new_zeros(target_tokens.shape[0], 0, target_tokens.shape[-1])
            cond_summary = target_tokens.new_zeros(target_tokens.shape[0], target_tokens.shape[-1])

        c = self.time_mlp(t) + self.condition_summary_proj(cond_summary)
        if self.maskdit_condition_token_mode == "append" and context.shape[1] > 0:
            tokens = torch.cat([target_tokens, context], dim=1)
        else:
            tokens = target_tokens

        for block in self.blocks:
            tokens = block(tokens, c)

        return self.norm(tokens[:, : target_tokens.shape[1]])
