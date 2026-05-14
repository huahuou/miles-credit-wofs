# WoFS MaskDiT Architecture

This document describes the `wofs-maskdit` model option added for WoFS precip
diffusion inpainting. It uses the same data pipeline, trainer, mask sampling,
diffusion objective, and sampling code as `wofs-diffmae`, but changes the
transformer decoder block to a MaskDiT-style conditional transformer.

Relevant files:

- Model: `credit/models/wofs_maskdit.py`
- Registry entry: `credit/models/__init__.py`
- Example config: `config/wofs_maskdit_4x4_patch_height_mask.yml`
- Baseline DiffMAE docs: `docs/wofs_diffmae_architecture.md`

## Training Interface

`wofs-maskdit` is designed to be a drop-in architecture option for the existing
DiffMAE training routine:

```yaml
model:
  type: wofs-maskdit

trainer:
  type: standard-wrf-diffmae
```

The trainer still calls the same model methods:

- `random_precip_mask`
- `random_channel_precip_mask`
- `random_height_precip_mask`
- `p_losses`
- `model_predictions`
- `sample_precip`

This means the existing `applications/train_wrf_wofs_mae.py` workflow does not
need a separate MaskDiT trainer.

## Shared DiffMAE Components

`WoFSMaskDiT` subclasses `WoFSDiffMAE`, so it keeps the same outer diffusion
model:

- Gaussian forward noising with configurable beta schedule.
- `pred_noise`, `pred_x0`, and `pred_v` objectives.
- DDIM, DDPM, and RePaint-style sampling support.
- Masked-pixel MSE loss.
- Spatial, channel/group, and height-aware precip masks.
- Grouped precip tokenization by hydrometeor.
- Optional visible-precip clamping during sampling.

The target and conditioning inputs are tokenized with the same
`WoFSInputAdapter` patch projection used by DiffMAE.

For the 4x4 height-mask config:

```text
image_size: 300 x 300
patch_size: 4
target grid: 75 x 75 = 5625 target tokens per precip group
precip groups: 8 hydrometeor groups
channels per group: 17 vertical levels
```

Vertical levels are still folded into each hydrometeor group's patch projection.
The model does not create one transformer token per level, so height masking
does not multiply the attention sequence length by 17.

## What MaskDiT Changes

The original `wofs-diffmae` decoder uses transformer blocks with explicit
conditioning context:

```text
target tokens -> cross-attention to condition tokens
target tokens -> target self-attention
target tokens -> MLP
```

`wofs-maskdit` replaces that block with a MaskDiT-style adaLN-Zero block:

```text
condition embedding = timestep embedding + pooled WoFS context embedding

target tokens -> adaptive LayerNorm modulation
target tokens -> self-attention
target tokens -> adaptive LayerNorm modulation
target tokens -> MLP
```

Each `MaskDiTBlock` predicts six modulation vectors from the condition
embedding:

```text
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
```

The shifts and scales modulate affine-free LayerNorm outputs. The gates control
how much attention and MLP residual signal is added back to the token stream.

With `maskdit_zero_init_adaln: true`, the final adaLN modulation layer is
zero-initialized. This follows the DiT/MaskDiT convention: at initialization,
the residual branches are gated off, which makes the transformer start from a
stable near-identity residual state.

## Conditioning Modes

`wofs-maskdit` supports two conditioning modes.

### `pooled_only`

```yaml
maskdit_condition_token_mode: pooled_only
```

This is the recommended default for the 4x4 WoFS config.

All conditioning modality tokens are concatenated, averaged into one context
summary per sample, projected, and added to the diffusion timestep embedding.
That combined vector drives adaLN modulation in every MaskDiT block.

The transformer sequence contains only target precip tokens. This preserves the
2-D target grid layout, so `target_attention_window_size` can be used:

```yaml
target_attention_window_size: 15
```

For the 4x4 config, each hydrometeor group has a `75 x 75` target grid. Windowed
attention with `15` attends over non-overlapping `15 x 15` target-token windows
instead of global attention over all 5625 tokens.

### `append`

```yaml
maskdit_condition_token_mode: append
```

This mode appends conditioning tokens to the target-token sequence before
MaskDiT self-attention. It gives the block direct token-level access to the
conditioning fields, but it is much more expensive for 4x4 patches because the
attention sequence becomes:

```text
target precip tokens + background tokens + forcing tokens + reflectivity tokens
```

Because the sequence is no longer a pure 2-D target grid, local target-window
attention is disabled in this mode. Use `append` mainly for smaller experiments,
larger patches, or ablation studies.

## Difference From `wofs-diffmae`

The key architectural difference is how conditioning enters the transformer.

`wofs-diffmae` uses explicit cross-attention:

```text
target query attends to condition key/value tokens
```

This keeps full spatial conditioning information available in every block, but
cross-attention cost scales with both target-token count and context-token
count.

`wofs-maskdit` in `pooled_only` mode uses conditioning through adaLN:

```text
condition tokens -> pooled summary -> timestep/context vector -> block modulation
```

This is cheaper at 4x4 resolution because the block attention operates only on
target tokens, and it can use local target windows. The tradeoff is that the
conditioning signal is summarized globally rather than cross-attended at every
spatial location.

In short:

| Aspect | `wofs-diffmae` | `wofs-maskdit` |
| --- | --- | --- |
| Decoder block | Cross-attention + self-attention | adaLN-Zero DiT block |
| Timestep conditioning | Added to tokens | Drives adaLN modulation |
| WoFS context in default config | Cross-attended as tokens | Pooled into adaLN condition |
| Local target windows | Supported in `cross_self` | Supported in `pooled_only` |
| 4x4 compute profile | Cross-attention to context plus target attention | Target attention only plus cheap condition pooling |
| Main tradeoff | Rich spatial conditioning, higher cost | Cheaper conditioning, less spatially explicit context |

## Output Path

The output path is inherited from DiffMAE. For each precip group:

```text
(B, N_patch, embed_dim)
  -> Linear(embed_dim, group_channels * patch_size * patch_size)
  -> patch tiling
  -> (B, group_channels, H, W)
```

All hydrometeor group outputs are concatenated back into:

```text
(B, 136, 300, 300)
```

The optional `anti_patch_refiner` can still be enabled because it is applied
after patch tiling and is independent of the transformer backbone.

## Config Notes

The current example config uses:

```yaml
model:
  type: wofs-maskdit
  precip_grouping: grouped
  grouped_decoder_scope: per_group
  maskdit_condition_token_mode: pooled_only
  maskdit_zero_init_adaln: true
  patch_size: 4
  image_size: [300, 300]
  embed_dim: 384
  depth: 8
  num_heads: 8
  target_attention_window_size: 15
```

`decoder_type` remains in the config for compatibility with the inherited
DiffMAE constructor, but the MaskDiT class replaces the decoder blocks after
the base model is initialized.

## When To Use Which Model

Use `wofs-diffmae` when spatially detailed conditioning through explicit
cross-attention is the priority and the memory budget is acceptable.

Use `wofs-maskdit` when testing whether DiT-style conditional modulation gives
a better compute/quality tradeoff, especially at small patch sizes where target
and context token counts are large.

For the 4x4 height-mask experiment, start with:

```bash
python applications/train_wrf_wofs_mae.py \
  -c config/wofs_maskdit_4x4_patch_height_mask.yml
```

On the cluster, use the same resource/module setup as the DiffMAE training
routine before launching.
