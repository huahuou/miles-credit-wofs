# WoFS DiffMAE Architecture

This document describes the current `wofs-diffmae` model used by
`config/wofs_diffmae_6x6_patch_height_mask.yml`.

## Goal

The model learns conditional diffusion inpainting for WoFS precip fields. It
predicts missing normalized precip channels while conditioning on background,
forcing, and reflectivity fields.

## Inputs

Configured input channel groups:

- `background`: 102 channels, from 6 variables x 17 levels.
- `precip`: 136 target channels, from 8 precip variables x 17 levels.
- `reflectivity`: 17 channels.
- `forcing`: 12 channels.
- `surface`: 0 channels in this config.

The target precip variables are grouped by hydrometeor:

- `QRAIN`
- `QNRAIN`
- `QHAIL`
- `QNHAIL`
- `QGRAUP`
- `QNGRAUPEL`
- `QSNOW`
- `QNSNOW`

Each group has 17 channels, one per vertical level.

## Tokenization

The configured spatial domain is `300 x 300` with `patch_size: 6`, producing a
`50 x 50` spatial patch grid, or 2500 patch tokens per tokenized field.

Conditioning modalities use `WoFSInputAdapter`, which is a patch projection:

```text
(B, C, H, W) -> (B, N_patch, embed_dim)
```

For this config, `embed_dim: 384`.

Precip target tokenization is grouped:

```text
each precip group: (B, 17, 300, 300) -> (B, 2500, 384)
```

The model does not create one attention token per vertical level. The 17 levels
inside a precip group are folded into the patch projection. This is the main
reason height masking does not multiply the target token count by 17.

## Conditioning Path

The conditioning modalities are:

```yaml
conditioned_modalities: [background, forcing, reflectivity]
```

Each modality is patch-projected independently and passed as context tokens to
the decoder. Surface is not used in this config because `surface: 0` and it is
not listed in `conditioned_modalities`.

## Target Path

For each precip group:

1. The noisy precip group is patch-projected.
2. Mask-token information is added where the precip mask marks that patch as
   hidden.
3. Fixed 2-D sine/cosine position embeddings are added.
4. A target-modality embedding and group-specific embedding are added.

Because `grouped_decoder_scope: per_group`, each hydrometeor group is decoded
separately. This avoids joint self-attention over all precip groups.

## Decoder

The config uses:

```yaml
decoder_type: cross_self
grouped_decoder_scope: per_group
depth: 8
num_heads: 8
mlp_ratio: 4.0
drop_path_rate: 0.05
```

Each decoder block performs:

```text
target tokens -> cross-attend conditioning context
target tokens -> self-attend target tokens
target tokens -> MLP
```

With `target_attention_window_size: 10`, target self-attention is local over
non-overlapping `10 x 10` windows in the `50 x 50` patch grid. This reduces
self-attention cost compared with global target self-attention. Cross-attention
to conditioning context remains global.

If `target_attention_window_size: 0`, target self-attention is global.

## Output Projection

Each decoded group token sequence is projected back to patch pixels:

```text
(B, 2500, 384) -> (B, 17 * 6 * 6) per patch -> (B, 17, 300, 300)
```

The outputs from all 8 precip groups are concatenated to recover:

```text
(B, 136, 300, 300)
```

## Diffusion Objective

The config uses:

```yaml
diffusion:
  timesteps: 1000
  sampling_timesteps: 200
  objective: pred_x0
  beta_schedule: sigmoid
  ddim_sampling_eta: 0.01
```

During training, the model receives a noised precip tensor `x_t` and predicts
the clean normalized precip field `x_0`.

The loss is MSE over masked pixels only.

## Mask Modes

The model supports three main precip mask modes:

### `spatial_patch`

Samples masked spatial patches. The same spatial mask is applied to all precip
channels.

Runtime mask shape:

```text
(B, N_patch)
```

### `channel_patch`

Samples masked group-patch entries independently. In grouped precip mode, this
means each hydrometeor group can have a different spatial mask.

Runtime mask shape:

```text
(B, N_group, N_patch)
```

### `height_patch`

Samples masked group-level-patch entries independently. This lets specific
vertical levels stay visible while other levels in the same hydrometeor group
are inpainted.

Runtime mask shape:

```text
(B, N_group, N_level, N_patch)
```

For this config:

```text
(B, 8, 17, 2500)
```

The attention token mask for a grouped precip token is reduced across level:

```text
group token is masked if any level in that group/patch is masked
```

The pixel loss and visible-precip clamping remain level-specific.

### `mixed`

Samples either `spatial_patch` or `channel_patch` each time a mask is built.
This is the legacy mixed mode and does not include height masking.

### `mixed_height`

Samples one of `spatial_patch`, `channel_patch`, or `height_patch` each time a
mask is built.

The relative sampling weights are:

```yaml
mixed_height_spatial_probability: 1.0
mixed_height_channel_probability: 1.0
mixed_height_height_probability: 1.0
```

These values are normalized internally, so `1, 1, 1` means one-third probability
for each mode. Setting one value to `0.0` removes that mode from the mixture.

## Height Mask Controls

`height_visible_levels` is a list of zero-based vertical levels that are never
sampled as masked. These levels remain visible during training loss masking and
are reinserted from `precip_visible` during sampling when
`clamp_visible_precip: true`.

Example:

```yaml
height_visible_levels: [0, 1, 2, 3]
```

This means levels 0, 1, 2, and 3 are treated as known background information.
Only levels 4 through 16 are eligible for random height masking unless
`height_mask_levels` is also set.

`height_mask_levels` is optional. If provided, only those zero-based levels are
eligible for masking, after excluding any levels listed in
`height_visible_levels`.

## Compute Notes

The current height-mask implementation is intentionally not full 3-D attention.
It does not create separate attention tokens for every vertical level. Instead:

- Grouped precip tokenization keeps the target token count at 2500 per group.
- `grouped_decoder_scope: per_group` decodes one hydrometeor group at a time.
- `target_attention_window_size: 10` limits target self-attention to local
  `10 x 10` patch windows.
- Height masks are applied at loss/clamping time with per-level precision.

This preserves the ability to use known sparse levels as background information
without paying the full token cost of `group x level x spatial_patch`
self-attention.
