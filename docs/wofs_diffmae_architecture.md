# WoFS DiffMAE 4x4 Patch Height-Mask Architecture

This README documents the DiffMAE implementation used by
`config/wofs_diffmae_4x4_patch_height_mask.yml`. It is based on the current
code in `credit/models/wofs_diffmae.py`,
`credit/trainers/trainerWRF_diffmae.py`, and the rollout helpers in
`applications/rollout_wrf_wofs_mae_da.py`.

The paper-level idea is Diffusion Masked Autoencoding: split a clean sample
`x_0` into visible content `x_0^v` and masked content `x_0^m`, diffuse only
the masked content, and train a decoder to recover the clean masked content
conditioned on encoded visible content. This WoFS model follows that contract
for precip inpainting, but adapts it to 3D precip cubes over vertical level,
latitude patch, and longitude patch.

## Config Summary

The active model section is:

```yaml
model:
  type: wofs-diffmae
  conditioned_modalities: [background, forcing, reflectivity]
  target_modality: precip
  precip_grouping: level
  decoder_type: cross_self
  precip_patch_size: [3, 4, 4]
  patch_size: 4
  surface_forcing_stride: 16
  image_size: [300, 300]
  embed_dim: 192
  condition_encoder_depth: 4
  depth: 4
  num_heads: 8
  target_attention_window_size: [2, 9, 9]
  diffusion:
    timesteps: 1000
    sampling_timesteps: 200
    objective: pred_x0
    beta_schedule: sigmoid
    ddim_sampling_eta: 0.01
```

The trainer uses:

```yaml
trainer:
  precip_mask_ratio: [0.6, 1.0]
  precip_mask_mode: cube_patch
  visible_precip_conditioning: true
  val_mask_ratio: 0.7
  val_mask_mode: cube_patch
  training_loss: mse
```

The important implementation detail is that this config does not use separate
runtime `channel_patch` or `height_patch` masks. The supported training mask
mode is `cube_patch`, which aliases to `random_precip_mask`. The mask is a
rank-sampled set of 3D precip cube tokens. Height enters through the token
geometry `precip_patch_size: [3, 4, 4]`.

## Data Layout

The target is normalized precip:

```text
precip: 136 channels = 8 hydrometeor variables x 17 vertical levels
```

Hydrometeor groups:

- `QRAIN`
- `QNRAIN`
- `QHAIL`
- `QNHAIL`
- `QGRAUP`
- `QNGRAUPEL`
- `QSNOW`
- `QNSNOW`

Conditioning fields:

```text
background:   102 channels = 6 variables x 17 levels
forcing:       12 channels
reflectivity:  17 channels
surface:        0 channels and not used by conditioned_modalities
```

All tensors are normalized before entering the model. Concentration precip
variables use the zero-inflated concentration transform configured under
`data.log_transform_params_json`, followed by the configured normalization
mode.

## Tokenization

### Conditioning Tokens

Each conditioning modality is handled by `WoFSInputAdapter`:

```text
(B, C, H, W)
-> pad to adapter image size if needed
-> Conv2d(C, embed_dim, kernel=patch_stride, stride=patch_stride)
-> flatten to (B, N_tokens, embed_dim)
-> add fixed 2D sin/cos position embedding
-> add learnable modality embedding
-> LayerNorm
```

For this config:

- `background` and `reflectivity` use `patch_size = 4`, so their grid is
  `75 x 75 = 5625` tokens each.
- `forcing` uses `surface_forcing_stride = 16`. The adapter pads
  `300 x 300` to `304 x 304`, so forcing uses a `19 x 19 = 361` token grid.
- `surface` is not constructed as context because it is absent from
  `conditioned_modalities`.

The raw context token list is therefore:

```text
background tokens   (B, 5625, 192)
forcing tokens      (B,  361, 192)
reflectivity tokens (B, 5625, 192)
optional visible precip tokens when visible_precip_conditioning=true
```

### Visible Precip Conditioning Tokens

Because `visible_precip_conditioning: true`, the clean precip tensor is also
used as conditioning, but only at visible locations.

Given a patch mask `M_patch`, the model expands it to a pixel-level level mask
`M_level` and repeats it across the 8 hydrometeor variables:

```text
visible_precip_image = precip * (1 - M_precip_pixel)
```

Masked precip pixels are zeroed before tokenization. The visible precip image
is tokenized with the same 3D precip patch adapter as the target, but with
visible-position embeddings and without adding the mask token. A learnable
`visible_precip_modality_emb` is then added.

### Target Precip Tokens

Precip is tokenized by `Precip3DPatchAdapter`, not by separate per-variable
2D tokens. The input shape is:

```text
(B, 136, 300, 300)
```

It is reshaped to:

```text
(B, 8 variables, 17 levels, 300, 300)
```

Then it is padded and projected by:

```text
Conv3d(
  in_channels=8,
  out_channels=192,
  kernel_size=(3, 4, 4),
  stride=(3, 4, 4)
)
```

The resulting token grid is:

```text
level axis: ceil(17 / 3) = 6
lat axis:   ceil(300 / 4) = 75
lon axis:   ceil(300 / 4) = 75
tokens:     6 x 75 x 75 = 33750
```

Each token represents a cube containing all 8 hydrometeor variables over up to
3 vertical levels and a `4 x 4` horizontal patch. The last level cube is padded
internally because 17 is not divisible by 3; predictions are cropped back to
17 levels after decoding.

Target tokens receive:

- fixed 3D sin/cos position embeddings over `(level_cube, y_patch, x_patch)`,
- a learnable `level_token_emb`,
- a learnable mask token added only where the patch mask is 1,
- a timestep embedding added before decoder blocks.

## Condition Encoder

`condition_encoder_depth: 4` adds a ViT-style encoder over the concatenated
conditioning tokens. Each block is standard pre-norm self-attention plus MLP:

```text
tokens = tokens + Attention(LayerNorm(tokens))
tokens = tokens + MLP(LayerNorm(tokens))
```

The encoder stores the output after every block. The last output is
LayerNormed. These four encoded context levels are later consumed by the
decoder in reverse order, giving the cross-self decoder a U-shaped connection
pattern similar to the DiffMAE paper description.

The condition encoder is run once per model forward during training. During
sampling, `_condition_tokens_once` runs it once before the reverse diffusion
loop and reuses the resulting context tokens for every denoising step.

## Decoder Architecture

The config uses `decoder_type: cross_self`, so every decoder block is
`CrossSelfDecoderBlock`:

```text
target tokens
-> LayerNorm target and LayerNorm context
-> cross-attention: target queries attend to context keys/values
-> residual add
-> LayerNorm target
-> target self-attention
-> residual add
-> LayerNorm target
-> MLP
-> residual add
```

There are 4 decoder blocks, `embed_dim = 192`, `num_heads = 8`,
`mlp_ratio = 4.0`, and stochastic depth ramps up to `drop_path_rate = 0.05`.

The target self-attention is local 3D window attention because
`target_attention_window_size: [2, 9, 9]`. Tokens are reshaped to the target
grid `(6, 75, 75)`, padded to multiples of `(2, 9, 9)`, partitioned into local
windows, self-attended inside each window, then unpadded and flattened back to
sequence form. Cross-attention to the context remains global.

Decoder context selection:

- If the condition encoder returns 4 layer outputs and the decoder has 4
  blocks, decoder block 0 uses encoder layer 3, block 1 uses layer 2, block 2
  uses layer 1, and block 3 uses layer 0.
- This is the implementation's reverse U-shaped connection.

## Output Projection

After the decoder blocks, target tokens are LayerNormed and projected by a
linear head:

```text
Linear(192, 8 variables * 3 levels * 4 * 4)
```

The projected patches are rearranged from:

```text
(B, 6, 75, 75, 8, 3, 4, 4)
```

to:

```text
(B, 8, 18, 300, 300)
```

Then the padded level is cropped:

```text
(B, 8, 17, 300, 300)
```

Finally the variable and level axes are flattened back to:

```text
(B, 136, 300, 300)
```

The optional `anti_patch_refiner` is disabled in this config, so the output is
returned directly.

## Diffusion Formulation

The implementation uses the same conditional diffusion structure described in
the DiffMAE paper:

```text
model p(x_0^m | x_0^v)
diffuse only masked precip
condition the denoiser on visible/context content
predict clean x_0 because objective: pred_x0
```

The configured schedule is sigmoid with `T = 1000`. The model registers:

```text
betas
alphas_cumprod
alphas_cumprod_prev
sqrt_alphas_cumprod
sqrt_one_minus_alphas_cumprod
sqrt_recip_alphas_cumprod
sqrt_recipm1_alphas_cumprod
posterior_variance
posterior_log_variance_clipped
posterior_mean_coef1
posterior_mean_coef2
```

Forward noising uses the closed-form DDPM equation:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
epsilon ~ N(0, I)
```

The implementation then applies `_masked_diffusion_state`:

```text
x_t <- x_t * M_precip_pixel
```

This keeps the Markov diffusion state only on masked target pixels. Visible
precip pixels are not part of the diffusion state; when enabled, they enter as
clean conditioning tokens instead.

## Mask Generation

Training calls `TrainerDiffMAE._sample_mask`. With this config:

```yaml
precip_mask_mode: cube_patch
precip_mask_ratio: [0.6, 1.0]
```

`cube_patch` is accepted as an alias of the model's 3D patch mask sampler:

```python
model.random_precip_mask(batch_size, ratio, device)
```

The random mask algorithm is:

1. Compute token count:

   ```text
   N = 6 * 75 * 75 = 33750
   ```

2. For each sample, draw a mask ratio. In training it is uniformly sampled
   from `[0.6, 1.0]`. In validation it is fixed at `0.7`.

3. Draw uniform random noise for all `N` cube tokens.

4. Sort the noise to create a random permutation.

5. Convert the permutation to ranks.

6. Mask the first `round(ratio * N)` ranked tokens:

   ```text
   M_patch = (rank < n_mask)
   shape: (B, 33750)
   values: 1.0 for masked, 0.0 for visible
   ```

7. To use the mask at pixel resolution, reshape it to `(B, 6, 75, 75)`, repeat
   by `(3, 4, 4)`, crop to `(B, 17, 300, 300)`, then repeat each level mask
   across the 8 hydrometeor variables to get `(B, 136, 300, 300)`.

Because each mask element is a cube token, masking one token masks the same
`3 x 4 x 4` level-space region for all 8 hydrometeor variables. This is the
current implementation's level-aware behavior. Older helper utilities still
contain `height_patch`, `channel_patch`, and `mixed_height` mask bundle logic
for rollout experiments, but the `WoFSDiffMAE` 3D token model used by this
config rejects those runtime mask modes in training.

## Training Algorithm

For each training batch:

1. Move tensor fields to the training device.

2. Build the conditioning dictionary:

   ```text
   background, surface, forcing, reflectivity
   ```

   Surface is present as an empty tensor when there are no surface channels,
   but it is not used by `conditioned_modalities`.

3. Sample a cube-patch precip mask `M_patch`.

4. Set `precip_visible = batch["precip"]` because
   `visible_precip_conditioning: true`.

5. Inside `model.p_losses`:

   - sample timestep `t ~ Uniform({0, ..., 999})`,
   - sample Gaussian noise `epsilon`,
   - compute `q_sample(x_0, t, epsilon)`,
   - zero all unmasked pixels with `_masked_diffusion_state`,
   - tokenize context and visible precip,
   - encode context with the 4-block condition encoder,
   - tokenize noisy masked precip and add mask/time/position embeddings,
   - decode through 4 cross-self decoder blocks,
   - project target tokens back to `(B, 136, 300, 300)`.

6. Because `objective: pred_x0`, the training target is the clean normalized
   precip tensor `x_0`.

7. Compute MSE only over masked precip pixels:

   ```text
   loss_raw = (model_out - x_0)^2
   loss = sum(loss_raw * M_precip_pixel) / sum(M_precip_pixel)
   ```

8. Use AMP when configured, scale the loss by `grad_accum_every`, backpropagate,
   clip gradients to `grad_max_norm = 1.0`, step the optimizer, update the AMP
   scaler, and update the cosine-restart scheduler on batch boundaries.

Validation repeats the same `p_losses` path with `val_mask_ratio: 0.7` and no
gradient updates.

## Sampling Algorithm

The model exposes `sample_precip` with samplers:

- `ddim`
- `ddpm`
- `repaint`
- `repaint_ddim`

The config's evaluation section uses DDIM:

```yaml
eval:
  sampler: ddim
  sampling_timesteps: 100
  ddim_sampling_eta: 0.02
  ensemble_size: 8
  visible_precip_conditioning: true
```

The general sampling setup is:

1. Build or load a precip mask.

2. Draw the initial image:

   ```text
   img ~ N(0, I), shape (B, 136, 300, 300)
   img <- img * M_precip_pixel
   ```

3. Encode conditioning once:

   ```text
   cond_tokens = _condition_tokens_once(cond, precip_visible, precip_mask)
   ```

   These tokens are reused for every reverse step. This matches the paper's
   statement that the encoder forwards visible patches only once during
   inference.

4. Iterate reverse diffusion steps, always reapplying `_masked_diffusion_state`
   after an update so the Markov state remains masked-only.

### DDIM Sampling

For `sampler: ddim`, the code builds a rounded unique descending timestep list
from `num_timesteps = 1000` to the requested sparse count. With
`sampling_timesteps: 100`, the loop uses 100 reverse steps.

For each pair `(time, time_next)`:

1. Predict `pred_x0` and `pred_noise` with the decoder.

2. If `time_next < 0`, return masked `pred_x0`.

3. Otherwise compute:

   ```text
   alpha_t    = alpha_bar[time]
   alpha_prev = alpha_bar[time_next]
   sigma^2 = eta^2
             * (1 - alpha_prev) / (1 - alpha_t)
             * (1 - alpha_t / alpha_prev)
   c = sqrt(1 - alpha_prev - sigma^2)
   img_prev = sqrt(alpha_prev) * pred_x0 + c * pred_noise + sigma * z
   ```

   where `z ~ N(0, I)` if `eta > 0`, otherwise `z = 0`.

4. Zero visible pixels in `img_prev`.

For deterministic DDIM, set `ddim_sampling_eta: 0`. This config uses nonzero
`eta`, so ensemble members can differ even with the same condition and mask.

### DDPM Sampling

For `sampler: ddpm`, every timestep from 999 to 0 is used. At each step:

1. Predict `x_0`.

2. Compute the DDPM posterior:

   ```text
   posterior_mean =
       posterior_mean_coef1[t] * x_0
     + posterior_mean_coef2[t] * x_t
   ```

3. Sample:

   ```text
   x_{t-1} = posterior_mean + exp(0.5 * posterior_log_variance[t]) * z
   ```

   with `z = 0` at `t = 0`.

4. Reapply the masked-state constraint.

### RePaint Sampling

For `sampler: repaint`, the code creates a schedule with backward denoising
steps and occasional forward-noising jumps. A forward jump from `from_time` to
`to_time` recursively applies:

```text
img = sqrt(1 - beta_time) * img + sqrt(beta_time) * z
```

Then normal DDPM-style posterior steps resume. The state is masked after every
forward and reverse move.

### RePaint DDIM Sampling

For `sampler: repaint_ddim`, the code uses the DDIM timestep grid but repeats
an inner resampling loop `repaint_jump_n_sample` times at each DDIM step. After
each inner DDIM step except the last, it re-noises from the predicted clean
state at the current alpha:

```text
img = sqrt(alpha_t) * x_start + sqrt(1 - alpha_t) * z
```

Again, the masked-state constraint is applied after each move.

## Alignment With The DiffMAE Paper

The implementation matches the paper in these core ways:

- It models masked precip conditioned on visible/context content,
  `p(x_0^m | x_0^v)`.
- The forward diffusion process is applied only to the masked region.
- The model uses the DDPM closed-form noising equation.
- The configured objective is clean-sample prediction, `pred_x0`, matching the
  paper's `||x_0^m - D_theta(x_t^m, t, E_phi(x_0^v))||^2`.
- The decoder receives noisy masked target tokens, timestep embeddings, and
  encoded visible/context tokens.
- The cross-self decoder block follows the paper's cross-attention then
  self-attention pattern.
- During sampling, conditioning is encoded once and reused through the reverse
  denoising loop.

Main WoFS-specific differences:

- The encoder context includes meteorological background, forcing, and
  reflectivity, not only visible patches from the same image.
- Clean visible precip can be added as an extra conditioning modality.
- Target tokens are 3D precip cube tokens over level, latitude patch, and
  longitude patch.
- Local 3D target self-attention is used to reduce attention cost.
- The sampling implementation supports DDIM, DDPM, RePaint, and RePaint DDIM.

## Relevant Files

- Config: `config/wofs_diffmae_4x4_patch_height_mask.yml`
- Model: `credit/models/wofs_diffmae.py`
- Trainer: `credit/trainers/trainerWRF_diffmae.py`
- Rollout sampler: `applications/rollout_wrf_wofs_mae_da.py`
- Deterministic mask bundle utility: `credit/wofs_diffmae_mask_utils.py`
- Mask generation CLI: `applications/generate_wofs_diffmae_mask.py`
- Tests: `tests/test_wofs_diffmae.py`
