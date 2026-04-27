# CrossFormerWRFEnsemble — Architecture & Configuration Guide

Model file: `credit/models/wxformer/crossformer_wrf_ensemble.py`  
Registry key: `crossformer_wrf_ensemble`  
Compatible trainers: `multi-step-wrf` (deterministic) · `multi-step-wrf-ensemble` (KCRPS)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Details](#2-component-details)
   - 2.1 [VariableTokenEmbedder — input projection](#21-variabletokenembedder--input-projection)
   - 2.2 [FiLM Conditioning](#22-film-conditioning)
   - 2.3 [CrossFormer Encoder](#23-crossformer-encoder)
   - 2.4 [AnticheckboardUpBlock — decoder](#24-anticheckboardupblock--decoder)
   - 2.5 [VariableTokenDecoder — output projection](#25-variabletokendecoder--output-projection)
   - 2.6 [StochasticDecompositionLayer — noise injection](#26-stochasticdecompositionlayer--noise-injection)
   - 2.7 [TensorPadding — boundary handling](#27-tensorpadding--boundary-handling)
3. [Data Flow Diagram](#3-data-flow-diagram)
4. [Config Reference: Input Feature Flexibility](#4-config-reference-input-feature-flexibility)
   - 4.1 [Variable lists](#41-variable-lists)
   - 4.2 [Adding a variable without retraining the backbone](#42-adding-a-variable-without-retraining-the-backbone)
   - 4.3 [Removing a variable at inference](#43-removing-a-variable-at-inference)
5. [Config Reference: Spatial Dimension Flexibility](#5-config-reference-spatial-dimension-flexibility)
   - 5.1 [Automatic window-divisibility padding](#51-automatic-window-divisibility-padding)
   - 5.2 [Boundary padding (TensorPadding)](#52-boundary-padding-tensorpadding)
   - 5.3 [Supported domain sizes](#53-supported-domain-sizes)
6. [Backbone Scaling Presets](#6-backbone-scaling-presets)
7. [Two-Phase Training Strategy](#7-two-phase-training-strategy)
8. [Loss Configuration](#8-loss-configuration)
   - 8.1 [Variable weights](#81-variable-weights)
   - 8.2 [Spectral loss](#82-spectral-loss)
9. [Launch Scripts](#9-launch-scripts)

---

## 1. Architecture Overview

CrossFormerWRFEnsemble is a **variable-flexible**, **resolution-flexible**, **ensemble-capable** LAM forecast model for WRF/WoFS domains. It is a symmetric U-Net whose encoder is a CrossFormer backbone and whose decoder is an anti-checkerboard upsampling stack.

```
Interior input x  (B, C_in, T, H, W)
Boundary input xb (B, C_b, T, H, W)
Time encoding     (B, time_encode_dim)
         │
         ▼
┌───────────────────────┐
│  TensorPadding (opt.) │  reflect at LAM boundaries → (B, C, T, H+2p, W+2p)
└───────────┬───────────┘
            │  flatten T into C
            ▼
┌───────────────────────────────────────────────────────────────┐
│  VariableTokenEmbedder                                        │
│  per-var 1×1 Conv2d → sigmoid gate → sum → LayerNorm          │
│  (B, C_total, H, W)  →  (B, embed_dim, H, W)                  │
└───────────┬───────────────────────────────────────────────────┘
            │                      ▲ FiLM γ,β from boundary pool + time enc
            ▼                      │
      x_emb × γ + β  ──────────────┘
            │
            │  pad H,W to multiple of local_window_size × 2⁴
            ▼

  ┌──────────────────────────────────────────────────┐
  │  Stage 0: CrossEmbedLayer(embed_dim → dim[0])    │─── enc[0] (H/2) ──────────────────┐
  │           Transformer(local, global=ws[0])  H/2  │                                   │
  └───────────────────────┬──────────────────────────┘                                   │
                          │                                                              │
  ┌──────────────────────────────────────────────────┐                                   │
  │  Stage 1: CrossEmbedLayer(dim[0] → dim[1])       │─── enc[1] (H/4) ──────────┐       │
  │           Transformer(local, global=ws[1])  H/4  │                           │       │
  └───────────────────────┬──────────────────────────┘                           │       │
                          │                                                      │       │
  ┌──────────────────────────────────────────────────┐                           │       │
  │  Stage 2: CrossEmbedLayer(dim[1] → dim[2])       │─── enc[2] (H/8) ──┐       │       │
  │           Transformer(local, global=ws[2])  H/8  │                   │       │       │
  └───────────────────────┬──────────────────────────┘                   │       │       │
                          │                                              │       │       │
  ┌──────────────────────────────────────────────────┐                   │       │       │
  │  Stage 3: CrossEmbedLayer(dim[2] → dim[3])       │  (bottleneck)     │       │       │
  │           Transformer(local, global=ws[3]) H/16  │                   │       │       │
  └───────────────────────┬──────────────────────────┘                   │       │       │
                          │ feat (H/16)                                  │       │       │
  ┌──────────────────────────────────────────────────┐                   │       │       │
  │  Up1: AnticheckboardUp  H/16 → H/8               │◄──────────────────┘       │       │
  │       nearest+Conv → [noise] → cat(enc[2])       │                           │       │
  └───────────────────────┬──────────────────────────┘                           │       │
                          │ H/8                                                  │       │
  ┌──────────────────────────────────────────────────┐                           │       │
  │  Up2: AnticheckboardUp  H/8 → H/4                │◄──────────────────────────┘       │
  │       nearest+Conv → [noise] → cat(enc[1])       │                                   │
  └───────────────────────┬──────────────────────────┘                                   │
                          │ H/4                                                          │
  ┌──────────────────────────────────────────────────┐                                   │
  │  Up3: AnticheckboardUp  H/4 → H/2                │◄──────────────────────────────────┘
  │       nearest+Conv → [noise] → cat(enc[0])       │
  └───────────────────────┬──────────────────────────┘
                          │ H/2
  ┌──────────────────────────────────────────────────┐
  │  Up4: AnticheckboardUp  H/2 → H                  │  (no skip)
  │       nearest+Conv → [noise]                     │
  └───────────────────────┬──────────────────────────┘
                          │  crop to orig_H × orig_W
                          ▼
┌───────────────────────────────────────────────────────────────┐
│  VariableTokenDecoder                                         │
│  per-var 1×1 Conv2d heads → concatenate                       │
│  (B, feat_dim, H, W)  →  (B, C_out, H, W)                     │
└───────────┬───────────────────────────────────────────────────┘
            │  unsqueeze T dim
            ▼
    TensorPadding.unpad (opt.)
            │
            ▼
    output (B, C_out, 1, H, W)
```

---

## 2. Component Details

### 2.1 VariableTokenEmbedder — input projection

**Purpose**: project an arbitrary flat-channel input into a fixed-width embedding so the backbone never needs to change when variables are added or removed.

**Mechanism**:
- For each variable (or variable group) a dedicated `nn.Conv2d(n_chans, embed_dim, kernel_size=1)` is registered in an `nn.ModuleDict` keyed by variable name.
- Each projector is multiplied by a learned sigmoid gate (`gate_logits` initialised to 0 → gate = 0.5 at init).
- All projections are **summed** into a single `(B, embed_dim, H, W)` tensor, followed by channel-wise `LayerNorm`.

**Why summation instead of concatenation**: Concatenation would change the embedding width every time a variable is added or dropped, requiring backbone weight surgery. Summation keeps `embed_dim` constant regardless of the variable set.

**Slice spec** is built automatically by `CrossFormerWRFEnsemble.__init__` from the variable name lists and `levels`. Channel order matches what `TrainerWRFMulti*` delivers:

```
upper_air[0] × levels, ..., upper_air[N-1] × levels,
surface[0] × 1, ..., surface[M-1] × 1,
dyn_forcing[0] × 1, ...,
forcing[0] × 1, ...,
static[0] × 1, ...
(all × frames)
```

A mirror `VariableTokenEmbedder` handles boundary fields using the `varname_boundary_upper` / `varname_boundary_surface` lists.

### 2.2 FiLM Conditioning

Boundary information and time encoding are fused into the interior embedding via **Feature-wise Linear Modulation (FiLM)**:

```
xb_pooled = mean_pool(boundary_embedding)         # (B, boundary_embed_dim)
cond      = cat(xb_pooled, x_time_encode, dim=1)  # (B, boundary_embed_dim + time_encode_dim)
γ, β      = split(MLP(cond), 2, dim=1)            # each (B, embed_dim)
x_emb     = γ[:,: ,None,None] × x_emb + β[:,:,None,None]
```

The MLP is `Linear → SiLU → Linear` producing `2 × embed_dim` outputs. This conditions every spatial token on the full boundary state without adding spatial dimensions.

### 2.3 CrossFormer Encoder

The encoder stacks 4 stages, each consisting of:

1. **CrossEmbedLayer**: multi-scale strided convolution that simultaneously halves spatial resolution and increases channel depth. At stage 0, kernel sizes `[4, 8, 16, 32]` capture convective cells (12 km), mesoscale features (24/48 km) and synoptic context (96 km) in a single layer.

2. **Transformer**: alternating **local short-range** and **global long-range** attention blocks.
   - Local attention: restricted to `local_window_size × local_window_size` windows (default 8×8 → 24 km at stage 0).
   - Global attention: restricted to `global_window_size × global_window_size` windows operating on the full (downsampled) feature map.
   - Both use the same `dim_head` head width; number of heads scales with channel dimension.

Encoder skip connections from stages 0, 1, 2 are stored and concatenated into the decoder.

### 2.4 AnticheckboardUpBlock — decoder

Each of the 4 decoder stages applies:

```
x  →  Upsample(2×, mode=nearest)  →  Conv2d(3×3)  →  x_up
x_up + Conv-GN-SiLU × num_residuals (x_up)         →  output
```

`nearest` (exact) upsampling duplicates grid values without interpolation artefacts. The subsequent 3×3 conv blends neighbours uniformly, removing any residual periodicity that would appear as a checkerboard in the power spectrum. This is superior to:
- `PixelShuffle`: hard-codes a 2×2 tile periodicity into the output spectrum.
- `ConvTranspose2d`: inserts learnable zeros at alternate positions, which project a 2× frequency spike into every output feature map.

After upsampling, the decoder concatenates the matching encoder skip connection along the channel axis before the next stage.

### 2.5 VariableTokenDecoder — output projection

The mirror of `VariableTokenEmbedder`. A per-variable `nn.Conv2d(feat_dim, n_chans, kernel_size=1)` projects the shared feature map into each output variable's channel count, then concatenates in order: `upper_air × levels`, `surface × 1`, `diagnostic × 1`.

### 2.6 StochasticDecompositionLayer — noise injection

When `noise_injection.activate: True`, a `StochasticDecompositionLayer` is inserted after each of the 4 decoder upsampling stages (and optionally after encoder stages 0–2 for `encoder_noise: True`).

The layer samples (or receives) a latent vector `z ∈ R^{latent_dim}` and injects scale-appropriate noise into the feature map. The noise magnitude is controlled by `noise_scales`:

| Injection site | Stage resolution (WoFS 300×300) | `noise_scales` index | Default |
|---|---|---|---|
| Decoder up1 | ~19×19 | 0 | 0.30 |
| Decoder up2 | ~38×38 | 1 | 0.15 |
| Decoder up3 | ~75×75 | 2 | 0.05 |
| Decoder up4 | ~150×150 | 3 | 0.01 |
| Encoder 0–2 | various | — | same as noise_scales[-1] |

Larger noise at coarser scales encodes mesoscale storm-placement uncertainty. Finer scales receive smaller noise to preserve resolved-scale structure in the mean.

Setting `correlated: True` draws a single `z` shared across all sites (spatially coherent noise). `correlated: False` (default) draws fresh `z` per site (independent multi-scale noise).

### 2.7 TensorPadding — boundary handling

WoFS is a **limited-area model (LAM)**. The padding mode must be `mirror` (reflection), not `earth`.

| Mode | Behaviour | Correct for |
|---|---|---|
| `mirror` | Reflects field at each edge | LAM domains (WoFS, WRF subdomains) |
| `earth` | Circular longitude wrap + 180° pole shift | Global models only |

The `earth` mode treats east and west edges as adjacent and applies a discontinuous pole transformation — both operations are physically wrong for a CONUS subdomain.

`pad_lat` and `pad_lon` specify symmetric padding in pixels. At 3 km grid spacing, `[16, 16]` adds 48 km of reflected boundary context on each side, filling the largest CrossEmbedLayer receptive field (32px kernel × 3 km = 96 km effective context).

---

## 3. Data Flow Diagram

```
Training input tensors from TrainerWRFMulti*
────────────────────────────────────────────
  x          : (B, C_in,  T,  H, W)   interior state, residual target
  x_boundary : (B, C_bnd, T,  H, W)   boundary conditions (lateral nudge zone)
  x_time_enc : (B, time_encode_dim)   lead time / calendar encoding

Channel layout of x
───────────────────
  [T × L, QVAPOR × L, U × L, V × L, W × L, GEOPOT × L,   <- upper_air × levels
   T2, Q2, U10, V10, COMPOSITE_REFL_10CM, RAIN_AMOUNT,     <- surface × 1
   cos_lat, sin_lat, cos_lon, sin_lon,                       <- dyn_forcing × 1
   cos_jd, sin_jd, cos_lt, sin_lt, cos_sza, insol]           (all × frames)

Channel layout of x_boundary
─────────────────────────────
  [T × L, QVAPOR × L, U × L, V × L, W × L, GEOPOT × L,   <- boundary upper × levels
   T2, Q2, U10, V10, COMPOSITE_REFL_10CM, RAIN_AMOUNT,     <- boundary surface × 1
   XLAND, HGT]                                               (all × frames)

Output tensor
─────────────
  out : (B, C_out, 1, H, W)
  C_out = upper_air × levels + surface × 1 + diagnostic × 1
        = 6×17 + 6 + 4 = 112  (WoFS default)
```

---

## 4. Config Reference: Input Feature Flexibility

### 4.1 Variable lists

The variable sets are declared in **two parallel places** in the config: `data.*` (controls data loading) and `model.varname_*` (controls the `VariableTokenEmbedder` slice spec). They must be consistent.

```yaml
# ── DATA SECTION ──────────────────────────────────────────────────────────
data:
  variables:                ['T', 'QVAPOR', 'U', 'V', 'W', 'GEOPOT']
  surface_variables:        ['T2', 'Q2', 'U10', 'V10', 'COMPOSITE_REFL_10CM', 'RAIN_AMOUNT']
  dynamic_forcing_variables: [cos_latitude, sin_latitude, cos_longitude, sin_longitude,
                              cos_julian_day, sin_julian_day, cos_local_time, sin_local_time,
                              cos_solar_zenith_angle, insolation]
  diagnostic_variables:     ['UP_HELI_MAX', 'WSPD80', 'W_UP_MAX', 'LWP']
  forcing_variables:        []
  static_variables:         []
  levels: 17
  boundary:
    variables:         ['T', 'QVAPOR', 'U', 'V', 'W', 'GEOPOT']
    surface_variables: ['T2', 'Q2', 'U10', 'V10', 'COMPOSITE_REFL_10CM', 'RAIN_AMOUNT', 'XLAND', 'HGT']
    levels: 17

# ── MODEL SECTION ──────────────────────────────────────────────────────────
model:
  varname_upper_air:        ['T', 'QVAPOR', 'U', 'V', 'W', 'GEOPOT']   # must match data.variables
  varname_surface:          ['T2', 'Q2', 'U10', 'V10', 'COMPOSITE_REFL_10CM', 'RAIN_AMOUNT']
  varname_dyn_forcing:      [cos_latitude, sin_latitude, cos_longitude, sin_longitude,
                             cos_julian_day, sin_julian_day, cos_local_time, sin_local_time,
                             cos_solar_zenith_angle, insolation]
  varname_forcing:          []
  varname_static:           []
  varname_diagnostic:       ['UP_HELI_MAX', 'WSPD80', 'W_UP_MAX', 'LWP']
  varname_boundary_upper:   ['T', 'QVAPOR', 'U', 'V', 'W', 'GEOPOT']
  varname_boundary_surface: ['T2', 'Q2', 'U10', 'V10', 'COMPOSITE_REFL_10CM', 'RAIN_AMOUNT', 'XLAND', 'HGT']
  levels: 17
  boundary_levels: 17
```

**Rules**:
- `varname_upper_air` entries each occupy `levels` channels in the input.
- `varname_surface`, `varname_dyn_forcing`, `varname_forcing`, `varname_static` each occupy 1 channel (× `frames`).
- `varname_diagnostic` entries are **output-only** — they are in the output spec but not the input spec.
- The data `all_varnames` list must include every name that appears in any of the above lists (used internally by the data loader).

### 4.2 Adding a variable without retraining the backbone

Because embeddings are summed, a new variable can be added by inserting a new leaf projector. The backbone weights load unchanged.

**Step 1** — update the config:
```yaml
data:
  surface_variables: ['T2', 'Q2', 'U10', 'V10', 'COMPOSITE_REFL_10CM', 'RAIN_AMOUNT', 'PWAT']  # add PWAT
  all_varnames: [..., 'PWAT']

model:
  varname_surface: ['T2', 'Q2', 'U10', 'V10', 'COMPOSITE_REFL_10CM', 'RAIN_AMOUNT', 'PWAT']
```

**Step 2** — load the pre-trained checkpoint with `strict=False`. The new `PWAT` projector will be randomly initialised (gate = 0.5, projection weights from Conv2d default). Because gate_logits starts at 0, the sigmoid gate is 0.5 and the new variable contributes at half strength immediately.

**Step 3** — optionally zero-initialise the new projector for a fully conservative start:
```python
model.interior_embedder.add_variable('PWAT', n_chans=1, start_channel=<offset>)
# add_variable() already zero-inits the projection weights and gate.
```

### 4.3 Removing a variable at inference

Set `varname_surface` (or whichever list) to exclude the variable — the corresponding projector in `interior_embedder.projectors` simply won't be called. The checkpoint key is preserved (load with `strict=False`). No architecture change is required.

---

## 5. Config Reference: Spatial Dimension Flexibility

### 5.1 Automatic window-divisibility padding

The CrossFormer encoder halves spatial dimensions 4 times (one per stage with `cross_embed_strides=[2,2,2,2]`). For attention windows to tile perfectly at every stage the spatial dimensions must satisfy:

```
H, W  ≡  0  (mod  local_window_size × 2^4)
```

With the default `local_window_size: 8` this becomes `H, W ≡ 0 mod 128`.

`CrossFormerWRFEnsemble.forward` computes the required padding automatically:

```python
pad_H = (-H) % (local_window_size * 2**num_stages)
pad_W = (-W) % (local_window_size * 2**num_stages)
```

It pads zeros to the right and bottom before the encoder and crops back to the original size after the decoder. **This is transparent to the caller** — any input size works without changing the config.

**Example**:

| Input grid | pad needed | padded size |
|---|---|---|
| 300 × 300 | 28 × 28 | 328 × 328 |
| 256 × 256 | 0 × 0 | 256 × 256 |
| 400 × 400 | 0 × 128 | 400 × 400 wait — 400 mod 128 = 16, pad=112 | 400 × 512 |
| 150 × 150 | 106 × 106 | 256 × 256 |

> **Tip**: Prefer input sizes that are already multiples of 128 (128, 256, 384, 512) to avoid wasted compute from padding. For WoFS 300×300 the 28px overhead is ~9%.

### 5.2 Boundary padding (TensorPadding)

`padding_conf` adds a reflective ring around the domain before the `VariableTokenEmbedder`, providing context at the model boundary. It is removed after the decoder.

```yaml
model:
  padding_conf:
    activate: True          # False to disable (saves ~22% memory at 3km 300×300)
    mode: mirror            # REQUIRED for LAM. Do NOT use 'earth' for WRF/WoFS.
    pad_lat: [16, 16]       # pixels added to south / north edges
    pad_lon: [16, 16]       # pixels added to west / east edges
```

**Effect on spatial size** (before window-divisibility pad):

```
H_padded = H + pad_lat[0] + pad_lat[1]   →  300 + 16 + 16 = 332
W_padded = W + pad_lon[0] + pad_lon[1]   →  300 + 16 + 16 = 332
```

After TensorPadding + window-divisibility pad the internal resolution becomes 384×384 (next multiple of 128 ≥ 332) for the default WoFS config.

**Guidance for non-WoFS domains**:

| Condition | Recommendation |
|---|---|
| Grid spacing < 5 km, small domain (≤ 200px) | `pad_lat/lon: [8, 8]` to reduce compute overhead |
| Grid spacing 5–15 km, medium domain | `pad_lat/lon: [16, 16]` (default) |
| Grid spacing > 15 km, large domain | `pad_lat/lon: [24, 32]` — larger receptive field gap |
| Memory-constrained (< 40 GB) | `activate: False` — the hard domain-edge artifact is mild |
| Global or periodic domain | Use `mode: earth` **only** — not for LAM |

### 5.3 Supported domain sizes

Any `(H, W)` is supported — the automatic padding logic handles all cases. Practical limits are GPU memory and batch size:

| Domain | Padded (TensorPad 16) | Internal (after win-pad) | Memory note |
|---|---|---|---|
| WoFS 300×300 | 332×332 | 384×384 | ~20M param model, batch 8, H100 fits comfortably |
| HRRR 1059×1799 | 1091×1831 | 1152×1920 | reduce batch to 1; consider Compact config |
| 3-km CONUS 1300×2000 | 1332×2032 | 1408×2048 | multi-GPU only |
| 256×256 (test) | 288×288 | 384×384 (no TensorPad) | fast dev cycle |

---

## 6. Backbone Scaling Presets

All presets use `depth=[2, 2, D, 2]` where `D` controls the bottleneck transformer depth. `cross_embed_strides=[2,2,2,2]` and `local_window_size=8` are fixed for the symmetric U-Net decoder.

| Preset | `dim` | `depth` | Params | Use case |
|---|---|---|---|---|
| **Compact** | `[64, 128, 192, 256]` | `[2, 2, 8, 2]` | ~14.8M | Memory-constrained; fine-tuning with large `ensemble_size` |
| **Medium-A** (default) | `[64, 128, 192, 384]` | `[2, 2, 12, 2]` | ~20.1M | H100 96GB; Phase-1 deterministic + Phase-2 ensemble |
| **Medium-B** | `[96, 192, 256, 512]` | `[2, 2, 12, 2]` | ~33.4M | Single-GPU A100 80GB, batch ≤ 4 |
| **Large** | `[128, 256, 512, 768]` | `[2, 4, 16, 4]` | ~85M | Multi-node; not recommended for 20M constraint |

```yaml
# Medium-A (default, ~20M params, recommended for H100 96GB)
model:
  dim:   [64, 128, 192, 384]
  depth: [2, 2, 12, 2]
  dim_head: 32
  global_window_size: [8, 6, 4, 1]
```

**`global_window_size`** controls the long-range attention window at each encoder stage. Larger values increase long-range context but also attention memory cost quadratically. The default `[8, 6, 4, 1]` prioritises long-range at early (higher-resolution) stages and restricts it at the coarsest stage where context is already spatially dense.

**`embed_dim`** is the width of the token embedding that enters the encoder. Increasing it raises the cost of the `VariableTokenEmbedder` and the first `CrossEmbedLayer` but has no effect on the interior backbone dims.

```yaml
embed_dim: 128           # 64 for Compact, 192 for Medium-B, 256 for Large
boundary_embed_dim: 64   # scale with embed_dim / 2 is a safe heuristic
```

---

## 7. Two-Phase Training Strategy

### Phase 1 — Deterministic pre-training

Config: `config/ursa_wofscast_crossformer_wrf_det.yml`

Train the backbone to minimise MSE + spectral loss. No noise injection. This establishes a well-calibrated mean-state forecast that the noise layers can then diversify.

```yaml
model:
  noise_injection:
    activate: False

trainer:
  type: 'multi-step-wrf'

loss:
  training_loss: 'mse'
  use_spectral_loss: True
  spectral_lambda_reg: 0.05
  spectral_wavenum_init: 10
```

### Phase 2 — Ensemble fine-tuning

Config: `config/ursa_wofscast_crossformer_wrf_ensemble.yml`

Load Phase-1 weights and fine-tune the noise injection layers using KCRPS. Optionally freeze the base backbone to prevent catastrophic forgetting.

```yaml
model:
  noise_injection:
    activate: True
    freeze_base_model_weights: True   # True = only noise layers train
                                      # False = full model trains (slower convergence)

trainer:
  type: 'multi-step-wrf-ensemble'
  load_weights: True                  # path to Phase-1 checkpoint
  ensemble_size: 8                    # number of members per forward pass

loss:
  training_loss: 'KCRPS'
  use_spectral_loss: False            # KCRPS already penalises spread calibration
```

**Weight transfer**: Phase-1 → Phase-2 works with `strict=False` (noise layers are new). All backbone parameter names are identical between phases so the transfer is complete for matching `dim` configs.

---

## 8. Loss Configuration

### 8.1 Variable weights

`use_variable_weights: True` routes the loss through `VariableTotalLoss2D`, which applies per-variable (and per-level) scaling before computing the final loss. This is also required to activate spectral loss.

Upper-air variables must specify a **list of length `data.levels`** (one weight per vertical eta level, ordered surface → model-top). Surface and diagnostic variables can be a scalar.

```yaml
loss:
  use_variable_weights: True
  variable_weights:
    # 17 weights for 17 eta levels (surface → model top)
    T:      [1.6, 1.5, 1.4, 1.3, 1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6]
    QVAPOR: [2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
    U:      [1.4, 1.3, 1.3, 1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.7, 0.7]
    V:      [1.3, 1.2, 1.2, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.7, 0.7]
    W:      [0.5, 0.7, 0.9, 1.1, 1.3, 1.4, 1.5, 1.6, 1.6, 1.5, 1.4, 1.3, 1.1, 0.9, 0.7, 0.6, 0.5]
    GEOPOT: [1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    # surface / diagnostic — scalar is fine
    T2: 1.2
    Q2: 1.5
    U10: 1.1
    V10: 1.1
    COMPOSITE_REFL_10CM: 2.0
    RAIN_AMOUNT: 2.0
    UP_HELI_MAX: 3.0
    W_UP_MAX: 3.0
    WSPD80: 1.5
    LWP: 3.5
```

**Rationale for the default weights**:
- `T`, `QVAPOR`: highest near the surface (boundary layer and storm inflow); decay sharply above mid-troposphere where forecast skill is lower and physical impact is smaller.
- `W`: peaks at mid-troposphere (levels 7–9, ~500–300 hPa) where deep convective updrafts are most intense. Near-zero at surface and model top where W is constrained to be small.
- `COMPOSITE_REFL_10CM`, `UP_HELI_MAX`, `W_UP_MAX`: triple-weighted — these are the primary convective verification metrics for WoFS.

> **Important**: passing a scalar for an upper-air variable causes a `TypeError` in the loss parser (it calls `len()` on the weight to build a per-level tensor). Always use a 17-element list for `variables` entries.

### 8.2 Spectral loss

Penalises under-representation of fine-scale power (blurring artifact from MSE). Only active during Phase-1; disabled in Phase-2 (KCRPS serves a similar function).

```yaml
loss:
  use_spectral_loss: True
  spectral_wavenum_init: 10   # penalise wavenumbers >= 10
                               # at 300px / 3km: wavenumber 10 → 90km features
                               # convective cells (10-50km) are wavenumbers 18-90
  spectral_lambda_reg: 0.05   # start conservative; increase to 0.1 when stable
```

> `use_spectral_loss` requires `use_variable_weights: True` (or `use_latitude_weights: True`) to have any effect; it is gated inside `VariableTotalLoss2D`.

---

## 9. Launch Scripts

### Interactive training (single node, debug)

```bash
cd /home/Zhanxiang.Hua/miles-credit-wofs
bash scripts/ursa_interactive_forecast.sh
```

The script points to `config/ursa_wofscast_crossformer_wrf_det.yml` and calls  
`applications/train_wrf_wofs_multi_ensemble.py`.

### Batch training (Slurm)

```bash
# Phase-1 deterministic
sbatch scripts/ursa_wofscast.sh  config/ursa_wofscast_crossformer_wrf_det.yml

# Phase-2 ensemble
sbatch scripts/ursa_wofscast.sh  config/ursa_wofscast_crossformer_wrf_ensemble.yml
```

### Rollout / evaluation

```bash
bash scripts/ursa_interactive_rollout.sh
```

Edit `CONFIG` inside the script to point to the desired checkpoint config.

---

*This README covers the `CrossFormerWRFEnsemble` model as implemented in `credit/models/wxformer/crossformer_wrf_ensemble.py`. For the original CrossFormer backbone see `credit/models/crossformer.py`.*
