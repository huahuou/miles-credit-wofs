# Multi-Modal Masked Autoencoder for WoFS Data Assimilation
## Detailed Implementation Plan

---

## 1. Motivation and Conceptual Shift

The existing DA approach (Approach 1 in `implementation_DA_plan_detail.md`) learns a
direct increment mapping: `ΔREFL_10CM → ΔQRAIN/QNRAIN` via the WRF Transformer with a
dual-path (interior + boundary) architecture.

This new approach adapts the **EarthNet multi-modal MAE** paradigm to WRF/WoFS. Instead
of learning an explicit increment, we train a masked autoencoder that learns the **joint
distribution** of all WoFS physical modalities. At DA inference time, we supply the
observed reflectivity as a "conditioning" modality and allow the decoder to reconstruct
the precipitation state — effectively a physics-aware analysis step.

### Key conceptual advantages over Approach 1

| Property | Approach 1 (Increment) | Approach 2 (MAE-DA) |
|---|---|---|
| Training signal | Temporal tendency proxy | Joint multi-modal reconstruction |
| Observation role | Difference innovation | Conditioning modality |
| DA inference | Add predicted increment | Decode precip from obs REFL |
| Architecture | SwinV2 (fixed windows) | ViT (random masking compatible) |
| Missing obs | Not modeled | N/A — WoFS grids are always complete |
| Extendibility | Single obs type | Any-to-any modality |

### Why no VAE pre-training stage

EarthNet's per-modality VAE exists for two reasons that do not apply to WoFS:

1. **NaN holes in satellite observations** — orbital gaps and cloud occlusion leave
   entire spatial patches as `NaN`. The VAE + KL regularization bounds token embeddings
   even when large fractions of a modality are missing. WoFS output is a complete dense
   NWP grid — no NaN by construction.
2. **Multi-sensor resolution mismatch** — different EarthNet modalities arrive at
   different native resolutions, making direct joint tokenization difficult. All WoFS
   modalities share the same 300 × 300 Arakawa-C grid.

For WoFS, inputs are pre-normalized to `N(0,1)` per variable/level via the existing
`mean.nc`/`std.nc` statistics, so a direct modality-specific linear projection (Conv2d
with `kernel_size == stride == patch_size`) produces stable token embeddings without
any VAE bottleneck. This collapses the original three-stage pipeline to a single
end-to-end MAE training run.

---

## 2. Modality Definition for WoFS

Each group of physically related variables becomes a "modality" with its own input/output
adapter pair, following EarthNet's adapter pattern.

| Modality key | Variables | Raw shape | Channels (C×17 levels) |
|---|---|---|---|
| `background` | T, QVAPOR, U, V, W, GEOPOT | `(6, 17, H, W)` | 102 |
| `precip` | QRAIN, QNRAIN, QHAIL, QNHAIL, QGRAUP, QNGRAUPEL, QSNOW, QNSNOW | `(8, 17, H, W)` | 136 |
| `reflectivity` | REFL_10CM | `(1, 17, H, W)` | 17 |
| `surface` | XLAND, HGT | `(2, H, W)` | 2 (no level dim) |
| `forcing` | cos/sin lat/lon, julian day, local time, solar zenith, insolation | `(10, H, W)` | 10 (no level dim) |

For the model, levels are folded into channels:
- `background`: `(102, H, W)`, `precip`: `(136, H, W)`, `reflectivity`: `(17, H, W)`
- `surface`: `(2, H, W)`, `forcing`: `(10, H, W)`

`surface` and `forcing` tokens are **never masked** during training (always provided as
conditioning). `background`, `precip`, and `reflectivity` participate in Dirichlet masking.

---

## 3. Architecture Overview

```
Single Training Stage — End-to-end Multi-Modal MAE
┌──────────────────────────────────────────────────────────────────────────────┐
│ WoFSMultiModalMAE                                                            │
│                                                                              │
│  background           precip          reflectivity     surface   forcing    │
│  (102, 304, 304)      (136, 304, 304)  (17, 304, 304)  (2,H,W)  (10,H,W)   │
│        │                    │                 │            │         │       │
│        ▼                    ▼                 ▼            ▼         ▼       │
│  WoFSInputAdapter    WoFSInputAdapter  WoFSInputAdapter  (pooled to 19×19)  │
│  Conv2d(102→768,     Conv2d(136→768,   Conv2d(17→768,                       │
│   k=8,s=8)           k=8,s=8)          k=8,s=8)                             │
│  + sin-cos pos emb   + sin-cos pos emb + sin-cos pos emb                    │
│  + modality emb e_bg + modality emb e_p + modality emb e_r                  │
│  → (B,1444,768)      → (B,1444,768)   → (B,1444,768)  → (B,361,768) each   │
│                                                                              │
│         └─────────────────────── concat ────────────────────────┘           │
│                         all tokens + 2 global tokens                        │
│                                                                              │
│   Dirichlet masking on {background, precip, reflectivity}                   │
│   → keep num_encoded_tokens=256 visible maskable tokens                     │
│   → always keep all surface + forcing tokens (~722 total)                   │
│   → backbone sees ~978 tokens total                                         │
│                                                                              │
│                      ┌─────────────────────────────┐                        │
│                      │  Shared ViT Encoder         │                        │
│                      │  depth=12, heads=12, dim=768│                        │
│                      └──────────────┬──────────────┘                        │
│                                     ▼                                        │
│             ┌───────────────────────┴──────────────────────┐                │
│             │  Per-modality cross-attention Output Adapters │                │
│             │  (WoFSOutputAdapter — 6 decoder blocks each)  │                │
│             └─────────────────────────────────────────────┘                 │
│                                                                              │
│  Loss: per-modality weighted MSE on masked tokens only                      │
└──────────────────────────────────────────────────────────────────────────────┘

DA Inference (no training)
┌─────────────────────────────────────────────────────────────────────┐
│  Given: background (t0), observed REFL_10CM (from radar or t1 proxy)│
│                                                                      │
│  Provide:   background tokens → unmasked                            │
│             surface + forcing  → unmasked                           │
│             refl tokens        → replace with obs REFL_10CM tokens  │
│             precip tokens      → fully masked (to be decoded)       │
│                                                                      │
│  Run MAE forward → WoFSOutputAdapter["precip"] → Q_analysis         │
│                                                                      │
│  Optional blend: Q_analysis = α * Q_MAE + (1-α) * Q_background     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tokenization Design

### Direct patch-linear projection

Each modality is tokenized by a single `Conv2d` layer with `kernel_size == stride == P`
(i.e., no overlap, no VAE bottleneck):

```
token_{i,j} = LayerNorm(
    W_m · flatten(X_m[iP:(i+1)P, jP:(j+1)P])   # Conv2d — modality-specific
    + E^pos_{i,j}                                 # 2D sin-cos — shared grid
    + e_m                                         # learnable scalar modality emb
)
```

This is identical to the ViT patch embedding, extended with a per-modality projection
matrix `W_m ∈ R^{(C_m · P²) × d}` and a broadcast modality vector `e_m ∈ R^d`.

### Spatial patch size

WoFS domain: **300 × 300**. Padding to **304 × 304** makes it exactly divisible by 8.

Patch size **8 × 8** (24 km at 3 km grid spacing) balances convective-scale spatial
resolution against backbone memory:

| Patch size | Token spatial scale | Tokens/modality (304²) | Backbone total (masked+unmasked) |
|---|---|---|---|
| 4 × 4 | 12 km | 76² = 5,776 | 256 + 2×5,776 = **11,808** — too heavy |
| **8 × 8** | **24 km** | **38² = 1,444** | **256 + 2×361 = ~978** — recommended |
| 16 × 16 | 48 km | 19² = 361 | 256 + 2×361 = 978 — but loses convective detail |

The existing `WRFTransformer` in this codebase uses 4×4 patches for sub-25 km convective
resolution; 8×8 is a reasonable compromise for the MAE backbone where the output adapter
can recover fine-scale details via cross-attention.

### Surface and forcing tokenization

`surface` (2 ch) and `forcing` (10 ch) are slowly-varying fields. Rather than generating
1,444 full-resolution tokens each (which would dominate the always-visible backbone
cost), they are pooled to a **19 × 19** grid (16×16 stride over 304×304) before the
linear projection, yielding **361 tokens each**. The backbone always sees these
`2 × 361 = 722` unmasked tokens plus 256 masked tokens = **~978 total**, manageable
on A100 with batch size 4–8.

### Position embeddings

Use 2D sin-cos positional embeddings (same as EarthNet). Each modality also gets a
learnable **modality embedding** `e_m` (a single vector broadcast over all its tokens)
so the backbone can distinguish modality origin without seeing channel values.

---

## 5. Files to Create

### 5.1 Model files

#### `credit/models/wofs_mae_adapters.py`

Contains two building-block classes (no VAE):

```python
class WoFSInputAdapter(nn.Module):
    """
    2D patched input adapter for a single WoFS modality.
    Converts (B, C, H, W) → (B, N_tok, embed_dim) tokens via direct patch projection.

    Operation:
        1. Pad input to (H_pad, W_pad) divisible by patch_size.
        2. Conv2d(C, embed_dim, kernel=patch_size, stride=patch_size)  — modality-specific W_m
        3. Rearrange: (B, embed_dim, N_H, N_W) → (B, N_H*N_W, embed_dim)
        4. Add 2D sin-cos positional embedding (fixed, interpolatable)
        5. Add learnable modality embedding e_m (broadcast over all tokens)
        6. LayerNorm

    Key params:
        num_channels  — C (levels folded in, e.g. 102 for background)
        patch_size    — 8 for physics modalities; ignored for surface/forcing
                        (which use stride=16 pooling before this adapter)
        embed_dim     — 768 (shared token dim)
        image_size    — (304, 304) after padding
        sincos_pos_emb — True (fixed; interpolatable to other resolutions)
    """

class WoFSOutputAdapter(nn.Module):
    """
    Cross-attention output adapter for a single WoFS modality.
    Converts encoder output tokens (B, N_vis, embed_dim) → (B, C, H, W) reconstruction.

    Operation:
        1. proj_context: Linear(embed_dim → decoder_dim) on all encoder tokens
        2. Build query tokens: positional emb + modality emb (one query per output patch)
        3. Fill masked query positions with a learnable mask_token
        4. Cross-attention: queries attend to context (all encoder tokens per modality)
        5. depth self-attention blocks on query side
        6. out_proj: Linear(decoder_dim → C * P_H * P_W)
        7. Rearrange patches → (B, C, H_pad, W_pad); crop to (B, C, H, W)
    
    Key params:
        num_channels  — C
        patch_size    — 8
        embed_dim     — 768 (encoder token dim)
        decoder_dim   — 384
        decoder_depth — 6 (cross-attn block + depth self-attn blocks)
        num_heads     — 8
        image_size    — (304, 304)
        context_tasks — list of all modality keys (for per-modality context embeddings)
    """
```

#### `credit/models/wofs_mae.py`

```python
class WoFSMultiModalMAE(nn.Module):
    """
    Multi-modal masked autoencoder for WoFS data assimilation.
    Single end-to-end training stage — no VAE pre-training required.
    
    Constructor params (also settable from YAML):
        modality_channels: dict[str, int]   — channel count per modality
            default: {background: 102, precip: 136, reflectivity: 17,
                      surface: 2, forcing: 10}
        non_masked_modalities: list[str]    — always visible modalities
            default: [surface, forcing]
        patch_size: int                     — 8 (physics modalities)
        surface_forcing_stride: int         — 16 (pooling stride for surface/forcing)
        image_size: tuple[int,int]          — (304, 304)
        embed_dim: int                      — 768
        depth: int                          — 12
        num_heads: int                      — 12
        mlp_ratio: float                    — 4.0
        decoder_dim: int                    — 384
        decoder_depth: int                  — 6
        decoder_num_heads: int              — 8
        num_global_tokens: int              — 2
        num_encoded_tokens: int             — 256
        drop_path_rate: float               — 0.1
    
    Forward signature:
        forward(modality_dict: dict[str, Tensor],
                mask_inputs: bool = True,
                num_encoded_tokens: int | None = None,
                alphas: float = 1.0)
        → reconstruction_dict: dict[str, Tensor],
          task_masks: dict[str, Tensor]
    
    DA inference helper:
        assimilate(background: Tensor,         # (B, 102, H, W)
                   obs_refl: Tensor,           # (B, 17, H, W)
                   surface: Tensor,            # (B, 2, H, W)
                   forcing: Tensor,            # (B, 10, H, W)
                   blend_alpha: float = 1.0)   # 1.0 = pure obs REFL, 0.0 = pure model
        → precip_analysis: Tensor              # (B, 136, H, W) in normalized space
    """
```

### 5.2 Dataset file

#### `credit/datasets/wrf_wofs_mae.py`

```python
class WoFSMAEDataset(torch.utils.data.Dataset):
    """
    Returns per-modality normalized tensors for MAE training.
    Reuses normalization infrastructure from WoFSDAIncrementDataset.
    
    Config keys consumed (under data:):
        mae_modalities       — list of modality keys to return
        background_vars      — ['T','QVAPOR','U','V','W','GEOPOT']
        precip_vars          — ['QRAIN','QNRAIN',...]
        reflectivity_vars    — ['REFL_10CM']
        surface_vars         — ['XLAND','HGT']
        dynamic_forcing_vars — [cos_latitude, ...]
        levels               — 17
        mae_image_size       — [304, 304]  (padded from 300×300)
        mae_include_t1_refl  — bool, if True load t1 REFL as the obs modality
                               instead of t0 (for Approach-1-style training proxy)
    
    __getitem__ returns:
        {
          'background':    (102, 304, 304) float32 normalized
          'precip':        (136, 304, 304) float32 normalized
          'reflectivity':  (17, 304, 304)  float32 normalized
          'surface':       (2, 304, 304)   float32 normalized
          'forcing':       (10, 304, 304)  float32 normalized
          'time_encode':   (4,)            float32
          'index':         int
        }
        
        When mae_include_t1_refl=True, 'reflectivity' key returns t1 obs REFL
        and 'reflectivity_background' key returns t0 model REFL (both channels=17).
    
    Internal normalization:
        Same per-variable, per-level mean/std from mean.nc/std.nc.
        Same concentration log-transform for precip vars.
        Pads from (300,300) to (304,304) via torch.nn.functional.pad.
    """
```

### 5.3 Trainer files

#### `credit/trainers/trainerWRF_mae.py`

```python
class TrainerMAE(BaseTrainer):
    """
    Multi-modal MAE trainer.
    
    Config path: trainer.type = 'standard-wrf-mae'
    
    Required trainer config keys:
        num_encoded_tokens: int    — 256
        masking_alphas: float      — 1.0 (Dirichlet concentration)
        masked_loss_only: bool     — True (compute loss only on masked tokens)
        loss_weights:              — per-modality loss weights
            background: 0.5
            precip: 2.0
            reflectivity: 1.0
        skip_validation: bool      — False
    
    train_one_epoch():
        batch = {modality: (B, C, H, W), ...}
        recon_dict, task_masks = model(batch, mask_inputs=True)
        for mod in masked_modalities:
            loss += weight[mod] * masked_mse(recon_dict[mod], batch[mod],
                                              task_masks[mod])
        
    validate():
        run with mask_inputs=False for full reconstruction RMSE per modality
        also compute reflectivity → precip reconstruction RMSE (DA proxy metric)
    """
```

### 5.4 Application files

#### `applications/train_wrf_wofs_mae.py`

End-to-end MAE training. Structure mirrors `applications/train_wrf_wofs_da.py`.

```python
# Build modality_channels dict from config
modality_channels = {
    "background":    len(conf["data"]["background_vars"]) * levels,
    "precip":        len(conf["data"]["precip_vars"]) * levels,
    "reflectivity":  len(conf["data"]["reflectivity_vars"]) * levels,
    "surface":       len(conf["data"]["surface_vars"]),
    "forcing":       len(conf["data"]["dynamic_forcing_variables"]),
}

# Load model — no VAE checkpoint loading needed
from credit.models.wofs_mae import WoFSMultiModalMAE
model = WoFSMultiModalMAE(modality_channels=modality_channels, **conf["model"])

# Dataset / trainer
from credit.datasets.wrf_wofs_mae import WoFSMAEDataset
from credit.trainers.trainerWRF_mae import TrainerMAE
```

#### `applications/rollout_wrf_wofs_mae_da.py`

DA inference application. Loads a trained `WoFSMultiModalMAE` checkpoint and runs
`model.assimilate()` at each analysis time.

```python
# For each (background_zarr, obs_refl_zarr) pair:
#   1. Load background state → build per-modality tensors
#   2. Load observed REFL (radar or t1 proxy) → build reflectivity tensor
#   3. model.assimilate(background, obs_refl, surface, forcing, blend_alpha=alpha)
#   4. Inverse-normalize precip_analysis → physical units
#   5. Save to output_dir/analysis_YYYYMMDD_HHMM.zarr
```

---

## 6. Registry Updates

### `credit/models/__init__.py`

Add one entry to `model_types`:

```python
from credit.models.wofs_mae import WoFSMultiModalMAE

model_types["wofs-mae"] = (WoFSMultiModalMAE, "Multi-modal MAE for WoFS DA")
```

### `credit/trainers/__init__.py`

Add one entry to `trainer_types`:

```python
from credit.trainers.trainerWRF_mae import TrainerMAE

trainer_types["standard-wrf-mae"] = (TrainerMAE, "Multi-modal MAE trainer for WoFS DA")
```

---

## 7. Config Files

### 7.1 `config/wofs_mae.yml` — MAE training (single stage)

```yaml
save_loc: '/scratch/credit_runs/wofs_mae_experiment_01'
seed: 1000

data:
  scaler_type: 'std-wrf'
  precip_vars: ['QRAIN','QNRAIN','QHAIL','QNHAIL','QGRAUP','QNGRAUPEL','QSNOW','QNSNOW']
  background_vars: ['T','QVAPOR','U','V','W','GEOPOT']
  reflectivity_vars: ['REFL_10CM']
  surface_vars: ['XLAND','HGT']
  dynamic_forcing_variables: ['cos_latitude','sin_latitude','cos_longitude','sin_longitude',
                               'cos_julian_day','sin_julian_day','cos_local_time','sin_local_time',
                               'cos_solar_zenith_angle','insolation']
  mae_modalities: ['background','precip','reflectivity','surface','forcing']
  mae_image_size: [304, 304]
  mae_include_t1_refl: true      # use t1 REFL as obs proxy during training
  levels: 17
  save_loc: '/work2/zhanxianghua/wofs_preprocess_to_credit_0413/cases/wofs_*.zarr.zip'
  mean_path: '/work2/zhanxianghua/wofs_preprocess_to_credit_0413/stats/mean.nc'
  std_path:  '/work2/zhanxianghua/wofs_preprocess_to_credit_0413/stats/std.nc'
  concentration_params_json: '/work2/zhanxianghua/wofs_preprocess_to_credit_0413/stats/concentration_tuning.json'
  train_years: [2020]
  valid_years: [2021]
  train_date_range: ['20200301','20200830']
  valid_date_range: ['20210405','20210407']
  history_len: 1
  forecast_len: 0

model:
  type: 'wofs-mae'
  patch_size: 8
  image_size: [304, 304]
  embed_dim: 768
  depth: 12
  num_heads: 12
  mlp_ratio: 4.0
  decoder_dim: 384
  decoder_depth: 6
  decoder_num_heads: 8
  num_global_tokens: 2
  num_encoded_tokens: 256
  drop_path_rate: 0.1
  non_masked_modalities: ['surface', 'forcing']
  surface_forcing_stride: 16   # pool surface/forcing to 19×19 (361 tokens) before projection

trainer:
  type: 'standard-wrf-mae'
  epochs: 100
  train_batch_size: 4
  valid_batch_size: 2
  learning_rate: 1.0e-4
  weight_decay: 1.0e-6
  scheduler: 'cosine'
  warmup_epochs: 10
  num_encoded_tokens: 256
  masking_alphas: 1.0
  masked_loss_only: true
  loss_weights:
    background: 0.5
    precip: 2.0
    reflectivity: 1.0
  skip_validation: false
  save_every_n_epochs: 10
  mode: 'ddp'             # use DDP for MAE (larger model)
```

### 7.2 `config/wofs_mae_da.yml` — DA inference

```yaml
save_loc: '/scratch/credit_runs/wofs_mae_da_analysis'

data:
  # same variable lists as wofs_mae.yml
  mae_image_size: [304, 304]
  levels: 17
  save_loc: '/work2/zhanxianghua/wofs_preprocess_to_credit_0413/cases/wofs_*.zarr.zip'
  mean_path: '/work2/zhanxianghua/wofs_preprocess_to_credit_0413/stats/mean.nc'
  std_path:  '/work2/zhanxianghua/wofs_preprocess_to_credit_0413/stats/std.nc'
  # optional: path to actual radar observations (MRMS or WoFS obs proxy)
  obs_refl_source: 'wofs_t1'   # options: wofs_t1 | mrms | zarr_path

model:
  type: 'wofs-mae'
  ckpt_path: '/scratch/credit_runs/wofs_mae_experiment_01/checkpoint_epoch100.pt'
  # all architecture params inherited from training config via checkpoint

inference:
  blend_alpha: 1.0          # 1.0 = pure obs REFL, 0.0 = pure model REFL
  output_dir: '/scratch/credit_runs/wofs_mae_da_analysis/output'
  date_range: ['20210601', '20210630']
  save_format: 'zarr'       # zarr | netcdf
```

---

## 8. Channel Accounting

### Input adapter patch sizes and token counts

| Modality | C (channels) | Patch/stride | Tokens | Masking | Always-visible |
|---|---|---|---|---|---|
| `background` | 102 | 8×8 | 38²=1,444 | Dirichlet | No |
| `precip` | 136 | 8×8 | 38²=1,444 | Dirichlet | No |
| `reflectivity` | 17 | 8×8 | 38²=1,444 | Dirichlet | No |
| `surface` | 2 | stride 16 pool | 19²=361 | Never | Yes |
| `forcing` | 10 | stride 16 pool | 19²=361 | Never | Yes |
| global tokens | — | — | 2 | Never | Yes |

Each physics-modality token encodes `C × 8 × 8` raw values via a single `Conv2d`
projection (= the patch-linear projection from ViT). No bottleneck, no KL term.

### MAE backbone token budget

With `num_encoded_tokens=256` distributed via Dirichlet across 3 maskable modalities:
- ~85 masked tokens per maskable modality retained
- Surface + forcing + global: 361 + 361 + 2 = **724 always-visible tokens**
- Backbone total: 256 + 724 = **~980 tokens**

At `embed_dim=768`, attention matrix per head is 980² ≈ 960K entries — well within
A100 memory at batch size 4–8. No FlashAttention required, though it can be added.

---

## 9. Training Stages and Commands

### Stage 1: Train multi-modal MAE end-to-end (multi-GPU)
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  applications/train_wrf_wofs_mae.py \
  -c config/wofs_mae.yml
```

### Stage 2: DA inference
```bash
python applications/rollout_wrf_wofs_mae_da.py \
  -c config/wofs_mae_da.yml
```

Single-GPU smoke test (reduced batch and token count):
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  applications/train_wrf_wofs_mae.py \
  -c config/wofs_mae.yml \
  trainer.train_batch_size=1 model.num_encoded_tokens=64
```

---

## 10. Evaluation Metrics

### VAE pre-training (per modality)
- **Reconstruction RMSE** in normalized units per variable/level
- **KL divergence** tracking (should decrease from ~N(0,1) initialization)
- **Peak spatial correlation** of reconstructed vs input patches

### MAE training
- **Masked token MSE** per modality (primary loss, logged per modality separately)
- **Full-field RMSE** per variable/level on validation set (`mask_inputs=False`)
- **Cross-modal DA proxy**: at validation time, zero-mask all `precip` tokens and feed
  background + obs REFL; measure RMSE of reconstructed precip vs. ground truth.
  This is the primary indicator of useful DA skill before running full inference.

### DA inference
- **Innovation consistency**: does `REFL_analysis ≈ REFL_obs`?
- **Precip mass conservation**: column-integrated water should not diverge
- **Sign consistency**: in regions where ΔREFL > 0, does ΔQRAIN > 0 in the analysis?
- **Per-level ACC** of precip increment vs Approach-1 increment (cross-validation)
- **Reflectivity-to-hydrometeor retrieval error** using held-out WoFS members as truth

---

## 11. Key Design Decisions and Alternatives

### Why ViT backbone instead of SwinV2?

Random token masking (as done in MAE) breaks SwinV2's local window assumption because
windows may be entirely empty. The existing `WRFTransformer` (SwinV2) would need
full-sequence attention with masking, losing its efficiency advantage. A plain ViT
with global self-attention handles arbitrary masking naturally.

If memory becomes a bottleneck, consider **FlashAttention** or **window-free SwinV2**
variants as a drop-in backbone replacement.

### Direct patch projection vs. VAE pre-training

EarthNet uses a per-modality VAE to handle NaN-valued satellite observations and
multi-resolution sensor fusion. Neither issue exists in WoFS:
- WoFS output is a complete, co-located, NaN-free NWP grid.
- All modalities are normalized to `N(0,1)` via existing `mean.nc`/`std.nc` stats.

A direct `Conv2d(C, embed_dim, kernel=P, stride=P)` projection is therefore stable
from the start of training, produces tokens of the correct dimension without a KL
bottleneck, and eliminates three separate pre-training runs.

### Levels as channels vs. "time-like" axis

EarthNet uses a true 3D convolution tokenizer that treats time as an explicit axis
`(C, T, H, W)`. For WRF, levels play an analogous role. The approach here folds levels
into channels `(C×L, H, W)` for simplicity — consistent with all existing WRF models
in this codebase and avoids a new 3D tokenizer dependency.

A future extension could introduce a true 3D adapter `(C, 17, H, W) → patches
(C, p_l, p_h, p_w)` analogous to EarthNet's `PatchedInputAdapter3D`, which would
provide better vertical structure learning at the cost of a larger patch vocabulary.

### Observation blending at inference

`blend_alpha` controls how strongly the observation is imposed:
- `alpha = 1.0`: pure observed REFL, maximum DA correction
- `alpha = 0.5`: 50% blending, softer assimilation
- `alpha = 0.0`: pure model REFL, no DA (diagnostic / ablation run)

This is implemented in `WoFSMultiModalMAE.assimilate()` by linearly interpolating the
normalized reflectivity token values before passing to the encoder.

---

## 12. File Summary

| Path | Purpose | New/Edit |
|---|---|---|
| `credit/models/wofs_mae_adapters.py` | `WoFSInputAdapter`, `WoFSOutputAdapter` | New |
| `credit/models/wofs_mae.py` | `WoFSMultiModalMAE` backbone | New |
| `credit/datasets/wrf_wofs_mae.py` | `WoFSMAEDataset` | New |
| `credit/trainers/trainerWRF_mae.py` | `TrainerMAE` | New |
| `applications/train_wrf_wofs_mae.py` | MAE training script | New |
| `applications/rollout_wrf_wofs_mae_da.py` | DA inference script | New |
| `credit/models/__init__.py` | Add `wofs-mae` | Edit |
| `credit/trainers/__init__.py` | Add `standard-wrf-mae` | Edit |
| `config/wofs_mae.yml` | MAE training config (single stage) | New |
| `config/wofs_mae_da.yml` | DA inference config | New |

**Total: 10 files** (6 new Python files, 2 new YAML configs, 2 registry edits)
