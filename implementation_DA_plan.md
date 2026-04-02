# Data Assimilation Increment Learning — Implementation Record

## Objective

Learn QRAIN/QNRAIN increments from REFL_10CM innovations using the existing
WoFS/CREDIT framework. This enables neural-network-based data assimilation
where radar reflectivity observations correct hydrometeor state variables.

## Approach

**Approach 1 — Temporal Self-Supervision** (implemented here):
Use consecutive forecast timesteps (t₀, t₁) to construct training pairs:
- Innovation = normalized(REFL_10CM_t₁) − normalized(REFL_10CM_t₀)
- Increment  = normalized(QRAIN_t₁) − normalized(QRAIN_t₀), same for QNRAIN

**Approach 2 — Ensemble Cross-Member** (planned, separate script):
Use ensemble member departures from ensemble mean at each timestep.

---

## Architecture Mapping

The WRFTransformer two-branch architecture maps to the DA problem:

```
Interior Encoder (background state)     Boundary Encoder (innovation)
  T, QVAPOR, U, V, W, GEOPOT,             REFL_10CM(t₁) − REFL_10CM(t₀)
  QRAIN, QNRAIN, REFL_10CM                 (1 var × 17 levels = 17 ch)
  + dynamic forcing
  (2×17 prognostic + 129 context)
         │ CubeEmbedding                          │ CubeEmbedding
         ▼                                        ▼
     x_embed ◄──── add ──── FiLM(time) ──── x_obs_embed
         │
    UTransformer (SwinV2)
         │
    Output Head → Δ QRAIN, Δ QNRAIN (2 × 17 = 34 channels)
```

| Channel group          | Input `x` | Output `y` | Count |
|------------------------|-----------|------------|-------|
| Prognostic (QRAIN, QNRAIN × 17 lev) | ✅ | ✅ | 34  |
| Context (T,QVAPOR,U,V,W,GEOPOT,REFL_10CM × 17 lev) | ✅ input-only | ❌ | 119 |
| Dynamic forcing (10 vars) | ✅ input-only | ❌ | 10  |
| **Total input**        |           |            | **163** |
| **Total output**       |           |            | **34**  |
| Boundary (REFL_10CM innovation × 17 lev) | ✅ | — | 17  |

---

## Files Changed

### New Files

#### `config/wofs_credit_wrf_da_increment.yml`

DA-specific config. Key differences from forecasting config:
- `variables: ['QRAIN', 'QNRAIN']` — prognostic only
- `context_upper_air_variables: ['T', 'QVAPOR', 'U', 'V', 'W', 'GEOPOT', 'REFL_10CM']` — NEW key
- `observation_variables: ['REFL_10CM']` — NEW key
- `surface_variables: []`, `diagnostic_variables: []`
- `boundary.variables: ['REFL_10CM']`
- Model: `channels=2, input_only_channels=129, output_only_channels=0`
- Boundary model: `channels=1, surface_channels=0`
- `residual_prediction: False` (output IS the increment)

#### `credit/datasets/wrf_wofs_da_increment.py`

New dataset class `WoFSDAIncrementDataset`. Key design decisions:
- **Self-normalizing**: Loads `mean.nc`/`std.nc` internally and normalizes all
  variables in `__getitem__`. Bypasses NormalizeWRF/ToTensorWRF transforms entirely.
- **Innovation construction**: Computes `norm(REFL_10CM_t1) − norm(REFL_10CM_t0)`
  per sample and packages it as `x_boundary`.
- **Increment target**: Computes `norm(prog_t1) − norm(prog_t0)` and packages
  as `y` (the target the trainer compares against `y_pred`).
- **Context flattening**: 3D context variables (7 vars × 17 levels) are
  flattened to 119 channels and concatenated with 10 dynamic forcing channels
  → 129 total `x_forcing_static` channels.
- **Returns tensor dict directly** with same keys the single-step trainer expects:
  `x`, `x_forcing_static`, `x_boundary`, `y`, `x_time_encode`.
- No `x_surf`, `y_surf`, `y_diag`, `x_surf_boundary` (not needed for this task).

#### `applications/train_wrf_wofs_da.py`

Training application mirroring `train_wrf_wofs.py`. Differences:
- Imports `WoFSDAIncrementDataset` instead of `WoFSSingleStepDataset`
- `_build_params` passes `varname_context_upper_air` and `observation_variables`
- No transforms loaded (dataset self-normalizes)
- Uses the same `trainerWRF.py` single-step trainer — zero trainer modifications

### Unchanged Files

- `credit/trainers/trainerWRF.py` — No changes needed. The trainer checks for
  optional keys (`x_surf`, `y_diag`, etc.) and gracefully skips them.
- `credit/transforms/transforms_wrf.py` — No changes needed. DA dataset
  bypasses transforms entirely.
- `credit/models/swin_wrf.py` — No changes needed. The model is parameterized
  by config; channel counts flow through automatically.

---

## Normalization Strategy

All normalization uses the existing `mean.nc` / `std.nc` files.

| Data element | Normalization | Rationale |
|---|---|---|
| Prognostic at t₀ (input) | `(x − μ) / σ` | Standard z-score |
| Context at t₀ (input) | `(x − μ) / σ` | Standard z-score |
| Innovation (boundary) | `norm(t₁) − norm(t₀) = Δx / σ` | Difference of normalized values |
| Increment (target) | `norm(t₁) − norm(t₀) = Δx / σ` | Difference of normalized values |

The increment in normalized space is `Δx / σ`, which is a natural normalization
for differences (zero mean, unit-like variance).

---

## Verification Plan

```bash
# Smoke test
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  applications/train_wrf_wofs_da.py \
  -c config/wofs_credit_wrf_da_increment.yml

# Full DDP training
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  applications/train_wrf_wofs_da.py \
  -c config/wofs_credit_wrf_da_increment.yml
```

### Sanity Checks
- Initial loss ~ O(1) (properly normalized targets)
- Innovation and increment distributions approximately zero-mean
- Gradients flow through both interior and boundary encoders
- Loss decreases over training epochs
