# Data Assimilation Increment Learning — Current Implementation Status

## Goal (unchanged)

Learn a mapping from **REFL_10CM innovation** to **QRAIN/QNRAIN increment**:

- Input conditioning: `ΔREFL_10CM = REFL_10CM(t1) - REFL_10CM(t0)` (proxy innovation)
- Target increment: `ΔQ = Q(t1) - Q(t0)` in normalized space
- Inference concept: `Q_corrected = Q_background + ΔQ_pred`

This is currently implemented using **Approach 1 (Temporal Self-Supervision)**.

---

## Current Active Config

Source: `config/wofs_credit_wrf_da_increment.yml`

### Data

- Prognostic variables: `QRAIN`, `QNRAIN`
- Context upper-air variables: `T, QVAPOR, U, V, W, GEOPOT, REFL_10CM`
- Observation variable (innovation source): `REFL_10CM`
- Dynamic forcing: 10 channels (cos/sin lat/lon, julian day, local time, solar zenith, insolation)
- `history_len: 1`, `forecast_len: 0` (single-step DA increment learning)
- Levels: 17

### Model

- `type: wrf`
- Interior channels: 2 prognostic + 129 input-only channels
- Outside channels: 1 (`REFL_10CM` innovation)
- Patch size: `4x4`, depth 24, heads 8, window 7

### Trainer

- `type: standard-wrf`
- `mode: none` (single-process training by default)
- `residual_prediction: false` (model output is increment directly)
- `skip_validation: true`
- `train_batch_size: 16`

---

## Architecture Mapping (Current)

The current DA setup uses the existing WRF transformer with dual-path conditioning:

```text
┌─────────────────────────────────────────────┐      ┌──────────────────────────────────┐
│ Interior path (background state at t0)      │      │ Boundary path (innovation)       │
│                                             │      │                                  │
│ Prognostic: QRAIN, QNRAIN (2 x 17 levels)  │      │ ΔREFL_10CM = norm(REFL_t1)       │
│ Context: T,QVAPOR,U,V,W,GEOPOT,REFL_10CM   │      │             - norm(REFL_t0)      │
│ Dynamic forcing: 10 channels                │      │ (1 x 17 levels)                  │
└──────────────────────────┬──────────────────┘      └───────────────┬──────────────────┘
           │                                           │
           ▼                                           ▼
         Interior CubeEmbedding                      Boundary CubeEmbedding
           │                                           │
           └─────────────── fusion (add + FiLM) ───────┘
               │
               ▼
             UTransformer / SwinV2 trunk
               │
               ▼
             Output head: ΔQRAIN, ΔQNRAIN (34 channels)
```

### Channel Accounting

- Interior prognostic channels: `2 x 17 = 34`
- Interior input-only channels: `7 x 17 + 10 = 129`
- Interior total effective input channels: `34 + 129 = 163`
- Boundary channels: `1 x 17 = 17`
- Output channels: `2 x 17 = 34`

### Tensor Roles in Current Dataset

- `x`: prognostic background state at `t0` (normalized)
- `x_forcing_static`: flattened 3D context + dynamic forcing
- `x_boundary`: innovation tensor (`norm(t1) - norm(t0)` for `REFL_10CM`)
- `y`: increment target (`norm(t1) - norm(t0)` for `QRAIN/QNRAIN`)

---

## Implemented Files

### 1) Dataset

- File: `credit/datasets/wrf_wofs_da_increment.py`
- Class: `WoFSDAIncrementDataset`

Implemented behavior:

1. Reads consecutive timesteps `(t0, t1)` from each WoFS file.
2. Builds prognostic input `x` from normalized `QRAIN/QNRAIN` at `t0`.
3. Builds target `y` as normalized increment: `norm(t1) - norm(t0)`.
4. Builds boundary input `x_boundary` as normalized innovation: `norm(REFL_t1) - norm(REFL_t0)`.
5. Builds `x_forcing_static` from flattened context-3D channels + dynamic forcing.

Important fix already applied:

- Level-wise mean/std broadcasting now handles shapes like `(17,)` stats against `(17, 300, 300)` fields.

---

### 2) Training App

- File: `applications/train_wrf_wofs_da.py`

Implemented behavior:

- Uses `WoFSDAIncrementDataset` directly (no `NormalizeWRF` / `ToTensorWRF` transform pipeline).
- Builds train params from `context_upper_air_variables` and `observation_variables`.
- Skips validation dataset/loader creation when `trainer.skip_validation: true`.
- Ensures process-group cleanup in `finally` for distributed mode.

---

### 3) Trainer Registry

- File: `credit/trainers/__init__.py`

Current supported WRF single-step key:

- `standard-wrf`

Note:

- DA config now uses `standard-wrf` (canonical key).

---

## Metric Interpretation (Current Behavior)

Training metrics come from `LatWeightedMetrics` on **normalized tensors**:

- `train_acc` is an anomaly-correlation style metric over predicted vs target increment fields.
- It is computed **before any inverse transform**.
- In this DA setup, `train_acc` is averaged across 34 channels (`2 variables x 17 levels`).

So low early-epoch `train_acc` near zero can occur even when loss decreases.

---

## What Changed vs Original Plan

1. Trainer key standardized to `standard-wrf`.
2. Runtime currently uses `trainer.mode: none` in config (not DDP by default).
3. Validation is intentionally skipped (`skip_validation: true`) and app now handles that cleanly.
4. No changes required in `credit/transforms/transforms_wrf.py` for this path, since dataset emits ready-to-train tensors.
5. Added robustness fix for level-stat broadcasting in dataset normalization.

---

## Known Limitations / Next Steps

1. Domain gap remains: temporal tendency proxy is not identical to true obs-analysis increment statistics.
2. Ensemble-based Approach 2 is not yet implemented.
3. Validation is currently disabled; when enabling it, ensure valid date range contains matching files.
4. Add DA-specific diagnostics for interpretability:
   - per-variable/per-level ACC
   - increment variance by level
   - innovation-to-increment sign consistency checks

### Planned Next Extension — Approach 2 (Ensemble-Mean Training)

Keep this as the primary follow-up after stabilizing Approach 1.

- Group member files by case-time key: `wofs_YYYYMMDD_HHMM_memNN`.
- At fixed time `t`, compute ensemble-mean background for `REFL_10CM`, `QRAIN`, `QNRAIN`.
- Build training pairs as perturbations from ensemble mean:
  - innovation: `REFL_member(t) - REFL_ens_mean(t)`
  - target increment: `Q_member(t) - Q_ens_mean(t)` for `QRAIN/QNRAIN`
- Reuse the same model/trainer interface (`standard-wrf`) so only dataset logic changes.
- Compare Approach 1 vs Approach 2 with the same metrics and per-level ACC diagnostics.

---

## Current Smoke-Test Command

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  applications/train_wrf_wofs_da.py \
  -c config/wofs_credit_wrf_da_increment.yml
```

If running with distributed mode, switch `trainer.mode` and launch with matching `nproc-per-node`.
