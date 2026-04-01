# WRF DataLoader Optimization & Residual Prediction — Walkthrough

## Summary
Two sets of changes across the WoFS training pipeline:
1. **DataLoader performance** — Resolved GPU starvation during DDP training
2. **Residual prediction** — Optional config flag to predict timestep deltas instead of full states

## Single-Step vs Multi-Step

| Aspect | Single-Step | Multi-Step |
|---|---|---|
| **Rollout** | Returns one sample (one input→target pair) | Returns a stacked list of `forecast_len + 1` samples for autoregressive rollout |
| **Trainer** | One forward pass per batch, one `backward()` | Iterates `rollout_len` steps, feeds `y_pred` back as next `x` |
| **`forecast_len`** | Set to 0; loads `[t, t+1]` pairs | Set to 1+; loads `[t, ..., t+N]` |
| **Loss** | Single prediction → single loss | Accumulated loss over multiple steps |

---

## Feature: Residual Prediction

### Motivation
At WRF's fine resolution, consecutive timesteps are ~99% identical. Direct prediction forces the network to reconstruct the entire field from scratch, wasting capacity on the near-constant background. Residual prediction lets the model focus on the ~1% that actually changes between steps.

### How It Works

**Config flag** (`trainer` section):
```yaml
trainer:
  residual_prediction: True   # False = direct prediction (default/original behavior)
```

**Trainer logic** (applied identically in both `trainerWRF.py` and `trainerWRF_multi.py`):

```python
y_pred = self.model(x, x_boundary, x_time_encode)   # model outputs delta

if residual_prediction:
    num_prog = y_pred.shape[1] - varnum_diag         # prognostic channels only
    residual = x[:, :num_prog, -1:, ...]             # last input timestep
    y_pred[:, :num_prog] += residual                  # add state back
    # diagnostic channels (output-only) remain as direct predictions
```

**Channel layout:**

| Channel group | In input `x` | In output `y_pred` | Residual applied? |
|---|---|---|---|
| Upper-air (6 vars × 17 levels) | ✅ first 102 | ✅ first 102 | ✅ Yes |
| Surface (4 vars) | ✅ next 4 | ✅ next 4 | ✅ Yes |
| Forcing/static (10 vars) | ✅ next 10 | ❌ not in output | N/A |
| Diagnostics (6 vars) | ❌ not in input | ✅ last 6 | ❌ No (no prior state) |

**Multi-step rollout:** After adding the residual, `y_pred` is the full predicted state. This full state is fed back as `x` for the next step, exactly as before. The model always receives full states as input and outputs deltas. The residual skip is transparent to the rest of the pipeline.

---

## DataLoader Optimization Changes

### Multi-Step Files

#### `credit/datasets/wrf_wofs_multistep.py`
render_diffs(file:///home/zxhua_l/miles-credit-wofs/credit/datasets/wrf_wofs_multistep.py)

- Single contiguous `.load()` for the full rollout window
- Merge `dyn_forcing`/`forcing`/`static` once before the rollout loop
- Pre-cached & pre-normalized boundary anchors
- Fast zarr-metadata `__len__`
- Pass `_boundary_pre_normalized` flag

#### `credit/trainers/trainerWRF_multi.py`
render_diffs(file:///home/zxhua_l/miles-credit-wofs/credit/trainers/trainerWRF_multi.py)

- Removed per-step `torch.distributed.barrier()`
- Added residual prediction support

#### `applications/train_wrf_wofs_multi.py`
render_diffs(file:///home/zxhua_l/miles-credit-wofs/applications/train_wrf_wofs_multi.py)

- `pin_memory=True` for validation DataLoader

#### `config/wofs_credit_wrf_t0_multi_train_date_range_example.yml`
render_diffs(file:///home/zxhua_l/miles-credit-wofs/config/wofs_credit_wrf_t0_multi_train_date_range_example.yml)

- Reduced `thread_workers` 16 → 8
- Added `residual_prediction: False` flag

---

### Single-Step Files

#### `credit/datasets/wrf_wofs_singlestep.py`
render_diffs(file:///home/zxhua_l/miles-credit-wofs/credit/datasets/wrf_wofs_singlestep.py)

- Single contiguous `.load()` merging upper + surface chunks up front
- Pre-cached & pre-normalized boundary anchors
- Fast zarr-metadata `__len__`
- Pass `_boundary_pre_normalized` flag

#### `credit/trainers/trainerWRF.py`
render_diffs(file:///home/zxhua_l/miles-credit-wofs/credit/trainers/trainerWRF.py)

- Removed post-backward `torch.distributed.barrier()`
- Added residual prediction support

#### `applications/train_wrf_wofs.py`
render_diffs(file:///home/zxhua_l/miles-credit-wofs/applications/train_wrf_wofs.py)

- `pin_memory=True` for validation DataLoader

---

### Shared Transform

#### `credit/transforms/transforms_wrf.py`
render_diffs(file:///home/zxhua_l/miles-credit-wofs/credit/transforms/transforms_wrf.py)

- `.values` arithmetic (bypass xarray coordinate alignment)
- Skip boundary normalization when `_boundary_pre_normalized` flag is set

## Expected Impact

### DataLoader Optimizations

| Bottleneck | Before | After |
|---|---|---|
| Zarr reads per sample | Multiple fragmented `.load()` calls | 1 contiguous `.load()` |
| `xr.merge()` per sample | Called per sub-component, repeatedly | Called once up-front |
| Boundary disk reads | 1–2 per sample | 0 (pre-cached & pre-normalized) |
| Normalization overhead | Full xarray coordinate alignment | Raw numpy `.values` arithmetic |
| GPU sync stalls | `barrier()` every batch/step | Only post-batch `all_reduce` |
| Validation CPU→GPU | Synchronous (unpinned) | Asynchronous (pinned memory) |

### Residual Prediction

| Aspect | Impact |
|---|---|
| **Convergence** | Model learns small deltas → easier optimization landscape → faster convergence |
| **Accuracy** | Background fields preserved exactly → reduces "smoothing" artifacts |
| **Multi-step stability** | Error accumulation is on deltas, not absolute fields → more stable rollouts |
| **Backward compatibility** | `residual_prediction: False` (default) preserves original behavior exactly |
