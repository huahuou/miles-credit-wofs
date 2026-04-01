# WRF DataLoader Optimization & Residual Prediction — Implementation Plan

## Proposed Changes

---

### DataLoader Optimizations (both pipelines)

#### Datasets (`wrf_wofs_multistep.py`, `wrf_wofs_singlestep.py`)
* Pre-cache & pre-normalize boundary anchors in `_open_datasets`
* Single contiguous `.load()` for the full time window
* Merge auxiliary data once before loop
* Fast zarr-metadata `__len__`
* Pass `_boundary_pre_normalized` flag

#### Transforms (`transforms_wrf.py`)
* `.values` arithmetic (bypass xarray coordinate alignment)
* Skip boundary normalization when pre-normalized

#### Trainers (`trainerWRF_multi.py`, `trainerWRF.py`)
* Remove redundant `torch.distributed.barrier()` calls
* **Residual prediction** via config flag

#### Training Apps (`train_wrf_wofs_multi.py`, `train_wrf_wofs.py`)
* `pin_memory=True` for validation DataLoader

#### Config
* Reduce `thread_workers` 16 → 8
* Add `residual_prediction: False` flag

---

### Residual Prediction Feature

#### Design

The model's raw output is treated as a **delta** (change from the previous timestep). The trainer adds the last input timestep's prognostic channels back to produce the full predicted state.

```
y_pred_full = model_output_delta + x[:, :num_prognostic, -1:, ...]
```

**Key details:**
- Only prognostic channels (upper-air + surface) get the skip connection
- Diagnostic channels (output-only, e.g. `COMPOSITE_REFL_10CM`) remain direct predictions
- The loss is computed on the full state (`y_pred_full` vs ground truth `y`), so no target modifications needed
- Multi-step rollout feeds the full state back as input — transparent to the autoregressive loop

#### Files Modified

| File | Change |
|---|---|
| [trainerWRF_multi.py](file:///home/zxhua_l/miles-credit-wofs/credit/trainers/trainerWRF_multi.py) | Add residual skip after `self.model()` in `train_one_epoch` and `validate` |
| [trainerWRF.py](file:///home/zxhua_l/miles-credit-wofs/credit/trainers/trainerWRF.py) | Same residual skip in both methods |
| [config yml](file:///home/zxhua_l/miles-credit-wofs/config/wofs_credit_wrf_t0_multi_train_date_range_example.yml) | Add `residual_prediction: False` under `trainer` |

## Verification Plan

### Automated Tests
```bash
# Multi-step (direct prediction — should be identical to before)
torchrun --standalone --nnodes=1 --nproc-per-node=2 applications/train_wrf_wofs_multi.py -c config/wofs_credit_wrf_t0_multi_train_date_range_example.yml

# Multi-step (residual prediction — set residual_prediction: True in config)
torchrun --standalone --nnodes=1 --nproc-per-node=2 applications/train_wrf_wofs_multi.py -c config/wofs_credit_wrf_t0_multi_train_date_range_example.yml
```

### Manual Verification
- With `residual_prediction: False`, training loss should match pre-change behavior exactly
- With `residual_prediction: True`, initial loss should be significantly lower (model starts by predicting near-zero deltas)
- Monitor GPU utilization to confirm dataloader optimizations are effective
