# DA Occupancy-Aware Loss Plan (Segmentation-Inspired)

## Motivation

Current DA increment training uses only regression loss on normalized increments.
When a variable-level slice is physically empty in many pixels, plain MSE can learn noise.

Goal: add an occupancy objective so the model first learns where an increment is active, and apply increment regression only on active pixels (or soft-gated active probability), reducing noise from null-increment regions.

Time notation in this DA dataset:
- `t0`: background timestep used as model input state (`x`).
- `t1`: immediate next timestep used for supervision targets (`y` increment and occupancy labels).
- Current DA setting uses consecutive pairs `(t0, t1 = t0 + 1)`.

---

## Key Design Choices

1. Label type from your request:
- Build binary labels at data load time using increment activity in physical space:
  - `delta = t1 - t0`
  - label = 0 if `abs(delta) < delta_threshold`
  - label = 1 otherwise
- Done per prognostic variable, per level, per pixel.
- Apply to all 8 current prognostic variables by default, while allowing config to select a subset.
- Use global `1e-10` as startup threshold with optional per-variable override.

2. Loss family:
- For binary per-variable masks, **BCEWithLogits** is the mathematically consistent pixel-wise cross-entropy.
- This is equivalent to 2-class cross-entropy per channel.
- This matches your request for pixel-wise cross-entropy classification over variable/level/pixel occupancy.
- Note: this is multi-label across variables, not one mutually-exclusive class among rain/snow/graup/hail.

3. Regression gating:
- Use **soft gating** in the current stage.
- Gate regression by predicted occupancy probability from classifier logits, i.e. `gate = sigmoid(logits)`.
- Practical form: weighted regression where active pixels get larger weight and null pixels are down-weighted, not fully discarded.
- Keep an option for hard mask fallback (`active = (abs(t1 - t0) >= delta_threshold)`) for ablation and stability checks.

4. Architecture strategy:
- Phase 1 (minimal-risk): no new model head.
- Derive occupancy logits from predicted increment magnitude with a temperature-scaled margin:
  - `occ_logits = (abs(y_pred) - delta_thr_norm) / tau`
- Phase 2 (optional): explicit classifier head for occupancy logits.

---

## Files To Change

## 1) Dataset labels and masks
File:
- `credit/datasets/wrf_wofs_da_increment.py`

Changes:
1. Add occupancy threshold config fields:
- `data.occupancy_delta_threshold: 1.0e-10` (default)
- `data.occupancy_delta_threshold_by_var`: optional dict override per variable
- `data.occupancy_variables`: default to all prognostic variables currently in DA config:
  - `['QRAIN', 'QNRAIN', 'QHAIL', 'QNHAIL', 'QGRAUP', 'QNGRAUPEL', 'QSNOW', 'QNSNOW']`
- `data.occupancy_target_source: delta_abs` (default)

2. In `__getitem__`, during prognostic loop (where raw_t0/raw_t1 are already loaded):
- Build `delta_raw = reduced_t1 - reduced_t0`
- Build `y_occ = (abs(delta_raw) >= delta_threshold[var]).astype(float32)`
- Build optional hard regression reference mask `y_reg_mask_hard = y_occ.copy()` for diagnostics/ablation

3. Add tensors to sample dict:
- `y_occupancy`: shape same as `y` (time=1, var, level, H, W)
- `y_regression_mask_hard`: shape same as `y` (optional hard-mask reference)
- Optional helper tensor for trainer logits mapping:
  - `y_occ_delta_threshold_norm`: per var-level normalized increment threshold broadcastable to `y`.

Why here:
- Dataset already has raw concentration arrays and normalization stats.
- Label generation is deterministic and cheap here.

---

## 2) Loss module extension for masked regression and occupancy losses
File:
- `credit/losses/weighted_loss.py`

Changes:
1. Extend `VariableTotalLoss2D.forward` signature:
- from: `forward(target, pred)`
- to: `forward(target, pred, mask=None)` (optional arg for backward compatibility)

2. If `mask` is provided:
- Apply mask to elementwise loss before variable reduction.
- Compute masked mean robustly:
  - numerator = `(loss * mask).sum(...)`
  - denominator = `mask.sum(...).clamp_min(eps)`
  - var_loss = numerator / denominator
- Keep support for soft mask values in `[0, 1]` (not only binary mask).

3. Preserve latitude/variable weighting behavior exactly as now.

4. Add small helper losses (same file or new `credit/losses/da_occ_losses.py`):
- `binary_focal_with_logits(logits, target, alpha, gamma)`
- `soft_dice_loss_from_logits(logits, target, smooth)`

Why here:
- Keeps existing loss wiring and config style.
- Enables regression masking without custom trainer-only duplicated weighting logic.

---

## 3) Trainer integration for composite objective
File:
- `credit/trainers/trainerWRF.py`

Changes in `train_one_epoch` and `validate`:
1. Read new tensors from batch when present:
- `y_occ = batch.get("y_occupancy")`
- `reg_mask_hard = batch.get("y_regression_mask_hard")`

2. Compute regression loss with mask:
- Build soft gate from occupancy classifier: `gate = sigmoid(occ_logits).detach()` for stable first version.
- Optionally blend with hard mask floor to avoid vanishing supervision:
  - `reg_mask = max(reg_mask_min, gate)` where `reg_mask_min` default is small (e.g. 0.05).
- `reg_loss = criterion(y, y_pred, mask=reg_mask)` when enabled.
- fallback to existing behavior when mask is absent or disabled.

3. Compute occupancy logits and occupancy loss:
- Build logits from predicted increment magnitude in normalized increment space:
  - `occ_logits = (abs(y_pred) - delta_thr_norm) / tau`
- Occupancy CE:
  - `occ_ce = BCEWithLogits(occ_logits, y_occ)`

4. Composite loss:
- `total_loss = reg_lambda * reg_loss + occ_lambda * occ_ce`
- Optional:
  - `+ focal_lambda * occ_focal`
  - `+ dice_lambda * occ_dice`
- Initial default: `occ_lambda = 0.2`.

5. Log diagnostics:
- `train_reg_loss`, `train_occ_ce`, `train_occ_f1`, `train_occ_precision`, `train_occ_recall`, masked pixel fraction.
- Same for validation if enabled.

Why trainer-level composition:
- Reuses existing criterion for regression.
- Keeps occupancy as DA-specific optional behavior controlled by config.

---

## 4) Config schema and defaults
Files:
- `config/wofs_credit_wrf_da_increment.yml`
- `config/ursa_wofs_credit_wrf_da_increment.yml`
- `credit/parser.py`

Changes:
1. Add DA occupancy loss block under `loss`:

```yaml
loss:
  occupancy:
    enabled: true
    delta_threshold: 1.0e-10
    delta_threshold_by_var: {}
    variables: ['QRAIN', 'QNRAIN', 'QHAIL', 'QNHAIL', 'QGRAUP', 'QNGRAUPEL', 'QSNOW', 'QNSNOW']
    target_source: delta_abs
    use_masked_regression: true
    masked_regression_mode: soft   # soft | hard
    reg_mask_min: 0.05

    # CE/BCE
    ce_weight: 0.2
    pos_weight: 4.0
    logit_temperature: 0.25

    # optional extras
    use_focal: false
    focal_weight: 0.0
    focal_alpha: 0.25
    focal_gamma: 2.0

    use_dice: false
    dice_weight: 0.0
    dice_smooth: 1.0
```

2. Parser defaults and guards in `credit/parser.py`:
- Add safe defaults if `loss.occupancy` is absent.
- Validate ranges and enum fields.
- Keep backward compatibility for non-DA runs.
- Validate that configured `occupancy.variables` are a subset of `data.variables`.
- Validate optional `delta_threshold_by_var` keys are a subset of `data.variables`.

---

## 5) Documentation updates
Files:
- `implementation_DA_plan_detail.md`
- `README.md` (DA section)

Add:
1. Why masking helps sparse hydrometeor increments.
2. Difference between BCE (binary per channel) and categorical CE (mutually-exclusive class index).
3. When to use focal loss:
- Severe class imbalance (few active pixels).
4. When to use Dice loss:
- Better overlap learning for sparse structures.
5. Suggested starting weights:
- `occ_ce_weight = 0.1 to 0.3`
- Keep focal/dice off initially.

---

## Proposed Implementation Order

1. Add dataset occupancy labels and regression mask output.
2. Add optional mask support in `VariableTotalLoss2D`.
3. Integrate occupancy CE, soft-gated regression masking, and composite objective in `trainerWRF.py`.
4. Add parser defaults and config block.
5. Run smoke test (single GPU), then DDP test.
6. Add docs and diagnostics plots.

---

## Validation and Smoke Tests

1. Unit-like checks (small batch):
- Ensure `y_occupancy` and optional `y_regression_mask_hard` shapes exactly match `y`.
- Verify values are {0,1}.
- Verify non-zero masked denominator handling for soft masks.

2. Numerical checks:
- Confirm no NaN when an entire batch is empty for a variable-level.
- Confirm total loss decreases and occupancy metrics improve.
- Confirm soft gate does not collapse to all zeros (track mean gate value).

3. Runtime check:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  applications/train_wrf_wofs_da.py \
  -c config/wofs_credit_wrf_da_increment.yml
```

4. DDP sanity check:
- Run your existing multi-GPU launcher and ensure no collective mismatch.

---

## Confirmed Decisions

1. Occupancy loss applies to all 8 current DA prognostic variables by default, with config control for future subset selection.
2. Pixel-wise cross-entropy objective is used as BCEWithLogits over binary occupancy labels per variable/level/pixel.
3. Occupancy target is increment null/non-null (`abs(t1 - t0)` threshold), not state occupancy at `t0` or `t1`.
4. Current-stage regression gating is soft, using classifier-derived occupancy probability as multiplicative/weighting gate.
5. Initial occupancy CE weight is `0.2`.
