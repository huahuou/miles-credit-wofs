# DA Increment Loss Functions

This document summarizes the loss terms used for WoFS DA concentration-increment
training and gives tuning guidance for the physical-space concentration loss and
the occupancy loss.

## Current Training Objective

The model predicts normalized increments:

```text
dz = z(t1) - z(t0)
```

For concentration variables, `z` is the normalized concentration-transform
space selected by:

```yaml
data:
  concentration_normalization_mode: zscore
```

The base supervised loss is still normalized-space MSE:

```text
L_base = MSE(dz_pred, dz_true)
```

This is useful for stable training, but it does not reflect the nonlinear
physical impact of the inverse concentration transform.  A small normalized
error can map to a large physical error, especially for number concentrations.

## Physical Concentration Increment Loss

Implemented in:

```text
credit/losses/physical_concentration_loss.py
```

Enabled by:

```yaml
loss:
  use_physical_concentration_loss: true
  physical_concentration_loss_weight: 1.0e-7
```

The loss compares physical increments, not final states alone:

```text
x0      = inverse(z0)
x_pred  = inverse(z0 + dz_pred)
x_true  = inverse(z0 + dz_true)

dx_pred = x_pred - x0
dx_true = x_true - x0
```

The auxiliary loss is:

```text
L_phys =
  huber_weight * Huber(clip((dx_pred - dx_true) / scale))
  + log_ratio_weight * clip(log((abs(dx_pred) + eps) / (abs(dx_true) + eps)))^2
  + positive_mass_underprediction_weight * clip(ReLU(dx_true - dx_pred) / scale)^2
```

The positive-mass underprediction term applies only to mass variables, not `QN*`
variables.

The clipping is important.  Early tests showed that the physical auxiliary loss
can otherwise dominate the entire training objective.  In particular, ratio
losses explode when `dx_true` is near zero, and scaled Huber terms explode when
the configured physical scale is too small.

### Why This Helps

The urgent failure mode is:

- `QN*` values can look close in z-score space but become very different after
  inverse transform.
- mass variables often underpredict positive physical increments.

The log-ratio term directly penalizes multiplicative physical errors.  A 2x or
3x physical miss receives extra weight even if the normalized-space difference
is small.

The ratio term is only evaluated where the target physical increment is
meaningful:

```text
abs(dx_true) > increment_threshold
```

False positives where `dx_true` is effectively zero are handled by the Huber
term instead of by an unbounded ratio.

The mass-underprediction term is asymmetric by design.  It only activates when:

```text
dx_true > increment_threshold
```

and:

```text
dx_pred < dx_true
```

This avoids a blanket upward bias in clear air.

### Masking

The physical loss is masked to hydrometeor-relevant regions:

```text
mask = (
  abs(x0) > threshold
  OR abs(x_true) > threshold
  OR abs(dx_true) > increment_threshold
)
```

Use separate thresholds for mass and number concentration variables:

```yaml
threshold_by_var:
  QRAIN: 1.0e-12
  QNRAIN: 1.0e-6
```

These should eventually be replaced by per-variable/per-level training-set
statistics, but fixed thresholds are adequate for the first urgent mitigation.

### Tuning Guidance

Start conservatively:

```yaml
physical_concentration_loss_weight: 1.0e-7
```

Watch the logged values:

```text
physical_concentration_loss
physical_concentration_loss_weighted
```

The weighted physical term should not be the whole objective.  In early runs,
settings like `0.005` and `1.0e-5` made:

```text
physical_concentration_loss_weighted ~= train_loss
```

That is too large; the model then optimizes the auxiliary physical penalty while
normalized-space `acc` and `mae` can degrade.  A safer first target is:

```text
physical_concentration_loss_weighted ~= 1%-30% of the base regression objective
```

If it still dominates, reduce to:

```yaml
physical_concentration_loss_weight: 1.0e-8
```

If it becomes negligible and physical ratio errors remain large, increase
gradually by factors of 2-5.

The most sensitive settings are:

```yaml
scale_by_var
eps_qn
ratio_penalty_weight
max_log_ratio
max_ratio_weight
max_scaled_error
positive_mass_underprediction_weight
```

`scale_by_var` controls the Huber term.  It is intentionally simple in the
current config and should be calibrated from physical increment percentiles in a
later pass.

Current safety settings:

```yaml
physical_concentration_loss:
  max_log_ratio: 3.0
  max_ratio_weight: 25.0
  max_scaled_error: 50.0
  positive_mass_underprediction_weight: 0.1
```

`max_scaled_error` caps the normalized physical error used by the Huber and
mass-underprediction terms.  This prevents small `scale_by_var` values from
creating huge gradients.

## Reflectivity Operator Constraint

Implemented in:

```text
credit/losses/refl_operator_constraint.py
```

Enabled by:

```yaml
loss:
  use_refl_operator_constraint: true
  refl_operator_constraint_weight: 0.001
```

This loss maps predicted physical hydrometeor states through the NSSL
reflectivity operator and compares predicted reflectivity innovation against the
boundary `REFL_10CM` innovation:

```text
dBZ_pred_innov = H(x0 + dx_pred) - H(x0)
```

This is useful for observation-space consistency, but it does not guarantee
that each individual concentration variable has the correct physical increment.
For the current QN blowup issue, the physical concentration loss should be the
more direct control.

## Microphysics Constraint Loss

Implemented in:

```text
credit/losses/microphysics_constraint.py
```

This is a normalized-space consistency regularizer for mass/number pairs:

```text
QRAIN   <-> QNRAIN
QHAIL   <-> QNHAIL
QGRAUP  <-> QNGRAUPEL
QSNOW   <-> QNSNOW
```

It checks sign consistency, correlation alignment, and small 2x2 covariance
structure.  It is cheap and useful for pair coherence, but it does not solve
large inverse-transform amplification by itself because it operates in z-score
space.

## Occupancy Loss

The occupancy machinery is implemented in:

```text
credit/trainers/trainerWRF.py
```

For `trainer.type: standard-wrf`, it is active when:

```yaml
loss:
  occupancy:
    enabled: true
```

The dataset creates a binary target:

```text
y_occupancy = abs(physical_delta) >= delta_threshold
```

The trainer then builds occupancy logits from the predicted normalized
increment magnitude:

```text
occ_logits = (abs(dz_pred) - delta_threshold_norm) / logit_temperature
```

It can add binary cross entropy, focal loss, dice loss, and optionally mask the
regression loss.

### Usability For This Problem

Occupancy is mostly a detection/gating tool:

- it asks whether a grid point has a meaningful physical increment;
- it does not measure whether the physical increment magnitude is correct;
- it does not directly penalize 2x or 3x physical over/under prediction.

For the current issue, occupancy can help reduce clear-air dominance, but it is
not the main fix.  The physical concentration increment loss is more directly
aligned with the failure.

### Suggested Simplification

The current block exposes more switches than are needed for this DA increment
task:

```yaml
use_focal
focal_weight
focal_alpha
focal_gamma
use_dice
dice_weight
dice_smooth
masked_regression_mode
reg_mask_min
```

A simpler useful occupancy configuration would keep only:

```yaml
occupancy:
  enabled: true
  delta_threshold: 1.0e-8
  delta_threshold_by_var: {}
  variables: ['QRAIN', 'QNRAIN', 'QHAIL', 'QNHAIL', 'QGRAUP', 'QNGRAUPEL', 'QSNOW', 'QNSNOW']
  use_masked_regression: true
  ce_weight: 0.1
  pos_weight: 4.0
  logit_temperature: 0.25
```

Recommended defaults:

- leave `use_focal: false`;
- leave `use_dice: false`;
- use soft masked regression only;
- avoid hard masks until the physical thresholds are well calibrated.

For the active physical-increment problem, it is reasonable to leave occupancy
disabled initially and let the new physical loss handle hydrometeor-region
masking internally.

## Practical First Experiment

Use:

```yaml
loss:
  training_loss: mse
  use_physical_concentration_loss: true
  physical_concentration_loss_weight: 1.0e-7
  physical_concentration_loss:
    max_log_ratio: 3.0
    max_ratio_weight: 25.0
    max_scaled_error: 50.0
    positive_mass_underprediction_weight: 0.1
  use_refl_operator_constraint: true
  refl_operator_constraint_weight: 0.001
  occupancy:
    enabled: false
```

Then inspect:

```text
train_loss
physical_concentration_loss
physical_concentration_loss_weighted
refl_operator_constraint_loss_weighted
```

Do not interpret trainer `reg_loss` as pure normalized MSE in the current
`standard-wrf` path.  It is the return value of `criterion(y, y_pred)`, so it
includes enabled auxiliary losses.  Use `train_loss` and the named weighted aux
losses together to infer which term is dominating.  A future cleanup should add
an explicit `base_loss` log.

The first validation diagnostics to add should be physical-increment metrics:

```text
abs(dx_pred - dx_true)
ratio = max((abs(dx_pred)+eps)/(abs(dx_true)+eps),
            (abs(dx_true)+eps)/(abs(dx_pred)+eps))
fraction ratio > 2
fraction ratio > 3
positive mass underprediction bias
```
