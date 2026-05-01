# Physical Constraint Losses for QRAIN / QNRAIN Increment Learning

## Why Pure MSE Is Insufficient

MSE on normalized increments `(Δz_Q, Δz_N)` treats QRAIN and QNRAIN as
independent channels. The model can produce physically incoherent combinations
such as: large positive `ΔQ_r` (added rain mass) paired with negative `ΔN_r`
(fewer drops), implying the mean drop diameter jumped to implausibly large
values. These states cannot arise under Thompson bulk microphysics. Adding
physically motivated auxiliary losses during training steers the model away
from such degeneracies.

---

## Physical Background

### The Q–N Relationship in Thompson Microphysics

WoFS uses the Thompson (2008) scheme. Rain is described by a modified gamma DSD:

$$n(D) = N_0 \, D^\mu \, e^{-\Lambda D}$$

The slope parameter $\Lambda$ and intercept $N_0$ are diagnosed from the two
prognostic moments:

| Symbol | WRF variable | Units |
|--------|-------------|-------|
| $q_r$  | `QRAIN`     | kg kg⁻¹ |
| $N_r$  | `QNRAIN`    | kg⁻¹ (number per kg of air) |

From these, the **mean volume diameter** is:

$$D_m = \left(\frac{6\,\rho_\mathrm{air}\,q_r}{\pi\,\rho_w\,N_r}\right)^{1/3}$$

Physical constraints that follow directly:

1. **Positivity**: $q_r \geq 0$, $N_r \geq 0$ always.
2. **Joint zero**: $q_r = 0 \Rightarrow N_r = 0$ (no mass → no drops).
3. **$D_m$ bounds**: physically realizable rain occupies roughly
   $50\,\mu\mathrm{m} \lesssim D_m \lesssim 3\,\mathrm{mm}$.  Values outside
   this range indicate an unphysical $(q_r, N_r)$ pair.
4. **Comonotonicity of increments**: adding rain mass almost always requires
   adding drops and vice versa; $\Delta q_r$ and $\Delta N_r$ should be
   positively correlated across space.

---

## Normalization Complication

Both variables go through the **concentration transform** before z-scoring:

$$f(q) = c_1 \min(q, q_{\max}) + c_2 \frac{\log(\max(q, \varepsilon)) - \log\varepsilon}{-\log\varepsilon}$$

with defaults $c_1{=}0.5$, $c_2{=}0.5$, $\varepsilon{=}10^{-4}$, $q_{\max}{=}2.5$.

The model sees and predicts **normalized increments**:

$$\Delta z_Q = z(q_{r,1}) - z(q_{r,0}), \qquad z(q) = \frac{f(q) - \bar{f}_Q}{\sigma_Q}$$

and similarly for $\Delta z_N$.  This has two practical implications:

- **Proxy constraints in normalized space** are cheap but approximate — the
  distortion from the log-blend changes correlation structure.
- **Physical-space constraints** require an inverse transform inside the loss,
  which is differentiable but ~3× slower (bisection for the middle branch).

The implementation plan below separates these into *Tier 1* (normalized-space,
low overhead) and *Tier 2* (physical-space, higher fidelity).

---

## Proposed Loss Terms

### Tier 1 — Normalized-Space Proxy Constraints

#### 1. Sign-Consistency Regularization

Penalize grid points where $\Delta z_Q$ and $\Delta z_N$ have **opposite
large-magnitude signs**. This is a soft proxy for the comonotonicity of
increments.

$$\mathcal{L}_\mathrm{sign} = \frac{1}{|\Omega|} \sum_{k,i,j} \mathrm{ReLU}\!\left(
  -\Delta\hat{z}_{Q,kij} \cdot \Delta\hat{z}_{N,kij}
\right) \cdot w_{kij}$$

where $w_{kij}$ is an optional magnitude weight that suppresses the penalty
near zero (avoiding noise amplification in clear-air levels):

$$w_{kij} = \mathrm{ReLU}\!\left(\min(|\Delta\hat{z}_{Q,kij}|, |\Delta\hat{z}_{N,kij}|) - \tau\right)$$

with threshold $\tau \sim 0.1$ in normalized units.  This loss is zero when
every predicted increment pair is concordant (same sign).

**Why this works even in normalized space**: the concentration transform is
strictly monotone, so sign is preserved.  $\Delta z_Q > 0 \Leftrightarrow
q_{r,1} > q_{r,0}$ for any non-pathological normalization.

**Implementation cost**: two element-wise multiplies + ReLU. Negligible.

---

#### 2. Pearson Correlation Alignment (per-level)

For each vertical level $k$, the Pearson correlation between $\Delta z_Q$ and
$\Delta z_N$ computed over the batch × spatial dimensions should match the
empirical target correlation $\rho^\star_k$ estimated from training data:

$$\mathcal{L}_\mathrm{corr} = \frac{1}{K}\sum_{k=1}^{K}
  \left(\rho_k(\Delta\hat{z}_Q, \Delta\hat{z}_N) - \rho^\star_k\right)^2$$

where:

$$\rho_k(\mathbf{a}, \mathbf{b}) = \frac{(\mathbf{a}-\bar{a})\cdot(\mathbf{b}-\bar{b})}
  {\|\mathbf{a}-\bar{a}\|_2\,\|\mathbf{b}-\bar{b}\|_2}$$

with vectors formed by flattening $(\text{batch}, H, W)$ for level $k$.

$\rho^\star_k$ can be precomputed over the training set as a constant tensor and
stored in the config.  A soft version that only penalizes *below* a target
threshold (rather than matching exactly) is less brittle:

$$\mathcal{L}_\mathrm{corr} = \frac{1}{K}\sum_k \mathrm{ReLU}(\rho^\star_k - \rho_k)^2$$

This is closely related to how **CORAL loss** (Sun & Saenko 2016, "Deep CORAL:
Correlation Alignment for Deep Domain Adaptation") works — it minimizes the
Frobenius norm between source and target second-order statistics.  CORAL
operates on full covariance matrices; this is the 2×2 specialization for
(QRAIN, QNRAIN).

**Implementation cost**: one Pearson computation per level per batch.
Parallelizable with `torch.einsum`.

---

#### 3. 2×2 Covariance Structure Matching (CORAL-style)

Generalize the per-level correlation into a full **2×2 covariance matrix**
matching.  For each level $k$, form the vector:

$$\mathbf{v}_{kij} = [\Delta\hat{z}_{Q,kij},\; \Delta\hat{z}_{N,kij}]^\top$$

Compute the batch-spatial covariance $\hat{\Sigma}_k$ (2×2) and compare with
the target $\Sigma^\star_k$ estimated from data:

$$\mathcal{L}_\mathrm{CORAL} = \frac{1}{4K}\sum_{k=1}^{K}
  \left\|\hat{\Sigma}_k - \Sigma^\star_k\right\|_F^2$$

The off-diagonal entry of $\hat{\Sigma}_k$ captures the cross-covariance; the
diagonal entries match the per-variable variance.  Because QRAIN and QNRAIN
span very different physical scales but are in the same normalized space, this
is best done after whitening each channel by its batch standard deviation.

This directly answers the user's question: **yes, this style of loss
(covariance/second-moment matching) is established in the literature** under
the names CORAL, Gram-matrix loss (Gatys 2015, style transfer), and moment
matching (Gretton MMD, Li 2017 MMD-GAN).  `credit/losses/covariance.py`
already contains `CovarianceWeightedMSELoss` which does full covariance
weighting; the CORAL-style objective is a lighter sibling.

**Implementation cost**: one 2×2 matrix outer product per level, summed over
the batch.  Efficient.

---

#### 4. Relative-Change Ratio Regularization

The user's suggestion of constraining the *ratio of relative changes*.  Define
the relative increment in normalized space:

$$\delta_Q = \frac{\Delta\hat{z}_Q}{|z_{Q,t0}| + \varepsilon_r}, \qquad
  \delta_N = \frac{\Delta\hat{z}_N}{|z_{N,t0}| + \varepsilon_r}$$

A natural constraint is that $\delta_Q$ and $\delta_N$ should not diverge
wildly.  One form:

$$\mathcal{L}_\mathrm{ratio} = \left\|
  \log\!\left(\frac{|\delta_Q| + \varepsilon_r}{|\delta_N| + \varepsilon_r}\right)
\right\|_2^2 \cdot \mathbf{1}[\delta_Q \cdot \delta_N > 0]$$

The indicator restricts the penalty to concordant (same-sign) pairs to avoid
amplifying sign disagreements already handled by $\mathcal{L}_\mathrm{sign}$.
This penalizes cases where QRAIN changes by 10× but QNRAIN barely moves —
physically, that would imply drops grow 2.1× in radius (since $q_r \propto
N_r D_m^3$).

**Caveat**: near-zero backgrounds $z_{Q,t0} \approx 0$ (clear air) will
dominate.  Must gate by a background-magnitude mask.

---

### Tier 2 — Physical-Space Constraints (Higher Fidelity)

#### 5. Mean Diameter Bounds Penalty

The most physically principled option.  Inverse-transform predictions to
physical space, compute $D_m$, and penalize out-of-bounds values.

Algorithm:

1. Compute predicted physical state:
   $$q_{r,1}^{\mathrm{pred}} = f^{-1}((z_{Q,t0} + \Delta\hat{z}_Q) \cdot \sigma_Q + \bar{f}_Q)$$
   $$N_{r,1}^{\mathrm{pred}} = f^{-1}((z_{N,t0} + \Delta\hat{z}_N) \cdot \sigma_N + \bar{f}_N)$$
2. Clamp to physical positivity: $q_r \leftarrow \max(q_r, 0)$, likewise $N_r$.
3. Compute $D_m = \left(6\rho_\mathrm{air}q_r / (\pi\rho_w N_r)\right)^{1/3}$,
   guarding division-by-zero with a small floor on $N_r$.
4. Loss:
   $$\mathcal{L}_{D_m} = \left\|\mathrm{ReLU}(D_{m,\min} - D_m)\right\|_2^2
     + \left\|\mathrm{ReLU}(D_m - D_{m,\max})\right\|_2^2$$
   masking clear-air grid points where $q_r < \varepsilon_{D_m}$.

**Feasibility**: the inverse concentration transform's middle branch uses
bisection (40 iterations) which is **not differentiable** through
`numpy` but can be re-implemented with `torch` binary search (`torch.bucketize`
or a fixed-iteration bisection in float32) to allow gradient flow.  This is
non-trivial but the $f^{-1}$ regions outside the bisection segment have exact
closed-form inverses and cover the majority of the range.

An approximation: use only the log-branch of $f^{-1}$ (valid for $q_r <
\varepsilon$ and $q_r > q_{\max}$), skip the bisection, accept ~5% error in the
mid-range but gain full differentiability.

---

#### 6. Joint-Zero Consistency

If $q_{r,\mathrm{pred}} \leq \varepsilon_{q}$ then $N_{r,\mathrm{pred}}$
should also be near zero, and vice versa.  This can be enforced as:

---

## Phase 2 — H-Operator Reflectivity Constraint (dBZ)

### Motivation

Tier-1 constraints keep the $(\Delta q, \Delta N)$ pairs coherent in normalized space, but they do not directly tie the predicted hydrometeor increments to the observed reflectivity innovations that drive the DA task. This phase introduces a forward observation operator, $\mathcal{H}$, that maps physical hydrometeor states to S-band reflectivity (REFL_10CM), and a corresponding loss that aligns predicted reflectivity changes with the observed reflectivity innovation.

The operator $\mathcal{H}$ follows the NSSL 2-moment microphysics reflectivity diagnosis (as implemented in diag_nssl_refl.py), adapted for torch-efficient, batched computation. With the new closed-form log-zscore inverse transform, we can cheaply and stably convert normalized predictions back to physical $(q, N)$ prior to applying $\mathcal{H}$.

### High-Level Idea

- Given normalized background state $z_0$ (for each concentration variable) and the model-predicted normalized increment $\Delta \hat{z}$, form the predicted post-increment normalized state: $z_1^{\mathrm{pred}} = z_0 + \Delta \hat{z}$.
- Use the log-zscore inverse transform (see build_log_transform_params.py and test_log_transform.py) to recover physical-space hydrometeor fields $(q, N)$ at $t_0$ and $t_1$:
  - $q_{t0} = \exp(z_{Q,t0} \cdot \sigma_{\log,Q} + \mu_{\log,Q})$ clamped to $[\text{clip}_\min, \text{clip}_\max]$
  - $q_{t1}^{\mathrm{pred}} = \exp(z_{Q,t1}^{\mathrm{pred}} \cdot \sigma_{\log,Q} + \mu_{\log,Q})$ (same for $N$)
- Apply the forward operator $\mathcal{H}$ to obtain reflectivity in dBZ at $t_0$ and $t_1$ (predicted):
  - $\mathrm{dBZ}_{t0} = \mathcal{H}(q_{\cdot,t0}, N_{\cdot,t0}; \rho, T, \ldots)$
  - $\mathrm{dBZ}_{t1}^{\mathrm{pred}} = \mathcal{H}(q_{\cdot,t1}^{\mathrm{pred}}, N_{\cdot,t1}^{\mathrm{pred}}; \rho, T, \ldots)$
- Compare the predicted reflectivity innovation $\Delta \mathrm{dBZ}^{\mathrm{pred}} = \mathrm{dBZ}_{t1}^{\mathrm{pred}} - \mathrm{dBZ}_{t0}$ to the observed innovation (from the dataset boundary input) either in physical dBZ space or in the dataset’s normalized REFL_10CM space.

This closes the loop from predicted microphysics increments → reflectivity change, adding a physically grounded, observation-space regularizer.

### Forward Operator $\mathcal{H}$ (NSSL 2-Moment)

The operator sums the linear reflectivity contributions from species present in the training set:

- Rain: $Z_r \propto (\rho\, q_r)^2 / N_r$ with gamma-diameter shape factor $g_1(\alpha_r)$ and water density constants (as in diag_nssl_refl.py).
- Snow: Dry-snow (Cox 1988) formulation with optional bright-band enhancement when $T > T_f + 1\,\mathrm{K}$ and rain is present; depends on $q_s$, $N_s$, $\rho$, $T$, and $q_r$.
- Graupel and Hail: $Z \propto (\rho\, q)^2 / N$ with effective density treatment and mean-volume clamping, recomputing $N$ when clamped (matching radardd02 logic).
- Ice crystals: optional; if not provided, treated as zero contribution.

Then $Z_{\text{total}} = Z_r + Z_s + Z_g + Z_h + Z_i$ and $\mathrm{dBZ} = 10\log_{10}(\max(Z_{\text{total}}, \epsilon))$ with a configurable floor.

Implementation notes:
- We will provide a torch implementation, vectorized over batch/levels/spatial dimensions, and numerically aligned to python_scripts/diag_nssl_refl.py (Numpy reference).
- Inputs required by $\mathcal{H}$:
  - $q$ and $N$ for the species present: QRAIN/QNRAIN, QSNOW/QNSNOW, QGRAUP/QNGRAUPEL, QHAIL/QNHAIL.
  - Optional temperature $T$ (from context variable `T`) for bright-band logic; if unavailable, bright-band is disabled.
  - Dry-air density $\rho$ per level; when not directly available, we will provide an approximation path:
    - Preferred: dataset-provided $\rho$ (if added later to context).
    - Fallback: standard-atmosphere profile from `GEOPOT` height; final scale largely cancels in innovation mode.

### Loss Definition

Two operation modes are supported:

1) Innovation mode (default):
   - Predict $\Delta \mathrm{dBZ}^{\mathrm{pred}} = \mathcal{H}(\cdot)\big|_{t1} - \mathcal{H}(\cdot)\big|_{t0}$.
   - Compute target innovation from boundary REFL_10CM: either
     - physical dBZ difference (requires inverse-normalization of REFL_10CM), or
     - normalized-space difference if we normalize our diagnosed $\mathrm{dBZ}$ with the same mean/std per level.
   - Loss: masked mean-squared error (or Huber) over levels and spatial domain, with optional gating by reflectivity magnitude (e.g., only where either $\ge$ 5 dBZ).

2) Absolute mode:
   - Compare $\mathrm{dBZ}_{t1}^{\mathrm{pred}}$ directly to dataset REFL_10CM at $t_1$ (inverse-normalized), optionally also aligning $t_0$.

Masking and weighting:
- Reuse the occupancy gating already configured for concentrations; only evaluate where microphysics changes occur or where observed reflectivity is non-trivial.
- Optional latitude weights are supported consistently with other losses.

### Configuration (YAML)

Add a separate, opt-in loss block so it’s independently tunable:

```yaml
loss:
  use_refl_operator_constraint: true
  refl_operator_constraint_weight: 0.02
  refl_operator_constraint:
    mode: innovation           # innovation | absolute
    compare_space: dbz         # dbz | normalized
    dbz_floor: 0.0             # dBZ lower bound in H
    threshold_dbz: 5.0         # mask below this magnitude (either t0 or pred)
    include_species: [rain, snow, graupel, hail]   # ice optional
    rho_source: approx         # dataset | approx
    temp_source: context       # context | none
    level_reduction: none      # none | max | mean (if needed)
```

Dependencies and prerequisites:
- The closed-form log-zscore transform must be enabled via `data.log_transform_params_json` (present in the current config).
- For `compare_space: normalized`, we will read REFL_10CM per-level mean/std from the same stats files already configured (`mean_path`/`std_path`) to place both diagnosed and target reflectivity in the same normalized space.

### API Placement

- `credit/physics/refl_operator.py` — torch $\mathcal{H}$ implementation (species kernels + combiner).
- `credit/losses/refl_operator_constraint.py` — loss module that:
  - Takes predicted normalized increments and background normalized state for concentrations.
  - Inverts to physical $(q, N)$ using log-zscore parameters.
  - Calls $\mathcal{H}$ at $t_0$ and $t_1$ (predicted), forms innovation or absolute target, applies masks, and computes MSE/Huber.
- `credit/losses/weighted_loss.py` — wire-up like the existing `MicrophysicsConsistencyLoss` with `use_refl_operator_constraint` and a separate weight.

### Numerical Alignment and Validation

We will keep the torch implementation numerically aligned with the Numpy reference in python_scripts/diag_nssl_refl.py:
- Unit conversions, density constants, gamma-function shape factors, mean-volume clamping, and bright-band logic are mirrored.
- We will include tolerances in tests (e.g., |dBZ| differences < 0.2–0.5 dB on typical ranges) acknowledging small differences from float32 vs float64 and batched epsilons.

### Test Plan and Script

We will add a self-contained test under python_scripts to exercise both components — the forward operator and the training-style constraint:

- File: python_scripts/test_refl_operator_constraint.py
- Tests:
  1) Torch vs Numpy operator parity: generate random-but-physical $(q,N,\rho,T)$ fields and assert dBZ differences < tolerance per species and total.
  2) Round-trip with dataset normalization: sample a few tiles/levels from one WoFS case; compute $\mathrm{dBZ}_{t0}, \mathrm{dBZ}_{t1}$ from raw $(q,N)$ using torch $\mathcal{H}$; compare to dataset REFL_10CM in dBZ (after inverse-normalization) — report MAE/RMSE/correlation per level.
  3) Training mimic (innovation mode):
     - Build $z_0$ from raw $(q,N)$ with the forward log-zscore; build $z_1$ from $t_1$; form $\Delta z = z_1 - z_0$.
     - Reconstruct $q_1$ by inverting $z_0 + \Delta z$; ensure reconstruction error is < 1e-5 (as in test_log_transform.py).
     - Compute $\Delta \mathrm{dBZ}^{\mathrm{pred}}$ and compare to observed REFL_10CM innovation (either physical or normalized per config) — report metrics.

Example run on Ursa (CPU node is fine for tests):

```
srun -A gpu-ai4wp -p u1-compute --mem=128g -N 1 -t 1:20:00 --pty bash -il
source $MODULESHOME/init/bash
module load rdhpcs-conda
module load cuda/12.8
conda activate credit-wofs
python python_scripts/test_refl_operator_constraint.py \
  --stats-mean /scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/mean.nc \
  --stats-std  /scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/std.nc \
  --log-params /scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/log_transform_params.json \
  --case /scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/cases/wofs_20190429_0000_mem01.zarr.zip
```

### Risks and Mitigations

- Density/temperature availability: if high-fidelity $\rho$ and $T$ are unavailable from context, use innovation mode and approximate profiles — most multiplicative biases cancel in the difference. We gate the loss by reflectivity magnitude to avoid clear-air noise.
- Cost: The torch operator is elementwise/outer-product heavy but memory-bandwidth bound; cost is modest vs the model forward. We can disable species to trade accuracy for speed.
- Stability: All transforms are monotone and bounded; we clamp to physical ranges, apply small epsilons in denominators, and compute logs only on strictly positive Z.

### Summary

The H-operator reflectivity constraint ties predicted $(\Delta q, \Delta N)$ to observation-space innovations using a physically principled forward map. It complements Tier-1 normalized-space consistency terms and is fully compatible with the new closed-form log-zscore normalization, enabling stable gradients and low overhead.

$$\mathcal{L}_\mathrm{zero} = \left\|
  \mathbf{1}[q_r < \varepsilon_q] \cdot N_{r,\mathrm{pred}}
\right\|_2^2
+ \left\|\mathbf{1}[N_r < \varepsilon_N] \cdot q_{r,\mathrm{pred}}\right\|_2^2$$

Applied in physical space after inverse transform.  This is the DA analogue of
occupancy masking already present in the trainer.

---

## Literature Survey — Does Covariance-Structure Loss Exist?

Yes, under several names:

| Method | Reference | What it does |
|--------|-----------|-------------|
| **CORAL** | Sun & Saenko (2016) | Matches full covariance matrices between source and target; Frobenius norm loss |
| **Gram-matrix / style loss** | Gatys et al. (2015) | Matches feature covariances across CNN layers; used for texture/style |
| **MMD** | Gretton et al. (2012) | Kernel-based distribution divergence; includes moment matching |
| **MMD-GAN** | Li et al. (2017) | Uses MMD as adversarial objective; trains generator to match full distribution |
| **DeepJDOT** | Damodaran et al. (2018) | Joint distribution optimal transport; matches joint (feature, label) |
| **Wasserstein / Sinkhorn** | Peyré & Cuturi (2019) | Optimal transport distance between full distributions |
| **Physics-Informed Loss** | Raissi et al. (2019, PINN) | Direct residual of physical PDEs/algebraic constraints in loss |

For our specific problem, **CORAL** is the closest match: it is cheap (O(C²)
per layer), purely algebraic, and has been successfully used in atmospheric
science domain adaptation tasks.  The 2×2 specialization (Tier 1, Option 3) is
the recommended starting point.

---

## Recommended Implementation Path

### Phase 1 (Low Risk, Low Overhead)

Implement `MicrophysicsConsistencyLoss` in
`credit/losses/microphysics_constraint.py` with the following configurable
terms:

```python
class MicrophysicsConsistencyLoss(nn.Module):
    """
    Physical consistency losses for paired (QRAIN, QNRAIN) increment predictions.

    All inputs are expected in normalized (z-score) space.
    The loss is a weighted sum of configurable terms.

    Args:
        q_idx    : channel indices of QRAIN channels in the output tensor
        n_idx    : channel indices of QNRAIN channels in the output tensor
        sign_weight  : weight for sign-consistency term
        corr_weight  : weight for per-level correlation alignment
        coral_weight : weight for 2x2 CORAL covariance matching
        ratio_weight : weight for relative-change ratio term
        target_corr  : (K,) tensor of target per-level Pearson correlations
        target_cov   : (K, 2, 2) tensor of target per-level covariance matrices
        corr_threshold : minimum acceptable correlation (for soft lower-bound mode)
        sign_threshold : |Δz| threshold below which sign penalty is suppressed
    """
```

The loss integrates with `VariableTotalLoss2D` as an **additive auxiliary
term** that is gated by `loss.microphysics_constraint_weight` in the config.

```yaml
# In wofs_credit_wrf_da_increment.yml
loss:
  training_loss: mse
  microphysics_constraint_weight: 0.1   # total weight of auxiliary term
  microphysics_constraint:
    sign_weight: 1.0
    corr_weight: 1.0
    coral_weight: 0.5
    ratio_weight: 0.0   # start off
    sign_threshold: 0.1
    target_corr_path: /path/to/target_corr.npy   # precomputed
    target_cov_path:  /path/to/target_cov.npy    # precomputed
```

**Precomputation step** (run once, fast):

```python
# scripts/compute_qn_target_statistics.py
# -- reads training zarr files, computes per-level Pearson correlation and
#    2x2 covariance of (normalized_increment_QRAIN, normalized_increment_QNRAIN)
#    across all training samples.  Saves as .npy for config injection.
```

### Phase 2 (Physical-Space, More Effort)

Implement a differentiable inverse concentration transform in PyTorch:

```python
def inverse_conc_torch(y: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Differentiable inverse of the concentration transform.
    Uses closed-form branches for y < y1 and y > y2; fixed-iteration
    bisection (unrolled 20 steps in float32) for the middle range.
    """
```

Then add `PhysicalDiameterLoss` that calls this inverse to recover physical
QRAIN/QNRAIN and penalizes out-of-bounds $D_m$.

### Integration into `VariableTotalLoss2D`

The cleanest integration avoids modifying the existing loss class structure.
Instead:

```python
# In train_wrf_wofs_da.py  main()
aux_loss_fn = build_microphysics_constraint_loss(conf)   # returns None if not configured
train_criterion = VariableTotalLoss2D(conf)
# In trainer loop (trainerWRF.train_one_epoch equivalent):
loss = train_criterion(y_pred, y_true)
if aux_loss_fn is not None:
    loss = loss + aux_loss_fn(y_pred, y_true, x_background=batch["x"])
```

This keeps the trainer agnostic of the auxiliary loss; only the DA-specific
training app wires it in.

---

## Risk / Trade-off Summary

| Term | Benefit | Risk | Overhead |
|------|---------|------|----------|
| Sign consistency | Prevents physically impossible sign conflicts | May fight with MSE at initialization | Negligible |
| Per-level correlation alignment | Matches Q–N co-variability | Needs precomputed target stats; noisy for small batches | Low |
| CORAL 2×2 covariance | Matches second-order structure including variance ratio | Needs precomputed targets; batch-size sensitive | Low |
| Relative-change ratio | Constrains D_m changes implicitly | Near-zero instability; needs careful masking | Low–Medium |
| D_m bounds (physical space) | Most principled; Thompson-exact | Bisection gradient approximation; slow | High |
| Joint-zero consistency | Removes ghost rain artifacts | Small effect for increments | Low |

**Recommended starting order**: sign consistency → CORAL 2×2 → correlation
alignment.  The D_m penalty should be added only after the Tier-1 losses are
confirmed to not dominate training.

---

## Open Questions

1. **Which space is better for covariance targets?** — Normalized increments
   have approximately Gaussian marginals (by design of the concentration
   transform), making CORAL more meaningful than in raw physical space.
   Empirically validate by plotting the $(\Delta z_Q, \Delta z_N)$ scatter in
   training data.

2. **Level weighting** — Lower levels have more rain events.  Weight the
   per-level terms by rain frequency at that level (derived from training data
   occupancy fractions) to avoid the loss being dominated by near-zero upper
   levels.

3. **Batch-size sensitivity** — Pearson and covariance estimates are noisy for
   small batch sizes.  With batch=16 and H×W=300×300, we have 16×300×300 =
   1.44M samples per level per batch — this is large enough that sample
   statistics converge reliably.

4. **Conflict with MSE during early training** — Consider linearly ramping up
   auxiliary weights over the first ~10 epochs so the model first learns the
   main MSE signal before structural constraints tighten.
