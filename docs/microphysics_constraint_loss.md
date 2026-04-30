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
