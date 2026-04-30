"""
Compute physically meaningful clip values for phys_clip_by_var from:
1. The normalization stats (mean/std + concentration params)
2. A sample of real zarr files to get actual Q value distributions

Outputs a YAML snippet ready to paste into the eval section of model.yml.
"""
from __future__ import annotations

import sys
import os
import json
import numpy as np
import xarray as xr

sys.path.insert(0, "/home/Zhanxiang.Hua/miles-credit-wofs")

MEAN_PATH   = "/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/mean.nc"
STD_PATH    = "/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/std.nc"
CONC_JSON   = "/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/concentration_tuning.json"
NORM_ZARR   = "/scratch5/purged/Zhanxiang.Hua/credit_wofs_da_example/wofs_da_increment_experiment_0423/test5/test_samples_2020.zarr"

PROG_VARS   = ["QRAIN", "QNRAIN", "QHAIL", "QNHAIL", "QGRAUP", "QNGRAUPEL", "QSNOW", "QNSNOW"]

DEFAULT_PARAMS = {"c1": 0.5, "c2": 0.5, "conc_eps": 1e-4, "conc_max": 2.5,
                  "value_clip_min": None, "value_clip_max": None}


def forward_conc(x, p):
    x64 = np.asarray(x, dtype=np.float64)
    c1, c2, eps, cmax = p["c1"], p["c2"], p["conc_eps"], p["conc_max"]
    if p["value_clip_min"] is not None: x64 = np.maximum(x64, p["value_clip_min"])
    if p["value_clip_max"] is not None: x64 = np.minimum(x64, p["value_clip_max"])
    log_eps = np.log(eps)
    return c1 * np.minimum(x64, cmax) + c2 * (np.log(np.maximum(x64, eps)) - log_eps) / (-log_eps)


def inverse_conc(y, p):
    target = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    c1, c2, eps, cmax = p["c1"], p["c2"], p["conc_eps"], p["conc_max"]
    log_eps = np.log(eps)
    neg_log_eps = -log_eps
    y1 = c1 * eps
    y2 = c1 * cmax + c2 * (np.log(cmax) - log_eps) / neg_log_eps
    out = np.empty_like(target)
    m1 = target <= y1
    if np.any(m1): out[m1] = target[m1] / c1
    m3 = target >= y2
    if np.any(m3): out[m3] = eps * np.exp((target[m3] - c1 * cmax) * neg_log_eps / c2)
    m2 = ~m1 & ~m3
    if np.any(m2):
        t2 = target[m2]
        lo, hi = np.full_like(t2, eps), np.full_like(t2, cmax)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            f = c1 * mid + c2 * (np.log(mid) - log_eps) / neg_log_eps
            go_right = f < t2
            lo = np.where(go_right, mid, lo)
            hi = np.where(go_right, hi, mid)
        out[m2] = 0.5 * (lo + hi)
    if p["value_clip_min"] is not None: out = np.maximum(out, p["value_clip_min"])
    if p["value_clip_max"] is not None: out = np.minimum(out, p["value_clip_max"])
    return out


# ── Load stats ────────────────────────────────────────────────────────────────
print("Loading normalization stats...")
mean_ds = xr.open_dataset(MEAN_PATH).load()
std_ds  = xr.open_dataset(STD_PATH).load()

# Load concentration params
conc_params = {v: dict(DEFAULT_PARAMS) for v in PROG_VARS}
try:
    with open(CONC_JSON) as f:
        payload = json.load(f)
    params_dict = payload.get("variables", payload)
    for var, info in params_dict.items():
        if var not in conc_params: continue
        if "recommended" in info and isinstance(info["recommended"], dict):
            conc_params[var].update(info["recommended"])
        else:
            conc_params[var].update({k: info[k] for k in DEFAULT_PARAMS if k in info})
    print(f"Loaded concentration params from {CONC_JSON}")
except Exception as e:
    print(f"Could not load concentration JSON: {e}, using defaults")

# Also check per-variable attrs in mean_ds
for var in PROG_VARS:
    if var in mean_ds:
        attrs = mean_ds[var].attrs
        for key in ("concentration_transform_c1", "concentration_transform_c2",
                    "concentration_transform_conc_eps", "concentration_transform_conc_max"):
            short = key.replace("concentration_transform_", "")
            if key in attrs:
                conc_params[var][short] = float(attrs[key])

# ── Approach 1: from stats — compute what z0 extremes correspond to in Q ──────
print("\n" + "="*70)
print("APPROACH 1: From stats — z0 range → physical Q range")
print("="*70)
print(f"{'var':<12}  {'z0_max_in_data':>14}  {'q_at_z0max':>14}  {'q_99.9th_conc':>14}")

# The normalized space stores z-scores of forward_conc(Q).
# Typical large z0 values from test3: QNRAIN L0 max z0 ≈ +1.3, min ≈ -5
# We'll use the actual stats to figure out the physical maximum
stats_results = {}
for var in PROG_VARS:
    if var not in mean_ds or var not in std_ds:
        print(f"  {var}: not found in stats")
        continue
    mean_v = float(np.asarray(mean_ds[var].values).ravel()[0])
    std_v  = float(np.asarray(std_ds[var].values).ravel()[0])
    p = conc_params[var]

    # Find q corresponding to max physically meaningful forward_conc value = conc_max
    q_at_conc_max = float(p["conc_max"])  # inverse_conc(conc_max) = conc_max itself
    # z0 at conc_max:
    f_at_cmax = float(forward_conc(np.array([q_at_conc_max]), p)[0])
    z0_at_cmax = (f_at_cmax - mean_v) / std_v

    # What's q when z0 = +3 (99.87th percentile of z)?
    f_at_z3 = 3.0 * std_v + mean_v
    q_at_z3 = float(inverse_conc(np.array([max(f_at_z3, 0.0)]), p)[0])

    # What's q when z0 = +5?
    f_at_z5 = 5.0 * std_v + mean_v
    q_at_z5 = float(inverse_conc(np.array([max(f_at_z5, 0.0)]), p)[0])

    stats_results[var] = {"mean_f": mean_v, "std_f": std_v, "params": p,
                          "q_at_z3": q_at_z3, "q_at_z5": q_at_z5,
                          "z0_at_cmax": z0_at_cmax, "conc_max_q": q_at_conc_max}
    print(f"  {var:<12}  z0_at_cmax={z0_at_cmax:+.2f}  q(z0=+3)={q_at_z3:.3e}  q(z0=+5)={q_at_z5:.3e}")

# ── Approach 2: from real zarr — actual Q value distribution ─────────────────
print("\n" + "="*70)
print("APPROACH 2: From eval zarr — actual y_true physical value distribution")
print("="*70)

PHYS_ZARR = "/scratch5/purged/Zhanxiang.Hua/credit_wofs_da_example/wofs_da_increment_experiment_0423/test5/eval_physical.zarr"
clip_values = {}
try:
    ds = xr.open_zarr(PHYS_ZARR)
    channels = list(ds.output_channel.values)

    print(f"\n{'var':<12}  {'y_true_p99.9':>14}  {'y_true_max':>14}  {'y_pred_p99.9':>14}  {'y_pred_max':>14}  {'suggested_clip':>15}")
    for var in PROG_VARS:
        var_channels = [c for c in channels if c.startswith(var + "_L")]
        if not var_channels:
            print(f"  {var}: no channels")
            continue

        # Sample first level (L0) across all samples
        ch = var_channels[0]
        # Load full array for a meaningful percentile estimate — use absolute values since these are increments
        true_vals = np.abs(ds["y_true_phys"].sel(output_channel=ch).values.ravel())
        pred_vals = np.abs(ds["y_pred_phys"].sel(output_channel=ch).values.ravel())

        true_p999 = float(np.nanpercentile(true_vals, 99.9))
        true_max  = float(np.nanmax(true_vals))
        pred_p999 = float(np.nanpercentile(pred_vals, 99.9))
        pred_max  = float(np.nanmax(pred_vals))

        # Also check q_t1 absolute (not increment) via stats approach
        q_at_z3  = stats_results.get(var, {}).get("q_at_z3", float("nan"))
        q_at_z5  = stats_results.get(var, {}).get("q_at_z5", float("nan"))

        # Suggest clip = max of:
        #   - 10× true_p99.9 (generous but finite)
        #   - q(z0=+3) from stats  (physical Q at the outer edge of the normal distribution)
        # For the q_t1 clip, use q(z0=+5) as a soft ceiling — anything beyond is artifact
        suggested = max(10.0 * true_p999, q_at_z3)
        # Round up to a clean power-of-10 scale
        magnitude = 10 ** np.floor(np.log10(suggested))
        suggested_clean = float(np.ceil(suggested / magnitude) * magnitude)

        clip_values[var] = suggested_clean
        print(f"  {var:<12}  {true_p999:>14.3e}  {true_max:>14.3e}  {pred_p999:>14.3e}  {pred_max:>14.3e}  {suggested_clean:>15.3e}")

    ds.close()
except Exception as e:
    print(f"  Could not open physical zarr: {e}")
    print("  Falling back to stats-only approach")
    for var in PROG_VARS:
        r = stats_results.get(var)
        if r:
            clip_values[var] = float(r["q_at_z5"])

# ── Print YAML output ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("YAML snippet for eval.phys_clip_by_var")
print("="*70)
print("  phys_clip_by_var:           # physical-space Q clip (q_t1 >= 0 enforced)")
print("                              # suppresses inverse-conc exponential-Jacobian artifacts")
for var in PROG_VARS:
    if var in clip_values:
        val = clip_values[var]
        # Format: if small use sci notation, if large use sci notation
        if val < 0.1:
            fmt = f"{val:.1e}"
        else:
            fmt = f"{val:.3g}"
        print(f"    {var}: {fmt}")
