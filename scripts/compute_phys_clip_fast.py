"""
Fast version: read NORMALIZED test zarr to find z0_max for all 8 Q variables,
then compute q(z0_max) and suggest clip values.
Does NOT load the heavy physical zarr.
"""
from __future__ import annotations
import sys, json
import numpy as np
import xarray as xr

MEAN_PATH = "/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/mean.nc"
STD_PATH  = "/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/std.nc"
CONC_JSON = "/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/concentration_tuning.json"
NORM_ZARR = "/scratch5/purged/Zhanxiang.Hua/credit_wofs_da_example/wofs_da_increment_experiment_0423/test5/test_samples_2020.zarr"

PROG_VARS  = ["QRAIN", "QNRAIN", "QHAIL", "QNHAIL", "QGRAUP", "QNGRAUPEL", "QSNOW", "QNSNOW"]
N_LEVELS   = 17
DEFAULT_P  = {"c1": 0.5, "c2": 0.5, "conc_eps": 1e-4, "conc_max": 2.5,
              "value_clip_min": None, "value_clip_max": None}

# ── helpers ──────────────────────────────────────────────────────────────────
def inverse_conc(y, p):
    target = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    c1, c2, eps, cmax = p["c1"], p["c2"], p["conc_eps"], p["conc_max"]
    log_eps = np.log(eps); neg_log_eps = -log_eps
    y1 = c1 * eps
    y2 = c1 * cmax + c2 * (np.log(cmax) - log_eps) / neg_log_eps
    out = np.empty_like(target)
    m1 = target <= y1
    if np.any(m1): out[m1] = target[m1] / c1
    m3 = target >= y2
    if np.any(m3): out[m3] = eps * np.exp((target[m3] - c1 * cmax) * neg_log_eps / c2)
    m2 = ~m1 & ~m3
    if np.any(m2):
        t2, lo, hi = target[m2], np.full_like(target[m2], eps), np.full_like(target[m2], cmax)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            f = c1 * mid + c2 * (np.log(mid) - log_eps) / neg_log_eps
            go = f < t2; lo = np.where(go, mid, lo); hi = np.where(go, hi, mid)
        out[m2] = 0.5 * (lo + hi)
    return out

# ── Load stats ────────────────────────────────────────────────────────────────
print("Loading normalization stats (level 0 only)...")
mean_ds = xr.open_dataset(MEAN_PATH)
std_ds  = xr.open_dataset(STD_PATH)

conc_params = {v: dict(DEFAULT_P) for v in PROG_VARS}
try:
    with open(CONC_JSON) as f: payload = json.load(f)
    params_dict = payload.get("variables", payload)
    for var, info in params_dict.items():
        if var not in conc_params: continue
        src = info.get("recommended", info)
        conc_params[var].update({k: src[k] for k in DEFAULT_P if k in src})
    print(f"Loaded conc params from JSON")
except Exception as e:
    print(f"Conc JSON error: {e}, using defaults")

# Per-variable level-0 stats
stats = {}
for var in PROG_VARS:
    if var in mean_ds and var in std_ds:
        m = float(np.asarray(mean_ds[var].values).ravel()[0])
        s = float(np.asarray(std_ds[var].values).ravel()[0])
        stats[var] = (m, s)
        p = conc_params[var]
        cmax = p["conc_max"]; c1 = p["c1"]; c2 = p["c2"]
        eps  = p["conc_eps"]; log_eps = np.log(eps)
        y2   = c1*cmax + c2*(np.log(cmax)-log_eps)/(-log_eps)
        z_cmax = (y2 - m) / s
        print(f"  {var:<12} mean_f={m:.4f} std_f={s:.4f} z0_at_cmax={z_cmax:+.2f} "
              f"y2={y2:.4f} conc_max={cmax} eps={eps}")

# ── Load normalized zarr ─────────────────────────────────────────────────────
print(f"\nOpening normalized zarr: {NORM_ZARR}")
ds = xr.open_zarr(NORM_ZARR)
print(f"  Variables: {list(ds.data_vars)}")
channels = list(ds["input_channel"].values) if "input_channel" in ds else list(ds.coords.get("input_channel", []))

# Find x_input for prognostic channels
print(f"\n{'var':<12}  {'z0_max':>8}  {'z0_min':>8}  {'q(z0_max)':>12}  {'suggested_clip':>15}")
clip_values = {}

for var in PROG_VARS:
    var_chs = [c for c in channels if str(c).startswith(var + "_L")][:N_LEVELS]
    if not var_chs:
        print(f"  {var}: no channels found in {channels[:5]}...")
        continue
    
    mean_v, std_v = stats.get(var, (None, None))
    if mean_v is None:
        print(f"  {var}: no stats")
        continue
    
    p = conc_params[var]
    
    # Load ALL samples for all levels (axis 0=samples, axis 1=spatial) for this var
    # Use a subsample if too large
    z0_global_max = -np.inf
    z0_global_min = np.inf
    for ch in var_chs:
        ch_data = ds["x_input"].sel(input_channel=ch).values  # shape: (N, H, W) or (N, spatial)
        z0_global_max = max(z0_global_max, float(np.nanmax(ch_data)))
        z0_global_min = min(z0_global_min, float(np.nanmin(ch_data)))
    
    # Also look at y_target
    out_channels = [c for c in (list(ds.coords.get("output_channel", [])) 
                                 if "output_channel" in ds.coords 
                                 else []) if str(c).startswith(var + "_L")][:N_LEVELS]
    y_max_phys = None
    if out_channels and "y_target" in ds:
        # y_target is in normalized delta space; convert z1 = z0 + dz
        # just get the range of z1 = z0 + dz values
        for i, (ich, och) in enumerate(zip(var_chs[:len(out_channels)], out_channels)):
            x_ch = ds["x_input"].sel(input_channel=ich).values
            y_ch = ds["y_target"].sel(output_channel=och).values
            z1 = x_ch + y_ch  # z_t1 = z_t0 + delta_z
            if i == 0:
                z1_max = float(np.nanmax(np.abs(z1)))
    
    # Compute q(z0_max) using actual stats for level 0
    f_at_z0max = z0_global_max * std_v + mean_v
    q_at_z0max = float(inverse_conc(np.array([max(f_at_z0max, 0.0)]), p)[0])
    
    # Clip = 5× q(z0_max), rounded up to nearest power-of-10 scale
    suggested = 5.0 * q_at_z0max
    if suggested > 0:
        mag = 10 ** np.floor(np.log10(suggested))
        clip = float(np.ceil(suggested / mag) * mag)
    else:
        clip = 1e-3
    clip_values[var] = clip
    
    print(f"  {var:<12}  {z0_global_max:>8.3f}  {z0_global_min:>8.3f}  {q_at_z0max:>12.3e}  {clip:>15.3e}")

# ── YAML output ───────────────────────────────────────────────────────────────
print("\n======================================================================")
print("YAML snippet:")
print("======================================================================")
print("  phys_clip_by_var:")
for var in PROG_VARS:
    if var in clip_values:
        v = clip_values[var]
        print(f"    {var}: {v:.2e}")
