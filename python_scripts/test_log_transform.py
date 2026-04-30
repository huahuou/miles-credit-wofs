"""
Unit test for the log-zscore transform in WoFSDAIncrementDataset.
Run from the repo root with: python python_scripts/test_log_transform.py
"""
import json
import numpy as np
import sys
sys.path.insert(0, "/home/Zhanxiang.Hua/miles-credit-wofs")
from credit.datasets.wrf_wofs_da_increment import WoFSDAIncrementDataset

PARAMS_PATH = "/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/log_transform_params.json"

with open(PARAMS_PATH) as f:
    params_json = json.load(f)

print(f"{'Variable':<14} {'rt_err':>10} {'norm_err':>10} {'no_nan':>6} {'bounded':>8}  result")
print("-" * 65)

all_passed = True
for var, p in params_json["variables"].items():
    # ------------------------------------------------------------------
    # 1. Round-trip: forward → inverse must recover clip(x, min, max)
    # ------------------------------------------------------------------
    test_vals = np.array([p["clip_min"], 1e-10, 1e-8, 1e-6, 1e-3, p["clip_max"]], dtype=np.float32)
    fwd = WoFSDAIncrementDataset._forward_log_numpy(test_vals, p)
    inv = WoFSDAIncrementDataset._inverse_log_numpy(fwd, p)
    expected = np.clip(test_vals, p["clip_min"], p["clip_max"])
    rt_err = float(np.max(np.abs(inv - expected) / (np.abs(expected) + 1e-30)))

    # ------------------------------------------------------------------
    # 2. Full normalize → denormalize cycle with level-replicated stats
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    mask = rng.random((17, 5, 5)) > 0.935
    raw = np.where(mask, np.exp(rng.normal(-20, 5, (17, 5, 5))), p["clip_min"]).astype(np.float32)
    raw = np.clip(raw, p["clip_min"], p["clip_max"])
    lm = np.full(17, p["log_mean"])
    ls = np.full(17, p["log_std"])
    z   = (WoFSDAIncrementDataset._forward_log_numpy(raw, p) - lm[:, None, None]) / ls[:, None, None]
    rec = WoFSDAIncrementDataset._inverse_log_numpy(z * ls[:, None, None] + lm[:, None, None], p)
    norm_err = float(np.max(np.abs(rec - raw) / (np.abs(raw) + 1e-30)))

    # ------------------------------------------------------------------
    # 3. No NaN / Inf across a very wide z range; result stays bounded
    # ------------------------------------------------------------------
    z_wide = np.linspace(-10, 20, 100_000, dtype=np.float64)
    log_x  = z_wide * p["log_std"] + p["log_mean"]
    x_back = WoFSDAIncrementDataset._inverse_log_numpy(log_x, p)
    no_nan  = not (np.any(np.isnan(x_back)) or np.any(np.isinf(x_back)))
    bounded = float(x_back.min()) >= p["clip_min"] * 0.9999 and float(x_back.max()) <= p["clip_max"] * 1.0001

    # ------------------------------------------------------------------
    # 4. Increment denormalization: z0 + dz → q1 must match directly
    # ------------------------------------------------------------------
    q_t0 = raw.copy()
    q_t1 = np.clip(raw * (1.0 + 0.1 * rng.standard_normal((17, 5, 5))), p["clip_min"], p["clip_max"]).astype(np.float32)
    z0 = (WoFSDAIncrementDataset._forward_log_numpy(q_t0, p) - lm[:, None, None]) / ls[:, None, None]
    z1 = (WoFSDAIncrementDataset._forward_log_numpy(q_t1, p) - lm[:, None, None]) / ls[:, None, None]
    dz = z1 - z0
    z1_hat = z0 + dz
    q1_hat = WoFSDAIncrementDataset._inverse_log_numpy(z1_hat * ls[:, None, None] + lm[:, None, None], p)
    incr_err = float(np.max(np.abs(q1_hat - q_t1) / (np.abs(q_t1) + 1e-30)))

    passed = rt_err < 1e-5 and norm_err < 1e-5 and no_nan and bounded and incr_err < 1e-5
    if not passed:
        all_passed = False

    tag = "PASS" if passed else "FAIL"
    print(
        f"{var:<14} {rt_err:>10.2e} {norm_err:>10.2e} {str(no_nan):>6} {str(bounded):>8}  {tag}"
        + (f"  incr_err={incr_err:.2e}" if incr_err >= 1e-5 else "")
    )

print()
print("Overall:", "ALL PASSED" if all_passed else "SOME FAILED")
sys.exit(0 if all_passed else 1)
