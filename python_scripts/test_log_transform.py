"""
Quick validation script for generalized concentration transforms.

Run from the repo root with:
  python python_scripts/test_log_transform.py \
    --params /path/to/transform_params.json
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from credit.transforms.concentration import (
    concentration_transform_overrides_stats,
    forward_concentration_transform_numpy,
    inverse_concentration_transform_numpy,
    load_concentration_transform_json,
)


def _expected_physical(raw: np.ndarray, spec: dict) -> np.ndarray:
    transform_type = str(spec["transform_type"])
    if transform_type == "zero_inflated_lognormal_probit":
        zero_floor = float(spec["zero_floor"])
        clip_max = float(spec["clip_max"])
        return np.where(raw < zero_floor, 0.0, np.clip(raw, 0.0, clip_max))
    clip_min = float(spec["clip_min"])
    clip_max = float(spec["clip_max"])
    return np.clip(raw, clip_min, clip_max)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate concentration transform round-trip behavior")
    parser.add_argument(
        "--params",
        default="/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/zero_inflated_transform_params.json",
        help="Path to generalized concentration transform JSON",
    )
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as fh:
        raw_payload = json.load(fh)
    _, specs = load_concentration_transform_json(args.params)

    print(f"{'Variable':<14} {'rt_err':>10} {'inv_finite':>10} {'bounded':>8}  result")
    print("-" * 68)

    all_passed = True
    for var, spec in specs.items():
        if str(spec["transform_type"]) == "zero_inflated_lognormal_probit":
            clip_max = float(spec["clip_max"])
            zero_floor = float(spec["zero_floor"])
            raw = np.array(
                [
                    [0.0, zero_floor * 0.1, zero_floor, zero_floor * 10.0],
                    [1.0e-8, 1.0e-6, 1.0e-4, clip_max],
                ],
                dtype=np.float64,
            )
        else:
            clip_min = float(spec["clip_min"])
            clip_max = float(spec["clip_max"])
            raw = np.array(
                [[clip_min, clip_min * 10.0, 1.0e-6, 1.0e-3], [1.0e-2, 1.0e-1, 1.0, clip_max]],
                dtype=np.float64,
            )

        level_axis = 0 if raw.ndim > 1 else None
        latent = forward_concentration_transform_numpy(raw, spec, level_axis=level_axis)
        recovered = inverse_concentration_transform_numpy(latent, spec, level_axis=level_axis)
        expected = _expected_physical(raw, spec)
        rt_err = float(np.max(np.abs(recovered - expected) / (np.abs(expected) + 1.0e-30)))

        if concentration_transform_overrides_stats(spec):
            latent_grid = np.linspace(-10.0, 20.0, 5000, dtype=np.float64)
        else:
            latent_grid = np.linspace(-8.0, 8.0, 5000, dtype=np.float64)
        x_back = inverse_concentration_transform_numpy(latent_grid, spec)
        inv_finite = not (np.any(np.isnan(x_back)) or np.any(np.isinf(x_back)))
        bounded = bool(np.all(x_back >= 0.0) and np.all(x_back <= float(spec.get("clip_max", np.inf)) * 1.0001))

        passed = rt_err < 1.0e-5 and inv_finite and bounded
        all_passed = all_passed and passed
        print(f"{var:<14} {rt_err:>10.2e} {str(inv_finite):>10} {str(bounded):>8}  {'PASS' if passed else 'FAIL'}")

    print()
    transform_type = raw_payload.get("transform_type", "unknown") if isinstance(raw_payload, dict) else "unknown"
    print("transform_type:", transform_type)
    print("Overall:", "ALL PASSED" if all_passed else "SOME FAILED")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
