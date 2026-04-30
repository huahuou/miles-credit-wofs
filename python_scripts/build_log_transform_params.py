"""
Build log-zscore normalization parameters for concentration variables.

Reads the existing ``concentration_tuning.json`` (produced by
``compute_credit_stats.py``) and derives, **without opening any zarr files**,
the two scalars needed for the simple invertible log-zscore transform:

    Forward:  z(x) = (log(clip(x, clip_min, clip_max)) − log_mean) / log_std
    Inverse:  x̂   = clip(exp(z · log_std + log_mean), clip_min, clip_max)

Both transforms are exact closed-form operations — no bisection, no piecewise
regions, no exponential blow-up in the Jacobian.

Statistical derivation
----------------------
Each concentration variable has ~93–94 % of values at ``clip_min`` (the
"floor").  We model the marginal log-space distribution as a two-component
mixture:

    Y = log(clip(X, clip_min, clip_max))

    Component 1 (spike):   Y = log(clip_min)   with probability α
    Component 2 (tail):    Y | Y > log(clip_min) ~ Normal(μ_cond, σ_cond)

α and the raw quantiles (p99, p99.9) are read directly from the JSON.

The conditional log-normal parameters are estimated from the two available
quantile points using the inverse-normal trick:

    σ_cond = (log(p999) − log(p99)) / (Φ⁻¹(q₂) − Φ⁻¹(q₁))
    μ_cond = log(p99) − σ_cond · Φ⁻¹(q₁)

where  q₁ = (0.99 − α)/(1−α)  and  q₂ = (0.999 − α)/(1−α)  are the
conditional CDF values.  Pooled mixture moments then give:

    log_mean = α · log(clip_min) + (1−α) · μ_cond
    log_std  = sqrt(α·(log(clip_min)−log_mean)²
                    + (1−α)·(σ_cond² + (μ_cond−log_mean)²))

The output JSON can be passed directly to the dataset via the config key
``log_transform_params_json``.

Usage
-----
    python python_scripts/build_log_transform_params.py \\
        --input  /path/to/stats/concentration_tuning.json \\
        --output /path/to/stats/log_transform_params.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Normal percent-point function (inverse CDF) — implemented without scipy
# so this script has zero extra dependencies beyond the stdlib.
# Approximation by Peter J. Acklam (max |err| < 1.15e-9 over [1e-20, 1-1e-20]).
# ---------------------------------------------------------------------------

_NORM_PPF_A = [
    -3.969683028665376e+01,
     2.209460984245205e+02,
    -2.759285104469687e+02,
     1.383577518672690e+02,
    -3.066479806614716e+01,
     2.506628277459239e+00,
]
_NORM_PPF_B = [
    -5.447609879822406e+01,
     1.615858368580409e+02,
    -1.556989798598866e+02,
     6.680131188771972e+01,
    -1.328068155288572e+01,
]
_NORM_PPF_C = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
     4.374664141464968e+00,
     2.938163982698783e+00,
]
_NORM_PPF_D = [
     7.784695709041462e-03,
     3.224671290700398e-01,
     2.445134137142996e+00,
     3.754408661907416e+00,
]

_P_LOW  = 0.02425
_P_HIGH = 1.0 - _P_LOW


def _norm_ppf(p: float) -> float:
    """Inverse CDF of the standard normal distribution (scalar)."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("+inf")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((_NORM_PPF_C[0] * q + _NORM_PPF_C[1]) * q + _NORM_PPF_C[2]) * q
                + _NORM_PPF_C[3]) * q + _NORM_PPF_C[4]) * q + _NORM_PPF_C[5]
        den = (((_NORM_PPF_D[0] * q + _NORM_PPF_D[1]) * q + _NORM_PPF_D[2]) * q
               + _NORM_PPF_D[3]) * q + 1.0
        return num / den
    if p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        num = (((((_NORM_PPF_A[0] * r + _NORM_PPF_A[1]) * r + _NORM_PPF_A[2]) * r
                 + _NORM_PPF_A[3]) * r + _NORM_PPF_A[4]) * r + _NORM_PPF_A[5]) * q
        den = ((((_NORM_PPF_B[0] * r + _NORM_PPF_B[1]) * r + _NORM_PPF_B[2]) * r
                + _NORM_PPF_B[3]) * r + _NORM_PPF_B[4]) * r + 1.0
        return num / den
    # Upper tail: reflect
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = ((((_NORM_PPF_C[0] * q + _NORM_PPF_C[1]) * q + _NORM_PPF_C[2]) * q
            + _NORM_PPF_C[3]) * q + _NORM_PPF_C[4]) * q + _NORM_PPF_C[5]
    den = (((_NORM_PPF_D[0] * q + _NORM_PPF_D[1]) * q + _NORM_PPF_D[2]) * q
           + _NORM_PPF_D[3]) * q + 1.0
    return -(num / den)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_log_stats(var_name: str, var_info: dict) -> dict:
    """
    Estimate log_mean and log_std for one concentration variable.

    Parameters
    ----------
    var_name  : variable name (used only for diagnostics)
    var_info  : entry from concentration_tuning.json["variables"][var_name]

    Returns
    -------
    dict with keys: clip_min, clip_max, log_mean, log_std (+ diagnostics).
    """
    clip_min = float(var_info["value_clip_min"])
    clip_max = float(var_info["value_clip_max"])
    log_clip_min = math.log(clip_min)

    quants = var_info.get("raw_quantiles", {})
    p99  = float(quants.get("p9900", clip_min))   # 99.0th percentile
    p999 = float(quants.get("p9990", clip_min))   # 99.9th percentile

    # alpha = fraction at or below clip_min (the "floor")
    # Best available estimate: low_clamp_fraction from the recommended config.
    rec = var_info.get("recommended", {})
    alpha_raw = rec.get("low_clamp_fraction", None)
    if alpha_raw is None:
        # Fallback: p5000 = clip_min → alpha ≥ 0.5; use 0.935 as a conservative estimate
        alpha_raw = 0.935
    alpha = float(alpha_raw)
    alpha = max(0.0, min(0.9999, alpha))

    # Degenerate: essentially all values at floor
    if p99 <= clip_min * 1.001:
        return {
            "clip_min": clip_min,
            "clip_max": clip_max,
            "log_mean": log_clip_min,
            "log_std": 1.0,
            "alpha": alpha,
            "degenerate": True,
        }

    # Conditional quantile levels in the tail (x > clip_min)
    q1_cond = max(1e-9, min(1 - 1e-9, (0.99  - alpha) / (1.0 - alpha)))
    q2_cond = max(1e-9, min(1 - 1e-9, (0.999 - alpha) / (1.0 - alpha)))

    z1 = _norm_ppf(q1_cond)
    z2 = _norm_ppf(q2_cond)

    log_p99  = math.log(max(p99,  clip_min))
    log_p999 = math.log(max(p999, clip_min))

    # Conditional log-normal parameters (2-point quantile fit)
    dz = z2 - z1
    if abs(dz) < 1e-10 or log_p999 <= log_p99:
        sigma_cond = max(abs(log_p999 - log_p99), 0.1)
        mu_cond = log_p99
    else:
        sigma_cond = (log_p999 - log_p99) / dz
        mu_cond = log_p99 - sigma_cond * z1

    # Mixture moments
    log_mean = alpha * log_clip_min + (1.0 - alpha) * mu_cond
    var_spike = (log_clip_min - log_mean) ** 2
    var_tail  = sigma_cond ** 2 + (mu_cond - log_mean) ** 2
    var_total = alpha * var_spike + (1.0 - alpha) * var_tail
    log_std   = math.sqrt(max(var_total, 1e-10))

    # Sanity: check that p99 maps to z ≈ z1 after the full zscore
    z_check_p99  = (log_p99  - log_mean) / log_std
    z_check_p999 = (log_p999 - log_mean) / log_std

    return {
        "clip_min":  clip_min,
        "clip_max":  clip_max,
        "log_mean":  log_mean,
        "log_std":   log_std,
        "alpha":     alpha,
        "mu_cond":   mu_cond,
        "sigma_cond": sigma_cond,
        # Diagnostics: where do p99/p99.9 land in normalized space?
        "z_at_p99":  z_check_p99,
        "z_at_p999": z_check_p999,
        "degenerate": False,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to concentration_tuning.json (output of compute_credit_stats.py).",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to write log_transform_params.json.",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        raw = json.load(fh)

    variables_data: dict = raw.get("variables", raw)  # handle flat or nested layout

    result_vars: dict[str, dict] = {}
    print(f"{'Variable':<14} {'clip_min':>12} {'clip_max':>12} {'alpha':>7} "
          f"{'log_mean':>10} {'log_std':>8} {'z@p99':>8} {'z@p999':>8}")
    print("-" * 85)

    for var_name, var_info in variables_data.items():
        if not isinstance(var_info, dict):
            continue
        if var_info.get("status") != "ok":
            print(f"  {var_name}: skipped (status={var_info.get('status')})")
            continue

        params = _compute_log_stats(var_name, var_info)
        result_vars[var_name] = params

        degen_tag = " [DEGENERATE]" if params.get("degenerate") else ""
        print(
            f"{var_name:<14} {params['clip_min']:>12.2e} {params['clip_max']:>12.2e} "
            f"{params['alpha']:>7.4f} {params['log_mean']:>10.4f} {params['log_std']:>8.4f} "
            f"{params.get('z_at_p99', float('nan')):>8.3f} "
            f"{params.get('z_at_p999', float('nan')):>8.3f}"
            f"{degen_tag}"
        )

    output = {
        "transform_type": "log_zscore",
        "description": (
            "Simple invertible log-zscore transform for concentration variables. "
            "Forward: z = (log(clip(x, clip_min, clip_max)) - log_mean) / log_std. "
            "Inverse: x = clip(exp(z * log_std + log_mean), clip_min, clip_max)."
        ),
        "source_file": str(Path(args.input).resolve()),
        "variables": result_vars,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nWrote {len(result_vars)} variable(s) → {out_path}")


if __name__ == "__main__":
    main()
