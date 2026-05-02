"""Shared concentration-transform utilities for WoFS/CREDIT pipelines."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

CONCENTRATION_VARS = {
    "QRAIN",
    "QNRAIN",
    "QSNOW",
    "QNSNOW",
    "QGRAUP",
    "QNGRAUPEL",
    "QVGRAUPEL",
    "QHAIL",
    "QNHAIL",
    "QVHAIL",
    "QICE",
    "QNICE",
}

DEFAULT_ZERO_FLOOR = 1.0e-11
DEFAULT_PROBIT_EPS = 1.0e-6
DEFAULT_POSITIVE_SIGMA = 1.0
DEFAULT_Q_CLIP_MAX = 1.0
DEFAULT_QN_CLIP_MAX = 1.0e7
LOG_ZSCORE_TRANSFORM = "log_zscore"
ZERO_INFLATED_LOGNORMAL_PROBIT_TRANSFORM = "zero_inflated_lognormal_probit"
_TAIL_PROB_EPS = 1.0e-12

_SQRT_HALF = math.sqrt(0.5)


def default_clip_max_for_var(var_name: str) -> float:
    return DEFAULT_QN_CLIP_MAX if str(var_name).startswith("QN") else DEFAULT_Q_CLIP_MAX


def _numpy_norm_cdf(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    xt = torch.as_tensor(x64, dtype=torch.float64)
    out = 0.5 * (1.0 + torch.erf(xt * _SQRT_HALF))
    return out.cpu().numpy()


def _torch_norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x * _SQRT_HALF))


def _numpy_norm_ppf(p: np.ndarray) -> np.ndarray:
    p64 = np.asarray(p, dtype=np.float64)
    out = np.empty_like(p64, dtype=np.float64)
    mask_bad_low = p64 <= 0.0
    mask_bad_high = p64 >= 1.0
    mask_finite = (~mask_bad_low) & (~mask_bad_high)
    out[mask_bad_low] = -np.inf
    out[mask_bad_high] = np.inf
    if np.any(mask_finite):
        pt = torch.as_tensor(p64[mask_finite], dtype=torch.float64)
        out[mask_finite] = (math.sqrt(2.0) * torch.erfinv(2.0 * pt - 1.0)).cpu().numpy()
    return out


def _torch_norm_ppf(p: torch.Tensor) -> torch.Tensor:
    p64 = p.to(dtype=torch.float64)
    out = torch.empty_like(p64)
    mask_bad_low = p64 <= 0.0
    mask_bad_high = p64 >= 1.0
    mask_finite = (~mask_bad_low) & (~mask_bad_high)
    out[mask_bad_low] = -torch.inf
    out[mask_bad_high] = torch.inf
    if torch.any(mask_finite):
        out[mask_finite] = math.sqrt(2.0) * torch.erfinv(2.0 * p64[mask_finite] - 1.0)
    return out.to(dtype=p.dtype)


def infer_level_axis(shape: tuple[int, ...], n_levels: int, preferred_axis: int = 1) -> int | None:
    matches = [axis for axis, size in enumerate(shape) if size == n_levels]
    if not matches:
        return None
    if preferred_axis in matches:
        return preferred_axis
    return matches[0]


def _broadcast_level_vector_numpy(values: np.ndarray, target_shape: tuple[int, ...], level_axis: int | None) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim == 0:
        return values_arr
    if level_axis is None:
        level_axis = infer_level_axis(target_shape, values_arr.shape[0])
        if level_axis is None:
            return values_arr
    reshape = [1] * len(target_shape)
    reshape[level_axis] = values_arr.shape[0]
    return values_arr.reshape(reshape)


def _broadcast_level_vector_torch(values: np.ndarray, target: torch.Tensor, level_axis: int | None) -> torch.Tensor:
    vec = torch.as_tensor(values, dtype=target.dtype, device=target.device)
    if vec.ndim == 0:
        return vec
    if level_axis is None:
        level_axis = infer_level_axis(tuple(target.shape), int(vec.shape[0]))
        if level_axis is None:
            return vec
    reshape = [1] * target.ndim
    reshape[level_axis] = int(vec.shape[0])
    return vec.reshape(reshape)


def _coerce_log_zscore_spec(var_name: str, info: dict[str, Any]) -> dict[str, Any]:
    clip_min = float(info.get("clip_min", 1.0e-12))
    clip_max = float(info.get("clip_max", default_clip_max_for_var(var_name)))
    log_std = max(float(info.get("log_std", 1.0)), 1.0e-12)
    return {
        "transform_type": LOG_ZSCORE_TRANSFORM,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "log_mean": float(info.get("log_mean", math.log(max(clip_min, 1.0e-12)))),
        "log_std": log_std,
    }


def _coerce_zero_inflated_spec(var_name: str, info: dict[str, Any], default_zero_floor: float, default_probit_eps: float) -> dict[str, Any]:
    zero_floor = max(float(info.get("zero_floor", default_zero_floor)), 1.0e-20)
    probit_eps = min(max(float(info.get("probit_eps", default_probit_eps)), 1.0e-12), 1.0e-2)
    clip_max = float(info.get("clip_max", default_clip_max_for_var(var_name)))
    fallback_raw = info.get("fallback_positive_fit", {}) if isinstance(info.get("fallback_positive_fit"), dict) else {}
    fallback_mu = float(fallback_raw.get("mu", math.log(zero_floor)))
    fallback_sigma = max(float(fallback_raw.get("sigma", DEFAULT_POSITIVE_SIGMA)), 1.0e-6)

    levels_raw = info.get("levels", [])
    if not isinstance(levels_raw, list):
        levels_raw = []

    level_entries: list[dict[str, Any]] = []
    alpha_vals = []
    mu_vals = []
    sigma_vals = []
    status_vals = []
    for raw_level in levels_raw:
        if not isinstance(raw_level, dict):
            raw_level = {}
        status = str(raw_level.get("status", "ok")).strip() or "ok"
        alpha = min(max(float(raw_level.get("alpha", 1.0 if status == "degenerate_zero" else 0.0)), 0.0), 1.0)
        if status == "degenerate_zero":
            alpha = 1.0
            mu = fallback_mu
            sigma = fallback_sigma
        else:
            mu = float(raw_level.get("mu", fallback_mu))
            sigma = max(float(raw_level.get("sigma", fallback_sigma)), 1.0e-6)
        level_entry = {
            "status": status,
            "alpha": alpha,
            "mu": mu,
            "sigma": sigma,
            "n_total": int(raw_level.get("n_total", 0)),
            "n_positive": int(raw_level.get("n_positive", 0)),
        }
        level_entries.append(level_entry)
        alpha_vals.append(alpha)
        mu_vals.append(mu)
        sigma_vals.append(sigma)
        status_vals.append(status)

    if not level_entries:
        level_entries.append(
            {
                "status": "degenerate_zero",
                "alpha": 1.0,
                "mu": fallback_mu,
                "sigma": fallback_sigma,
                "n_total": 0,
                "n_positive": 0,
            }
        )
        alpha_vals.append(1.0)
        mu_vals.append(fallback_mu)
        sigma_vals.append(fallback_sigma)
        status_vals.append("degenerate_zero")

    return {
        "transform_type": ZERO_INFLATED_LOGNORMAL_PROBIT_TRANSFORM,
        "zero_floor": zero_floor,
        "probit_eps": probit_eps,
        "clip_max": clip_max,
        "fallback_positive_fit": {"mu": fallback_mu, "sigma": fallback_sigma},
        "levels": level_entries,
        "alpha": np.asarray(alpha_vals, dtype=np.float64),
        "mu": np.asarray(mu_vals, dtype=np.float64),
        "sigma": np.asarray(sigma_vals, dtype=np.float64),
        "status": status_vals,
    }


def parse_concentration_transform_payload(payload: dict[str, Any], variables: set[str] | None = None) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}

    top_transform_type = str(payload.get("transform_type", "")).strip() or None
    top_zero_floor = float(payload.get("zero_floor", DEFAULT_ZERO_FLOOR))
    top_probit_eps = float(payload.get("probit_eps", DEFAULT_PROBIT_EPS))
    vars_dict = payload.get("variables", payload)
    if not isinstance(vars_dict, dict):
        return {}

    parsed: dict[str, dict[str, Any]] = {}
    for var_name, info in vars_dict.items():
        if variables is not None and var_name not in variables:
            continue
        if not isinstance(info, dict):
            continue
        transform_type = str(info.get("transform_type", top_transform_type or LOG_ZSCORE_TRANSFORM)).strip()
        if transform_type == ZERO_INFLATED_LOGNORMAL_PROBIT_TRANSFORM:
            parsed[var_name] = _coerce_zero_inflated_spec(
                var_name,
                info,
                default_zero_floor=top_zero_floor,
                default_probit_eps=top_probit_eps,
            )
        elif transform_type == LOG_ZSCORE_TRANSFORM:
            parsed[var_name] = _coerce_log_zscore_spec(var_name, info)
    return parsed


def load_concentration_transform_json(path: str | None, variables: set[str] | None = None) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    if not path:
        return None, {}
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload, parse_concentration_transform_payload(payload, variables=variables)


def concentration_transform_overrides_stats(spec: dict[str, Any]) -> bool:
    return str(spec.get("transform_type")) == LOG_ZSCORE_TRANSFORM


def build_log_zscore_stats_override(spec: dict[str, Any], n_levels: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full(int(max(1, n_levels)), float(spec["log_mean"]), dtype=np.float64)
    std = np.full(int(max(1, n_levels)), max(float(spec["log_std"]), 1.0e-12), dtype=np.float64)
    return mean, std


def forward_concentration_transform_numpy(x: np.ndarray, spec: dict[str, Any], level_axis: int | None = None) -> np.ndarray:
    transform_type = str(spec.get("transform_type", ""))
    x64 = np.asarray(x, dtype=np.float64)

    if transform_type == LOG_ZSCORE_TRANSFORM:
        clip_min = float(spec["clip_min"])
        clip_max = float(spec["clip_max"])
        out = np.log(np.clip(x64, clip_min, clip_max))
        return out.astype(x.dtype if hasattr(x, "dtype") else np.float32, copy=False)

    if transform_type != ZERO_INFLATED_LOGNORMAL_PROBIT_TRANSFORM:
        raise ValueError(f"Unsupported concentration transform_type: {transform_type!r}")

    zero_floor = float(spec["zero_floor"])
    probit_eps = float(spec["probit_eps"])
    clip_max = float(spec["clip_max"])
    x_clip = np.clip(np.where(x64 < zero_floor, 0.0, x64), 0.0, clip_max)

    alpha = np.broadcast_to(_broadcast_level_vector_numpy(spec["alpha"], x_clip.shape, level_axis), x_clip.shape)
    mu = np.broadcast_to(_broadcast_level_vector_numpy(spec["mu"], x_clip.shape, level_axis), x_clip.shape)
    sigma = np.broadcast_to(_broadcast_level_vector_numpy(spec["sigma"], x_clip.shape, level_axis), x_clip.shape)
    sigma = np.maximum(sigma, 1.0e-6)

    zero_u = np.clip(alpha * 0.5, probit_eps, 1.0 - probit_eps)
    out_u = np.array(zero_u, copy=True, dtype=np.float64)

    pos_mask = x_clip > 0.0
    if np.any(pos_mask):
        logx = np.log(np.maximum(x_clip[pos_mask], zero_floor))
        alpha_pos = np.asarray(alpha[pos_mask] if np.ndim(alpha) > 0 else alpha, dtype=np.float64)
        mu_pos = np.asarray(mu[pos_mask] if np.ndim(mu) > 0 else mu, dtype=np.float64)
        sigma_pos = np.asarray(sigma[pos_mask] if np.ndim(sigma) > 0 else sigma, dtype=np.float64)
        tail_cdf = _numpy_norm_cdf((logx - mu_pos) / sigma_pos)
        out_u[pos_mask] = alpha_pos + (1.0 - alpha_pos) * tail_cdf

    out_u = np.clip(out_u, probit_eps, 1.0 - probit_eps)
    out = _numpy_norm_ppf(out_u)
    return out.astype(x.dtype if hasattr(x, "dtype") else np.float32, copy=False)


def inverse_concentration_transform_numpy(z: np.ndarray, spec: dict[str, Any], level_axis: int | None = None) -> np.ndarray:
    transform_type = str(spec.get("transform_type", ""))
    z64 = np.asarray(z, dtype=np.float64)

    if transform_type == LOG_ZSCORE_TRANSFORM:
        clip_min = float(spec["clip_min"])
        clip_max = float(spec["clip_max"])
        out = np.clip(np.exp(z64), clip_min, clip_max)
        return out.astype(z.dtype if hasattr(z, "dtype") else np.float32, copy=False)

    if transform_type != ZERO_INFLATED_LOGNORMAL_PROBIT_TRANSFORM:
        raise ValueError(f"Unsupported concentration transform_type: {transform_type!r}")

    zero_floor = float(spec["zero_floor"])
    probit_eps = float(spec["probit_eps"])
    clip_max = float(spec["clip_max"])

    alpha = np.broadcast_to(_broadcast_level_vector_numpy(spec["alpha"], z64.shape, level_axis), z64.shape)
    mu = np.broadcast_to(_broadcast_level_vector_numpy(spec["mu"], z64.shape, level_axis), z64.shape)
    sigma = np.broadcast_to(_broadcast_level_vector_numpy(spec["sigma"], z64.shape, level_axis), z64.shape)
    sigma = np.maximum(sigma, 1.0e-6)

    u = np.clip(_numpy_norm_cdf(z64), probit_eps, 1.0 - probit_eps)
    out = np.zeros_like(z64, dtype=np.float64)
    pos_mask = u > alpha
    if np.any(pos_mask):
        alpha_pos = np.asarray(alpha[pos_mask] if np.ndim(alpha) > 0 else alpha, dtype=np.float64)
        q = np.clip((u[pos_mask] - alpha_pos) / np.maximum(1.0 - alpha_pos, 1.0e-12), _TAIL_PROB_EPS, 1.0 - _TAIL_PROB_EPS)
        mu_pos = np.asarray(mu[pos_mask] if np.ndim(mu) > 0 else mu, dtype=np.float64)
        sigma_pos = np.asarray(sigma[pos_mask] if np.ndim(sigma) > 0 else sigma, dtype=np.float64)
        out[pos_mask] = np.exp(mu_pos + sigma_pos * _numpy_norm_ppf(q))

    out = np.clip(out, 0.0, clip_max)
    out = np.where(out < zero_floor, 0.0, out)
    return out.astype(z.dtype if hasattr(z, "dtype") else np.float32, copy=False)


def inverse_concentration_transform_torch(z: torch.Tensor, spec: dict[str, Any], level_axis: int | None = None) -> torch.Tensor:
    transform_type = str(spec.get("transform_type", ""))

    if transform_type == LOG_ZSCORE_TRANSFORM:
        clip_min = float(spec["clip_min"])
        clip_max = float(spec["clip_max"])
        return torch.clamp(torch.exp(z), min=clip_min, max=clip_max)

    if transform_type != ZERO_INFLATED_LOGNORMAL_PROBIT_TRANSFORM:
        raise ValueError(f"Unsupported concentration transform_type: {transform_type!r}")

    zero_floor = float(spec["zero_floor"])
    probit_eps = float(spec["probit_eps"])
    clip_max = float(spec["clip_max"])

    alpha = torch.broadcast_to(_broadcast_level_vector_torch(spec["alpha"], z, level_axis), z.shape)
    mu = torch.broadcast_to(_broadcast_level_vector_torch(spec["mu"], z, level_axis), z.shape)
    sigma = torch.clamp(torch.broadcast_to(_broadcast_level_vector_torch(spec["sigma"], z, level_axis), z.shape), min=1.0e-6)

    u = torch.clamp(_torch_norm_cdf(z), min=probit_eps, max=1.0 - probit_eps)
    out = torch.zeros_like(z)
    pos_mask = u > alpha
    if torch.any(pos_mask):
        q = torch.clamp(
            (u[pos_mask] - alpha[pos_mask]) / torch.clamp(1.0 - alpha[pos_mask], min=1.0e-12),
            min=_TAIL_PROB_EPS,
            max=1.0 - _TAIL_PROB_EPS,
        )
        out[pos_mask] = torch.exp(mu[pos_mask] + sigma[pos_mask] * _torch_norm_ppf(q))

    out = torch.clamp(out, min=0.0, max=clip_max)
    return torch.where(out < zero_floor, torch.zeros_like(out), out)
