from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
import xarray as xr
from credit.transforms.concentration import (
    forward_concentration_transform_numpy,
    load_concentration_transform_json,
)

def _zarr_open(file_path: str, **kwargs):
    file_path = str(file_path)
    if file_path.endswith(".zarr.zip") or file_path.endswith(".zarr.zip/"):
        zip_file = file_path.rstrip("/")
        zip_basename = Path(zip_file).stem
        zip_roots = [f"zip://{zip_basename}::{zip_file}", f"zip://{zip_basename}/::{zip_file}"]

        last_exc = None
        for uri in zip_roots:
            try:
                return xr.open_zarr(uri, consolidated=True, zarr_format=2, **kwargs)
            except Exception as exc:
                last_exc = exc

        for uri in zip_roots:
            try:
                return xr.open_zarr(uri, consolidated=False, zarr_format=2, **kwargs)
            except Exception as exc:
                last_exc = exc

        raise last_exc
    return xr.open_zarr(file_path, consolidated=True, zarr_format=2, **kwargs)

from dask.distributed import Client, LocalCluster, as_completed


_CASE_DATE_RE = re.compile(r"wofs_(?:boundary_)?(\d{8})_.+_mem\d+\.zarr(?:\.zip)?/?$")
_CONCENTRATION_VARS = {
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
_DEFAULT_CONCENTRATION_PARAMS = {
    "c1": 0.5,
    "c2": 0.5,
    "conc_max": 2.5,
    "conc_eps": 1e-4,
    "value_clip_min": None,
    "value_clip_max": None,
}


def _parse_yyyymmdd(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if not re.fullmatch(r"\d{8}", value):
        raise ValueError(f"Expected YYYYMMDD date string, got: {value}")
    return int(value)


def _extract_case_date(file_path: str) -> int:
    match = _CASE_DATE_RE.search(file_path)
    if match is None:
        raise ValueError(
            "Could not infer case date from file path. Expected names like "
            f"'wofs_YYYYMMDD_<init_slug>_memNN.zarr(.zip)' or 'wofs_boundary_YYYYMMDD_<init_slug>_memNN.zarr(.zip)', got: {file_path}"
        )
    return int(match.group(1))


def _filter_files_by_date(files: list[str], start_date: int | None, end_date: int | None) -> list[str]:
    filtered = []
    for file_path in files:
        case_date = _extract_case_date(file_path)
        if start_date is not None and case_date < start_date:
            continue
        if end_date is not None and case_date > end_date:
            continue
        filtered.append(file_path)
    return filtered


def _chunk_files(files: list[str], files_per_task: int) -> list[list[str]]:
    if files_per_task <= 0:
        raise ValueError(f"files_per_task must be positive, got {files_per_task}")
    return [files[i : i + files_per_task] for i in range(0, len(files), files_per_task)]


def _parse_float_list(value: str) -> list[float]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Expected at least one numeric value")
    return out


def _select_files(files: list[str], max_files: int, seed: int) -> list[str]:
    if max_files <= 0:
        raise ValueError(f"max_files must be positive, got {max_files}")
    if len(files) <= max_files:
        return files
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(files), size=max_files, replace=False)
    indices.sort()
    return [files[i] for i in indices]


def _sample_array(arr: np.ndarray, target_size: int, rng: np.random.Generator) -> np.ndarray:
    if arr.size <= target_size:
        return arr
    idx = rng.choice(arr.size, size=target_size, replace=False)
    return arr[idx]


def _build_accumulator_schema(ds: xr.Dataset, data_vars: list[str]) -> dict[str, dict[str, np.ndarray]]:
    accum = {}
    for var in data_vars:
        da = ds[var]
        shape = (da.sizes["level"],) if "level" in da.dims else ()
        accum[var] = {
            "count": np.zeros(shape, dtype=np.float64),
            "sum": np.zeros(shape, dtype=np.float64),
            "sum_sq": np.zeros(shape, dtype=np.float64),
        }
    return accum


def _merge_accumulators(
    accum: dict[str, dict[str, np.ndarray]],
    partial: dict[str, dict[str, np.ndarray]],
) -> None:
    for var in accum:
        accum[var]["count"] += partial[var]["count"]
        accum[var]["sum"] += partial[var]["sum"]
        accum[var]["sum_sq"] += partial[var]["sum_sq"]


def _transform_concentration(da: xr.DataArray, params: dict[str, float] | None = None) -> xr.DataArray:
    """Apply the CREDIT concentration-variable transform before statistics."""

    arr = np.asarray(da.values, dtype=np.float64)
    level_axis = da.dims.index("level") if "level" in da.dims else None
    transform_spec = None
    if params is not None and "transform_type" in params:
        transform_spec = params
    if transform_spec is not None:
        transformed = forward_concentration_transform_numpy(arr, transform_spec, level_axis=level_axis)
        return xr.DataArray(transformed, dims=da.dims, coords=da.coords)

    merged = dict(_DEFAULT_CONCENTRATION_PARAMS)
    if params is not None:
        merged.update(params)
    c1 = float(merged["c1"])
    c2 = float(merged["c2"])
    conc_max = float(merged["conc_max"])
    conc_eps = float(merged["conc_eps"])
    value_clip_min = merged.get("value_clip_min")
    value_clip_max = merged.get("value_clip_max")
    if value_clip_min is not None:
        arr = np.maximum(arr, float(value_clip_min))
    if value_clip_max is not None:
        arr = np.minimum(arr, float(value_clip_max))
    log_eps = math.log(conc_eps)
    neg_log_eps = -log_eps
    transformed = c1 * np.minimum(arr, conc_max) + c2 * (np.log(np.maximum(arr, conc_eps)) - log_eps) / neg_log_eps
    return xr.DataArray(transformed, dims=da.dims, coords=da.coords)


def _get_transform_clip_bounds(params: dict[str, float] | None) -> tuple[float | None, float | None]:
    if not isinstance(params, dict):
        return None, None
    if "transform_type" in params:
        transform_type = str(params.get("transform_type", "")).strip()
        if transform_type == "zero_inflated_lognormal_probit":
            return float(params.get("zero_floor", 0.0)), float(params.get("clip_max", np.inf))
        if transform_type == "log_zscore":
            return float(params.get("clip_min", 0.0)), float(params.get("clip_max", np.inf))
    return params.get("value_clip_min"), params.get("value_clip_max")


def _score_concentration_candidate(
    x: np.ndarray,
    c1: float,
    c2: float,
    conc_max: float,
    conc_eps: float,
    target_low_clamp: float,
    target_high_clamp: float,
    min_spread: float,
) -> dict[str, float]:
    log_eps = math.log(conc_eps)
    neg_log_eps = -log_eps
    transformed = c1 * np.minimum(x, conc_max) + c2 * (np.log(np.maximum(x, conc_eps)) - log_eps) / neg_log_eps
    p01, p99 = np.quantile(transformed, [0.01, 0.99])
    mean = float(np.mean(transformed))
    std = float(np.std(transformed))
    skew = 0.0
    if std > 0:
        skew = float(np.mean(((transformed - mean) / std) ** 3))
    spread = float(p99 - p01)
    low_clamp_fraction = float(np.mean(x <= conc_eps))
    high_clamp_fraction = float(np.mean(x >= conc_max))

    score = (
        3.0 * abs(low_clamp_fraction - target_low_clamp)
        + 2.0 * abs(high_clamp_fraction - target_high_clamp)
        + 4.0 * max(0.0, min_spread - spread)
        + 0.2 * abs(skew)
    )
    return {
        "score": float(score),
        "c1": float(c1),
        "c2": float(c2),
        "conc_max": float(conc_max),
        "conc_eps": float(conc_eps),
        "low_clamp_fraction": low_clamp_fraction,
        "high_clamp_fraction": high_clamp_fraction,
        "transformed_spread_p99_p01": spread,
        "transformed_skew": skew,
    }


def _auto_tune_concentration_params(
    files: list[str],
    max_files: int,
    samples_per_file_per_var: int,
    seed: int,
    eps_quantiles: list[float],
    max_quantiles: list[float],
    c1_values: list[float],
    target_low_clamp: float,
    target_high_clamp: float,
    min_spread: float,
) -> dict[str, dict[str, float]]:
    selected_files = _select_files(files, max_files=max_files, seed=seed)
    print(f"[tune] selected_files={len(selected_files)} max_files={max_files} seed={seed}")
    rng = np.random.default_rng(seed)
    samples: dict[str, list[np.ndarray]] = {var: [] for var in _CONCENTRATION_VARS}

    for idx, file_path in enumerate(selected_files, start=1):
        print(f"[tune] sampling file {idx}/{len(selected_files)}: {file_path}")
        ds = _zarr_open(file_path)
        try:
            for var in _CONCENTRATION_VARS:
                if var not in ds.data_vars:
                    continue
                arr = np.asarray(ds[var].values, dtype=np.float64).ravel()
                finite = np.isfinite(arr)
                if not np.any(finite):
                    continue
                arr = arr[finite]
                arr = arr[arr >= 0.0]
                if arr.size == 0:
                    continue
                arr = _sample_array(arr, samples_per_file_per_var, rng)
                samples[var].append(arr)
        finally:
            ds.close()

    tuned: dict[str, dict[str, float]] = {}
    for var in sorted(_CONCENTRATION_VARS):
        if not samples[var]:
            tuned[var] = dict(_DEFAULT_CONCENTRATION_PARAMS)
            continue

        x = np.concatenate(samples[var], axis=0)
        x_pos = x[x > 0]
        if x_pos.size == 0:
            tuned[var] = dict(_DEFAULT_CONCENTRATION_PARAMS)
            continue

        eps_candidates = [1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4]
        eps_candidates.extend(float(np.quantile(x_pos, q)) for q in eps_quantiles)
        eps_candidates = sorted({float(e) for e in eps_candidates if np.isfinite(e) and 0.0 < e < 1.0})
        if not eps_candidates:
            eps_candidates = [1e-6, 1e-5, 1e-4]

        max_candidates = [0.5, 1.0, 2.5, 5.0]
        max_candidates.extend(float(np.quantile(x_pos, q)) for q in max_quantiles)
        max_candidates = sorted({float(m) for m in max_candidates if np.isfinite(m) and m > 0.0})
        if not max_candidates:
            max_candidates = [2.5]

        best: dict[str, float] | None = None
        for c1 in c1_values:
            c2 = 1.0 - c1
            for conc_eps in eps_candidates:
                for conc_max in max_candidates:
                    if conc_max <= conc_eps:
                        continue
                    candidate = _score_concentration_candidate(
                        x=x,
                        c1=float(c1),
                        c2=float(c2),
                        conc_max=float(conc_max),
                        conc_eps=float(conc_eps),
                        target_low_clamp=target_low_clamp,
                        target_high_clamp=target_high_clamp,
                        min_spread=min_spread,
                    )
                    if best is None or candidate["score"] < best["score"]:
                        best = candidate

        if best is None:
            tuned[var] = dict(_DEFAULT_CONCENTRATION_PARAMS)
            continue

        tuned[var] = {
            "c1": float(best["c1"]),
            "c2": float(best["c2"]),
            "conc_eps": float(best["conc_eps"]),
            "conc_max": float(best["conc_max"]),
        }
        print(
            f"[tune] best {var}: score={best['score']:.5f} c1={best['c1']:.3f} c2={best['c2']:.3f} "
            f"conc_eps={best['conc_eps']:.3e} conc_max={best['conc_max']:.3e}"
        )

    return tuned


def _load_concentration_params_json(path: str) -> dict[str, dict[str, float]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "variables" in payload and isinstance(payload["variables"], dict):
        out: dict[str, dict[str, float]] = {}
        for var, info in payload["variables"].items():
            if not isinstance(info, dict):
                continue
            if "recommended" in info and isinstance(info["recommended"], dict):
                rec = info["recommended"]
                out[var] = {
                    "c1": float(rec.get("c1", _DEFAULT_CONCENTRATION_PARAMS["c1"])),
                    "c2": float(rec.get("c2", _DEFAULT_CONCENTRATION_PARAMS["c2"])),
                    "conc_max": float(rec.get("conc_max", _DEFAULT_CONCENTRATION_PARAMS["conc_max"])),
                    "conc_eps": float(rec.get("conc_eps", _DEFAULT_CONCENTRATION_PARAMS["conc_eps"])),
                    "value_clip_min": rec.get(
                        "value_clip_min",
                        info.get("value_clip_min", _DEFAULT_CONCENTRATION_PARAMS["value_clip_min"]),
                    ),
                    "value_clip_max": rec.get(
                        "value_clip_max",
                        info.get("value_clip_max", _DEFAULT_CONCENTRATION_PARAMS["value_clip_max"]),
                    ),
                }
            elif {"c1", "c2", "conc_max", "conc_eps"}.issubset(info.keys()):
                out[var] = {
                    "c1": float(info["c1"]),
                    "c2": float(info["c2"]),
                    "conc_max": float(info["conc_max"]),
                    "conc_eps": float(info["conc_eps"]),
                    "value_clip_min": info.get("value_clip_min", _DEFAULT_CONCENTRATION_PARAMS["value_clip_min"]),
                    "value_clip_max": info.get("value_clip_max", _DEFAULT_CONCENTRATION_PARAMS["value_clip_max"]),
                }
        return out
    if isinstance(payload, dict):
        out: dict[str, dict[str, float]] = {}
        for var, info in payload.items():
            if not isinstance(info, dict):
                continue
            if {"c1", "c2", "conc_max", "conc_eps"}.issubset(info.keys()):
                out[var] = {
                    "c1": float(info["c1"]),
                    "c2": float(info["c2"]),
                    "conc_max": float(info["conc_max"]),
                    "conc_eps": float(info["conc_eps"]),
                    "value_clip_min": info.get("value_clip_min", _DEFAULT_CONCENTRATION_PARAMS["value_clip_min"]),
                    "value_clip_max": info.get("value_clip_max", _DEFAULT_CONCENTRATION_PARAMS["value_clip_max"]),
                }
        return out
    raise ValueError(f"Unsupported concentration params JSON structure in {path}")


def _accumulate(stats, da: xr.DataArray, valid_mask: np.ndarray | None = None):
    arr = da.values
    finite = np.isfinite(arr)
    if valid_mask is not None:
        finite = finite & valid_mask
    if "level" in da.dims:
        reduce_axes = tuple(i for i, dim in enumerate(da.dims) if dim != "level")
    else:
        reduce_axes = tuple(range(arr.ndim))

    count = finite.sum(axis=reduce_axes)
    sums = np.where(finite, arr, 0.0).sum(axis=reduce_axes)
    sums_sq = np.where(finite, arr * arr, 0.0).sum(axis=reduce_axes)

    stats["count"] += count
    stats["sum"] += sums
    stats["sum_sq"] += sums_sq


def _compute_partial_stats(
    file_batch: list[str],
    data_vars: list[str],
    concentration_params: dict[str, dict[str, float]],
    concentration_transform_specs: dict[str, dict],
    exclude_clipped_boundary_samples: bool,
    clip_boundary_tolerance: float,
) -> dict[str, dict[str, np.ndarray]]:
    if not file_batch:
        raise ValueError("Received empty file batch")

    first = _zarr_open(file_batch[0])
    try:
        accum = _build_accumulator_schema(first, data_vars)
    finally:
        first.close()

    for file_path in file_batch:
        ds = _zarr_open(file_path)
        try:
            for var in data_vars:
                da = ds[var]
                valid_mask = None
                if var in _CONCENTRATION_VARS:
                    params = concentration_transform_specs.get(var)
                    if params is None:
                        params = concentration_params.get(var)
                    if exclude_clipped_boundary_samples and params is not None:
                        clip_min, clip_max = _get_transform_clip_bounds(params)
                        if clip_min is not None or clip_max is not None:
                            raw = np.asarray(da.values, dtype=np.float64)
                            valid_mask = np.isfinite(raw)
                            tol = float(clip_boundary_tolerance)
                            if clip_min is not None:
                                valid_mask = valid_mask & (raw > float(clip_min) + tol)
                            if clip_max is not None:
                                valid_mask = valid_mask & (raw < float(clip_max) - tol)
                    da = _transform_concentration(da, params)
                _accumulate(accum[var], da, valid_mask=valid_mask)
        finally:
            ds.close()

    return accum


def main():
    parser = argparse.ArgumentParser(description="Compute CREDIT-style mean/std files from converted WoFS CREDIT-WRF zarr archives")
    parser.add_argument("--glob", required=True, help="Glob matching converted WoFS CREDIT-WRF zarr files")
    parser.add_argument("--start-date", help="Inclusive lower date bound in YYYYMMDD format")
    parser.add_argument("--end-date", help="Inclusive upper date bound in YYYYMMDD format")
    parser.add_argument("--mean-out", required=True, help="Output path for mean NetCDF")
    parser.add_argument("--std-out", required=True, help="Output path for std NetCDF")
    parser.add_argument("--n-workers", type=int, default=1, help="Number of Dask worker processes to use")
    parser.add_argument("--threads-per-worker", type=int, default=1, help="Threads per Dask worker")
    parser.add_argument(
        "--memory-limit",
        default="4GiB",
        help="Per-worker memory limit passed to the local Dask cluster, e.g. 2GiB or 8000MiB",
    )
    parser.add_argument(
        "--files-per-task",
        type=int,
        default=4,
        help="How many zarr stores each worker processes sequentially per submitted task",
    )
    parser.add_argument(
        "--latweights-out",
        help="Optional output path for a placeholder latitude-weights file. Use with `use_latitude_weights: False` in CREDIT.",
    )
    parser.add_argument(
        "--transform-params-json",
        "--log-transform-params-json",
        dest="transform_params_json",
        help="Optional generalized concentration transform JSON path (e.g. zero-inflated probit or log-zscore)",
    )
    parser.add_argument(
        "--concentration-params-json",
        help="Optional JSON path with per-variable concentration transform params (c1,c2,conc_eps,conc_max)",
    )
    parser.add_argument(
        "--auto-tune-concentration",
        action="store_true",
        help="Auto-tune concentration transform params from a subset of files before computing statistics",
    )
    parser.add_argument("--tune-max-files", type=int, default=150, help="Max files to sample when auto-tuning")
    parser.add_argument(
        "--tune-samples-per-file-per-var",
        type=int,
        default=20000,
        help="Max random samples per variable per sampled file when auto-tuning",
    )
    parser.add_argument("--tune-seed", type=int, default=1234, help="Random seed for auto-tuning")
    parser.add_argument(
        "--tune-eps-quantiles",
        default="0.0005,0.001,0.002,0.005,0.01,0.02,0.05",
        help="Comma-separated quantiles used to build conc_eps candidates during auto-tuning",
    )
    parser.add_argument(
        "--tune-max-quantiles",
        default="0.95,0.98,0.99,0.995,0.999",
        help="Comma-separated quantiles used to build conc_max candidates during auto-tuning",
    )
    parser.add_argument(
        "--tune-c1-values",
        default="0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated c1 candidates during auto-tuning. c2 is set to (1-c1)",
    )
    parser.add_argument(
        "--tune-target-low-clamp",
        type=float,
        default=0.025,
        help="Target fraction of samples clamped at conc_eps during auto-tuning",
    )
    parser.add_argument(
        "--tune-target-high-clamp",
        type=float,
        default=0.025,
        help="Target fraction of samples capped at conc_max during auto-tuning",
    )
    parser.add_argument(
        "--tune-min-spread",
        type=float,
        default=0.4,
        help="Minimum desired transformed spread p99-p01 during auto-tuning",
    )
    parser.add_argument(
        "--tune-json-out",
        help="Optional JSON output path for the auto-tuned concentration parameters",
    )
    parser.add_argument(
        "--exclude-clipped-boundary-samples",
        action="store_true",
        help="Exclude concentration samples at/over clip_min or clip_max from mean/std accumulation",
    )
    parser.add_argument(
        "--clip-boundary-tolerance",
        type=float,
        default=0.0,
        help="Tolerance used with --exclude-clipped-boundary-samples when testing clip boundaries",
    )
    args = parser.parse_args()

    selected_transform_opts = sum(
        1
        for enabled in (
            bool(args.transform_params_json),
            bool(args.concentration_params_json),
            bool(args.auto_tune_concentration),
        )
        if enabled
    )
    if selected_transform_opts > 1:
        raise SystemExit(
            "Use only one of --transform-params-json/--log-transform-params-json, "
            "--concentration-params-json, or --auto-tune-concentration"
        )

    files = sorted(glob.glob(args.glob))
    start_date = _parse_yyyymmdd(args.start_date)
    end_date = _parse_yyyymmdd(args.end_date)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise SystemExit("--start-date must be <= --end-date")

    files = _filter_files_by_date(files, start_date, end_date)
    if not files:
        raise SystemExit(
            f"No files matched: {args.glob} after applying date filter "
            f"start={args.start_date!r} end={args.end_date!r}"
        )

    concentration_params: dict[str, dict[str, float]] = {
        var: dict(_DEFAULT_CONCENTRATION_PARAMS) for var in _CONCENTRATION_VARS
    }
    concentration_transform_specs: dict[str, dict] = {}
    concentration_params_source = "default"
    raw_transform_payload: dict | None = None
    if args.transform_params_json:
        raw_transform_payload, concentration_transform_specs = load_concentration_transform_json(
            args.transform_params_json,
            variables=_CONCENTRATION_VARS,
        )
        concentration_params_source = f"transform_json:{args.transform_params_json}"
        print(
            f"[transform] loaded generalized transform specs for "
            f"{len(concentration_transform_specs)} variables from {args.transform_params_json}"
        )
    if args.concentration_params_json:
        loaded = _load_concentration_params_json(args.concentration_params_json)
        for var, params in loaded.items():
            if var in _CONCENTRATION_VARS:
                concentration_params[var].update(params)
        concentration_params_source = f"json:{args.concentration_params_json}"
        print(f"[transform] loaded params for {len(loaded)} variables from {args.concentration_params_json}")
    elif args.auto_tune_concentration:
        tuned = _auto_tune_concentration_params(
            files,
            max_files=args.tune_max_files,
            samples_per_file_per_var=args.tune_samples_per_file_per_var,
            seed=args.tune_seed,
            eps_quantiles=_parse_float_list(args.tune_eps_quantiles),
            max_quantiles=_parse_float_list(args.tune_max_quantiles),
            c1_values=_parse_float_list(args.tune_c1_values),
            target_low_clamp=args.tune_target_low_clamp,
            target_high_clamp=args.tune_target_high_clamp,
            min_spread=args.tune_min_spread,
        )
        for var, params in tuned.items():
            concentration_params[var].update(params)
        concentration_params_source = "auto_tuned"
        if args.tune_json_out:
            tune_path = Path(args.tune_json_out)
            tune_path.parent.mkdir(parents=True, exist_ok=True)
            with tune_path.open("w", encoding="utf-8") as f:
                json.dump({"variables": tuned}, f, indent=2)
            print(f"[tune] wrote tuned concentration params to {tune_path}")

    first = _zarr_open(files[0])
    try:
        level_coord = first["level"].copy() if "level" in first.coords else None
        latitude = first["latitude"].values.copy() if args.latweights_out else None
        longitude = first["longitude"].values.copy() if args.latweights_out else None
        data_vars = [str(v) for v in first.data_vars if v not in {"trajectory_id"}]
        accum = _build_accumulator_schema(first, data_vars)

        file_batches = _chunk_files(files, args.files_per_task)
        print(
            f"[stats] matched_files={len(files)} batches={len(file_batches)} "
            f"workers={args.n_workers} files_per_task={args.files_per_task}"
        )

        if args.n_workers <= 1:
            for index, file_batch in enumerate(file_batches, start=1):
                print(f"[stats] processing batch {index}/{len(file_batches)}")
                partial = _compute_partial_stats(
                    file_batch,
                    data_vars,
                    concentration_params,
                    concentration_transform_specs,
                    exclude_clipped_boundary_samples=args.exclude_clipped_boundary_samples,
                    clip_boundary_tolerance=args.clip_boundary_tolerance,
                )
                _merge_accumulators(accum, partial)
        else:
            cluster = LocalCluster(
                n_workers=args.n_workers,
                threads_per_worker=args.threads_per_worker,
                processes=True,
                memory_limit=args.memory_limit,
            )
            client = Client(cluster)
            print(f"[dask] dashboard={client.dashboard_link}")
            try:
                futures = [
                    client.submit(
                        _compute_partial_stats,
                        file_batch,
                        data_vars,
                        concentration_params,
                        concentration_transform_specs,
                        args.exclude_clipped_boundary_samples,
                        args.clip_boundary_tolerance,
                        pure=False,
                    )
                    for file_batch in file_batches
                ]
                completed = 0
                for future in as_completed(futures):
                    partial = future.result()
                    _merge_accumulators(accum, partial)
                    completed += 1
                    print(f"[stats] completed batch {completed}/{len(file_batches)}")
            finally:
                client.close()
                cluster.close()
    finally:
        first.close()

    mean_ds = xr.Dataset()
    std_ds = xr.Dataset()
    if level_coord is not None:
        mean_ds = mean_ds.assign_coords(level=level_coord)
        std_ds = std_ds.assign_coords(level=level_coord)

    transform_metadata = {
        "source": concentration_params_source,
        "parameters": concentration_params,
        "transform_specs_count": len(concentration_transform_specs),
        "exclude_clipped_boundary_samples": bool(args.exclude_clipped_boundary_samples),
        "clip_boundary_tolerance": float(args.clip_boundary_tolerance),
    }
    transform_metadata_str = json.dumps(transform_metadata, sort_keys=True)
    mean_ds.attrs["concentration_transform_params_json"] = transform_metadata_str
    std_ds.attrs["concentration_transform_params_json"] = transform_metadata_str
    mean_ds.attrs["concentration_transform_source"] = concentration_params_source
    std_ds.attrs["concentration_transform_source"] = concentration_params_source
    if raw_transform_payload is not None:
        raw_transform_str = json.dumps(raw_transform_payload, sort_keys=True)
        mean_ds.attrs["concentration_transform_spec_json"] = raw_transform_str
        std_ds.attrs["concentration_transform_spec_json"] = raw_transform_str

    for var in data_vars:
        count = accum[var]["count"]
        valid = count > 0
        if not np.any(valid):
            raise SystemExit(f"Variable {var} has zero valid samples in the selected file/date range")
        if np.any(~valid):
            if np.ndim(count) == 1:
                zero_levels = np.where(~valid)[0].tolist()
                preview = zero_levels[:10]
                suffix = "" if len(zero_levels) <= 10 else " ..."
                print(f"[warn] {var}: {len(zero_levels)} level(s) have zero valid samples; filling mean/std with NaN at levels {preview}{suffix}")
            else:
                print(f"[warn] {var}: some elements have zero valid samples; filling mean/std with NaN")

        sum_arr = accum[var]["sum"]
        sum_sq_arr = accum[var]["sum_sq"]
        mean = np.full_like(sum_arr, np.nan, dtype=np.float64)
        mean = np.divide(sum_arr, count, out=mean, where=valid)

        second_moment = np.full_like(sum_sq_arr, np.nan, dtype=np.float64)
        second_moment = np.divide(sum_sq_arr, count, out=second_moment, where=valid)
        variance = np.where(valid, np.maximum(second_moment - mean * mean, 0.0), np.nan)
        std = np.sqrt(variance)
        std = np.where(valid & (std == 0.0), 1.0, std)

        if mean.ndim == 1:
            mean_da = xr.DataArray(mean.astype(np.float32), dims=("level",), coords={"level": level_coord})
            std_da = xr.DataArray(std.astype(np.float32), dims=("level",), coords={"level": level_coord})
        else:
            mean_da = xr.DataArray(np.float32(mean))
            std_da = xr.DataArray(np.float32(std))

        if var in _CONCENTRATION_VARS:
            transform_spec = concentration_transform_specs.get(var)
            if transform_spec is not None:
                for da in (mean_da, std_da):
                    da.attrs["concentration_transform_type"] = str(transform_spec["transform_type"])
                    if str(transform_spec["transform_type"]) == "zero_inflated_lognormal_probit":
                        da.attrs["concentration_transform_zero_floor"] = float(transform_spec["zero_floor"])
                        da.attrs["concentration_transform_probit_eps"] = float(transform_spec["probit_eps"])
                        da.attrs["concentration_transform_clip_max"] = float(transform_spec["clip_max"])
                    elif str(transform_spec["transform_type"]) == "log_zscore":
                        da.attrs["concentration_transform_clip_min"] = float(transform_spec["clip_min"])
                        da.attrs["concentration_transform_clip_max"] = float(transform_spec["clip_max"])
            else:
                params = concentration_params.get(var, _DEFAULT_CONCENTRATION_PARAMS)
                for da in (mean_da, std_da):
                    da.attrs["concentration_transform_c1"] = float(params["c1"])
                    da.attrs["concentration_transform_c2"] = float(params["c2"])
                    da.attrs["concentration_transform_conc_eps"] = float(params["conc_eps"])
                    da.attrs["concentration_transform_conc_max"] = float(params["conc_max"])
                    clip_min = params.get("value_clip_min")
                    clip_max = params.get("value_clip_max")
                    if clip_min is not None:
                        da.attrs["concentration_transform_clip_min"] = float(clip_min)
                    if clip_max is not None:
                        da.attrs["concentration_transform_clip_max"] = float(clip_max)

        mean_ds[var] = mean_da
        std_ds[var] = std_da

    Path(args.mean_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.std_out).parent.mkdir(parents=True, exist_ok=True)
    mean_ds.to_netcdf(args.mean_out)
    std_ds.to_netcdf(args.std_out)

    if args.latweights_out:
        # Placeholder support file for CREDIT configs when use_latitude_weights=False.
        # We collapse the first case's 2-D geolocation to 1-D representative axes.
        assert latitude is not None and longitude is not None
        lat2d = latitude
        lon2d = longitude
        lat1d = lat2d.mean(axis=1).astype(np.float32)
        lon1d = lon2d.mean(axis=0).astype(np.float32)
        weights_ds = xr.Dataset(coords={"latitude": ("latitude", lat1d), "longitude": ("longitude", lon1d)})
        Path(args.latweights_out).parent.mkdir(parents=True, exist_ok=True)
        weights_ds.to_netcdf(args.latweights_out)


if __name__ == "__main__":
    main()
