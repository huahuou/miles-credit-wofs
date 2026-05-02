from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

def _zarr_open(file_path, **kwargs):
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def _parse_value_clip_ranges(value: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if value is None:
        return out
    value = value.strip()
    if not value:
        return out
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Invalid --value-clip-ranges entry. Expected format VAR:min:max, got: "
                f"{token!r}"
            )
        var_name = parts[0].strip()
        clip_min = float(parts[1])
        clip_max = float(parts[2])
        if not var_name:
            raise ValueError(f"Invalid variable name in --value-clip-ranges entry: {token!r}")
        if not np.isfinite(clip_min) or not np.isfinite(clip_max) or clip_min >= clip_max:
            raise ValueError(f"Invalid clip bounds in --value-clip-ranges entry: {token!r}")
        out[var_name] = (clip_min, clip_max)
    return out


def _select_files(files: list[str], max_files: int, rng: np.random.Generator) -> list[str]:
    if max_files <= 0:
        raise ValueError(f"max_files must be positive, got {max_files}")
    if len(files) <= max_files:
        return files
    indices = rng.choice(len(files), size=max_files, replace=False)
    indices.sort()
    return [files[i] for i in indices]


def _sample_array(arr: np.ndarray, target_size: int, rng: np.random.Generator) -> np.ndarray:
    if arr.size <= target_size:
        return arr
    idx = rng.choice(arr.size, size=target_size, replace=False)
    return arr[idx]


def _chunk_files(files: list[str], files_per_task: int) -> list[list[str]]:
    if files_per_task <= 0:
        raise ValueError(f"files_per_task must be positive, got {files_per_task}")
    return [files[i : i + files_per_task] for i in range(0, len(files), files_per_task)]


def _take_topk_by_key(values: np.ndarray, keys: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0 or values.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if values.size <= k:
        return values, keys
    idx = np.argpartition(keys, -k)[-k:]
    return values[idx], keys[idx]


def _reservoir_add_with_rng(
    values: np.ndarray,
    keys: np.ndarray,
    new_values: np.ndarray,
    rng: np.random.Generator,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_size <= 0 or new_values.size == 0:
        return values, keys
    new_keys = rng.random(new_values.size)
    if values.size == 0:
        return _take_topk_by_key(new_values, new_keys, max_size)
    merged_values = np.concatenate([values, new_values], axis=0)
    merged_keys = np.concatenate([keys, new_keys], axis=0)
    return _take_topk_by_key(merged_values, merged_keys, max_size)


def _reservoir_merge(
    values: np.ndarray,
    keys: np.ndarray,
    incoming_values: np.ndarray,
    incoming_keys: np.ndarray,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_size <= 0 or incoming_values.size == 0:
        return values, keys
    if values.size == 0:
        return _take_topk_by_key(incoming_values, incoming_keys, max_size)
    merged_values = np.concatenate([values, incoming_values], axis=0)
    merged_keys = np.concatenate([keys, incoming_keys], axis=0)
    return _take_topk_by_key(merged_values, merged_keys, max_size)


def _safe_quantiles(values: np.ndarray, probs: list[float]) -> list[float]:
    if values.size == 0:
        return [float("nan") for _ in probs]
    return [float(x) for x in np.quantile(values, probs)]


def _transform_np(x64: np.ndarray, c1: float, c2: float, conc_max: float, conc_eps: float) -> np.ndarray:
    log_eps = math.log(conc_eps)
    neg_log_eps = -log_eps
    return c1 * np.minimum(x64, conc_max) + c2 * (np.log(np.maximum(x64, conc_eps)) - log_eps) / neg_log_eps


def _transformed_stats(z: np.ndarray) -> dict[str, float]:
    p01, p50, p99 = np.quantile(z, [0.01, 0.5, 0.99])
    mean = float(np.mean(z))
    std = float(np.std(z))
    skew = 0.0
    if std > 0:
        centered = z - mean
        skew = float(np.mean(centered * centered * centered) / (std ** 3))
    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "p01": float(p01),
        "p50": float(p50),
        "p99": float(p99),
        "spread_p99_p01": float(p99 - p01),
        "min": float(np.min(z)),
        "max": float(np.max(z)),
    }


def _score_candidate(
    x: np.ndarray,
    c1: float,
    c2: float,
    conc_max: float,
    conc_eps: float,
    target_low_clamp: float,
    target_high_clamp: float,
    min_spread: float,
) -> dict[str, float]:
    low_clamp_fraction = float(np.mean(x <= conc_eps))
    high_clamp_fraction = float(np.mean(x >= conc_max))
    z = _transform_np(x, c1=c1, c2=c2, conc_max=conc_max, conc_eps=conc_eps)
    stats = _transformed_stats(z)

    score = (
        3.0 * abs(low_clamp_fraction - target_low_clamp)
        + 2.0 * abs(high_clamp_fraction - target_high_clamp)
        + 4.0 * max(0.0, min_spread - stats["spread_p99_p01"])
        + 0.2 * abs(stats["skew"])
    )

    return {
        "score": float(score),
        "c1": float(c1),
        "c2": float(c2),
        "conc_max": float(conc_max),
        "conc_eps": float(conc_eps),
        "low_clamp_fraction": low_clamp_fraction,
        "high_clamp_fraction": high_clamp_fraction,
        "transformed_mean": stats["mean"],
        "transformed_std": stats["std"],
        "transformed_skew": stats["skew"],
        "transformed_p01": stats["p01"],
        "transformed_p50": stats["p50"],
        "transformed_p99": stats["p99"],
        "transformed_spread_p99_p01": stats["spread_p99_p01"],
        "transformed_min": stats["min"],
        "transformed_max": stats["max"],
    }


def _evaluate_candidates(
    x: np.ndarray,
    eps_candidates: list[float],
    max_candidates: list[float],
    c1_candidates: list[float],
    target_low_clamp: float,
    target_high_clamp: float,
    min_spread: float,
    progress_prefix: str,
) -> list[dict[str, float]]:
    total = len(c1_candidates) * len(eps_candidates) * len(max_candidates)
    all_candidates: list[dict[str, float]] = []
    idx = 0
    for c1 in c1_candidates:
        c2 = 1.0 - c1
        for conc_eps in eps_candidates:
            for conc_max in max_candidates:
                idx += 1
                if idx % 100 == 0 or idx == total:
                    print(f"[{progress_prefix}] scoring candidate {idx}/{total}")
                if conc_max <= conc_eps:
                    continue
                candidate = _score_candidate(
                    x=x,
                    c1=c1,
                    c2=c2,
                    conc_max=conc_max,
                    conc_eps=conc_eps,
                    target_low_clamp=target_low_clamp,
                    target_high_clamp=target_high_clamp,
                    min_spread=min_spread,
                )
                all_candidates.append(candidate)
    all_candidates.sort(key=lambda c: c["score"])
    return all_candidates


def _build_candidates_from_data(
    x: np.ndarray,
    eps_quantiles: list[float],
    max_quantiles: list[float],
    c1_values: list[float],
) -> tuple[list[float], list[float], list[float]]:
    x_pos = x[x > 0]
    if x_pos.size == 0:
        raise ValueError("No positive values found for concentration variable")

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

    c1_candidates = sorted({float(c1) for c1 in c1_values if 0.0 <= c1 <= 1.0})
    if not c1_candidates:
        c1_candidates = [0.5]
    return eps_candidates, max_candidates, c1_candidates


def _plot_variable_distributions(
    var_name: str,
    raw: np.ndarray,
    best: dict[str, float],
    plot_path: Path,
) -> None:
    conc_eps = best["conc_eps"]
    conc_max = best["conc_max"]
    c1 = best["c1"]
    c2 = best["c2"]
    transformed = _transform_np(raw, c1=c1, c2=c2, conc_max=conc_max, conc_eps=conc_eps)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    raw_positive = raw[raw > 0]
    if raw_positive.size > 0:
        xmin = max(float(np.min(raw_positive)), 1e-12)
        xmax = float(np.quantile(raw_positive, 0.9995))
        if xmax <= xmin:
            xmax = xmin * 10.0
        bins = np.logspace(np.log10(xmin), np.log10(xmax), 80)
        ax.hist(raw_positive, bins=bins, color="#1f77b4", alpha=0.8)
        ax.set_xscale("log")
    ax.axvline(conc_eps, color="#d62728", linestyle="--", linewidth=1.5, label="conc_eps")
    ax.axvline(conc_max, color="#2ca02c", linestyle="--", linewidth=1.5, label="conc_max")
    ax.set_title(f"{var_name}: raw (>0) distribution")
    ax.set_xlabel("raw value (log scale)")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.legend(loc="best")

    ax = axes[1]
    ax.hist(transformed, bins=100, color="#9467bd", alpha=0.85)
    ax.set_title(f"{var_name}: transformed distribution")
    ax.set_xlabel("transformed value")
    ax.set_ylabel("count")
    ax.set_yscale("log")

    text = (
        f"c1={c1:.3f}, c2={c2:.3f}\n"
        f"conc_eps={conc_eps:.3e}, conc_max={conc_max:.3e}\n"
        f"low_clamp={best['low_clamp_fraction']:.4f}, high_clamp={best['high_clamp_fraction']:.4f}\n"
        f"spread(p99-p01)={best['transformed_spread_p99_p01']:.4f}, skew={best['transformed_skew']:.4f}"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=9)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _collect_samples(
    files: list[str],
    variables: list[str],
    samples_per_file_per_var: int,
    rng: np.random.Generator,
    max_reservoir_samples_per_var: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    reservoirs: dict[str, np.ndarray] = {var: np.array([], dtype=np.float64) for var in variables}
    reservoir_keys: dict[str, np.ndarray] = {var: np.array([], dtype=np.float64) for var in variables}
    seen_counts: dict[str, int] = {var: 0 for var in variables}

    for idx, file_path in enumerate(files, start=1):
        print(f"[sample] file {idx}/{len(files)}: {file_path}")
        ds = _zarr_open(file_path)
        try:
            for var in variables:
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
                seen_counts[var] += int(arr.size)
                arr = _sample_array(arr, samples_per_file_per_var, rng)
                reservoirs[var], reservoir_keys[var] = _reservoir_add_with_rng(
                    reservoirs[var],
                    reservoir_keys[var],
                    arr,
                    rng,
                    max_reservoir_samples_per_var,
                )
        finally:
            ds.close()

    sampled = {}
    for var in variables:
        sampled[var] = reservoirs[var]
    return sampled, seen_counts


def _collect_samples_batch(
    file_batch: list[str],
    variables: list[str],
    samples_per_file_per_var: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray | int]]:
    rng = np.random.default_rng(seed)
    reservoirs: dict[str, np.ndarray] = {var: np.array([], dtype=np.float64) for var in variables}
    reservoir_keys: dict[str, np.ndarray] = {var: np.array([], dtype=np.float64) for var in variables}
    seen_counts: dict[str, int] = {var: 0 for var in variables}
    max_reservoir_samples_per_var = samples_per_file_per_var

    for file_path in file_batch:
        ds = _zarr_open(file_path)
        try:
            for var in variables:
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
                seen_counts[var] += int(arr.size)
                arr = _sample_array(arr, samples_per_file_per_var, rng)
                reservoirs[var], reservoir_keys[var] = _reservoir_add_with_rng(
                    reservoirs[var],
                    reservoir_keys[var],
                    arr,
                    rng,
                    max_reservoir_samples_per_var,
                )
        finally:
            ds.close()

    sampled: dict[str, dict[str, np.ndarray | int]] = {}
    for var in variables:
        sampled[var] = {
            "values": reservoirs[var],
            "keys": reservoir_keys[var],
            "n_seen": int(seen_counts[var]),
        }
    return sampled


def _collect_samples_parallel(
    files: list[str],
    variables: list[str],
    samples_per_file_per_var: int,
    seed: int,
    n_workers: int,
    threads_per_worker: int,
    memory_limit: str,
    files_per_task: int,
    max_reservoir_samples_per_var: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    file_batches = _chunk_files(files, files_per_task)
    merged_values: dict[str, np.ndarray] = {var: np.array([], dtype=np.float64) for var in variables}
    merged_keys: dict[str, np.ndarray] = {var: np.array([], dtype=np.float64) for var in variables}
    seen_counts: dict[str, int] = {var: 0 for var in variables}
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit=memory_limit,
    )
    client = Client(cluster)
    print(f"[dask] dashboard={client.dashboard_link}")
    print(f"[dask] sampling batches={len(file_batches)} workers={n_workers} files_per_task={files_per_task}")
    try:
        futures = []
        for idx, file_batch in enumerate(file_batches):
            batch_seed = int(seed + idx * 7919)
            futures.append(
                client.submit(
                    _collect_samples_batch,
                    file_batch,
                    variables,
                    samples_per_file_per_var,
                    batch_seed,
                    pure=False,
                )
            )
        completed = 0
        for future in as_completed(futures):
            partial = future.result()
            completed += 1
            print(f"[dask] completed sampling batch {completed}/{len(file_batches)}")
            for var in variables:
                payload = partial.get(var)
                if not isinstance(payload, dict):
                    continue
                values = payload.get("values")
                keys = payload.get("keys")
                n_seen = payload.get("n_seen", 0)
                seen_counts[var] += int(n_seen)
                if isinstance(values, np.ndarray) and isinstance(keys, np.ndarray) and values.size > 0:
                    merged_values[var], merged_keys[var] = _reservoir_merge(
                        merged_values[var],
                        merged_keys[var],
                        values,
                        keys,
                        max_reservoir_samples_per_var,
                    )
    finally:
        client.close()
        cluster.close()

    merged = {}
    for var in variables:
        merged[var] = merged_values[var]
    return merged, seen_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune concentration-variable transform parameters from a sample of WoFS zarr files and generate plots + JSON recommendations"
    )
    parser.add_argument("--glob", required=True, help="Glob for WoFS zarr files")
    parser.add_argument("--start-date", help="Inclusive YYYYMMDD lower bound")
    parser.add_argument("--end-date", help="Inclusive YYYYMMDD upper bound")
    parser.add_argument("--max-files", type=int, default=150, help="Maximum number of zarr files to sample")
    parser.add_argument(
        "--samples-per-file-per-var",
        type=int,
        default=100000,
        help="Maximum random samples taken from each file for each variable",
    )
    parser.add_argument(
        "--variables",
        default=",".join(sorted(_CONCENTRATION_VARS)),
        help="Comma-separated concentration variables to tune",
    )
    parser.add_argument(
        "--eps-quantiles",
        default="0.0005,0.001,0.002,0.005,0.01,0.02,0.05",
        help="Comma-separated quantiles used to create conc_eps candidates from positive raw samples",
    )
    parser.add_argument(
        "--max-quantiles",
        default="0.95,0.98,0.99,0.995,0.999",
        help="Comma-separated quantiles used to create conc_max candidates from positive raw samples",
    )
    parser.add_argument(
        "--c1-values",
        default="0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated c1 candidates. c2 is set to (1-c1)",
    )
    parser.add_argument(
        "--target-low-clamp",
        type=float,
        default=0.01,
        help="Target fraction of samples clamped at conc_eps (small values ignored)",
    )
    parser.add_argument(
        "--target-high-clamp",
        type=float,
        default=0.01,
        help="Target fraction of samples capped at conc_max",
    )
    parser.add_argument(
        "--min-spread",
        type=float,
        default=0.4,
        help="Minimum desired transformed spread p99-p01",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Top candidates to keep per variable in JSON")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--max-score-samples-per-var",
        type=int,
        default=400000,
        help="Maximum samples per variable used in coarse candidate scoring (major speed knob)",
    )
    parser.add_argument(
        "--refine-top-k",
        type=int,
        default=20,
        help="Re-score top-K coarse candidates on a larger sample",
    )
    parser.add_argument(
        "--refine-score-samples-per-var",
        type=int,
        default=1200000,
        help="Maximum samples per variable used for refine-stage scoring; set <=0 to disable refine stage",
    )
    parser.add_argument(
        "--max-plot-samples-per-var",
        type=int,
        default=300000,
        help="Maximum samples per variable used for plotting",
    )
    parser.add_argument(
        "--max-reservoir-samples-per-var",
        type=int,
        default=1500000,
        help="Maximum retained samples per variable during streaming collection/merge (memory bound)",
    )
    parser.add_argument(
        "--value-clip-ranges",
        default="",
        help=(
            "Optional per-variable clip bounds applied before transform scoring/plots, "
            "format 'VAR:min:max,VAR:min:max' (e.g. 'QNRAIN:1e-13:1e-1,QRAIN:1e-12:1e7')"
        ),
    )
    parser.add_argument("--n-workers", type=int, default=1, help="Number of Dask workers used for sampling")
    parser.add_argument("--threads-per-worker", type=int, default=1, help="Threads per Dask worker")
    parser.add_argument(
        "--memory-limit",
        default="4GiB",
        help="Per-worker memory limit passed to local Dask cluster, e.g. 2GiB",
    )
    parser.add_argument(
        "--files-per-task",
        type=int,
        default=4,
        help="How many zarr stores each Dask sampling task processes sequentially",
    )
    parser.add_argument("--plots-dir", required=True, help="Directory where per-variable plot PNGs are written")
    parser.add_argument("--json-out", required=True, help="Output JSON path with recommendations")
    args = parser.parse_args()

    start_date = _parse_yyyymmdd(args.start_date)
    end_date = _parse_yyyymmdd(args.end_date)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise SystemExit("--start-date must be <= --end-date")

    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(args.glob))
    files = _filter_files_by_date(files, start_date, end_date)
    if not files:
        raise SystemExit(f"No files matched {args.glob!r} after date filtering")

    selected_files = _select_files(files, args.max_files, rng)
    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    if not variables:
        raise SystemExit("--variables resolved to an empty list")

    eps_quantiles = _parse_float_list(args.eps_quantiles)
    max_quantiles = _parse_float_list(args.max_quantiles)
    c1_values = _parse_float_list(args.c1_values)
    value_clip_ranges = _parse_value_clip_ranges(args.value_clip_ranges)

    print(f"[config] matched_files={len(files)} selected_files={len(selected_files)} seed={args.seed}")
    print(f"[config] variables={variables}")
    if value_clip_ranges:
        print(f"[config] value_clip_ranges={value_clip_ranges}")

    if args.n_workers <= 1:
        sampled, seen_counts = _collect_samples(
            selected_files,
            variables=variables,
            samples_per_file_per_var=args.samples_per_file_per_var,
            rng=rng,
            max_reservoir_samples_per_var=args.max_reservoir_samples_per_var,
        )
    else:
        sampled, seen_counts = _collect_samples_parallel(
            selected_files,
            variables=variables,
            samples_per_file_per_var=args.samples_per_file_per_var,
            seed=args.seed,
            n_workers=args.n_workers,
            threads_per_worker=args.threads_per_worker,
            memory_limit=args.memory_limit,
            files_per_task=args.files_per_task,
            max_reservoir_samples_per_var=args.max_reservoir_samples_per_var,
        )

    results: dict[str, object] = {
        "metadata": {
            "glob": args.glob,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "matched_files": len(files),
            "selected_files": len(selected_files),
            "seed": args.seed,
            "max_files": args.max_files,
            "samples_per_file_per_var": args.samples_per_file_per_var,
            "eps_quantiles": eps_quantiles,
            "max_quantiles": max_quantiles,
            "c1_values": c1_values,
            "target_low_clamp": args.target_low_clamp,
            "target_high_clamp": args.target_high_clamp,
            "min_spread": args.min_spread,
            "n_workers": args.n_workers,
            "threads_per_worker": args.threads_per_worker,
            "memory_limit": args.memory_limit,
            "files_per_task": args.files_per_task,
            "max_score_samples_per_var": args.max_score_samples_per_var,
            "refine_top_k": args.refine_top_k,
            "refine_score_samples_per_var": args.refine_score_samples_per_var,
            "max_plot_samples_per_var": args.max_plot_samples_per_var,
            "max_reservoir_samples_per_var": args.max_reservoir_samples_per_var,
            "value_clip_ranges": {k: [v[0], v[1]] for k, v in value_clip_ranges.items()},
        },
        "variables": {},
    }

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for var in variables:
        x = sampled.get(var, np.array([], dtype=np.float64))
        n_seen = int(seen_counts.get(var, 0))
        clip_bounds = value_clip_ranges.get(var)
        if x.size == 0:
            print(f"[warn] {var}: no valid non-negative finite samples")
            results["variables"][var] = {
                "status": "no_samples",
                "n_samples": 0,
                "n_seen": n_seen,
            }
            continue

        n_clipped_low = 0
        n_clipped_high = 0
        x_eval = x
        if clip_bounds is not None:
            clip_min, clip_max = clip_bounds
            n_clipped_low = int(np.sum(x < clip_min))
            n_clipped_high = int(np.sum(x > clip_max))
            x_eval = np.clip(x, clip_min, clip_max)
            print(
                f"[clip] {var}: clip_min={clip_min:.3e} clip_max={clip_max:.3e} "
                f"n_low={n_clipped_low} n_high={n_clipped_high}"
            )

        x_pos = x_eval[x_eval > 0]
        if x_pos.size == 0:
            print(f"[warn] {var}: no positive samples")
            results["variables"][var] = {
                "status": "no_positive_samples",
                "n_samples": int(x_eval.size),
                "n_seen": n_seen,
                "fraction_nonpositive": float(np.mean(x_eval <= 0.0)),
            }
            continue

        eps_candidates, max_candidates, c1_candidates = _build_candidates_from_data(
            x=x_eval,
            eps_quantiles=eps_quantiles,
            max_quantiles=max_quantiles,
            c1_values=c1_values,
        )

        x_coarse = _sample_array(x_eval, args.max_score_samples_per_var, rng)
        print(f"[score] {var}: full_samples={x_eval.size} coarse_samples={x_coarse.size}")
        all_candidates = _evaluate_candidates(
            x=x_coarse,
            eps_candidates=eps_candidates,
            max_candidates=max_candidates,
            c1_candidates=c1_candidates,
            target_low_clamp=args.target_low_clamp,
            target_high_clamp=args.target_high_clamp,
            min_spread=args.min_spread,
            progress_prefix=f"score:{var}",
        )

        if not all_candidates:
            print(f"[warn] {var}: no candidate combinations generated")
            results["variables"][var] = {
                "status": "no_candidates",
                "n_samples": int(x_eval.size),
                "n_seen": n_seen,
            }
            continue

        if args.refine_score_samples_per_var > 0 and args.refine_top_k > 0:
            x_refine = _sample_array(x_eval, args.refine_score_samples_per_var, rng)
            top = all_candidates[: args.refine_top_k]
            refined: list[dict[str, float]] = []
            print(f"[refine] {var}: refine_samples={x_refine.size} top_k={len(top)}")
            for cand in top:
                refined.append(
                    _score_candidate(
                        x=x_refine,
                        c1=float(cand["c1"]),
                        c2=float(cand["c2"]),
                        conc_max=float(cand["conc_max"]),
                        conc_eps=float(cand["conc_eps"]),
                        target_low_clamp=args.target_low_clamp,
                        target_high_clamp=args.target_high_clamp,
                        min_spread=args.min_spread,
                    )
                )
            refined.sort(key=lambda c: c["score"])
            best = refined[0]
            for cand in all_candidates:
                if (
                    float(cand["c1"]) == float(best["c1"])
                    and float(cand["c2"]) == float(best["c2"])
                    and float(cand["conc_max"]) == float(best["conc_max"])
                    and float(cand["conc_eps"]) == float(best["conc_eps"])
                ):
                    cand.update(best)
                    break
            all_candidates.sort(key=lambda c: c["score"])
        else:
            best = all_candidates[0]

        plot_path = plots_dir / f"{var.lower()}_distribution.png"
        x_plot = _sample_array(x_eval, args.max_plot_samples_per_var, rng)
        _plot_variable_distributions(var, x_plot, best, plot_path)

        print(
            f"[best] {var}: score={best['score']:.5f} c1={best['c1']:.3f} c2={best['c2']:.3f} "
            f"conc_eps={best['conc_eps']:.3e} conc_max={best['conc_max']:.3e}"
        )

        raw_q = _safe_quantiles(x_eval, [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0])
        results["variables"][var] = {
            "status": "ok",
            "n_samples": int(x_eval.size),
            "n_seen": n_seen,
            "fraction_nonpositive": float(np.mean(x_eval <= 0.0)),
            "value_clip_min": float(clip_bounds[0]) if clip_bounds is not None else None,
            "value_clip_max": float(clip_bounds[1]) if clip_bounds is not None else None,
            "n_clipped_low": int(n_clipped_low),
            "n_clipped_high": int(n_clipped_high),
            "raw_quantiles": {
                "p0000": raw_q[0],
                "p0001": raw_q[1],
                "p0010": raw_q[2],
                "p5000": raw_q[3],
                "p9900": raw_q[4],
                "p9990": raw_q[5],
                "p10000": raw_q[6],
            },
            "recommended": best,
            "top_candidates": all_candidates[: args.top_k],
            "plot": str(plot_path),
        }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=False)

    print(f"[done] wrote recommendations to {json_out}")
    print(f"[done] plots in {plots_dir}")


if __name__ == "__main__":
    main()