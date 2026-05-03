from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import re
import time
from collections import Counter
from pathlib import Path
from builtins import TimeoutError as BuiltinTimeoutError

import matplotlib
import numpy as np
from dask.distributed import Client, LocalCluster, wait

from credit.data import get_forward_data
from credit.transforms.concentration import (
    CONCENTRATION_VARS,
    DEFAULT_PROBIT_EPS,
    DEFAULT_ZERO_FLOOR,
    default_clip_max_for_var,
    forward_concentration_transform_numpy,
    parse_concentration_transform_payload,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _zarr_open(file_path: str, **kwargs):
    if kwargs:
        raise TypeError(f"_zarr_open does not accept extra keyword arguments, got: {sorted(kwargs)}")
    return get_forward_data(str(file_path), zarr_chunks=None)


_CASE_INFO_RE = re.compile(
    r"wofs_(?:boundary_)?(?P<date>\d{8})_(?:.+_)?(?P<init>\d{4})_mem\d+\.zarr(?:\.zip)?/?$"
)


def _parse_yyyymmdd(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if not re.fullmatch(r"\d{8}", value):
        raise ValueError(f"Expected YYYYMMDD date string, got: {value}")
    return int(value)


def _extract_case_info(file_path: str) -> tuple[int, str]:
    match = _CASE_INFO_RE.search(file_path)
    if match is None:
        raise ValueError(
            "Could not infer case date from file path. Expected names like "
            f"'wofs_YYYYMMDD_<slug_>HHMM_memNN.zarr(.zip)', got: {file_path}"
        )
    return int(match.group("date")), match.group("init")


def _extract_case_date(file_path: str) -> int:
    case_date, _ = _extract_case_info(file_path)
    return case_date


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


def _select_spread_indices(init_minutes: np.ndarray, target_size: int, rng: np.random.Generator) -> list[int]:
    if target_size <= 0 or init_minutes.size <= target_size:
        return list(range(int(init_minutes.size)))

    selected = [int(rng.integers(init_minutes.size))]
    min_distance = np.abs(init_minutes - init_minutes[selected[0]])

    while len(selected) < target_size:
        remaining = np.array([idx for idx in range(init_minutes.size) if idx not in selected], dtype=np.int64)
        farthest_distance = np.max(min_distance[remaining])
        candidates = remaining[min_distance[remaining] == farthest_distance]
        chosen = int(rng.choice(candidates))
        selected.append(chosen)
        min_distance = np.minimum(min_distance, np.abs(init_minutes - init_minutes[chosen]))

    selected.sort()
    return selected


def _select_cases_per_day(files: list[str], num_cases_per_day: int, seed: int) -> list[str]:
    if num_cases_per_day <= 0:
        return files

    files_by_day: dict[int, list[tuple[int, str]]] = {}
    for file_path in files:
        case_date, init_hhmm = _extract_case_info(file_path)
        init_minutes = int(init_hhmm[:2]) * 60 + int(init_hhmm[2:])
        files_by_day.setdefault(case_date, []).append((init_minutes, file_path))

    selected_files: list[str] = []
    for case_date in sorted(files_by_day):
        entries = sorted(files_by_day[case_date], key=lambda item: (item[0], item[1]))
        if len(entries) <= num_cases_per_day:
            selected_files.extend(file_path for _, file_path in entries)
            continue

        day_rng = np.random.default_rng(int(seed + case_date))
        init_minutes = np.asarray([item[0] for item in entries], dtype=np.int64)
        chosen_indices = _select_spread_indices(init_minutes, num_cases_per_day, day_rng)
        selected_files.extend(entries[idx][1] for idx in chosen_indices)

    return selected_files


def _select_files(files: list[str], max_files: int, seed: int) -> list[str]:
    if max_files <= 0 or len(files) <= max_files:
        return files
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(files), size=max_files, replace=False)
    indices.sort()
    return [files[i] for i in indices]


def _sample_array(arr: np.ndarray, target_size: int, rng: np.random.Generator) -> np.ndarray:
    if target_size <= 0 or arr.size <= target_size:
        return arr
    idx = rng.choice(arr.size, size=target_size, replace=False)
    return arr[idx]


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


def _parse_variables(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return sorted(CONCENTRATION_VARS)
    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(token)
    return out


def _parse_clip_max_overrides(value: str | None) -> dict[str, float]:
    if value is None or not value.strip():
        return {}
    out: dict[str, float] = {}
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        name, raw_max = token.split(":", 1)
        out[name.strip()] = float(raw_max)
    return out


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{timestamp} {message}", flush=True)


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _summarize_file_batch(file_batch: list[str], max_files_to_list: int = 4) -> str:
    names = [Path(file_path).name for file_path in file_batch]
    if len(names) <= max_files_to_list:
        return ", ".join(names)
    return f"{names[0]} ... {names[-1]}"


def _pending_state_summary(client: Client, pending_futures: set, future_meta: dict, max_items: int = 3) -> tuple[str, list[str]]:
    try:
        processing = client.processing()
    except Exception:
        processing = {}

    running_keys = set()
    for task_keys in processing.values():
        running_keys.update(task_keys)

    counts = Counter()
    for future in pending_futures:
        if future.key in running_keys or future.status == "running":
            counts["running"] += 1
        else:
            counts["waiting"] += 1

    now = time.monotonic()
    slowest = sorted(pending_futures, key=lambda future: future_meta[future]["submitted_at"])[:max_items]
    details = []
    for future in slowest:
        meta = future_meta[future]
        age = _format_elapsed(now - meta["submitted_at"])
        state = "running" if future.key in running_keys else future.status
        details.append(
            f"batch {meta['batch_number']}/{meta['batch_total']} age={age} "
            f"state={state} files={meta['file_count']} [{meta['label']}]"
        )

    summary = f"running={counts['running']} waiting={counts['waiting']}"
    return summary, details


def _emit_dask_heartbeat(
    client: Client,
    pending_futures: set,
    future_meta: dict,
    completed: int,
    total: int,
    started_at: float,
    last_progress_at: float,
    stalled_warning_seconds: float,
) -> None:
    pending_summary, oldest_pending = _pending_state_summary(client, pending_futures, future_meta)
    now = time.monotonic()
    idle_for = now - last_progress_at
    warning = " stalled_warning=yes" if idle_for >= stalled_warning_seconds else ""
    _log(
        f"[dask] heartbeat elapsed={_format_elapsed(now - started_at)} "
        f"completed={completed}/{total} {pending_summary} "
        f"idle_since_last_complete={_format_elapsed(idle_for)}{warning}"
    )
    for item in oldest_pending:
        _log(f"[dask] oldest_pending {item}")


def _fit_positive_lognormal(sample: np.ndarray, zero_floor: float, winsorize_upper_quantile: float) -> dict[str, float]:
    if sample.size == 0:
        return {"mu": math.log(zero_floor), "sigma": 1.0}
    log_values = np.log(np.maximum(sample.astype(np.float64), zero_floor))
    if 0.0 < winsorize_upper_quantile < 1.0 and log_values.size > 1:
        hi = float(np.quantile(log_values, winsorize_upper_quantile))
        log_values = np.minimum(log_values, hi)
    mu = float(np.mean(log_values))
    sigma = max(float(np.std(log_values)), 1.0e-6)
    return {"mu": mu, "sigma": sigma}


def _chunk_files(files: list[str], files_per_task: int) -> list[list[str]]:
    if files_per_task <= 0:
        raise ValueError(f"files_per_task must be positive, got {files_per_task}")
    return [files[i : i + files_per_task] for i in range(0, len(files), files_per_task)]


def _empty_batch_payload(level_count: int, variables: list[str]) -> dict[str, dict]:
    payload = {}
    for var in variables:
        payload[var] = {
            "count_total": np.zeros(level_count, dtype=np.int64),
            "count_positive": np.zeros(level_count, dtype=np.int64),
            "level_positive_values": [np.array([], dtype=np.float64) for _ in range(level_count)],
            "level_positive_keys": [np.array([], dtype=np.float64) for _ in range(level_count)],
            "level_plot_values": [np.array([], dtype=np.float64) for _ in range(level_count)],
            "level_plot_keys": [np.array([], dtype=np.float64) for _ in range(level_count)],
            "pooled_values": np.array([], dtype=np.float64),
            "pooled_keys": np.array([], dtype=np.float64),
        }
    return payload


def _collect_batch_statistics(
    file_batch: list[str],
    variables: list[str],
    level_count: int,
    zero_floor: float,
    positive_samples_per_file_per_level: int,
    plot_samples_per_file_per_level: int,
    max_level_samples: int,
    max_plot_samples_per_level: int,
    max_pooled_samples: int,
    seed: int,
) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    payload = _empty_batch_payload(level_count, variables)

    for file_path in file_batch:
        ds = _zarr_open(file_path)
        try:
            for var in variables:
                if var not in ds.data_vars:
                    continue
                da = ds[var]
                if "level" not in da.dims:
                    continue
                level_axis = da.dims.index("level")
                raw = np.asarray(da.values, dtype=np.float64)
                raw = np.moveaxis(raw, level_axis, 0).reshape(level_count, -1)

                for lev in range(level_count):
                    values = raw[lev]
                    valid = np.isfinite(values) & (values >= 0.0)
                    if not np.any(valid):
                        continue
                    values = values[valid]
                    values = np.where(values < zero_floor, 0.0, values)
                    payload[var]["count_total"][lev] += int(values.size)

                    pos = values[values > 0.0]
                    payload[var]["count_positive"][lev] += int(pos.size)

                    if pos.size > 0:
                        pos_sample = _sample_array(pos, positive_samples_per_file_per_level, rng)
                        level_values = payload[var]["level_positive_values"][lev]
                        level_keys = payload[var]["level_positive_keys"][lev]
                        level_values, level_keys = _reservoir_add_with_rng(
                            level_values,
                            level_keys,
                            pos_sample,
                            rng,
                            max_level_samples,
                        )
                        payload[var]["level_positive_values"][lev] = level_values
                        payload[var]["level_positive_keys"][lev] = level_keys

                        pooled_values = payload[var]["pooled_values"]
                        pooled_keys = payload[var]["pooled_keys"]
                        pooled_values, pooled_keys = _reservoir_add_with_rng(
                            pooled_values,
                            pooled_keys,
                            pos_sample,
                            rng,
                            max_pooled_samples,
                        )
                        payload[var]["pooled_values"] = pooled_values
                        payload[var]["pooled_keys"] = pooled_keys

                    plot_sample = _sample_array(values, plot_samples_per_file_per_level, rng)
                    plot_values = payload[var]["level_plot_values"][lev]
                    plot_keys = payload[var]["level_plot_keys"][lev]
                    plot_values, plot_keys = _reservoir_add_with_rng(
                        plot_values,
                        plot_keys,
                        plot_sample,
                        rng,
                        max_plot_samples_per_level,
                    )
                    payload[var]["level_plot_values"][lev] = plot_values
                    payload[var]["level_plot_keys"][lev] = plot_keys
        finally:
            ds.close()

    return payload


def _merge_batch_payload(
    merged: dict[str, dict],
    partial: dict[str, dict],
    variables: list[str],
    level_count: int,
    max_level_samples: int,
    max_plot_samples_per_level: int,
    max_pooled_samples: int,
) -> None:
    for var in variables:
        merged[var]["count_total"] += partial[var]["count_total"]
        merged[var]["count_positive"] += partial[var]["count_positive"]
        for lev in range(level_count):
            values, keys = _reservoir_merge(
                merged[var]["level_positive_values"][lev],
                merged[var]["level_positive_keys"][lev],
                partial[var]["level_positive_values"][lev],
                partial[var]["level_positive_keys"][lev],
                max_level_samples,
            )
            merged[var]["level_positive_values"][lev] = values
            merged[var]["level_positive_keys"][lev] = keys

            plot_values, plot_keys = _reservoir_merge(
                merged[var]["level_plot_values"][lev],
                merged[var]["level_plot_keys"][lev],
                partial[var]["level_plot_values"][lev],
                partial[var]["level_plot_keys"][lev],
                max_plot_samples_per_level,
            )
            merged[var]["level_plot_values"][lev] = plot_values
            merged[var]["level_plot_keys"][lev] = plot_keys

        pooled_values, pooled_keys = _reservoir_merge(
            merged[var]["pooled_values"],
            merged[var]["pooled_keys"],
            partial[var]["pooled_values"],
            partial[var]["pooled_keys"],
            max_pooled_samples,
        )
        merged[var]["pooled_values"] = pooled_values
        merged[var]["pooled_keys"] = pooled_keys


def _build_plot_spec(
    output_payload: dict,
    zero_floor: float,
    probit_eps: float,
    var_name: str,
) -> dict:
    wrapped = {
        "transform_type": "zero_inflated_lognormal_probit",
        "zero_floor": zero_floor,
        "probit_eps": probit_eps,
        "variables": {
            var_name: output_payload["variables"][var_name],
        },
    }
    return parse_concentration_transform_payload(wrapped, variables={var_name})[var_name]


def _single_level_spec(spec: dict, level_index: int) -> dict:
    return {
        "transform_type": spec["transform_type"],
        "zero_floor": spec["zero_floor"],
        "probit_eps": spec["probit_eps"],
        "clip_max": spec["clip_max"],
        "fallback_positive_fit": spec["fallback_positive_fit"],
        "levels": [spec["levels"][level_index]],
        "alpha": np.asarray([spec["alpha"][level_index]], dtype=np.float64),
        "mu": np.asarray([spec["mu"][level_index]], dtype=np.float64),
        "sigma": np.asarray([spec["sigma"][level_index]], dtype=np.float64),
        "status": [spec["status"][level_index]],
    }


def _plot_variable_distributions(
    var_name: str,
    spec: dict,
    raw_positive_by_level: list[np.ndarray],
    raw_plot_by_level: list[np.ndarray],
    counts_total: np.ndarray,
    counts_positive: np.ndarray,
    plot_path: Path,
) -> None:
    pooled_positive = [arr for arr in raw_positive_by_level if arr.size > 0]
    pooled_positive = np.concatenate(pooled_positive, axis=0) if pooled_positive else np.array([], dtype=np.float64)

    transformed_samples = []
    raw_all_samples = []
    for lev, raw_level in enumerate(raw_plot_by_level):
        if raw_level.size == 0:
            continue
        raw_all_samples.append(raw_level)
        lev_spec = _single_level_spec(spec, lev)
        transformed = forward_concentration_transform_numpy(raw_level[np.newaxis, :], lev_spec, level_axis=0)[0]
        transformed_samples.append(np.asarray(transformed, dtype=np.float64))

    pooled_raw_all = np.concatenate(raw_all_samples, axis=0) if raw_all_samples else np.array([], dtype=np.float64)
    pooled_transformed = np.concatenate(transformed_samples, axis=0) if transformed_samples else np.array([], dtype=np.float64)

    zero_fraction = 1.0
    total_n = int(np.sum(counts_total))
    pos_n = int(np.sum(counts_positive))
    if total_n > 0:
        zero_fraction = 1.0 - (pos_n / total_n)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    if pooled_positive.size > 0:
        xmin = max(float(np.min(pooled_positive)), spec["zero_floor"])
        xmax = max(float(np.quantile(pooled_positive, 0.999)), xmin * 10.0)
        bins = np.logspace(np.log10(xmin), np.log10(xmax), 80)
        ax.hist(pooled_positive, bins=bins, color="#1f77b4", alpha=0.85)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(f"{var_name}: raw positive values")
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.text(
        0.02,
        0.98,
        f"total={total_n:,}\npositive={pos_n:,}\nzero_frac={zero_fraction:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    ax = axes[1]
    if pooled_transformed.size > 0:
        ax.hist(pooled_transformed, bins=100, color="#d62728", alpha=0.85)
        ax.set_yscale("log")
    ax.set_title(f"{var_name}: transformed values")
    ax.set_xlabel("latent value")
    ax.set_ylabel("count")

    ax = axes[2]
    levels = np.arange(1, len(spec["levels"]) + 1)
    alpha = np.asarray([item["alpha"] for item in spec["levels"]], dtype=np.float64)
    ax.plot(levels, alpha, marker="o", color="#2ca02c", linewidth=1.5)
    ok_levels = [idx + 1 for idx, item in enumerate(spec["levels"]) if item["status"] == "ok"]
    borrow_levels = [idx + 1 for idx, item in enumerate(spec["levels"]) if item["status"] == "borrow_fallback"]
    zero_levels = [idx + 1 for idx, item in enumerate(spec["levels"]) if item["status"] == "degenerate_zero"]
    if ok_levels:
        ax.scatter(ok_levels, alpha[np.array(ok_levels) - 1], color="#2ca02c", label="ok")
    if borrow_levels:
        ax.scatter(borrow_levels, alpha[np.array(borrow_levels) - 1], color="#ff7f0e", label="borrow")
    if zero_levels:
        ax.scatter(zero_levels, alpha[np.array(zero_levels) - 1], color="#7f7f7f", label="zero-only")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{var_name}: per-level zero mass")
    ax.set_xlabel("level")
    ax.set_ylabel("alpha")
    ax.legend(loc="best")

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-variable, per-level zero-inflated lognormal-probit transform parameters from WoFS zarr files."
    )
    parser.add_argument("--glob", required=True, help="Glob matching WoFS zarr files")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--start-date", help="Inclusive lower date bound in YYYYMMDD format")
    parser.add_argument("--end-date", help="Inclusive upper date bound in YYYYMMDD format")
    parser.add_argument("--variables", help="Comma-separated concentration variables to fit")
    parser.add_argument("--max-files", type=int, default=0, help="Optional max number of files to sample; 0 means all matched files")
    parser.add_argument(
        "--num-cases-per-day",
        type=int,
        default=0,
        help="Optional max number of init times kept per day after date filtering; 0 means keep all cases",
    )
    parser.add_argument(
        "--case-selection-seed",
        type=int,
        default=1234,
        help="Random seed used when subsampling init times within each day",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--zero-floor", type=float, default=DEFAULT_ZERO_FLOOR, help="Values below this are treated as exact zero")
    parser.add_argument("--probit-eps", type=float, default=DEFAULT_PROBIT_EPS, help="Probability clipping used before probit")
    parser.add_argument(
        "--samples-per-file-per-level",
        type=int,
        default=20000,
        help="Maximum positive samples drawn from each file for each variable/level",
    )
    parser.add_argument(
        "--plot-samples-per-file-per-level",
        type=int,
        default=20000,
        help="Maximum all-value samples drawn from each file for each variable/level for plotting",
    )
    parser.add_argument(
        "--max-level-samples",
        type=int,
        default=250000,
        help="Maximum retained positive samples per variable-level after reservoir sampling",
    )
    parser.add_argument(
        "--max-plot-samples-per-level",
        type=int,
        default=50000,
        help="Maximum retained raw samples per variable-level for plotting",
    )
    parser.add_argument(
        "--max-pooled-samples",
        type=int,
        default=500000,
        help="Maximum retained positive samples per variable for pooled fallback fitting",
    )
    parser.add_argument(
        "--min-positive-samples-per-level",
        type=int,
        default=2000,
        help="Minimum positive sample count required for a level-specific fit",
    )
    parser.add_argument(
        "--winsorize-upper-quantile",
        type=float,
        default=0.999,
        help="Upper quantile used to winsorize positive log-values during fitting",
    )
    parser.add_argument(
        "--clip-max-overrides",
        help="Optional comma-separated overrides in VAR:max format",
    )
    parser.add_argument(
        "--plots-dir",
        help="Directory for output diagnostic plots. Defaults to <output_stem>_plots next to the output JSON",
    )
    parser.add_argument("--n-workers", type=int, default=1, help="Number of Dask worker processes to use")
    parser.add_argument("--threads-per-worker", type=int, default=1, help="Threads per Dask worker")
    parser.add_argument(
        "--memory-limit",
        default="8GiB",
        help="Per-worker memory limit for LocalCluster, e.g. 8GiB",
    )
    parser.add_argument(
        "--files-per-task",
        type=int,
        default=8,
        help="Number of zarr files processed sequentially in each submitted task",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=60.0,
        help="Heartbeat interval for progress logs while Dask batches are still running",
    )
    parser.add_argument(
        "--stalled-warning-seconds",
        type=float,
        default=900.0,
        help="Emit a stalled warning when no batch completes for this long",
    )
    args = parser.parse_args()

    variables = _parse_variables(args.variables)
    clip_max_overrides = _parse_clip_max_overrides(args.clip_max_overrides)
    start_date = _parse_yyyymmdd(args.start_date)
    end_date = _parse_yyyymmdd(args.end_date)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise SystemExit("--start-date must be <= --end-date")

    files = sorted(glob.glob(args.glob))
    files = _filter_files_by_date(files, start_date, end_date)
    files_before_day_sampling = len(files)
    files = _select_cases_per_day(files, num_cases_per_day=args.num_cases_per_day, seed=args.case_selection_seed)
    files = _select_files(files, max_files=args.max_files, seed=args.seed)
    if not files:
        raise SystemExit("No files matched after applying filters")
    if args.num_cases_per_day > 0:
        selected_days = len({_extract_case_date(file_path) for file_path in files})
        _log(
            f"[fit] day_sampling days={selected_days} "
            f"num_cases_per_day={args.num_cases_per_day} "
            f"selected_files={len(files)}/{files_before_day_sampling}"
        )

    first = _zarr_open(files[0])
    try:
        level_count = int(first.sizes["level"])
    finally:
        first.close()

    merged = _empty_batch_payload(level_count, variables)
    file_batches = _chunk_files(files, args.files_per_task)
    _log(
        f"[fit] matched_files={len(files)} batches={len(file_batches)} "
        f"workers={args.n_workers} files_per_task={args.files_per_task}"
    )

    if args.n_workers <= 1:
        for batch_idx, file_batch in enumerate(file_batches, start=1):
            _log(
                f"[fit] processing batch {batch_idx}/{len(file_batches)} "
                f"files={len(file_batch)} [{_summarize_file_batch(file_batch)}]"
            )
            partial = _collect_batch_statistics(
                file_batch=file_batch,
                variables=variables,
                level_count=level_count,
                zero_floor=args.zero_floor,
                positive_samples_per_file_per_level=args.samples_per_file_per_level,
                plot_samples_per_file_per_level=args.plot_samples_per_file_per_level,
                max_level_samples=args.max_level_samples,
                max_plot_samples_per_level=args.max_plot_samples_per_level,
                max_pooled_samples=args.max_pooled_samples,
                seed=int(args.seed + batch_idx * 7919),
            )
            _merge_batch_payload(
                merged,
                partial,
                variables,
                level_count,
                args.max_level_samples,
                args.max_plot_samples_per_level,
                args.max_pooled_samples,
            )
    else:
        cluster = LocalCluster(
            n_workers=args.n_workers,
            threads_per_worker=args.threads_per_worker,
            processes=True,
            memory_limit=args.memory_limit,
            silence_logs=logging.ERROR,
        )
        client = Client(cluster)
        heartbeat_interval = max(float(args.progress_interval_seconds), 1.0)
        stalled_warning_seconds = max(float(args.stalled_warning_seconds), heartbeat_interval)
        _log(f"[dask] dashboard={client.dashboard_link}")
        _log(
            f"[dask] heartbeat_interval={_format_elapsed(heartbeat_interval)} "
            f"stalled_warning_after={_format_elapsed(stalled_warning_seconds)}"
        )
        try:
            futures = []
            future_meta = {}
            for batch_idx, file_batch in enumerate(file_batches, start=1):
                batch_seed = int(args.seed + (batch_idx - 1) * 7919)
                future = client.submit(
                    _collect_batch_statistics,
                    file_batch,
                    variables,
                    level_count,
                    args.zero_floor,
                    args.samples_per_file_per_level,
                    args.plot_samples_per_file_per_level,
                    args.max_level_samples,
                    args.max_plot_samples_per_level,
                    args.max_pooled_samples,
                    batch_seed,
                    pure=False,
                )
                futures.append(future)
                future_meta[future] = {
                    "batch_number": batch_idx,
                    "batch_total": len(file_batches),
                    "file_count": len(file_batch),
                    "label": _summarize_file_batch(file_batch),
                    "submitted_at": time.monotonic(),
                }
            completed = 0
            started_at = time.monotonic()
            last_progress_at = started_at
            pending = set(futures)
            while pending:
                try:
                    done_now, pending_after_wait = wait(
                        pending,
                        timeout=heartbeat_interval,
                        return_when="FIRST_COMPLETED",
                    )
                    done_now = set(done_now)
                    pending = set(pending_after_wait)
                except BuiltinTimeoutError:
                    done_now = set()
                if not done_now:
                    _emit_dask_heartbeat(
                        client=client,
                        pending_futures=pending,
                        future_meta=future_meta,
                        completed=completed,
                        total=len(file_batches),
                        started_at=started_at,
                        last_progress_at=last_progress_at,
                        stalled_warning_seconds=stalled_warning_seconds,
                    )
                    continue

                for future in sorted(done_now, key=lambda item: future_meta[item]["batch_number"]):
                    meta = future_meta[future]
                    try:
                        partial = future.result()
                    except Exception:
                        age = _format_elapsed(time.monotonic() - meta["submitted_at"])
                        _log(
                            f"[dask][error] batch {meta['batch_number']}/{meta['batch_total']} "
                            f"failed after {age} [{meta['label']}]"
                        )
                        raise
                    _merge_batch_payload(
                        merged,
                        partial,
                        variables,
                        level_count,
                        args.max_level_samples,
                        args.max_plot_samples_per_level,
                        args.max_pooled_samples,
                    )
                    completed += 1
                    last_progress_at = time.monotonic()
                    age = _format_elapsed(last_progress_at - meta["submitted_at"])
                    _log(
                        f"[dask] completed batch {completed}/{len(file_batches)} "
                        f"in {age} files={meta['file_count']} [{meta['label']}]"
                    )
        finally:
            try:
                client.shutdown()
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass
            try:
                cluster.close()
            except Exception:
                pass

    output = {
        "transform_type": "zero_inflated_lognormal_probit",
        "zero_floor": float(args.zero_floor),
        "probit_eps": float(args.probit_eps),
        "source_glob": args.glob,
        "selected_files": len(files),
        "variables": {},
    }

    plots_dir = Path(args.plots_dir) if args.plots_dir else Path(args.output).with_suffix("").with_name(Path(args.output).stem + "_plots")

    for var in variables:
        pooled_fit = _fit_positive_lognormal(
            merged[var]["pooled_values"],
            zero_floor=args.zero_floor,
            winsorize_upper_quantile=args.winsorize_upper_quantile,
        )
        levels = []
        counts_total = merged[var]["count_total"]
        counts_positive = merged[var]["count_positive"]
        for lev in range(level_count):
            n_total = int(counts_total[lev])
            n_positive = int(counts_positive[lev])
            alpha = 1.0 if n_total <= 0 else float(max(0.0, min(1.0, 1.0 - (n_positive / n_total))))
            if n_positive <= 0:
                entry = {
                    "status": "degenerate_zero",
                    "alpha": 1.0,
                    "n_total": n_total,
                    "n_positive": 0,
                }
            elif n_positive < args.min_positive_samples_per_level or merged[var]["level_positive_values"][lev].size < max(32, args.min_positive_samples_per_level // 4):
                entry = {
                    "status": "borrow_fallback",
                    "alpha": alpha,
                    "mu": float(pooled_fit["mu"]),
                    "sigma": float(pooled_fit["sigma"]),
                    "n_total": n_total,
                    "n_positive": n_positive,
                }
            else:
                level_fit = _fit_positive_lognormal(
                    merged[var]["level_positive_values"][lev],
                    zero_floor=args.zero_floor,
                    winsorize_upper_quantile=args.winsorize_upper_quantile,
                )
                entry = {
                    "status": "ok",
                    "alpha": alpha,
                    "mu": float(level_fit["mu"]),
                    "sigma": float(level_fit["sigma"]),
                    "n_total": n_total,
                    "n_positive": n_positive,
                }
            levels.append(entry)

        output["variables"][var] = {
            "clip_max": float(clip_max_overrides.get(var, default_clip_max_for_var(var))),
            "fallback_positive_fit": pooled_fit,
            "levels": levels,
        }

        spec = _build_plot_spec(output, args.zero_floor, args.probit_eps, var)
        plot_path = plots_dir / f"{var.lower()}_zero_inflated_transform.png"
        _plot_variable_distributions(
            var_name=var,
            spec=spec,
            raw_positive_by_level=merged[var]["level_positive_values"],
            raw_plot_by_level=merged[var]["level_plot_values"],
            counts_total=counts_total,
            counts_positive=counts_positive,
            plot_path=plot_path,
        )
        output["variables"][var]["plot"] = str(plot_path)

        nondeg = sum(1 for item in levels if item["status"] == "ok")
        borrowed = sum(1 for item in levels if item["status"] == "borrow_fallback")
        deg = sum(1 for item in levels if item["status"] == "degenerate_zero")
        _log(
            f"[fit] {var}: ok_levels={nondeg} borrow_levels={borrowed} zero_levels={deg} "
            f"pooled_mu={pooled_fit['mu']:.4f} pooled_sigma={pooled_fit['sigma']:.4f}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    _log(f"[fit] wrote transform parameters to {out_path}")
    _log(f"[fit] wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
