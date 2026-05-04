from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import xarray as xr
from credit.data import get_forward_data
from credit.transforms.concentration import (
    CONCENTRATION_VARS,
    forward_concentration_transform_numpy,
    load_concentration_transform_json,
)

def _zarr_open(file_path: str, **kwargs):
    if kwargs:
        raise TypeError(f"_zarr_open does not accept extra keyword arguments, got: {sorted(kwargs)}")
    return get_forward_data(str(file_path), zarr_chunks=None)

from dask.distributed import Client, LocalCluster, as_completed


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
            f"'wofs_YYYYMMDD_<slug_>HHMM_memNN.zarr(.zip)' or "
            f"'wofs_boundary_YYYYMMDD_<slug_>HHMM_memNN.zarr(.zip)', got: {file_path}"
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


def _chunk_files(files: list[str], files_per_task: int) -> list[list[str]]:
    if files_per_task <= 0:
        raise ValueError(f"files_per_task must be positive, got {files_per_task}")
    return [files[i : i + files_per_task] for i in range(0, len(files), files_per_task)]


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


def _clone_accumulator_schema(accum: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    return {
        var: {
            "count": np.zeros_like(stats["count"]),
            "sum": np.zeros_like(stats["sum"]),
            "sum_sq": np.zeros_like(stats["sum_sq"]),
        }
        for var, stats in accum.items()
    }


def _merge_accumulators(
    accum: dict[str, dict[str, np.ndarray]],
    partial: dict[str, dict[str, np.ndarray]],
) -> None:
    for var in accum:
        accum[var]["count"] += partial[var]["count"]
        accum[var]["sum"] += partial[var]["sum"]
        accum[var]["sum_sq"] += partial[var]["sum_sq"]


def _transform_concentration(da: xr.DataArray, transform_spec: dict | None) -> xr.DataArray:
    """Apply the configured concentration transform before statistics."""

    if transform_spec is None:
        return da

    arr = np.asarray(da.values, dtype=np.float64)
    level_axis = da.dims.index("level") if "level" in da.dims else None
    transformed = forward_concentration_transform_numpy(arr, transform_spec, level_axis=level_axis)
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
    return None, None


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
    accum_template: dict[str, dict[str, np.ndarray]],
    concentration_transform_specs: dict[str, dict],
    exclude_clipped_boundary_samples: bool,
    clip_boundary_tolerance: float,
) -> dict[str, dict[str, np.ndarray]]:
    if not file_batch:
        raise ValueError("Received empty file batch")

    accum = _clone_accumulator_schema(accum_template)
    data_vars = list(accum.keys())

    for file_path in file_batch:
        ds = _zarr_open(file_path)
        try:
            available_vars = set(ds.data_vars)
            if not available_vars:
                print(f"[warn] skipping empty dataset: {Path(file_path).name}")
                continue
            for var in data_vars:
                if var not in available_vars:
                    continue
                da = ds[var]
                valid_mask = None
                if var in CONCENTRATION_VARS:
                    transform_spec = concentration_transform_specs.get(var)
                    if exclude_clipped_boundary_samples and transform_spec is not None:
                        clip_min, clip_max = _get_transform_clip_bounds(transform_spec)
                        if clip_min is not None or clip_max is not None:
                            raw = np.asarray(da.values, dtype=np.float64)
                            valid_mask = np.isfinite(raw)
                            tol = float(clip_boundary_tolerance)
                            if clip_min is not None:
                                valid_mask = valid_mask & (raw > float(clip_min) + tol)
                            if clip_max is not None:
                                valid_mask = valid_mask & (raw < float(clip_max) - tol)
                    da = _transform_concentration(da, transform_spec)
                _accumulate(accum[var], da, valid_mask=valid_mask)
        finally:
            ds.close()

    return accum


def main():
    parser = argparse.ArgumentParser(description="Compute CREDIT-style mean/std files from converted WoFS CREDIT-WRF zarr archives")
    parser.add_argument("--glob", required=True, help="Glob matching converted WoFS CREDIT-WRF zarr files")
    parser.add_argument("--start-date", help="Inclusive lower date bound in YYYYMMDD format")
    parser.add_argument("--end-date", help="Inclusive upper date bound in YYYYMMDD format")
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
        help="Generalized concentration transform JSON path (e.g. zero-inflated probit or log-zscore)",
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
    if not args.transform_params_json:
        raise SystemExit(
            "Missing --transform-params-json/--log-transform-params-json. "
            "The deprecated piecewise concentration transform path has been removed."
        )

    files = sorted(glob.glob(args.glob))
    start_date = _parse_yyyymmdd(args.start_date)
    end_date = _parse_yyyymmdd(args.end_date)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise SystemExit("--start-date must be <= --end-date")

    files = _filter_files_by_date(files, start_date, end_date)
    files_before_day_sampling = len(files)
    files = _select_cases_per_day(files, args.num_cases_per_day, args.case_selection_seed)
    if not files:
        raise SystemExit(
            f"No files matched: {args.glob} after applying date filter "
            f"start={args.start_date!r} end={args.end_date!r}"
        )
    if args.num_cases_per_day > 0:
        selected_days = len({_extract_case_date(file_path) for file_path in files})
        print(
            f"[stats] day_sampling days={selected_days} "
            f"num_cases_per_day={args.num_cases_per_day} "
            f"selected_files={len(files)}/{files_before_day_sampling}"
        )

    concentration_transform_specs: dict[str, dict] = {}
    raw_transform_payload: dict | None = None
    raw_transform_payload, concentration_transform_specs = load_concentration_transform_json(
        args.transform_params_json,
        variables=CONCENTRATION_VARS,
    )
    concentration_transform_source = f"transform_json:{args.transform_params_json}"
    print(
        f"[transform] loaded generalized transform specs for "
        f"{len(concentration_transform_specs)} variables from {args.transform_params_json}"
    )

    first = None
    for file_path in files:
        candidate = _zarr_open(file_path)
        candidate_data_vars = [str(v) for v in candidate.data_vars if v not in {"trajectory_id"}]
        if candidate_data_vars:
            first = candidate
            data_vars = candidate_data_vars
            break
        candidate.close()

    if first is None:
        raise SystemExit("No usable variables found in any selected file")

    try:
        level_coord = first["level"].copy() if "level" in first.coords else None
        latitude = first["latitude"].values.copy() if args.latweights_out else None
        longitude = first["longitude"].values.copy() if args.latweights_out else None
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
                    accum,
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
                        accum,
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
        "source": concentration_transform_source,
        "transform_specs_count": len(concentration_transform_specs),
        "exclude_clipped_boundary_samples": bool(args.exclude_clipped_boundary_samples),
        "clip_boundary_tolerance": float(args.clip_boundary_tolerance),
    }
    transform_metadata_str = json.dumps(transform_metadata, sort_keys=True)
    mean_ds.attrs["concentration_transform_params_json"] = transform_metadata_str
    std_ds.attrs["concentration_transform_params_json"] = transform_metadata_str
    mean_ds.attrs["concentration_transform_source"] = concentration_transform_source
    std_ds.attrs["concentration_transform_source"] = concentration_transform_source
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

        if var in CONCENTRATION_VARS:
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
