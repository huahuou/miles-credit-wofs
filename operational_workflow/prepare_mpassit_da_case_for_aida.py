#!/usr/bin/env python3
"""
Convert MPASSIT WoFS DA-cycle files into CREDIT-WRF zarr cases for AIDA rollout.

The output layout matches the single-member CREDIT-WRF archives consumed by
applications/rollout_wrf_wofs_mae_da_metrics.py:

    wofs_YYYYMMDD_HHMM_memXX.zarr
        time, level, y, x
        T, QVAPOR, U, V, W, GEOPOT, REFL_10CM, precip vars, XLAND, HGT, ...

MPASSIT fields are preprocessed the same way as the ufs2arco WRF raw source:
destagger U/V/W and PH/PHB-derived GEOPOT, add 300 K to WRF perturbation
potential temperature T, clip negative QVAPOR, and select the configured
17-level subset.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import yaml


DEFAULT_INPUT_DIR = Path("/scratch3/NAGAPE/wof/cliu/misc/mldata/dacycles_Z_nssl_20240508_2000_bg2wrf")
DEFAULT_CONFIG = Path("/home/Zhanxiang.Hua/miles-credit-wofs/config/wofs_diffmae_4x4_patch_height_mask.yml")
DEFAULT_OUT_DIR = Path("/scratch5/purged/Zhanxiang.Hua/wofs_mpassit_da_aida/cases")
DEFAULT_LEVEL_INDICES = tuple(range(0, 51, 3))[:17]
REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_members(raw: str, input_dir: Path) -> list[int]:
    if raw.strip().lower() == "all":
        members = []
        for path in sorted(input_dir.glob("mem_*")):
            match = re.fullmatch(r"mem_(\d+)", path.name)
            if match:
                members.append(int(match.group(1)))
        if not members:
            raise FileNotFoundError(f"No mem_XX directories found under {input_dir}")
        return members

    members: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = [int(x) for x in part.split("-", 1)]
            members.extend(range(start, stop + 1))
        else:
            members.append(int(part))
    if not members:
        raise ValueError(f"No members parsed from {raw!r}")
    return sorted(set(members))


def _parse_level_indices(raw: str | None) -> list[int]:
    if raw is None or raw.strip().lower() == "default":
        return list(DEFAULT_LEVEL_INDICES)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _infer_case_info(input_dir: Path, event_date: str | None, cycle: str | None) -> tuple[str, str]:
    if event_date and cycle:
        return event_date, cycle
    match = re.search(r"_(\d{8})_(\d{4})_bg2wrf$", input_dir.name)
    if not match:
        missing = []
        if not event_date:
            missing.append("--event-date")
        if not cycle:
            missing.append("--cycle")
        raise ValueError(f"Could not infer {' and '.join(missing)} from {input_dir.name}; pass them explicitly.")
    return event_date or match.group(1), cycle or match.group(2)


def _valid_time(event_date: str, cycle: str, valid_time: str | None) -> np.datetime64:
    if valid_time:
        return np.datetime64(pd.Timestamp(valid_time).to_datetime64())
    ts = pd.Timestamp(
        year=int(event_date[:4]),
        month=int(event_date[4:6]),
        day=int(event_date[6:8]),
        hour=int(cycle[:2]),
        minute=int(cycle[2:]),
    )
    return np.datetime64(ts.to_datetime64())


def _load_config_vars(config_path: Path) -> dict[str, list[str]]:
    with config_path.open() as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    data = conf["data"]
    return {
        "background": list(data.get("background_vars", [])),
        "precip": list(data.get("precip_vars", [])),
        "reflectivity": list(data.get("reflectivity_vars", [])),
        "surface": list(data.get("surface_vars", [])),
        "forcing": list(data.get("dynamic_forcing_variables", [])),
    }


def _member_bg_path(input_dir: Path, member: int, event_date: str, cycle: str) -> Path:
    mem_dir = input_dir / f"mem_{member:02d}"
    stamp = f"{event_date[:4]}-{event_date[4:6]}-{event_date[6:8]}_{cycle[:2]}.{cycle[2:]}.00"
    expected = mem_dir / f"MPASSIT_{member:02d}.{stamp}.nc"
    if expected.exists():
        return expected
    matches = sorted(mem_dir.glob(f"MPASSIT_{member:02d}.*.nc"))
    if not matches:
        raise FileNotFoundError(f"No MPASSIT file found for member {member:02d} in {mem_dir}")
    return matches[0]


def _member_innov_path(input_dir: Path, member: int) -> Path:
    per_member = input_dir / f"innov_mrms_refl_wrf_mem{member:02d}.nc"
    if per_member.exists():
        return per_member
    mean_file = input_dir / "innov_mrms_refl_wrf.nc"
    if mean_file.exists():
        return mean_file
    raise FileNotFoundError(f"No innovation file found for member {member:02d} under {input_dir}")


def _time0(arr: xr.DataArray) -> xr.DataArray:
    for dim in ("Time", "time"):
        if dim in arr.dims:
            return arr.isel({dim: 0}, drop=True)
    return arr


def _as_float32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _destagger_np(values: np.ndarray, axis: int) -> np.ndarray:
    left = [slice(None)] * values.ndim
    right = [slice(None)] * values.ndim
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)
    return 0.5 * (values[tuple(left)] + values[tuple(right)])


def _read_3d_var(bg: xr.Dataset, var: str, level_indices: list[int], template: np.ndarray | None = None) -> np.ndarray:
    if var == "GEOPOT":
        if "PH" not in bg or "PHB" not in bg:
            raise KeyError("GEOPOT requested, but PH/PHB are missing from background file")
        geopot_stag = _as_float32(_time0(bg["PH"]).values + _time0(bg["PHB"]).values)
        return _destagger_np(geopot_stag, axis=0)[level_indices]

    if var not in bg:
        if template is None:
            raise KeyError(var)
        return np.zeros_like(template, dtype=np.float32)

    arr = _as_float32(_time0(bg[var]).values)
    dims = _time0(bg[var]).dims
    if "west_east_stag" in dims:
        arr = _destagger_np(arr, axis=dims.index("west_east_stag"))
        dims = tuple("west_east" if d == "west_east_stag" else d for d in dims)
    if "south_north_stag" in dims:
        arr = _destagger_np(arr, axis=dims.index("south_north_stag"))
        dims = tuple("south_north" if d == "south_north_stag" else d for d in dims)
    if "bottom_top_stag" in dims:
        arr = _destagger_np(arr, axis=dims.index("bottom_top_stag"))
        dims = tuple("bottom_top" if d == "bottom_top_stag" else d for d in dims)

    if "bottom_top" not in dims:
        raise ValueError(f"{var} is not a 3-D WRF variable; dims={dims}")
    lev_axis = dims.index("bottom_top")
    arr = np.moveaxis(arr, lev_axis, 0)[level_indices]

    if var == "T":
        arr = arr + np.float32(300.0)
    elif var == "QVAPOR":
        arr = np.where(arr > 0.0, arr, 0.0).astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def _read_2d_var(bg: xr.Dataset, var: str, shape: tuple[int, int]) -> np.ndarray:
    if var not in bg:
        return np.zeros(shape, dtype=np.float32)
    arr = _as_float32(_time0(bg[var]).values)
    if arr.shape != shape:
        raise ValueError(f"{var} has shape {arr.shape}, expected {shape}")
    return arr


def _read_grid(bg: xr.Dataset) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    lat = _as_float32(_time0(bg["XLAT"]).values)
    lon = _as_float32(_time0(bg["XLONG"]).values)
    if lat.shape != lon.shape:
        raise ValueError(f"XLAT shape {lat.shape} != XLONG shape {lon.shape}")
    return lat, lon, lat.shape


def _read_da_grid(path: Path, var: str, level_indices: list[int]) -> np.ndarray:
    with xr.open_dataset(path, decode_times=False) as ds:
        if var not in ds:
            raise KeyError(f"{var} not found in {path}")
        arr = _as_float32(_time0(ds[var]).values)
    return arr[level_indices]


def _read_da_count(path: Path, var: str, level_indices: list[int], shape: tuple[int, int]) -> np.ndarray:
    with xr.open_dataset(path, decode_times=False) as ds:
        if var not in ds:
            return np.zeros((len(level_indices), *shape), dtype=np.int32)
        arr = np.asarray(_time0(ds[var]).values)
    return arr[level_indices].astype(np.int32, copy=False)


def _forcing_arrays(forcing_vars: Iterable[str], valid_time: np.datetime64, lat2d: np.ndarray, lon2d: np.ndarray) -> dict[str, np.ndarray]:
    ts = pd.Timestamp(valid_time.astype("datetime64[ns]").astype("int64"), unit="ns")
    lat_rad = np.deg2rad(lat2d).astype(np.float32)
    lon_rad = np.deg2rad(lon2d).astype(np.float32)
    day_of_year = ts.dayofyear
    total_days = 366 if ts.is_leap_year else 365
    julian_angle = np.float32(2.0 * np.pi * day_of_year / total_days)
    utc_hour = np.float32(ts.hour + ts.minute / 60.0 + ts.second / 3600.0)
    local_hour = (utc_hour + lon2d.astype(np.float32) / 15.0) % 24.0
    local_angle = (2.0 * np.pi * local_hour / 24.0).astype(np.float32)
    decl = np.float32(0.409 * np.sin(julian_angle - 1.39))
    h_angle = np.deg2rad((local_hour - 12.0) * 15.0).astype(np.float32)
    cos_zenith = (
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(h_angle)
    ).astype(np.float32)
    computed = {
        "cos_latitude": np.cos(lat_rad).astype(np.float32),
        "sin_latitude": np.sin(lat_rad).astype(np.float32),
        "cos_longitude": np.cos(lon_rad).astype(np.float32),
        "sin_longitude": np.sin(lon_rad).astype(np.float32),
        "cos_julian_day": np.full(lat2d.shape, np.cos(julian_angle), dtype=np.float32),
        "sin_julian_day": np.full(lat2d.shape, np.sin(julian_angle), dtype=np.float32),
        "cos_local_time": np.cos(local_angle).astype(np.float32),
        "sin_local_time": np.sin(local_angle).astype(np.float32),
        "cos_solar_zenith_angle": cos_zenith,
        "insolation": np.clip(cos_zenith, 0.0, None).astype(np.float32),
    }
    return {name: computed[name] for name in forcing_vars if name in computed}


def _finite_or_zero(name: str, arr: np.ndarray, replacements: dict[str, int]) -> np.ndarray:
    mask = ~np.isfinite(arr)
    count = int(mask.sum())
    if count:
        replacements[name] = replacements.get(name, 0) + count
        arr = np.where(mask, 0.0, arr).astype(np.float32, copy=False)
    return arr


def _write_zarr_v2(ds: xr.Dataset, out_path: Path, overwrite: bool) -> None:
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")
        shutil.rmtree(out_path)
    encoding = {}
    for name, da in ds.data_vars.items():
        if da.ndim == 4:
            encoding[name] = {"chunks": (1, da.shape[1], da.shape[2], da.shape[3])}
        elif da.ndim == 3:
            encoding[name] = {"chunks": (1, da.shape[1], da.shape[2])}
        elif da.ndim == 2:
            encoding[name] = {"chunks": da.shape}
        elif da.ndim == 1:
            encoding[name] = {"chunks": da.shape}
    try:
        ds.to_zarr(out_path, mode="w", consolidated=True, encoding=encoding, zarr_format=2)
    except TypeError:
        ds.to_zarr(out_path, mode="w", consolidated=True, encoding=encoding, zarr_version=2)


def _build_member_dataset(
    *,
    bg_path: Path,
    obs_path: Path,
    innov_path: Path,
    config_vars: dict[str, list[str]],
    level_indices: list[int],
    valid_time: np.datetime64,
    event_date: str,
    cycle: str,
    member: int,
    refl_condition: str,
    strict_vars: bool,
) -> xr.Dataset:
    replacements: dict[str, int] = {}
    missing_required: list[str] = []
    missing_filled: list[str] = []

    with xr.open_dataset(bg_path, decode_times=False) as bg:
        lat2d, lon2d, shape = _read_grid(bg)
        y = np.arange(shape[0], dtype=np.int32)
        x = np.arange(shape[1], dtype=np.int32)
        level = np.arange(len(level_indices), dtype=np.int32)

        template = _read_3d_var(bg, "T", level_indices)
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

        upper_vars = []
        for group in ("background", "precip", "reflectivity"):
            for var in config_vars[group]:
                if var not in upper_vars:
                    upper_vars.append(var)

        bg_refl = None
        for var in upper_vars:
            try:
                arr = _read_3d_var(bg, var, level_indices, template=template)
                if var not in bg and var != "GEOPOT":
                    missing_filled.append(var)
            except KeyError:
                if strict_vars:
                    missing_required.append(var)
                    continue
                arr = np.zeros_like(template, dtype=np.float32)
                missing_filled.append(var)
            arr = _finite_or_zero(var, arr, replacements)
            if var == "REFL_10CM":
                bg_refl = arr.copy()
            data_vars[var] = (("time", "level", "y", "x"), arr[np.newaxis, ...])

        if missing_required:
            raise KeyError(f"Missing required variables in {bg_path}: {missing_required}")

        if bg_refl is None:
            raise KeyError("REFL_10CM is required by this workflow")

        obs_refl = _read_da_grid(obs_path, "REFL", level_indices)
        innov = _read_da_grid(innov_path, "INNOV", level_indices)
        if obs_refl.shape != bg_refl.shape or innov.shape != bg_refl.shape:
            raise ValueError(
                f"DA grid shape mismatch: bg={bg_refl.shape}, obs={obs_refl.shape}, innov={innov.shape}"
            )

        if refl_condition == "background":
            cond_refl = bg_refl
        elif refl_condition == "obs":
            cond_refl = np.where(np.isfinite(obs_refl), obs_refl, bg_refl)
        elif refl_condition == "background_plus_innov":
            cond_refl = np.where(np.isfinite(innov), bg_refl + innov, bg_refl)
        else:
            raise ValueError(f"Unknown refl condition {refl_condition}")
        data_vars["REFL_10CM"] = (("time", "level", "y", "x"), _finite_or_zero("REFL_10CM", cond_refl, replacements)[np.newaxis, ...])

        for var in sorted(set(config_vars["surface"] + config_vars["forcing"])):
            if var in data_vars:
                continue
            if var in {"cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude",
                       "cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time",
                       "cos_solar_zenith_angle", "insolation"}:
                continue
            arr2d = _read_2d_var(bg, var, shape)
            arr2d = _finite_or_zero(var, arr2d, replacements)
            data_vars[var] = (("time", "y", "x"), arr2d[np.newaxis, ...])

    for var, arr2d in _forcing_arrays(config_vars["forcing"], valid_time, lat2d, lon2d).items():
        data_vars[var] = (("time", "y", "x"), arr2d[np.newaxis, ...])

    data_vars["latitude"] = (("y", "x"), lat2d)
    data_vars["longitude"] = (("y", "x"), lon2d)
    data_vars["trajectory_id"] = (("time",), np.array([0], dtype=np.int64))
    data_vars["REFL_10CM_BACKGROUND"] = (("time", "level", "y", "x"), bg_refl[np.newaxis, ...])
    data_vars["MRMS_REFL"] = (("time", "level", "y", "x"), obs_refl[np.newaxis, ...].astype(np.float32))
    data_vars["MRMS_INNOV"] = (("time", "level", "y", "x"), innov[np.newaxis, ...].astype(np.float32))
    data_vars["MRMS_OBS_COUNT"] = (("time", "level", "y", "x"), _read_da_count(obs_path, "OBS_COUNT", level_indices, lat2d.shape)[np.newaxis, ...])
    data_vars["MRMS_INNOV_COUNT"] = (("time", "level", "y", "x"), _read_da_count(innov_path, "INNOV_COUNT", level_indices, lat2d.shape)[np.newaxis, ...])

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.array([valid_time], dtype="datetime64[ns]"),
            "level": level,
            "y": y,
            "x": x,
            "lat": ("y", np.asarray(lat2d.mean(axis=1), dtype=np.float32)),
            "lon": ("x", np.asarray(lon2d.mean(axis=0), dtype=np.float32)),
        },
        attrs={
            "description": "MPASSIT WoFS DA-cycle case converted to CREDIT-WRF layout for AIDA DiffMAE rollout.",
            "event_date": event_date,
            "cycle": cycle,
            "member": f"{member:02d}",
            "source_background": str(bg_path),
            "source_observation": str(obs_path),
            "source_innovation": str(innov_path),
            "source_level_indices": ",".join(str(i) for i in level_indices),
            "reflectivity_condition": refl_condition,
            "missing_variables_filled_with_zero": ",".join(sorted(set(missing_filled))),
            "nonfinite_replacements": json.dumps(replacements, sort_keys=True),
        },
    )
    return ds


def _validate_output(out_path: Path, config_vars: dict[str, list[str]]) -> None:
    required = []
    for group in ("background", "precip", "reflectivity", "surface"):
        required.extend(config_vars[group])
    required.extend(var for var in config_vars["forcing"] if var in {"XLAND", "HGT"})
    required = sorted(set(required))

    with xr.open_zarr(out_path, consolidated=True, zarr_format=2) as ds:
        missing = [var for var in required if var not in ds]
        if missing:
            raise RuntimeError(f"Validation failed: missing variables in {out_path}: {missing}")
        if ds.sizes.get("time") != 1:
            raise RuntimeError(f"Validation failed: expected time=1, got {ds.sizes.get('time')}")
        if "T" in ds and ds["T"].sizes.get("level") != ds.sizes.get("level"):
            raise RuntimeError(f"Validation failed: inconsistent T level count in {out_path}")
        for var in required:
            values = ds[var].values
            if not np.isfinite(values).all():
                raise RuntimeError(f"Validation failed: {var} contains non-finite values")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from credit.data import filter_ds, get_forward_data

        ds = get_forward_data(str(out_path))
        try:
            filter_ds(ds, required)
        finally:
            ds.close()
    except Exception as exc:
        raise RuntimeError(f"Validation failed through credit.data.get_forward_data/filter_ds: {exc}") from exc


def _write_rollout_config(base_config: Path, config_out: Path, out_dir: Path) -> None:
    with base_config.open() as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    conf.setdefault("data", {})["save_loc"] = str(out_dir / "wofs_*.zarr")
    config_out.parent.mkdir(parents=True, exist_ok=True)
    with config_out.open("w") as f:
        yaml.safe_dump(conf, f, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--members", default="all", help='Member list, e.g. "1", "1,3,5", "1-36", or "all".')
    parser.add_argument("--event-date", default=None, help="YYYYMMDD. Inferred from input directory by default.")
    parser.add_argument("--cycle", default=None, help="HHMM. Inferred from input directory by default.")
    parser.add_argument("--valid-time", default=None, help="Override valid time, e.g. 2024-05-08T20:00:00.")
    parser.add_argument("--level-indices", default="default", help="Comma-separated source bottom_top indices. Default: 0,3,...,48.")
    parser.add_argument(
        "--refl-condition",
        choices=["background_plus_innov", "obs", "background"],
        default="obs",
        help="How to build REFL_10CM used by AIDA. Obs/innov NaNs fall back to background.",
    )
    parser.add_argument("--strict-vars", action="store_true", help="Fail instead of zero-filling missing config variables.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--config-out", type=Path, default=None, help="Optional rollout config copy with data.save_loc set to this out-dir.")
    args = parser.parse_args()

    event_date, cycle = _infer_case_info(args.input_dir, args.event_date, args.cycle)
    valid_time = _valid_time(event_date, cycle, args.valid_time)
    level_indices = _parse_level_indices(args.level_indices)
    config_vars = _load_config_vars(args.config)
    members = _parse_members(args.members, args.input_dir)
    obs_path = args.input_dir / "obs_mrms_refl_wrf.nc"
    if not obs_path.exists():
        raise FileNotFoundError(obs_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for member in members:
        bg_path = _member_bg_path(args.input_dir, member, event_date, cycle)
        innov_path = _member_innov_path(args.input_dir, member)
        out_path = args.out_dir / f"wofs_{event_date}_{cycle}_mem{member:02d}.zarr"
        print(f"[member {member:02d}] background={bg_path}")
        print(f"[member {member:02d}] observation={obs_path}")
        print(f"[member {member:02d}] innovation={innov_path}")
        ds = _build_member_dataset(
            bg_path=bg_path,
            obs_path=obs_path,
            innov_path=innov_path,
            config_vars=config_vars,
            level_indices=level_indices,
            valid_time=valid_time,
            event_date=event_date,
            cycle=cycle,
            member=member,
            refl_condition=args.refl_condition,
            strict_vars=args.strict_vars,
        )
        _write_zarr_v2(ds, out_path, overwrite=args.overwrite)
        ds.close()
        if not args.skip_validation:
            _validate_output(out_path, config_vars)
        print(f"[member {member:02d}] wrote {out_path}")
        written.append(out_path)

    if args.config_out is not None:
        _write_rollout_config(args.config, args.config_out, args.out_dir)
        print(f"Wrote rollout config: {args.config_out}")

    print(f"Converted {len(written)} member case(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
