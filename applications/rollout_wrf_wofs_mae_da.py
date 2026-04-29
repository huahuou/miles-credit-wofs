"""
WoFS MAE Data Assimilation Rollout Script
------------------------------------------
Loads a trained WoFSMultiModalMAE checkpoint, processes one or more WoFS
zarr cases, and writes the precip analysis (Q_analysis) back to disk.

Workflow per case file:
    1. Load background, reflectivity, surface, forcing from zarr
    2. Normalize using mean.nc / std.nc + concentration transform
    3. Run model.assimilate(background, obs_refl, surface, forcing)
    4. Inverse-normalize the precip output
    5. Write QRAIN/QNRAIN/… analysis arrays to output zarr

Usage:
    python applications/rollout_wrf_wofs_mae_da.py \\
        -c config/wofs_mae_da.yml \\
        --checkpoint /path/to/checkpoint.pt \\
        --start-date 20230501 --end-date 20230531 \\
        --out-dir /work2/zhanxianghua/wofs_mae_analysis

Output layout:
    <out-dir>/<YYYYMMDD>/<case_stem>_analysis.zarr.zip
"""

import argparse
import json
import logging
import os
import re
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import xarray as xr
import yaml
import zarr
import zarr.storage

from credit.datasets.wrf_wofs_mae import (
    WoFSMAEDataset,
    _coerce_concentration_params,
    _forward_concentration_numpy,
    DEFAULT_BACKGROUND_VARS,
    DEFAULT_PRECIP_VARS,
    DEFAULT_REFLECTIVITY_VARS,
    DEFAULT_SURFACE_VARS,
    DEFAULT_DYNAMIC_FORCING_VARS,
    CONCENTRATION_VARS,
    _DEFAULT_CONCENTRATION_PARAMS,
)
from credit.data import filter_ds, get_forward_data
from credit.models import load_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inverse normalization
# ---------------------------------------------------------------------------

def _inverse_concentration_numpy(arr: np.ndarray, params: dict) -> np.ndarray:
    """Piecewise inverse of the concentration transform used in WoFSDAIncrementDataset.

    Forward: f(x) = c1*min(x, cmax) + c2*(log(max(x,eps)) - log(eps)) / (-log(eps))

    Regions:
        1  x < eps   : f(x) = c1*x               → x = y / c1  (closed-form)
        2  eps<=x<cmax: mixed linear+log term      → bisection on [eps, cmax]
        3  x >= cmax : c1*cmax + c2*log(x/eps)/L  → x = eps*exp((y-c1*cmax)*L/c2)  (closed-form)
    """
    target = np.asarray(arr, dtype=np.float64)
    target = np.maximum(target, 0.0)

    c1 = float(params["c1"])
    c2 = float(params["c2"])
    eps = float(params["conc_eps"])
    cmax = float(params["conc_max"])
    clip_min = params.get("value_clip_min")
    clip_max = params.get("value_clip_max")
    log_eps = float(np.log(eps))
    neg_log_eps = float(-log_eps)

    # Boundary values in transform space
    y1 = c1 * eps                                                           # f(eps) with log term = 0
    y2 = c1 * cmax + c2 * (np.log(cmax) - log_eps) / neg_log_eps           # f(cmax)

    out = np.empty_like(target)

    # Region 1: target <= y1  →  x = target / c1
    mask1 = target <= y1
    if np.any(mask1):
        out[mask1] = target[mask1] / c1

    # Region 3: target >= y2  →  x = eps * exp((target - c1*cmax) * neg_log_eps / c2)
    mask3 = target >= y2
    if np.any(mask3):
        out[mask3] = eps * np.exp((target[mask3] - c1 * cmax) * neg_log_eps / c2)

    # Region 2: y1 < target < y2  →  bisect on [eps, cmax] (width ~2.5, 40 steps → ~2e-12)
    mask2 = ~mask1 & ~mask3
    if np.any(mask2):
        t2 = target[mask2]
        lo = np.full_like(t2, eps)
        hi = np.full_like(t2, cmax)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            f_mid = c1 * mid + c2 * (np.log(mid) - log_eps) / neg_log_eps
            go_right = f_mid < t2
            lo = np.where(go_right, mid, lo)
            hi = np.where(go_right, hi, mid)
        out[mask2] = 0.5 * (lo + hi)

    if clip_min is not None:
        out = np.maximum(out, float(clip_min))
    if clip_max is not None:
        out = np.minimum(out, float(clip_max))
    return out.astype(np.float32)


def _denormalize_array(
    arr: np.ndarray,
    var_name: str,
    mean_values: Dict[str, np.ndarray],
    std_values: Dict[str, np.ndarray],
    concentration_params: Dict[str, dict],
) -> np.ndarray:
    """Inverse z-score + inverse concentration transform."""
    mean = mean_values[var_name]
    std = std_values[var_name]

    # Broadcast stats to match arr shape
    def _bcast(stats, values):
        stats = np.asarray(stats)
        if stats.ndim != 1:
            return stats
        n = stats.shape[0]
        for ax, sz in enumerate(values.shape):
            if sz == n:
                shape = [1] * values.ndim
                shape[ax] = n
                return stats.reshape(shape)
        return stats

    mean = _bcast(mean, arr)
    std = _bcast(std, arr)

    x = arr * std + mean  # inverse z-score (in concentration space if applicable)
    if var_name in CONCENTRATION_VARS:
        params = concentration_params.get(var_name, _DEFAULT_CONCENTRATION_PARAMS)
        x = _inverse_concentration_numpy(x, params)
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Single-step assimilation
# ---------------------------------------------------------------------------

def assimilate_one_timestep(
    model: torch.nn.Module,
    t0_data: Dict[str, np.ndarray],
    mean_values: Dict[str, np.ndarray],
    std_values: Dict[str, np.ndarray],
    concentration_params: Dict[str, dict],
    dummy_dataset: WoFSMAEDataset,
    device: torch.device,
    target_size: int = 304,
    blend_alpha: float = 1.0,
    precip_mask_ratio: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Run MAE DA for one timestep; return denormalized precip arrays.

    Args:
        t0_data    : {var_name: np.ndarray (level, H, W) or (H, W)}
        Returns    : {var_name: (level, H, W)} denormalized precip analysis
    """

    def load_group(var_list, ndim):
        parts = []
        for var in var_list:
            if var not in t0_data:
                raise KeyError(f"Missing variable '{var}' in t0_data for rollout")
            raw = t0_data[var]
            normed = dummy_dataset._normalize_array(raw, var)
            normed_padded = dummy_dataset._pad_to_size(normed, target_size)
            parts.append(normed_padded.astype(np.float32))
        return np.concatenate(parts, axis=0)  # (C, H, W) for 3D or (C, H, W) for 2D

    H_orig = next(iter(t0_data.values())).shape[-2]
    W_orig = next(iter(t0_data.values())).shape[-1]

    bg_arr = load_group(dummy_dataset.background_vars, ndim=3)
    refl_arr = load_group(dummy_dataset.reflectivity_vars, ndim=3)
    surf_arr_parts = []
    for var in dummy_dataset.surface_vars:
        if var in t0_data and var in dummy_dataset._mean_values:
            raw = t0_data[var]
            normed = dummy_dataset._normalize_array(raw, var)
            normed_padded = dummy_dataset._pad_to_size(normed[np.newaxis], target_size)
            surf_arr_parts.append(normed_padded)
    if surf_arr_parts:
        surf_arr = np.concatenate(surf_arr_parts, axis=0)
    else:
        surf_arr = np.zeros((len(dummy_dataset.surface_vars), target_size, target_size), dtype=np.float32)

    # Build forcing (skip forcing normalization — they are already in [-1,1] range)
    forcing_arr = np.zeros((len(dummy_dataset.forcing_vars), target_size, target_size), dtype=np.float32)

    def to_tensor(arr):
        return torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        precip_recon = model.assimilate(
            background=to_tensor(bg_arr),
            obs_refl=to_tensor(refl_arr),
            surface=to_tensor(surf_arr),
            forcing=to_tensor(forcing_arr),
            blend_alpha=blend_alpha,
            precip_mask_ratio=precip_mask_ratio,
        )  # (1, 136, H_pad, W_pad)

    precip_norm = precip_recon[0].cpu().numpy()  # (136, H_pad, W_pad)
    precip_norm = precip_norm[:, :H_orig, :W_orig]

    # Denormalize each precip variable
    levels = precip_norm.shape[0] // len(dummy_dataset.precip_vars)
    out: Dict[str, np.ndarray] = {}
    out_norm: Dict[str, np.ndarray] = {}  # raw network output in normalized space
    offset = 0
    for var in dummy_dataset.precip_vars:
        n_lev = levels
        slc = precip_norm[offset: offset + n_lev, ...]
        out_norm[var] = slc.copy()  # save normalized slice before inversion
        out[var] = _denormalize_array(slc, var, mean_values, std_values, concentration_params)
        offset += n_lev
    return out, out_norm


# ---------------------------------------------------------------------------
# Main rollout loop
# ---------------------------------------------------------------------------

def _load_stats_and_params(conf: dict):
    data_conf = conf["data"]
    mean_ds = xr.open_dataset(data_conf["mean_path"]).load()
    std_ds = xr.open_dataset(data_conf["std_path"]).load()

    all_vars = (
        data_conf.get("background_vars", DEFAULT_BACKGROUND_VARS)
        + data_conf.get("precip_vars", DEFAULT_PRECIP_VARS)
        + data_conf.get("reflectivity_vars", DEFAULT_REFLECTIVITY_VARS)
        + data_conf.get("surface_vars", DEFAULT_SURFACE_VARS)
    )

    mean_values: Dict[str, np.ndarray] = {}
    std_values: Dict[str, np.ndarray] = {}
    for var in all_vars:
        if var in mean_ds and var in std_ds:
            mv = mean_ds[var].values
            sv = std_ds[var].values
            if var in CONCENTRATION_VARS:
                mv = np.where(np.isnan(mv), 0.0, mv)
                sv = np.where(np.isnan(sv), 1.0, sv)
            sv = np.where(sv <= 0.0, 1.0, sv)
            mean_values[var] = mv
            std_values[var] = sv

    # Load concentration params
    concentration_params: Dict[str, dict] = {v: dict(_DEFAULT_CONCENTRATION_PARAMS) for v in CONCENTRATION_VARS}
    json_path = data_conf.get("concentration_params_json")
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            params_dict = payload.get("variables", payload) if isinstance(payload, dict) else {}
            for var, info in params_dict.items():
                if var not in CONCENTRATION_VARS or not isinstance(info, dict):
                    continue
                if "recommended" in info and isinstance(info["recommended"], dict):
                    concentration_params[var] = _coerce_concentration_params(info["recommended"])
                else:
                    concentration_params[var] = _coerce_concentration_params(info)
        except Exception as exc:
            logger.warning("Failed to load concentration params JSON: %s", exc)

    return mean_values, std_values, concentration_params


def _extract_case_date(file_path: str) -> Optional[str]:
    match = re.search(r"wofs_(\d{8})_\d{4}_mem\d+\.zarr(?:\.zip)?$", Path(file_path).name)
    return match.group(1) if match else None


def run_rollout(args, conf: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running DA rollout on device: %s", device)

    # Load model
    conf_model = conf.copy()
    m = load_model(conf_model)
    checkpoint_path = args.checkpoint
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state, strict=False)
    m.to(device)
    m.eval()
    logger.info("Loaded checkpoint from %s", checkpoint_path)

    # Load stats
    mean_values, std_values, concentration_params = _load_stats_and_params(conf)

    # Build a dummy dataset instance for normalization methods (no file loading)
    dummy_ds = WoFSMAEDataset(filenames=[], conf=conf, seed=0)
    dummy_ds._mean_values = mean_values
    dummy_ds._std_values = std_values
    dummy_ds._concentration_params = concentration_params
    dummy_ds._opened = True

    target_size = int(conf["data"].get("mae_pad_to", 304))
    blend_alpha = float(conf.get("rollout", {}).get("blend_alpha", 1.0))
    precip_mask_ratio = float(conf.get("rollout", {}).get("precip_mask_ratio", 1.0))
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Enumerate files
    file_pattern = conf["data"]["save_loc"]
    all_files = sorted(glob(file_pattern))
    if args.start_date and args.end_date:
        all_files = [
            fp for fp in all_files
            if (cd := _extract_case_date(fp)) is not None and args.start_date <= cd <= args.end_date
        ]

    if not all_files:
        logger.error("No files found matching %s for the given date range", file_pattern)
        return

    logger.info("Processing %d case files", len(all_files))

    for case_file in all_files:
        case_stem = Path(case_file).stem.replace(".zarr", "")
        case_date = _extract_case_date(case_file) or "unknown"
        case_out_dir = os.path.join(out_dir, case_date)
        os.makedirs(case_out_dir, exist_ok=True)
        out_path = os.path.join(case_out_dir, f"{case_stem}_analysis.zarr.zip")

        if os.path.exists(out_path):
            logger.info("Skipping %s (already exists)", out_path)
            continue

        try:
            logger.info("Processing %s", case_file)
            all_vars = list(set(
                dummy_ds.background_vars
                + dummy_ds.precip_vars
                + dummy_ds.reflectivity_vars
                + dummy_ds.surface_vars
            ))
            ds = get_forward_data(case_file)
            ds = filter_ds(ds, all_vars)
            n_time = ds.sizes.get("time", 0)

            recon_by_var: Dict[str, List[np.ndarray]] = {v: [] for v in dummy_ds.precip_vars}
            recon_by_var_norm: Dict[str, List[np.ndarray]] = {v: [] for v in dummy_ds.precip_vars}

            for t_idx in range(n_time):
                t0_data: Dict[str, np.ndarray] = {}
                chunk = ds.isel(time=t_idx).load()
                for var in all_vars:
                    if var in chunk:
                        t0_data[var] = chunk[var].values

                result, result_norm = assimilate_one_timestep(
                    model=m,
                    t0_data=t0_data,
                    mean_values=mean_values,
                    std_values=std_values,
                    concentration_params=concentration_params,
                    dummy_dataset=dummy_ds,
                    device=device,
                    target_size=target_size,
                    blend_alpha=blend_alpha,
                    precip_mask_ratio=precip_mask_ratio,
                )
                for var, arr in result.items():
                    recon_by_var[var].append(arr)
                for var, arr in result_norm.items():
                    recon_by_var_norm[var].append(arr)

            # Stack along time and write zarr
            out_store = zarr.storage.ZipStore(out_path, mode="w")
            out_root = zarr.open_group(store=out_store, mode="w")
            time_vals = ds["time"].values if "time" in ds.coords else np.arange(n_time)
            out_root.create_array("time", data=time_vals)

            # Denormalized (physical units)
            for var, frames in recon_by_var.items():
                stacked = np.stack(frames, axis=0).astype(np.float32)  # (T, level, H, W)
                out_root.create_array(var, data=stacked, chunks=(1,) + stacked.shape[1:])

            # Raw network output in normalized space (debugging: is jaggedness pre- or post-inversion?)
            norm_grp = out_root.require_group("norm_output")
            for var, frames in recon_by_var_norm.items():
                stacked = np.stack(frames, axis=0).astype(np.float32)  # (T, level, H, W)
                norm_grp.create_array(var, data=stacked, chunks=(1,) + stacked.shape[1:])

            out_store.close()
            logger.info("Wrote analysis to %s", out_path)

        except Exception as exc:
            logger.error("Failed to process %s: %s", case_file, exc, exc_info=True)

    logger.info("Rollout complete.")


def primary_main():
    parser = argparse.ArgumentParser(
        description="WoFS MAE DA rollout: assimilate REFL_10CM → precip analysis"
    )
    parser.add_argument("-c", "--config", dest="model_config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained MAE checkpoint (.pt)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="YYYYMMDD start date filter (inclusive)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="YYYYMMDD end date filter (inclusive)")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for analysis zarr files")
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(ch)

    with open(args.model_config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    run_rollout(args, conf)


if __name__ == "__main__":
    primary_main()
