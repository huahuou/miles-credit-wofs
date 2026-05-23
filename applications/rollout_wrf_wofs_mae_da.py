"""
WoFS DiffMAE Data Assimilation Rollout Script
---------------------------------------------
Loads a trained WoFSDiffMAE checkpoint, processes one or more WoFS zarr cases,
samples masked precip analyses, and writes them back to disk.

Workflow per case file:
    1. Load background, reflectivity, surface, forcing from zarr
    2. Normalize using mean.nc / std.nc + concentration transform
    3. Build a precip mask from the eval/rollout config
    4. Run model.sample_precip(...) with DDIM, DDPM, or RePaint sampling
    5. Inverse-normalize the precip output
    6. Write QRAIN/QNRAIN/... analysis arrays to output zarr

Usage:
    python applications/rollout_wrf_wofs_mae_da.py \\
        -c config/wofs_mae_da.yml \\
        --checkpoint /path/to/checkpoint.pt \\
        --start-date 20230501 --end-date 20230531 \\
        --out-dir /work2/zhanxianghua/wofs_mae_analysis

Output layout:
    <out-dir>/<YYYYMMDD>/<case_stem>_analysis.zarr
Open groups within each zarr:
    xr.open_zarr(path)
    xr.open_zarr(path, group="norm_output")
    xr.open_zarr(path, group="denoise_trajectory")

"""

import argparse
import copy
import logging
import os
import re
import shutil
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import xarray as xr
import yaml

from credit.datasets.wrf_wofs_mae import (
    WoFSMAEDataset,
    DEFAULT_BACKGROUND_VARS,
    DEFAULT_PRECIP_VARS,
    DEFAULT_REFLECTIVITY_VARS,
    DEFAULT_SURFACE_VARS,
)
from credit.data import filter_ds, get_forward_data
from credit.distributed import distributed_model_wrapper, get_rank_info, setup
from credit.models import load_model
from credit.models.checkpoint import load_model_state, load_state_dict_error_handler
from credit.transforms.concentration import (
    CONCENTRATION_VARS,
    build_log_zscore_stats_override,
    concentration_transform_overrides_stats,
    inverse_concentration_transform_numpy,
    load_concentration_transform_json,
)

logger = logging.getLogger(__name__)


def _broadcast_stats(stats: np.ndarray, values: np.ndarray) -> np.ndarray:
    stats_arr = np.asarray(stats)
    if stats_arr.ndim != 1:
        return stats_arr
    stat_len = stats_arr.shape[0]
    matching = [axis for axis, size in enumerate(values.shape) if size == stat_len]
    if not matching:
        return stats_arr
    axis = 1 if (len(matching) > 1 and 1 in matching) else matching[0]
    shape = [1] * values.ndim
    shape[axis] = stat_len
    return stats_arr.reshape(shape)


def _find_level_axis(values: np.ndarray, stats: np.ndarray) -> int | None:
    stats_arr = np.asarray(stats)
    if stats_arr.ndim != 1:
        return None
    matching_axes = [axis for axis, size in enumerate(values.shape) if size == stats_arr.shape[0]]
    if not matching_axes:
        return None
    return 1 if (len(matching_axes) > 1 and 1 in matching_axes) else matching_axes[0]


def _denormalize_array(
    arr: np.ndarray,
    var_name: str,
    mean_values: Dict[str, np.ndarray],
    std_values: Dict[str, np.ndarray],
    concentration_specs: Dict[str, dict],
    normalization_mode: str = "zscore",
) -> np.ndarray:
    """Inverse z-score + inverse concentration transform."""
    mode = str(normalization_mode).strip().lower()
    if var_name in CONCENTRATION_VARS and mode == "transform_only":
        x = arr
    elif var_name in CONCENTRATION_VARS and mode == "scale_only":
        x = arr * _broadcast_stats(std_values[var_name], arr)
    else:
        mean = _broadcast_stats(mean_values[var_name], arr)
        std = _broadcast_stats(std_values[var_name], arr)
        x = arr * std + mean

    if var_name in CONCENTRATION_VARS:
        spec = concentration_specs.get(var_name)
        if spec is None:
            raise KeyError(f"Missing concentration transform spec for {var_name}")
        x = inverse_concentration_transform_numpy(
            x,
            spec,
            level_axis=_find_level_axis(np.asarray(x), np.asarray(std_values[var_name])),
        )
    return x.astype(np.float32)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    seen = set()
    while True:
        obj_id = id(model)
        if obj_id in seen:
            return model
        seen.add(obj_id)
        if hasattr(model, "sample_precip"):
            return model
        if hasattr(model, "module"):
            model = model.module
            continue
        if hasattr(model, "unwrap"):
            model = model.unwrap()
            continue
        return model


def _eval_conf(conf: dict) -> dict:
    eval_conf = dict(conf.get("eval", {}))
    rollout_conf = conf.get("rollout", {})
    for key, value in rollout_conf.items():
        eval_conf.setdefault(key, value)
    return eval_conf


def _distributed_conf(conf: dict) -> dict:
    eval_conf = _eval_conf(conf)
    rollout_conf = copy.deepcopy(conf)
    rollout_conf.setdefault("trainer", {})
    rollout_conf["trainer"]["mode"] = eval_conf.get("mode", rollout_conf["trainer"].get("mode", "none"))
    rollout_conf["trainer"].setdefault("activation_checkpoint", False)
    rollout_conf["trainer"].setdefault("ddp_find_unused_parameters", False)
    return rollout_conf


def _load_diffmae_model(conf: dict, checkpoint_path: Optional[str], device: torch.device) -> torch.nn.Module:
    eval_conf = _eval_conf(conf)
    mode = str(eval_conf.get("mode", "none")).strip().lower()
    rollout_conf = _distributed_conf(conf)

    if mode == "none":
        model = load_model(conf).to(device)
        ckpt = checkpoint_path or os.path.join(os.path.expandvars(conf["save_loc"]), "checkpoint.pt")
        state = torch.load(ckpt, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        load_msg = model.load_state_dict(state, strict=False)
        load_state_dict_error_handler(load_msg)
        model.eval()
        logger.info("Loaded checkpoint from %s", ckpt)
        return model

    if mode == "ddp":
        model = load_model(conf).to(device)
        model = distributed_model_wrapper(rollout_conf, model, device)
        ckpt = checkpoint_path or os.path.join(os.path.expandvars(conf["save_loc"]), "checkpoint.pt")
        state = torch.load(ckpt, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        load_msg = model.module.load_state_dict(state, strict=False)
        load_state_dict_error_handler(load_msg)
        model.eval()
        logger.info("Loaded DDP checkpoint from %s", ckpt)
        return model

    if mode == "fsdp":
        model = load_model(conf).to(device)
        model = distributed_model_wrapper(rollout_conf, model, device)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                load_msg = _unwrap_model(model).load_state_dict(state["model_state_dict"], strict=False)
                load_state_dict_error_handler(load_msg)
            else:
                from credit.models.checkpoint import TorchFSDPCheckpointIO

                TorchFSDPCheckpointIO().load_unsharded_model(model, checkpoint_path)
        else:
            model = load_model_state(rollout_conf, model, device)
        model.eval()
        logger.info("Loaded FSDP checkpoint%s", f" from {checkpoint_path}" if checkpoint_path else "")
        return model

    raise ValueError(f"Unsupported eval.mode: {mode!r}")


def _sample_mask(model: torch.nn.Module, eval_conf: dict, batch_size: int, device: torch.device) -> torch.Tensor:
    core = _unwrap_model(model)
    mask_ratio = eval_conf.get("precip_mask_ratio", eval_conf.get("mask_ratio", 1.0))
    mask_mode = str(eval_conf.get("precip_mask_mode", eval_conf.get("mask_mode", "spatial_patch"))).strip().lower()
    if mask_mode in {"mixed_height", "mixed_with_height", "spatial_channel_height"}:
        probs = torch.tensor(
            [
                float(eval_conf.get("mixed_height_spatial_probability", 1.0)),
                float(eval_conf.get("mixed_height_channel_probability", 0.0)),
                float(eval_conf.get("mixed_height_height_probability", 1.0)),
            ],
            device=device,
            dtype=torch.float32,
        )
        if torch.any(probs < 0):
            raise ValueError("Mixed height mask probabilities must be non-negative")
        total = probs.sum()
        if total <= 0:
            raise ValueError("At least one mixed height mask probability must be positive")
        mask_mode = ["spatial_patch", "channel_patch", "height_patch"][int(torch.multinomial(probs / total, 1).item())]
    if mask_mode in {"height_patch", "height", "level_patch", "vertical_patch"}:
        return core.random_height_precip_mask(
            batch_size,
            mask_ratio,
            device,
            masked_levels=eval_conf.get("height_mask_levels"),
            visible_levels=eval_conf.get("height_visible_levels"),
        )
    if mask_mode in {"channel_patch", "random_channel", "channel", "group_patch", "variable_patch", "grouped"}:
        return core.random_channel_precip_mask(batch_size, mask_ratio, device)
    if mask_mode in {"spatial_patch", "spatial", "patch"}:
        return core.random_precip_mask(batch_size, mask_ratio, device)
    raise ValueError(f"Unsupported eval precip_mask_mode: {mask_mode!r}")


def _global_channel_to_var_level(dummy_dataset: WoFSMAEDataset, channel_idx: int) -> tuple[str, int]:
    offset = 0
    for var in dummy_dataset.precip_vars:
        n_ch = dummy_dataset._mean_values[var].shape[0] if np.asarray(dummy_dataset._mean_values[var]).ndim > 0 else 1
        if offset <= channel_idx < offset + n_ch:
            return var, channel_idx - offset
        offset += n_ch
    raise IndexError(f"Precip channel {channel_idx} is outside configured precip channels")


# ---------------------------------------------------------------------------
# Single-step assimilation
# ---------------------------------------------------------------------------

def _build_condition_dict(
    batch: dict,
    background_vars: List[str],
    reflectivity_vars: List[str],
    surface_vars: List[str],
    forcing_vars: List[str],
    target_size: int,
) -> Dict[str, torch.Tensor]:
    def _stack_group(var_list: List[str]) -> torch.Tensor:
        parts = [batch[var] for var in var_list if var in batch]
        if not parts:
            return torch.zeros((batch["precip"].shape[0], 0, target_size, target_size), device=batch["precip"].device)
        return torch.cat(parts, dim=1)

    return {
        "background": _stack_group(background_vars),
        "reflectivity": _stack_group(reflectivity_vars),
        "surface": _stack_group(surface_vars),
        "forcing": batch.get(
            "forcing",
            torch.zeros((batch["precip"].shape[0], 0, target_size, target_size), device=batch["precip"].device),
        ),
    }


def _prepare_batch_from_case(
    chunk,
    dummy_dataset: WoFSMAEDataset,
    target_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    snap_t = 1 if dummy_dataset.mae_include_t1_refl else 0
    batch = {}
    first_var = dummy_dataset.background_vars[0] if dummy_dataset.background_vars else dummy_dataset.precip_vars[0]
    orig_h, orig_w = chunk[first_var].isel(time=snap_t).values.shape[-2:]
    for var in dummy_dataset.background_vars:
        raw = chunk[var].isel(time=snap_t).values
        batch[var] = torch.from_numpy(dummy_dataset._pad_to_size(dummy_dataset._normalize_array(raw, var), target_size)).unsqueeze(0).to(device)
    for var in dummy_dataset.precip_vars:
        raw = chunk[var].isel(time=snap_t).values
        batch[var] = torch.from_numpy(dummy_dataset._pad_to_size(dummy_dataset._normalize_array(raw, var), target_size)).unsqueeze(0).to(device)
    for var in dummy_dataset.reflectivity_vars:
        raw = chunk[var].isel(time=snap_t).values
        batch[var] = torch.from_numpy(dummy_dataset._pad_to_size(dummy_dataset._normalize_array(raw, var), target_size)).unsqueeze(0).to(device)
    for var in dummy_dataset.surface_vars:
        if var in chunk:
            raw = chunk[var].isel(time=0).values
            norm = dummy_dataset._normalize_array(raw, var)
            batch[var] = torch.from_numpy(dummy_dataset._pad_to_size(norm[np.newaxis], target_size)).unsqueeze(0).to(device)
    forcing_arr = dummy_dataset._compute_forcing(chunk.time.values[0], chunk, target_size)
    if forcing_arr.shape[0] > 0:
        batch["forcing"] = torch.from_numpy(forcing_arr).unsqueeze(0).to(device)
    else:
        batch["forcing"] = torch.zeros((1, 0, target_size, target_size), device=device)
    batch["precip"] = torch.cat([batch[var] for var in dummy_dataset.precip_vars], dim=1)
    batch["_orig_hw"] = (int(orig_h), int(orig_w))
    return batch


def rollout_one_timestep(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    mean_values: Dict[str, np.ndarray],
    std_values: Dict[str, np.ndarray],
    concentration_specs: Dict[str, dict],
    dummy_dataset: WoFSMAEDataset,
    eval_conf: dict,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    core = _unwrap_model(model)
    target = batch["precip"]
    precip_mask = _sample_mask(model, eval_conf, target.shape[0], device)
    use_visible_precip = bool(eval_conf.get("visible_precip_conditioning", eval_conf.get("clamp_visible_precip", True)))
    precip_visible = target if use_visible_precip else None
    sampler = str(eval_conf.get("sampler", "ddim")).strip().lower()
    sampling_timesteps = int(eval_conf.get("sampling_timesteps", getattr(core, "sampling_timesteps", 50)))
    eta = eval_conf.get("ddim_sampling_eta", None)
    repaint_jump_length = int(eval_conf.get("repaint_jump_length", 10))
    repaint_jump_n_sample = int(eval_conf.get("repaint_jump_n_sample", 10))
    save_trajectory = bool(eval_conf.get("save_denoise_trajectory", False))
    inpaint_mode = eval_conf.get("inpaint_mode", None)

    cond = _build_condition_dict(
        batch,
        dummy_dataset.background_vars,
        dummy_dataset.reflectivity_vars,
        dummy_dataset.surface_vars,
        dummy_dataset.forcing_vars,
        target.shape[-1],
    )

    with torch.no_grad():
        pred_norm = core.sample_precip(
            cond,
            precip_mask,
            precip_visible=precip_visible,
            sampling_timesteps=sampling_timesteps,
            eta=eta,
            sampler=sampler,
            repaint_jump_length=repaint_jump_length,
            repaint_jump_n_sample=repaint_jump_n_sample,
            inpaint_mode=inpaint_mode,
            return_all_timesteps=save_trajectory,
        )

    trajectory = None
    if save_trajectory:
        all_steps = pred_norm[0].detach().cpu().numpy()
        pred_norm = pred_norm[:, -1]
        channels = eval_conf.get("trajectory_channels", [int(eval_conf.get("channel", 0))])
        if isinstance(channels, int):
            channels = [channels]
        channels = [max(0, min(int(channel), all_steps.shape[1] - 1)) for channel in channels]
        stride = max(1, int(eval_conf.get("trajectory_stride", 1)))
        all_steps = all_steps[::stride, channels]
        trajectory = {
            "values": all_steps.astype(np.float32, copy=False),
            "channels": np.asarray(channels, dtype=np.int32),
        }

    pred_norm = pred_norm[0].detach().cpu().numpy()
    orig_h, orig_w = batch.get("_orig_hw", pred_norm.shape[-2:])
    pred_norm = pred_norm[:, :orig_h, :orig_w]
    out: Dict[str, np.ndarray] = {}
    out_norm: Dict[str, np.ndarray] = {}
    offset = 0
    for var in dummy_dataset.precip_vars:
        n_ch = dummy_dataset._mean_values[var].shape[0] if np.asarray(dummy_dataset._mean_values[var]).ndim > 0 else 1
        slc = pred_norm[offset : offset + n_ch, ...]
        out_norm[var] = slc.copy()
        out[var] = _denormalize_array(
            slc,
            var,
            mean_values,
            std_values,
            concentration_specs,
            normalization_mode=dummy_dataset._concentration_normalization_mode,
        )
        offset += n_ch
    return out, out_norm, trajectory


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

    _, concentration_specs = load_concentration_transform_json(
        data_conf.get("log_transform_params_json"),
        variables=CONCENTRATION_VARS,
    )
    for var, spec in concentration_specs.items():
        if not concentration_transform_overrides_stats(spec) or var not in mean_values:
            continue
        n_lev = int(np.asarray(mean_values[var]).shape[0]) if np.asarray(mean_values[var]).ndim >= 1 else 1
        mean_override, std_override = build_log_zscore_stats_override(spec, n_lev)
        mean_values[var] = mean_override
        std_values[var] = std_override

    return mean_values, std_values, concentration_specs


def _extract_case_date(file_path: str) -> Optional[str]:
    match = re.search(r"wofs_(\d{8})_\d{4}_mem\d+\.zarr(?:\.zip)?$", Path(file_path).name)
    return match.group(1) if match else None


def _forcing_source_vars(dummy_ds: WoFSMAEDataset) -> list[str]:
    computed = {
        "cos_lat", "sin_lat", "cos_lon", "sin_lon",
        "cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude",
        "cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time",
        "cos_solar_zenith", "cos_solar_zenith_angle", "insolation",
    }
    return [var for var in dummy_ds.forcing_vars if var not in computed]


def run_rollout(args, conf: dict):
    eval_conf = _eval_conf(conf)
    mode = str(eval_conf.get("mode", "none")).strip().lower()
    local_rank, world_rank, world_size = get_rank_info(mode)
    distributed = mode in {"ddp", "fsdp"}
    backend = str(eval_conf.get("backend", args.backend))
    if distributed:
        setup(world_rank, world_size, mode, backend)

    try:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        logger.info("Running DiffMAE rollout on rank %d/%d device=%s", world_rank, world_size, device)

        m = _load_diffmae_model(conf, args.checkpoint, device)
        mean_values, std_values, concentration_specs = _load_stats_and_params(conf)

        dummy_ds = WoFSMAEDataset(filenames=[], conf=conf, seed=0)
        dummy_ds._mean_values = mean_values
        dummy_ds._std_values = std_values
        dummy_ds._concentration_transform_specs = concentration_specs
        dummy_ds._opened = True

        target_size = int(conf["data"].get("mae_pad_to", 304))
        out_dir = args.out_dir
        if world_rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        file_pattern = conf["data"]["save_loc"]
        all_files = sorted(glob(file_pattern))
        if args.start_date and args.end_date:
            all_files = [
                fp for fp in all_files
                if (cd := _extract_case_date(fp)) is not None and args.start_date <= cd <= args.end_date
            ]
        if args.max_files is not None:
            all_files = all_files[: int(args.max_files)]

        if not all_files:
            logger.error("No files found matching %s for the given date range", file_pattern)
            return

        rank_files = all_files[world_rank::world_size]
        logger.info("Processing %d/%d case files on rank %d", len(rank_files), len(all_files), world_rank)

        for case_file in rank_files:
            case_stem = Path(case_file).stem.replace(".zarr", "")
            case_date = _extract_case_date(case_file) or "unknown"
            case_out_dir = os.path.join(out_dir, case_date)
            os.makedirs(case_out_dir, exist_ok=True)
            out_path = os.path.join(case_out_dir, f"{case_stem}_analysis.zarr")

            if os.path.exists(out_path) and not bool(eval_conf.get("overwrite", False)):
                logger.info("Skipping %s (already exists)", out_path)
                continue

            try:
                logger.info("Processing %s", case_file)
                all_vars = list(set(
                    dummy_ds.background_vars
                    + dummy_ds.precip_vars
                    + dummy_ds.reflectivity_vars
                    + dummy_ds.surface_vars
                    + _forcing_source_vars(dummy_ds)
                ))
                ds = get_forward_data(case_file)
                ds = filter_ds(ds, all_vars)
                n_time = ds.sizes.get("time", 0)
                if dummy_ds.mae_include_t1_refl:
                    n_time = max(0, n_time - 1)
                if args.max_times is not None:
                    n_time = min(n_time, int(args.max_times))

                recon_by_var: Dict[str, List[np.ndarray]] = {v: [] for v in dummy_ds.precip_vars}
                recon_by_var_norm: Dict[str, List[np.ndarray]] = {v: [] for v in dummy_ds.precip_vars}
                trajectories: List[np.ndarray] = []
                trajectory_channels = None

                for t_idx in range(n_time):
                    stop = t_idx + 2 if dummy_ds.mae_include_t1_refl else t_idx + 1
                    chunk = ds.isel(time=slice(t_idx, stop)).load()
                    try:
                        batch = _prepare_batch_from_case(chunk, dummy_ds, target_size, device)
                        result, result_norm, trajectory = rollout_one_timestep(
                            model=m,
                            batch=batch,
                            mean_values=mean_values,
                            std_values=std_values,
                            concentration_specs=concentration_specs,
                            dummy_dataset=dummy_ds,
                            eval_conf=eval_conf,
                            device=device,
                        )
                    finally:
                        try:
                            chunk.close()
                        except Exception:
                            pass

                    for var, arr in result.items():
                        recon_by_var[var].append(arr)
                    for var, arr in result_norm.items():
                        recon_by_var_norm[var].append(arr)
                    if trajectory is not None:
                        trajectories.append(trajectory["values"])
                        trajectory_channels = trajectory["channels"]

                time_vals = ds["time"].values[:n_time] if "time" in ds.coords else np.arange(n_time)
                data_vars = {}
                for var, frames in recon_by_var.items():
                    stacked = np.stack(frames, axis=0).astype(np.float32)
                    data_vars[var] = (("time", "level", "y", "x"), stacked)
                if not data_vars:
                    raise RuntimeError(f"No rollout frames were produced for {case_file}")
                first_data = next(iter(data_vars.values()))[1]

                analysis_ds = xr.Dataset(
                    data_vars=data_vars,
                    coords={
                        "time": time_vals,
                        "level": np.arange(first_data.shape[1], dtype=np.int32),
                        "y": np.arange(first_data.shape[2], dtype=np.int32),
                        "x": np.arange(first_data.shape[3], dtype=np.int32),
                    },
                    attrs={
                        "units": "physical_units_after_inverse_normalization",
                        "sampler": str(eval_conf.get("sampler", "ddim")),
                        "sampling_timesteps": int(eval_conf.get("sampling_timesteps", 50)),
                        "ddim_sampling_eta": float(eval_conf.get("ddim_sampling_eta", 0.0)),
                        "repaint_jump_length": int(eval_conf.get("repaint_jump_length", 10)),
                        "repaint_jump_n_sample": int(eval_conf.get("repaint_jump_n_sample", 10)),
                        "precip_mask_mode": str(eval_conf.get("precip_mask_mode", "spatial_patch")),
                        "precip_mask_ratio": str(eval_conf.get("precip_mask_ratio", 1.0)),
                    },
                )
                if os.path.exists(out_path):
                    shutil.rmtree(out_path)
                analysis_ds.to_zarr(out_path, mode="w")

                if bool(eval_conf.get("save_norm_output", True)):
                    norm_vars = {}
                    for var, frames in recon_by_var_norm.items():
                        stacked = np.stack(frames, axis=0).astype(np.float32)
                        norm_vars[var] = (("time", "level", "y", "x"), stacked)
                    xr.Dataset(
                        data_vars=norm_vars,
                        coords=analysis_ds.coords,
                        attrs={"units": "normalized_model_output"},
                    ).to_zarr(out_path, mode="a", group="norm_output")

                if trajectories:
                    traj_arr = np.stack(trajectories, axis=0).astype(np.float32)
                    channel_vars = []
                    channel_levels = []
                    for channel_idx in trajectory_channels:
                        var_name, level_idx = _global_channel_to_var_level(dummy_ds, int(channel_idx))
                        channel_vars.append(var_name)
                        channel_levels.append(level_idx)
                    xr.Dataset(
                        data_vars={
                            "precip": (("time", "denoise_step", "channel", "y_padded", "x_padded"), traj_arr)
                        },
                        coords={
                            "time": time_vals[: traj_arr.shape[0]],
                            "denoise_step": np.arange(traj_arr.shape[1], dtype=np.int32),
                            "channel": trajectory_channels,
                            "channel_level": ("channel", np.asarray(channel_levels, dtype=np.int32)),
                            "y_padded": np.arange(traj_arr.shape[-2], dtype=np.int32),
                            "x_padded": np.arange(traj_arr.shape[-1], dtype=np.int32),
                        },
                        attrs={
                            "units": "normalized_model_output",
                            "description": "Selected channels over the reverse diffusion sampling trajectory.",
                            "channel_vars": ",".join(channel_vars),
                        },
                    ).to_zarr(out_path, mode="a", group="denoise_trajectory")

                logger.info("Wrote analysis to %s", out_path)

            except Exception as exc:
                logger.error("Failed to process %s: %s", case_file, exc, exc_info=True)

        logger.info("Rollout complete on rank %d.", world_rank)
    finally:
        if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def primary_main():
    parser = argparse.ArgumentParser(
        description="WoFS DiffMAE DA rollout: sample masked precip analysis conditioned on WoFS context"
    )
    parser.add_argument("-c", "--config", dest="model_config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained DiffMAE checkpoint (.pt). Defaults to save_loc/checkpoint.pt.")
    parser.add_argument("--start-date", type=str, default=None,
                        help="YYYYMMDD start date filter (inclusive)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="YYYYMMDD end date filter (inclusive)")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for analysis zarr files")
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo", "mpi"],
                        help="Distributed backend for eval.mode=ddp/fsdp")
    parser.add_argument("--mode", type=str, default=None, choices=["none", "ddp", "fsdp"],
                        help="Override eval.mode from config for rollout.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Optional smoke-test limit on number of case files")
    parser.add_argument("--max-times", type=int, default=None,
                        help="Optional smoke-test limit on timesteps per case")
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(ch)

    with open(args.model_config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    if args.mode is not None:
        conf.setdefault("eval", {})
        conf["eval"]["mode"] = args.mode

    run_rollout(args, conf)


if __name__ == "__main__":
    primary_main()
