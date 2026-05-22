"""
WoFS DiffMAE rollout with persistent masks and per-case metrics.

Adds the following on top of rollout_wrf_wofs_mae_da.py:
    1. Metrics JSON with normalized-space MSE/MAE/SSIM and physical-space MSE/MAE
    2. Config YAML copied into the output directory
    3. Deterministic custom mask support via an .npz mask bundle
    4. Mask arrays saved into each analysis zarr
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import yaml

# Support direct script execution: `python applications/rollout_wrf_wofs_mae_da_metrics.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from applications.rollout_wrf_wofs_mae_da import (
    _build_condition_dict,
    _eval_conf,
    _extract_case_date,
    _forcing_source_vars,
    _global_channel_to_var_level,
    _load_diffmae_model,
    _load_stats_and_params,
    _prepare_batch_from_case,
    _sample_mask as _runtime_sample_mask,
    _unwrap_model,
)
from credit.data import filter_ds, get_forward_data
from credit.datasets.wrf_wofs_mae import WoFSMAEDataset
from credit.distributed import get_rank_info, setup
from credit.wofs_diffmae_mask_utils import (
    build_grouped_patch_masks,
    grouped_patch_mask_time_slice,
    load_mask_bundle,
    normalize_mask_mode,
    resolve_precip_group_layout,
    save_mask_bundle,
    token_grid_shape,
)

logger = logging.getLogger(__name__)


def _resolve_eval_ensemble_size(conf: dict) -> tuple[int, int, bool, float]:
    eval_conf = _eval_conf(conf)
    requested = int(eval_conf.get("ensemble_size", eval_conf.get("ensemble", 1)))
    if requested < 1:
        raise ValueError(f"eval.ensemble_size must be >= 1, got {requested}")

    sampler = str(eval_conf.get("sampler", "ddim")).strip().lower()
    eta = eval_conf.get("ddim_sampling_eta", conf.get("model", {}).get("diffusion", {}).get("ddim_sampling_eta", 0.0))
    eta = float(0.0 if eta is None else eta)
    deterministic = sampler == "ddim" and np.isclose(eta, 0.0)
    effective = 1 if deterministic else requested
    return requested, effective, deterministic, eta


def _resolve_eval_batch_size(conf: dict, ensemble_size: int) -> int:
    eval_conf = _eval_conf(conf)
    batch_size = int(eval_conf.get("eval_batch_size", 1))
    if batch_size < 1:
        raise ValueError(f"eval.eval_batch_size must be >= 1, got {batch_size}")
    return min(batch_size, int(ensemble_size))


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)
    return kernel


def _gaussian_blur(img: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    channel = img.shape[1]
    dtype = img.dtype if img.dtype.is_floating_point else torch.float32
    x = img.to(dtype)
    kernel_1d = _gaussian_kernel_1d(window_size, sigma, x.device, dtype)
    kernel_x = kernel_1d.view(1, 1, 1, window_size).expand(channel, 1, 1, window_size)
    kernel_y = kernel_1d.view(1, 1, window_size, 1).expand(channel, 1, window_size, 1)
    padding = window_size // 2
    x = F.conv2d(x, kernel_x, padding=(0, padding), groups=channel)
    x = F.conv2d(x, kernel_y, padding=(padding, 0), groups=channel)
    return x


def _ssim_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    mu1 = _gaussian_blur(pred, window_size=window_size, sigma=sigma)
    mu2 = _gaussian_blur(target, window_size=window_size, sigma=sigma)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _gaussian_blur(pred * pred, window_size=window_size, sigma=sigma) - mu1_sq
    sigma2_sq = _gaussian_blur(target * target, window_size=window_size, sigma=sigma) - mu2_sq
    sigma12 = _gaussian_blur(pred * target, window_size=window_size, sigma=sigma) - mu1_mu2

    dr = data_range.reshape(1, -1, 1, 1).to(pred.device, pred.dtype).clamp_min(torch.finfo(pred.dtype).eps)
    c1 = (k1 * dr) ** 2
    c2 = (k2 * dr) ** 2

    cs_map = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2.0 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean(dim=(-2, -1))


def _compute_basic_error_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    diff = pred.astype(np.float32, copy=False) - target.astype(np.float32, copy=False)
    denom = float(mask.sum())
    if denom <= 0.0:
        raise ValueError("Metric requested with zero selected pixels")
    return {
        "mse": float(np.sum((diff**2) * mask) / denom),
        "mae": float(np.sum(np.abs(diff) * mask) / denom),
        "fraction": float(mask.mean()),
    }


def compute_masked_normalized_metrics(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    pixel_mask: np.ndarray,
) -> dict[str, float]:
    if pred_norm.shape != target_norm.shape:
        raise ValueError(f"pred_norm shape {pred_norm.shape} does not match target_norm shape {target_norm.shape}")
    if pixel_mask.shape != pred_norm.shape:
        raise ValueError(f"pixel_mask shape {pixel_mask.shape} does not match precip shape {pred_norm.shape}")

    mask = pixel_mask.astype(np.float32, copy=False)
    basic = _compute_basic_error_metrics(pred_norm, target_norm, mask)

    pred_t = torch.from_numpy(pred_norm.astype(np.float32, copy=False))
    target_t = torch.from_numpy(target_norm.astype(np.float32, copy=False))
    mask_t = torch.from_numpy(mask.astype(np.float32, copy=False))

    finite = torch.isfinite(pred_t) & torch.isfinite(target_t)
    mask_bool = mask_t > 0.5
    if pred_t.ndim != 3:
        raise ValueError(f"Expected precip arrays shaped (channel, y, x); got {pred_t.shape}")

    data_range = []
    channel_scores = []
    for channel_idx in range(pred_t.shape[0]):
        valid = finite[channel_idx] & mask_bool[channel_idx]
        if not torch.any(valid):
            continue
        channel_pred = pred_t[channel_idx : channel_idx + 1].clone()
        channel_target = target_t[channel_idx : channel_idx + 1].clone()
        fill_target = torch.where(valid, channel_target[0], torch.zeros_like(channel_target[0]))
        fill_pred = torch.where(valid, channel_pred[0], fill_target)
        channel_pred[0] = fill_pred
        channel_target[0] = fill_target
        ch_max = torch.max(torch.stack([fill_pred[valid], fill_target[valid]]))
        ch_min = torch.min(torch.stack([fill_pred[valid], fill_target[valid]]))
        data_range.append(torch.clamp(ch_max - ch_min, min=1.0e-6))
        score = _ssim_per_channel(
            channel_pred.unsqueeze(0),
            channel_target.unsqueeze(0),
            data_range=torch.tensor([float(data_range[-1])]),
        )
        channel_scores.append(float(score.squeeze().item()))

    ssim = float(np.mean(channel_scores)) if channel_scores else float("nan")
    return {
        "mse": basic["mse"],
        "mae": basic["mae"],
        "ssim": ssim,
        "masked_fraction": basic["fraction"],
    }


def compute_masked_physical_metrics(
    pred_phys: np.ndarray,
    target_phys: np.ndarray,
    pixel_mask: np.ndarray,
) -> dict[str, float]:
    if pred_phys.shape != target_phys.shape:
        raise ValueError(f"pred_phys shape {pred_phys.shape} does not match target_phys shape {target_phys.shape}")
    if pixel_mask.shape != pred_phys.shape:
        raise ValueError(f"pixel_mask shape {pixel_mask.shape} does not match precip shape {pred_phys.shape}")

    mask = pixel_mask.astype(np.float32, copy=False)
    basic = _compute_basic_error_metrics(pred_phys, target_phys, mask)
    return {"mse": basic["mse"], "mae": basic["mae"], "masked_fraction": basic["fraction"]}


def _grouped_mask_to_runtime_mask(
    grouped_mask: np.ndarray,
    group_channels: list[int],
    mode: str,
) -> np.ndarray:
    if grouped_mask.ndim == 4:
        return np.asarray(grouped_mask, dtype=np.float32).reshape(len(group_channels), int(group_channels[0]), -1)
    if mode == "spatial_patch":
        return np.asarray(grouped_mask[0], dtype=np.float32).reshape(-1)
    if mode == "height_patch":
        return np.asarray(grouped_mask, dtype=np.float32).reshape(len(group_channels), int(group_channels[0]), -1)

    expanded: list[np.ndarray] = []
    for group_index, n_channels in enumerate(group_channels):
        expanded.append(np.repeat(grouped_mask[group_index : group_index + 1], int(n_channels), axis=0))
    return np.concatenate(expanded, axis=0).astype(np.float32, copy=False)


def _grouped_patch_to_pixel_mask(
    grouped_mask: np.ndarray,
    group_channels: list[int],
    patch_size: int,
    height: int,
    width: int,
) -> np.ndarray:
    expanded = []
    for group_index, n_channels in enumerate(group_channels):
        group = grouped_mask[group_index]
        if group.ndim == 3:
            channel_pixels = []
            for level in range(int(n_channels)):
                pixel = np.repeat(np.repeat(group[level], patch_size, axis=0), patch_size, axis=1)
                channel_pixels.append(pixel[:height, :width])
            expanded.append(np.stack(channel_pixels, axis=0))
            continue
        pixel = np.repeat(np.repeat(group, patch_size, axis=0), patch_size, axis=1)
        pixel = pixel[:height, :width]
        expanded.append(np.repeat(pixel[None, :, :], int(n_channels), axis=0))
    return np.concatenate(expanded, axis=0).astype(np.float32, copy=False)


def _blend_mask_boundary(
    pred_norm: torch.Tensor,
    pixel_mask: torch.Tensor,
    width_px: int,
    strength: float,
    passes: int,
) -> torch.Tensor:
    """Optionally smooth the hard data-consistency seam in normalized space."""
    width_px = int(width_px)
    if width_px <= 0:
        return pred_norm
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return pred_norm
    passes = max(1, int(passes))

    mask = pixel_mask.to(device=pred_norm.device, dtype=pred_norm.dtype)
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    visible = 1.0 - mask
    kernel = 2 * width_px + 1
    dilated_mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=width_px)
    dilated_visible = F.max_pool2d(visible, kernel_size=kernel, stride=1, padding=width_px)
    boundary_band = (dilated_mask * dilated_visible).clamp(0.0, 1.0)

    out = pred_norm
    for _ in range(passes):
        smoothed = F.avg_pool2d(out, kernel_size=kernel, stride=1, padding=width_px)
        out = out * (1.0 - strength * boundary_band) + smoothed * (strength * boundary_band)
    return out


def _stack_precip_target(batch: Dict[str, torch.Tensor], dummy_dataset: WoFSMAEDataset) -> np.ndarray:
    return torch.cat([batch[var] for var in dummy_dataset.precip_vars], dim=1)[0].detach().cpu().numpy().astype(np.float32)


def _denormalize_precip_target(
    target_norm: np.ndarray,
    mean_values: Dict[str, np.ndarray],
    std_values: Dict[str, np.ndarray],
    concentration_specs: Dict[str, dict],
    dummy_dataset: WoFSMAEDataset,
) -> Dict[str, np.ndarray]:
    from applications.rollout_wrf_wofs_mae_da import _denormalize_array

    offset = 0
    out: Dict[str, np.ndarray] = {}
    for var in dummy_dataset.precip_vars:
        n_ch = dummy_dataset._mean_values[var].shape[0] if np.asarray(dummy_dataset._mean_values[var]).ndim > 0 else 1
        slc = target_norm[offset : offset + n_ch]
        out[var] = _denormalize_array(
            slc,
            var,
            mean_values,
            std_values,
            concentration_specs,
            normalization_mode=dummy_dataset._concentration_normalization_mode,
        )
        offset += n_ch
    return out


def _concat_var_dict(data_by_var: Dict[str, np.ndarray], precip_vars: list[str]) -> np.ndarray:
    return np.concatenate([np.asarray(data_by_var[var], dtype=np.float32) for var in precip_vars], axis=0)


def _metrics_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [
        "normalized_mse_masked",
        "normalized_mae_masked",
        "normalized_ssim_masked",
        "normalized_mse_full",
        "normalized_mae_full",
        "normalized_ssim_full",
        "physical_mse_masked",
        "physical_mae_masked",
        "physical_mse_full",
        "physical_mae_full",
        "masked_fraction",
    ]
    summary = {"n_samples": len(rows)}
    for key in keys:
        vals = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
    return summary


def _copy_config_to_output(config_path: str, out_dir: str, world_rank: int) -> None:
    if world_rank != 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, Path(config_path).name)
    shutil.copy2(config_path, dst)


def _write_case_metrics_json(case_out_dir: str, case_stem: str, rows: list[dict], metadata: dict) -> str:
    out_path = os.path.join(case_out_dir, f"{case_stem}_metrics.json")
    payload = {
        "case": case_stem,
        "n_samples": len(rows),
        "summary": _metrics_summary(rows),
        "samples": rows,
        "metadata": metadata,
    }
    with open(out_path, "w", encoding="ascii") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    return out_path


def _write_case_mask_bundle(
    out_dir: str,
    case_stem: str,
    conf: dict,
    n_times: int,
    seed: int,
    case_file: str,
    config_path: str,
) -> dict:
    eval_conf = _eval_conf(conf)
    group_names, group_channels = resolve_precip_group_layout(conf)
    token_h, token_w = token_grid_shape(conf)
    bundle = build_grouped_patch_masks(
        n_times=n_times,
        n_groups=len(group_names),
        token_h=token_h,
        token_w=token_w,
        mask_ratio=eval_conf.get("precip_mask_ratio", 1.0),
        mask_mode=eval_conf.get("precip_mask_mode", "spatial_patch"),
        seed=seed,
        channel_patch_mask_probability=float(conf.get("trainer", {}).get("channel_patch_mask_probability", 0.5)),
        mixed_height_spatial_probability=float(eval_conf.get("mixed_height_spatial_probability", 1.0)),
        mixed_height_channel_probability=float(eval_conf.get("mixed_height_channel_probability", 0.0)),
        mixed_height_height_probability=float(eval_conf.get("mixed_height_height_probability", 1.0)),
        group_channels=group_channels,
        height_mask_levels=eval_conf.get("height_mask_levels"),
        height_visible_levels=eval_conf.get("height_visible_levels"),
    )
    mask_path = os.path.join(out_dir, f"{case_stem}_mask.npz")
    save_mask_bundle(
        mask_path,
        patch_mask_grouped=bundle["patch_mask_grouped"],
        mask_mode=bundle["mask_mode"],
        requested_mask_ratio=bundle["requested_mask_ratio"],
        actual_group_mask_fraction=bundle["actual_group_mask_fraction"],
        group_names=group_names,
        group_channels=group_channels,
        image_size=tuple(int(v) for v in conf["model"]["image_size"]),
        patch_size=int(conf["model"]["patch_size"]),
        seed=seed,
        source_case=case_file,
        config_path=config_path,
    )
    return load_mask_bundle(mask_path)


def _build_precip_mask_for_rollout(
    model: torch.nn.Module,
    conf: dict,
    batch_size: int,
    device: torch.device,
    mask_bundle: Optional[dict],
    time_index: int,
) -> tuple[torch.Tensor, dict]:
    if mask_bundle is None:
        eval_conf = dict(_eval_conf(conf))
        mode = normalize_mask_mode(eval_conf.get("precip_mask_mode", "spatial_patch"))
        needs_level_axis = mode == "mixed_height"
        if mode == "mixed":
            channel_prob = float(conf.get("trainer", {}).get("channel_patch_mask_probability", 0.5))
            mode = "channel_patch" if torch.rand((), device=device).item() < channel_prob else "spatial_patch"
            eval_conf["precip_mask_mode"] = mode
        elif mode == "mixed_height":
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
            mode = ["spatial_patch", "channel_patch", "height_patch"][int(torch.multinomial(probs / total, 1).item())]
            eval_conf["precip_mask_mode"] = mode
        runtime_mask = _runtime_sample_mask(model, eval_conf, batch_size, device)
        token_h, token_w = token_grid_shape(conf)
        _, group_channels = resolve_precip_group_layout(conf)
        grouped_count = len(group_channels)
        runtime_np = runtime_mask.detach().cpu().numpy()
        if runtime_mask.ndim == 2:
            base = runtime_np[0].reshape(token_h, token_w)
            grouped_mask = np.repeat(base[None, :, :], grouped_count, axis=0)
        elif runtime_mask.ndim == 4:
            grouped_mask = runtime_np[0].reshape(grouped_count, int(group_channels[0]), token_h, token_w)
        else:
            grouped_mask = runtime_np[0].reshape(runtime_np.shape[1], token_h, token_w)
        if needs_level_axis and grouped_mask.ndim == 3:
            grouped_mask = np.repeat(grouped_mask[:, None, :, :], int(group_channels[0]), axis=1)
        return runtime_mask, {
            "mode": mode,
            "grouped_patch_mask": grouped_mask,
            "requested_ratio": float(_eval_conf(conf).get("precip_mask_ratio", 1.0)),
        }

    grouped_mask, mode, requested_ratio = grouped_patch_mask_time_slice(mask_bundle, time_index)
    runtime_np = _grouped_mask_to_runtime_mask(grouped_mask, mask_bundle["group_channels"], mode)
    runtime_mask = torch.from_numpy(runtime_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    return runtime_mask, {
        "mode": mode,
        "grouped_patch_mask": grouped_mask,
        "requested_ratio": requested_ratio,
    }


def rollout_one_timestep_with_metrics(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    mean_values: Dict[str, np.ndarray],
    std_values: Dict[str, np.ndarray],
    concentration_specs: Dict[str, dict],
    dummy_dataset: WoFSMAEDataset,
    conf: dict,
    device: torch.device,
    time_index: int,
    mask_bundle: Optional[dict],
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[dict], dict]:
    precip_mask, mask_info = _build_precip_mask_for_rollout(
        model=model,
        conf=conf,
        batch_size=batch["precip"].shape[0],
        device=device,
        mask_bundle=mask_bundle,
        time_index=time_index,
    )

    core = _unwrap_model(model)
    target = batch["precip"]
    use_visible_precip = bool(
        _eval_conf(conf).get("visible_precip_conditioning", _eval_conf(conf).get("clamp_visible_precip", True))
    )
    precip_visible = target if use_visible_precip else None
    sampler = str(_eval_conf(conf).get("sampler", "ddim")).strip().lower()
    sampling_timesteps = int(_eval_conf(conf).get("sampling_timesteps", getattr(core, "sampling_timesteps", 50)))
    eta = _eval_conf(conf).get("ddim_sampling_eta", None)
    repaint_jump_length = int(_eval_conf(conf).get("repaint_jump_length", 10))
    repaint_jump_n_sample = int(_eval_conf(conf).get("repaint_jump_n_sample", 10))
    clamp_final_visible = bool(_eval_conf(conf).get("clamp_final_visible_precip", False))
    save_trajectory = bool(_eval_conf(conf).get("save_denoise_trajectory", False))
    _, ensemble_size, _, _ = _resolve_eval_ensemble_size(conf)
    eval_batch_size = _resolve_eval_batch_size(conf, ensemble_size)

    cond = _build_condition_dict(
        batch,
        dummy_dataset.background_vars,
        dummy_dataset.reflectivity_vars,
        dummy_dataset.surface_vars,
        dummy_dataset.forcing_vars,
        target.shape[-1],
    )

    orig_h, orig_w = batch.get("_orig_hw", target.shape[-2:])
    target_norm = _stack_precip_target(batch, dummy_dataset)[:, :orig_h, :orig_w]
    target_phys = _concat_var_dict(
        _denormalize_precip_target(
            target_norm,
            mean_values=mean_values,
            std_values=std_values,
            concentration_specs=concentration_specs,
            dummy_dataset=dummy_dataset,
        ),
        dummy_dataset.precip_vars,
    )
    pixel_mask = _grouped_patch_to_pixel_mask(
        np.asarray(mask_info["grouped_patch_mask"], dtype=np.float32),
        resolve_precip_group_layout(conf)[1],
        patch_size=int(conf["model"]["patch_size"]),
        height=orig_h,
        width=orig_w,
    )

    out_by_var: Dict[str, List[np.ndarray]] = {var: [] for var in dummy_dataset.precip_vars}
    out_norm_by_var: Dict[str, List[np.ndarray]] = {var: [] for var in dummy_dataset.precip_vars}
    trajectory_values: List[np.ndarray] = []
    trajectory_channels = None
    metrics_rows: list[dict] = []
    full_mask = np.ones_like(pixel_mask, dtype=np.float32)

    for ensemble_start in range(0, ensemble_size, eval_batch_size):
        member_count = min(eval_batch_size, ensemble_size - ensemble_start)
        cond_batch = {
            key: value.repeat(member_count, *((1,) * (value.ndim - 1)))
            for key, value in cond.items()
        }
        precip_mask_batch = precip_mask.repeat(member_count, *((1,) * (precip_mask.ndim - 1)))
        precip_visible_batch = None
        if precip_visible is not None:
            precip_visible_batch = precip_visible.repeat(member_count, *((1,) * (precip_visible.ndim - 1)))

        with torch.no_grad():
            pred_norm = core.sample_precip(
                cond_batch,
                precip_mask_batch,
                precip_visible=precip_visible_batch,
                sampling_timesteps=sampling_timesteps,
                eta=eta,
                sampler=sampler,
                repaint_jump_length=repaint_jump_length,
                repaint_jump_n_sample=repaint_jump_n_sample,
                clamp_final_visible=clamp_final_visible,
                return_all_timesteps=save_trajectory,
            )

        if save_trajectory:
            all_steps_batch = pred_norm.detach().cpu().numpy()
            pred_norm = pred_norm[:, -1]
            channels = _eval_conf(conf).get("trajectory_channels", [int(_eval_conf(conf).get("channel", 0))])
            if isinstance(channels, int):
                channels = [channels]
            channels = [max(0, min(int(channel), all_steps_batch.shape[2] - 1)) for channel in channels]
            stride = max(1, int(_eval_conf(conf).get("trajectory_stride", 1)))
            for member_steps in all_steps_batch:
                trajectory_values.append(member_steps[::stride, channels].astype(np.float32, copy=False))
            trajectory_channels = np.asarray(channels, dtype=np.int32)

        blend_width_px = int(_eval_conf(conf).get("visible_blend_width_px", 0))
        if blend_width_px > 0:
            pixel_mask_batch = core.expand_patch_mask(precip_mask_batch, target.shape[-2], target.shape[-1]).to(pred_norm.dtype)
            pred_norm = _blend_mask_boundary(
                pred_norm,
                pixel_mask_batch,
                width_px=blend_width_px,
                strength=float(_eval_conf(conf).get("visible_blend_strength", 0.5)),
                passes=int(_eval_conf(conf).get("visible_blend_passes", 1)),
            )

        pred_norm_batch = pred_norm.detach().cpu().numpy()[:, :, :orig_h, :orig_w]

        for batch_member_index, pred_norm_member in enumerate(pred_norm_batch):
            ensemble_index = ensemble_start + batch_member_index
            member_out_norm: Dict[str, np.ndarray] = {}
            offset = 0
            for var in dummy_dataset.precip_vars:
                n_ch = dummy_dataset._mean_values[var].shape[0] if np.asarray(dummy_dataset._mean_values[var]).ndim > 0 else 1
                member_out_norm[var] = pred_norm_member[offset : offset + n_ch].copy()
                offset += n_ch

            member_out = _denormalize_precip_target(
                pred_norm_member,
                mean_values=mean_values,
                std_values=std_values,
                concentration_specs=concentration_specs,
                dummy_dataset=dummy_dataset,
            )
            pred_phys = _concat_var_dict(member_out, dummy_dataset.precip_vars)

            normalized_metrics = compute_masked_normalized_metrics(pred_norm_member, target_norm, pixel_mask)
            physical_metrics = compute_masked_physical_metrics(pred_phys, target_phys, pixel_mask)
            normalized_metrics_full = compute_masked_normalized_metrics(pred_norm_member, target_norm, full_mask)
            physical_metrics_full = compute_masked_physical_metrics(pred_phys, target_phys, full_mask)
            metrics_rows.append(
                {
                    "time_index": int(time_index),
                    "ensemble_index": int(ensemble_index),
                    "normalized_mse_masked": normalized_metrics["mse"],
                    "normalized_mae_masked": normalized_metrics["mae"],
                    "normalized_ssim_masked": normalized_metrics["ssim"],
                    "normalized_mse_full": normalized_metrics_full["mse"],
                    "normalized_mae_full": normalized_metrics_full["mae"],
                    "normalized_ssim_full": normalized_metrics_full["ssim"],
                    "physical_mse_masked": physical_metrics["mse"],
                    "physical_mae_masked": physical_metrics["mae"],
                    "physical_mse_full": physical_metrics_full["mse"],
                    "physical_mae_full": physical_metrics_full["mae"],
                    "masked_fraction": normalized_metrics["masked_fraction"],
                    "mask_mode": mask_info["mode"],
                    "requested_mask_ratio": float(mask_info["requested_ratio"]),
                }
            )

            for var, arr in member_out.items():
                out_by_var[var].append(arr)
            for var, arr in member_out_norm.items():
                out_norm_by_var[var].append(arr)

    trajectory = None
    if trajectory_values:
        trajectory = {
            "values": np.stack(trajectory_values, axis=0).astype(np.float32, copy=False),
            "channels": trajectory_channels,
        }

    mask_record = {
        "grouped_patch_mask": np.asarray(mask_info["grouped_patch_mask"], dtype=np.float32),
        "pixel_mask": pixel_mask,
        "mask_mode": str(mask_info["mode"]),
        "requested_mask_ratio": float(mask_info["requested_ratio"]),
    }
    return (
        {var: np.stack(frames, axis=0).astype(np.float32, copy=False) for var, frames in out_by_var.items()},
        {var: np.stack(frames, axis=0).astype(np.float32, copy=False) for var, frames in out_norm_by_var.items()},
        trajectory,
        {"metrics": metrics_rows, "mask": mask_record},
    )


def run_rollout(args, conf: dict):
    eval_conf = _eval_conf(conf)
    requested_ensemble_size, ensemble_size, deterministic_ensemble, effective_eta = _resolve_eval_ensemble_size(conf)
    eval_batch_size = _resolve_eval_batch_size(conf, ensemble_size)
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
        logger.info("Running DiffMAE rollout+metrics on rank %d/%d device=%s", world_rank, world_size, device)
        logger.info("Using eval ensemble_size=%d with eval_batch_size=%d", ensemble_size, eval_batch_size)
        if deterministic_ensemble and world_rank == 0:
            logger.info(
                "Eval sampler=ddim with ddim_sampling_eta=%s is deterministic; forcing rollout ensemble_size=1 (requested %d).",
                effective_eta,
                requested_ensemble_size,
            )

        _copy_config_to_output(args.model_config, args.out_dir, world_rank)
        model = _load_diffmae_model(conf, args.checkpoint, device)
        mean_values, std_values, concentration_specs = _load_stats_and_params(conf)

        dummy_ds = WoFSMAEDataset(filenames=[], conf=conf, seed=0)
        dummy_ds._mean_values = mean_values
        dummy_ds._std_values = std_values
        dummy_ds._concentration_transform_specs = concentration_specs
        dummy_ds._opened = True

        out_dir = args.out_dir
        if world_rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        file_pattern = conf["data"]["save_loc"]
        from glob import glob

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

        for case_index, case_file in enumerate(rank_files, start=1):
            case_start_time = time.monotonic()
            case_stem = Path(case_file).stem.replace(".zarr", "")
            case_date = _extract_case_date(case_file) or "unknown"
            case_out_dir = os.path.join(out_dir, case_date)
            os.makedirs(case_out_dir, exist_ok=True)
            out_path = os.path.join(case_out_dir, f"{case_stem}_analysis.zarr")
            metrics_path = os.path.join(case_out_dir, f"{case_stem}_metrics.json")
            logger.info(
                "Rank %d starting case %d/%d: %s",
                world_rank,
                case_index,
                len(rank_files),
                case_stem,
            )

            if os.path.exists(out_path) and not bool(eval_conf.get("overwrite", False)):
                logger.info("Skipping %s (already exists)", out_path)
                continue

            try:
                io_start_time = time.monotonic()
                all_vars = list(
                    set(
                        dummy_ds.background_vars
                        + dummy_ds.precip_vars
                        + dummy_ds.reflectivity_vars
                        + dummy_ds.surface_vars
                        + _forcing_source_vars(dummy_ds)
                    )
                )
                ds = get_forward_data(case_file)
                ds = filter_ds(ds, all_vars)
                n_time = ds.sizes.get("time", 0)
                if dummy_ds.mae_include_t1_refl:
                    n_time = max(0, n_time - 1)
                if args.max_times is not None:
                    n_time = min(n_time, int(args.max_times))
                logger.info(
                    "Rank %d case %s opened in %.1fs; processing %d timesteps, ensemble_size=%d, eval_batch_size=%d",
                    world_rank,
                    case_stem,
                    time.monotonic() - io_start_time,
                    n_time,
                    ensemble_size,
                    eval_batch_size,
                )

                if args.mask_file:
                    logger.info("Rank %d case %s loading mask bundle: %s", world_rank, case_stem, args.mask_file)
                    mask_bundle = load_mask_bundle(args.mask_file)
                else:
                    mask_seed = int(args.mask_seed if args.mask_seed is not None else conf.get("seed", 0))
                    mask_start_time = time.monotonic()
                    mask_bundle = _write_case_mask_bundle(
                        out_dir=case_out_dir,
                        case_stem=case_stem,
                        conf=conf,
                        n_times=n_time,
                        seed=mask_seed,
                        case_file=case_file,
                        config_path=args.model_config,
                    )
                    logger.info(
                        "Rank %d case %s wrote mask bundle in %.1fs",
                        world_rank,
                        case_stem,
                        time.monotonic() - mask_start_time,
                    )

                recon_by_var: Dict[str, List[np.ndarray]] = {v: [] for v in dummy_ds.precip_vars}
                recon_by_var_norm: Dict[str, List[np.ndarray]] = {v: [] for v in dummy_ds.precip_vars}
                trajectories: List[np.ndarray] = []
                trajectory_channels = None
                metric_rows: list[dict] = []
                grouped_patch_masks: list[np.ndarray] = []
                pixel_masks: list[np.ndarray] = []
                mask_modes: list[str] = []
                requested_mask_ratios: list[float] = []

                for t_idx in range(n_time):
                    timestep_start_time = time.monotonic()
                    valid_time = str(ds["time"].values[t_idx]) if "time" in ds.coords else str(t_idx)
                    logger.info(
                        "Rank %d case %s timestep %d/%d valid_time=%s: sampling %d ensemble members in batches of %d",
                        world_rank,
                        case_stem,
                        t_idx + 1,
                        n_time,
                        valid_time,
                        ensemble_size,
                        eval_batch_size,
                    )
                    stop = t_idx + 2 if dummy_ds.mae_include_t1_refl else t_idx + 1
                    chunk = ds.isel(time=slice(t_idx, stop)).load()
                    try:
                        batch = _prepare_batch_from_case(chunk, dummy_ds, int(conf["data"].get("mae_pad_to", 304)), device)
                        result, result_norm, trajectory, extra = rollout_one_timestep_with_metrics(
                            model=model,
                            batch=batch,
                            mean_values=mean_values,
                            std_values=std_values,
                            concentration_specs=concentration_specs,
                            dummy_dataset=dummy_ds,
                            conf=conf,
                            device=device,
                            time_index=t_idx,
                            mask_bundle=mask_bundle,
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

                    for metrics_row in extra["metrics"]:
                        metrics_row["valid_time"] = valid_time
                        metric_rows.append(metrics_row)
                    grouped_patch_masks.append(extra["mask"]["grouped_patch_mask"])
                    pixel_masks.append(extra["mask"]["pixel_mask"])
                    mask_modes.append(extra["mask"]["mask_mode"])
                    requested_mask_ratios.append(extra["mask"]["requested_mask_ratio"])
                    logger.info(
                        "Rank %d case %s timestep %d/%d finished in %.1fs",
                        world_rank,
                        case_stem,
                        t_idx + 1,
                        n_time,
                        time.monotonic() - timestep_start_time,
                    )

                time_vals = ds["time"].values[:n_time] if "time" in ds.coords else np.arange(n_time)
                data_vars = {}
                for var, frames in recon_by_var.items():
                    stacked = np.stack(frames, axis=0).astype(np.float32)
                    data_vars[var] = (("time", "ensemble", "level", "y", "x"), stacked)
                if not data_vars:
                    raise RuntimeError(f"No rollout frames were produced for {case_file}")
                first_data = next(iter(data_vars.values()))[1]
                analysis_ds = xr.Dataset(
                    data_vars=data_vars,
                    coords={
                        "time": time_vals,
                        "ensemble": np.arange(first_data.shape[1], dtype=np.int32),
                        "level": np.arange(first_data.shape[2], dtype=np.int32),
                        "y": np.arange(first_data.shape[3], dtype=np.int32),
                        "x": np.arange(first_data.shape[4], dtype=np.int32),
                    },
                    attrs={
                        "units": "physical_units_after_inverse_normalization",
                        "sampler": str(eval_conf.get("sampler", "ddim")),
                        "sampling_timesteps": int(eval_conf.get("sampling_timesteps", 50)),
                        "ddim_sampling_eta": float(effective_eta),
                        "repaint_jump_length": int(eval_conf.get("repaint_jump_length", 10)),
                        "repaint_jump_n_sample": int(eval_conf.get("repaint_jump_n_sample", 10)),
                        "precip_mask_mode": str(eval_conf.get("precip_mask_mode", "spatial_patch")),
                        "precip_mask_ratio": str(eval_conf.get("precip_mask_ratio", 1.0)),
                        "mask_file": str(args.mask_file) if args.mask_file else "",
                        "requested_ensemble_size": requested_ensemble_size,
                        "ensemble_size": ensemble_size,
                        "eval_batch_size": eval_batch_size,
                        "deterministic_ensemble_override": int(deterministic_ensemble),
                    },
                )
                if os.path.exists(out_path):
                    shutil.rmtree(out_path)
                write_start_time = time.monotonic()
                logger.info("Rank %d case %s writing analysis zarr: %s", world_rank, case_stem, out_path)
                analysis_ds.to_zarr(out_path, mode="w")

                if bool(eval_conf.get("save_norm_output", True)):
                    logger.info("Rank %d case %s writing normalized output group", world_rank, case_stem)
                    xr.Dataset(
                        data_vars={
                            var: (("time", "ensemble", "level", "y", "x"), np.stack(frames, axis=0).astype(np.float32))
                            for var, frames in recon_by_var_norm.items()
                        },
                        coords=analysis_ds.coords,
                        attrs={"units": "normalized_model_output"},
                    ).to_zarr(out_path, mode="a", group="norm_output")

                if trajectories:
                    logger.info("Rank %d case %s writing denoise trajectory group", world_rank, case_stem)
                    traj_arr = np.stack(trajectories, axis=0).astype(np.float32)
                    channel_vars = []
                    channel_levels = []
                    for channel_idx in trajectory_channels:
                        var_name, level_idx = _global_channel_to_var_level(dummy_ds, int(channel_idx))
                        channel_vars.append(var_name)
                        channel_levels.append(level_idx)
                    xr.Dataset(
                        data_vars={
                            "precip": (
                                ("time", "ensemble", "denoise_step", "channel", "y_padded", "x_padded"),
                                traj_arr,
                            )
                        },
                        coords={
                            "time": time_vals[: traj_arr.shape[0]],
                            "ensemble": np.arange(traj_arr.shape[1], dtype=np.int32),
                            "denoise_step": np.arange(traj_arr.shape[2], dtype=np.int32),
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

                group_names, group_channels = resolve_precip_group_layout(conf)
                grouped_mask_arr = np.stack(grouped_patch_masks, axis=0).astype(np.float32)
                patch_h, patch_w = grouped_mask_arr.shape[-2:]
                grouped_mask_dims = ("time", "mask_group", "patch_y", "patch_x")
                grouped_mask_coords = {}
                if grouped_mask_arr.ndim == 5:
                    grouped_mask_dims = ("time", "mask_group", "level", "patch_y", "patch_x")
                    grouped_mask_coords["level"] = np.arange(grouped_mask_arr.shape[2], dtype=np.int32)
                mask_ds = xr.Dataset(
                    data_vars={
                        "patch_mask_grouped": (
                            grouped_mask_dims,
                            grouped_mask_arr,
                        ),
                        "pixel_mask_channel": (
                            ("time", "channel", "y", "x"),
                            np.stack(pixel_masks, axis=0).astype(np.float32),
                        ),
                        "requested_mask_ratio": (("time",), np.asarray(requested_mask_ratios, dtype=np.float32)),
                        "mask_mode": (("time",), np.asarray(mask_modes, dtype="<U16")),
                    },
                    coords={
                        "time": time_vals,
                        "mask_group": np.arange(len(group_names), dtype=np.int32),
                        "patch_y": np.arange(patch_h, dtype=np.int32),
                        "patch_x": np.arange(patch_w, dtype=np.int32),
                        "channel": np.arange(int(sum(group_channels)), dtype=np.int32),
                        "y": analysis_ds.coords["y"].values,
                        "x": analysis_ds.coords["x"].values,
                        "mask_group_name": ("mask_group", np.asarray(group_names, dtype="<U64")),
                        "mask_group_channels": ("mask_group", np.asarray(group_channels, dtype=np.int32)),
                        **grouped_mask_coords,
                    },
                    attrs={
                        "description": "Patch-aligned precip masks used during rollout. pixel_mask_channel is expanded from patch_mask_grouped.",
                        "patch_size": int(conf["model"]["patch_size"]),
                    },
                )
                logger.info("Rank %d case %s writing mask group", world_rank, case_stem)
                mask_ds.to_zarr(out_path, mode="a", group="mask")

                logger.info("Rank %d case %s writing metrics JSON", world_rank, case_stem)
                _write_case_metrics_json(
                    case_out_dir=case_out_dir,
                    case_stem=case_stem,
                    rows=metric_rows,
                    metadata={
                        "case_file": case_file,
                        "analysis_zarr": out_path,
                        "mask_file": str(args.mask_file) if args.mask_file else os.path.join(case_out_dir, f"{case_stem}_mask.npz"),
                        "config_copy": os.path.join(out_dir, Path(args.model_config).name),
                        "requested_ensemble_size": requested_ensemble_size,
                        "ensemble_size": ensemble_size,
                        "eval_batch_size": eval_batch_size,
                        "deterministic_ensemble_override": bool(deterministic_ensemble),
                    },
                )
                logger.info(
                    "Rank %d case %s complete in %.1fs; writes took %.1fs; analysis=%s metrics=%s",
                    world_rank,
                    case_stem,
                    time.monotonic() - case_start_time,
                    time.monotonic() - write_start_time,
                    out_path,
                    metrics_path,
                )
            except Exception as exc:
                logger.error("Failed to process %s: %s", case_file, exc, exc_info=True)

        logger.info("Rollout complete on rank %d.", world_rank)
    finally:
        if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def primary_main():
    parser = argparse.ArgumentParser(
        description="WoFS DiffMAE DA rollout with metrics JSON, config copy, and persistent masks."
    )
    parser.add_argument("-c", "--config", dest="model_config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo", "mpi"])
    parser.add_argument("--mode", type=str, default=None, choices=["none", "ddp", "fsdp"])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-times", type=int, default=None)
    parser.add_argument(
        "--mask-file",
        type=str,
        default=None,
        help="Optional .npz mask bundle created by generate_wofs_diffmae_mask.py. Overrides random mask sampling.",
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=None,
        help="Seed for generated per-case masks when --mask-file is not provided.",
    )
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

    if args.mask_file:
        mask_bundle = load_mask_bundle(args.mask_file)
        expected_groups, expected_channels = resolve_precip_group_layout(conf)
        if mask_bundle["group_names"] != expected_groups:
            raise ValueError(
                f"Mask group names {mask_bundle['group_names']} do not match config groups {expected_groups}"
            )
        if mask_bundle["group_channels"] != expected_channels:
            raise ValueError(
                f"Mask group channels {mask_bundle['group_channels']} do not match config {expected_channels}"
            )
        if tuple(mask_bundle["image_size"]) != tuple(int(v) for v in conf["model"]["image_size"]):
            raise ValueError("Mask image_size does not match config model.image_size")
        if int(mask_bundle["patch_size"]) != int(conf["model"]["patch_size"]):
            raise ValueError("Mask patch_size does not match config model.patch_size")

    run_rollout(args, conf)


if __name__ == "__main__":
    primary_main()
