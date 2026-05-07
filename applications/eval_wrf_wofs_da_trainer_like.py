from __future__ import annotations

import logging
import os
import re
import csv
import shutil
import multiprocessing as mp
import time
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

from credit.data import reshape_only
from credit.datasets.wrf_wofs_da_increment import WoFSDAIncrementDataset
from credit.metrics import LatWeightedMetrics
from credit.models import load_model
from credit.models.checkpoint import load_state_dict_error_handler
from credit.parser import credit_main_parser
from credit.seed import seed_everything
from credit.transforms.concentration import CONCENTRATION_VARS
from train_wrf_wofs_da import _configure_aurora_da_model

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    if hasattr(dataset, "_open_datasets") and callable(dataset._open_datasets):
        dataset._open_datasets()


def _extract_case_date(file_path: str) -> str | None:
    match = re.search(r"wofs_(\d{8})_\d{4}_mem\d+\.zarr(?:\.zip)?$", Path(file_path).name)
    return match.group(1) if match else None


def _select_files(glob_pattern: str, date_range: list[str] | tuple[str, str] | None = None) -> list[str]:
    files = sorted(glob(glob_pattern))
    if date_range is None:
        return files

    start_date, end_date = str(date_range[0]), str(date_range[1])
    selected = []
    for file_path in files:
        case_date = _extract_case_date(file_path)
        if case_date is not None and start_date <= case_date <= end_date:
            selected.append(file_path)
    return selected


def _sync_prognostic_levels(conf: dict) -> int:
    model_levels = int(conf["model"]["param_interior"]["levels"])
    data_levels = int(conf["data"].get("prognostic_levels", model_levels))

    level_candidates = []
    level_ops = conf["data"].get("concentration_level_ops", {})
    if isinstance(level_ops, dict):
        for var_name in conf["data"].get("variables", []):
            var_cfg = level_ops.get(var_name)
            if isinstance(var_cfg, dict) and "ceiling_level" in var_cfg:
                level_candidates.append(int(var_cfg["ceiling_level"]))

    if len(level_candidates) > 0:
        unique_levels = sorted(set(level_candidates))
        if len(unique_levels) > 1:
            raise ValueError(
                "All prognostic concentration variables must share one ceiling_level for reduced-level evaluation. "
                f"Got: {unique_levels}"
            )
        data_levels = int(unique_levels[0])

    data_levels = max(1, int(data_levels))
    conf["data"]["prognostic_levels"] = data_levels
    conf["model"]["param_interior"]["levels"] = data_levels
    return data_levels


def _build_test_dataset(
    conf: dict,
    test_path: str,
    test_dyn_path: str | None,
    date_range: list[str] | tuple[str, str] | None,
) -> WoFSDAIncrementDataset:
    filenames = _select_files(test_path, date_range)
    if len(filenames) == 0:
        raise ValueError(f"No test files found for pattern {test_path} with date_range={date_range}")

    if test_dyn_path:
        filename_dyn_forcing = _select_files(test_dyn_path, date_range)
    elif conf["data"].get("save_loc_dynamic_forcing"):
        filename_dyn_forcing = _select_files(conf["data"]["save_loc_dynamic_forcing"], date_range)
    else:
        filename_dyn_forcing = None

    if filename_dyn_forcing is not None and len(filename_dyn_forcing) != len(filenames):
        raise ValueError(
            "Mismatch between selected test files and selected dynamic forcing files. "
            f"Got {len(filenames)} test files and {len(filename_dyn_forcing)} forcing files."
        )

    param_interior = {
        "varname_upper_air": conf["data"]["variables"],
        "varname_context_upper_air": conf["data"]["context_upper_air_variables"],
        "varname_dyn_forcing": conf["data"]["dynamic_forcing_variables"],
        "filenames": filenames,
        "filename_dyn_forcing": filename_dyn_forcing,
        "history_len": conf["data"].get("valid_history_len", conf["data"]["history_len"]),
        "forecast_len": conf["data"].get("valid_forecast_len", conf["data"]["forecast_len"]),
    }
    param_outside = {
        "varname_upper_air": conf["data"]["observation_variables"],
    }

    dataset = WoFSDAIncrementDataset(param_interior, param_outside, conf=conf, seed=conf["seed"])
    logger.info("Loaded test dataset with %s files and %s samples", len(filenames), len(dataset))
    return dataset


def _load_eval_model(conf: dict, device: torch.device) -> torch.nn.Module:
    model_type = conf.get("model", {}).get("type")
    if model_type in {"aurora_crossformer_wrf", "aurora_crossformer_wrf_da"}:
        checkpoint_path = _resolve_checkpoint_path(conf["save_loc"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        _reconcile_aurora_model_conf_from_checkpoint(conf, state_dict)

        model = load_model(conf, load_weights=False)
        load_msg = model.load_state_dict(state_dict, strict=False)
        load_state_dict_error_handler(load_msg)
        model = model.to(device)
    else:
        model = load_model(conf, load_weights=True).to(device)
    model.eval()
    return model


def _resolve_checkpoint_path(save_loc: str) -> str:
    expanded_save_loc = os.path.expandvars(save_loc)
    model_checkpoint = os.path.join(expanded_save_loc, "model_checkpoint.pt")
    if os.path.isfile(model_checkpoint):
        return model_checkpoint

    checkpoint = os.path.join(expanded_save_loc, "checkpoint.pt")
    if os.path.isfile(checkpoint):
        return checkpoint

    raise ValueError(f"No saved checkpoint exists under {expanded_save_loc}")


def _state_tensor_shape(state_dict: dict, key_prefix: str) -> tuple[int, ...] | None:
    for suffix in ("weight_orig", "weight", "bias"):
        tensor = state_dict.get(f"{key_prefix}.{suffix}")
        if tensor is not None:
            return tuple(int(dim) for dim in tensor.shape)
    return None


def _reconcile_aurora_model_conf_from_checkpoint(conf: dict, state_dict: dict) -> None:
    model_conf = conf.get("model", {})
    if model_conf.get("type") not in {"aurora_crossformer_wrf", "aurora_crossformer_wrf_da"}:
        return

    resolved: dict[str, object] = {}

    spectral_norm_enabled = any(key.endswith("weight_orig") for key in state_dict.keys())
    resolved["use_spectral_norm"] = spectral_norm_enabled

    interior_norm = state_dict.get("interior_embedder.norm.weight")
    if interior_norm is not None:
        resolved["embed_dim"] = int(interior_norm.shape[0])

    boundary_norm = state_dict.get("boundary_embedder.norm.weight")
    if boundary_norm is not None:
        resolved["boundary_embed_dim"] = int(boundary_norm.shape[0])

    film_in_shape = _state_tensor_shape(state_dict, "film_mlp.0")
    if film_in_shape is not None and "boundary_embed_dim" in resolved:
        time_encode_dim = int(film_in_shape[1]) - 5 * int(resolved["boundary_embed_dim"])
        if time_encode_dim > 0:
            resolved["time_encode_dim"] = time_encode_dim

    encoder_dims: list[int] = []
    encoder_depths: list[int] = []
    stage_index = 0
    while True:
        stage_conv_prefix = f"encoder_layers.{stage_index}.0.convs"
        conv_index = 0
        stage_dim = 0
        while True:
            shape = _state_tensor_shape(state_dict, f"{stage_conv_prefix}.{conv_index}")
            if shape is None:
                break
            stage_dim += int(shape[0])
            conv_index += 1

        if stage_dim == 0:
            break

        encoder_dims.append(stage_dim)

        layer_index = 0
        while True:
            attn_shape = _state_tensor_shape(
                state_dict,
                f"encoder_layers.{stage_index}.1.layers.{layer_index}.0.to_qkv",
            )
            if attn_shape is None:
                break
            layer_index += 1
        if layer_index > 0:
            encoder_depths.append(layer_index)

        stage_index += 1

    if encoder_dims:
        resolved["dim"] = encoder_dims
    if len(encoder_depths) == len(encoder_dims):
        resolved["depth"] = encoder_depths

    noise_conf = dict(model_conf.get("noise_injection", {}))
    latent_shape = _state_tensor_shape(state_dict, "dec_noise1.noise_transform")
    if latent_shape is not None:
        noise_conf["activate"] = True
        noise_conf["latent_dim"] = int(latent_shape[1])
        noise_conf["encoder_noise"] = any(key.startswith("enc_noise_layers.") for key in state_dict.keys())

        noise_scales: list[float] = []
        dec_noise_index = 1
        while True:
            tensor = state_dict.get(f"dec_noise{dec_noise_index}.noise_factor")
            if tensor is None:
                break
            noise_scales.append(float(tensor.reshape(-1)[0].item()))
            dec_noise_index += 1
        if noise_scales:
            noise_conf["noise_scales"] = noise_scales
    elif "activate" not in noise_conf:
        noise_conf["activate"] = False

    resolved["noise_injection"] = noise_conf

    changed = {}
    for key, value in resolved.items():
        if model_conf.get(key) != value:
            changed[key] = {"from": model_conf.get(key), "to": value}
            model_conf[key] = value

    if changed:
        logger.info("Reconciled Aurora model config from checkpoint: %s", changed)


def _sample_to_model_inputs(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = reshape_only(batch["x"]).to(device, non_blocking=True)

    if "x_forcing_static" in batch:
        x_forcing = batch["x_forcing_static"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4)
        x = torch.cat((x, x_forcing), dim=1)

    x_boundary = reshape_only(batch["x_boundary"]).to(device, non_blocking=True)
    x_time_encode = batch["x_time_encode"].to(device, non_blocking=True)
    y_true = reshape_only(batch["y"]).to(device, non_blocking=True)
    return x, x_boundary, x_time_encode, y_true


def _repeat_for_ensemble(tensor: torch.Tensor, ensemble_size: int) -> torch.Tensor:
    if ensemble_size <= 1:
        return tensor
    return torch.repeat_interleave(tensor, ensemble_size, dim=0)


def _ensemble_mean(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    if y_pred.shape[0] == y_true.shape[0] or y_pred.shape[0] % y_true.shape[0] != 0:
        return y_pred
    ensemble_size = y_pred.shape[0] // y_true.shape[0]
    return y_pred.view(y_true.shape[0], ensemble_size, *y_pred.shape[1:]).mean(dim=1)


def _ensemble_members(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor | None:
    if y_pred.shape[0] == y_true.shape[0] or y_pred.shape[0] % y_true.shape[0] != 0:
        return None
    ensemble_size = y_pred.shape[0] // y_true.shape[0]
    return y_pred.view(y_true.shape[0], ensemble_size, *y_pred.shape[1:])


def _apply_residual_prediction(
    y_pred: torch.Tensor,
    x_model: torch.Tensor,
    residual_prediction: bool,
    varnum_diag: int,
) -> torch.Tensor:
    if not residual_prediction:
        return y_pred

    num_prog = y_pred.shape[1] - varnum_diag
    residual = x_model[:, :num_prog, -1:, ...]
    if varnum_diag > 0:
        return torch.cat([y_pred[:, :num_prog, ...] + residual, y_pred[:, num_prog:, ...]], dim=1)
    return y_pred + residual


def _denormalize_prog_increments(
    dataset: "WoFSDAIncrementDataset",
    y_norm_incr: np.ndarray,
    x_norm_z0: np.ndarray,
) -> np.ndarray:
    """Convert normalized-space prognostic increments to physical-space increments.

    Inverts ``norm(Q_t1) - norm(Q_t0)`` back to ``Q_t1 - Q_t0`` in original units.
    Uses the dataset's configured inverse variable transform, which for the
    current WoFS DA setup is driven by ``data.log_transform_params_json``.

    Args:
        dataset: An opened ``WoFSDAIncrementDataset`` providing normalization stats.
        y_norm_incr: Normalized increments, shape ``(batch, n_vars*n_levels, time, H, W)``.
        x_norm_z0: Normalized prognostic-only state at t0,
            shape ``(batch, n_vars*n_levels, time, H, W)``.

    Returns:
        Physical increments in original variable units, same shape as inputs.
    """
    if not getattr(dataset, "_opened", False):
        dataset._open_datasets()

    n_levels = dataset._prognostic_levels
    phys = np.empty_like(y_norm_incr, dtype=np.float32)

    for var_idx, var_name in enumerate(dataset.varname_prognostic):
        ch0 = var_idx * n_levels
        ch1 = ch0 + n_levels

        dz = y_norm_incr[:, ch0:ch1, ...].astype(np.float64)
        z0 = x_norm_z0[:, ch0:ch1, ...].astype(np.float64)
        z1 = z0 + dz

        # Step 1: invert the dataset normalization mode to recover transformed-space values.
        transformed_t0 = dataset._denormalize_transformed_values(z0, var_name).astype(np.float64)
        transformed_t1 = dataset._denormalize_transformed_values(z1, var_name).astype(np.float64)

        # Step 2: inverse concentration transform → physical Q values
        q_t0 = dataset._inverse_var_transform(transformed_t0.astype(np.float32), var_name).astype(np.float64)
        q_t1 = dataset._inverse_var_transform(transformed_t1.astype(np.float32), var_name).astype(np.float64)

        phys[:, ch0:ch1, ...] = (q_t1 - q_t0).astype(np.float32)

    return phys


def _strip_deprecated_concentration_config(conf: dict) -> None:
    data_conf = conf.get("data", {})
    deprecated_path = data_conf.pop("concentration_params_json", None)
    if deprecated_path:
        logger.info(
            "Ignoring deprecated data.concentration_params_json=%s during evaluation; "
            "using data.log_transform_params_json only.",
            deprecated_path,
        )


def _validate_eval_transform_config(conf: dict) -> None:
    data_conf = conf["data"]
    prognostic_concentration_vars = [var for var in data_conf.get("variables", []) if var in CONCENTRATION_VARS]
    if prognostic_concentration_vars and not data_conf.get("log_transform_params_json"):
        raise ValueError(
            "Evaluation for concentration prognostic variables requires data.log_transform_params_json. "
            "The deprecated data.concentration_params_json path is ignored by this eval script. "
            f"Missing log transform config for variables: {prognostic_concentration_vars}"
        )


def _make_output_channel_names(conf: dict) -> list[str]:
    varnames = conf["data"]["variables"]
    num_levels = int(conf["model"]["param_interior"]["levels"])
    return [f"{var}_L{level}" for var in varnames for level in range(num_levels)]


def _make_input_channel_names(conf: dict) -> list[str]:
    names: list[str] = []

    prog_vars = conf["data"]["variables"]
    prog_levels = int(conf["model"]["param_interior"]["levels"])
    for var in prog_vars:
        for level in range(prog_levels):
            names.append(f"{var}_L{level}_t0")

    ctx_vars = conf["data"].get("context_upper_air_variables", [])
    data_levels = int(conf["data"].get("levels", prog_levels))
    for var in ctx_vars:
        for level in range(data_levels):
            names.append(f"{var}_L{level}_ctx")

    for var in conf["data"].get("dynamic_forcing_variables", []):
        names.append(f"{var}_dyn")

    return names


def _make_boundary_channel_names(conf: dict) -> list[str]:
    obs_vars = conf["data"].get("observation_variables", [])
    data_levels = int(conf["data"].get("levels", conf["model"]["param_interior"]["levels"]))
    return [f"{var}_L{level}_innov" for var in obs_vars for level in range(data_levels)]


def _collect_sample_metadata(dataset: WoFSDAIncrementDataset, sample_index: int) -> tuple[str, str, np.datetime64]:
    if not getattr(dataset, "_opened", False):
        dataset._open_datasets()

    file_idx, ind_in_file = dataset._locate_index(sample_index)
    case_path = dataset.filenames[file_idx]
    case_name = Path(case_path).stem
    upper_ds = dataset._get_upper_ds(file_idx)
    valid_time = np.datetime64(upper_ds["time"].isel(time=ind_in_file + 1).values)
    return case_name, case_path, valid_time


def _fixed_bytes(values: list[str], width: int) -> np.ndarray:
    return np.asarray(values, dtype=f"S{width}")


def _build_batch_dataset_for_zarr(
    conf: dict,
    sample_offset: int,
    x_inputs: np.ndarray,
    x_boundaries: np.ndarray,
    y_trues: np.ndarray,
    y_preds: np.ndarray,
    y_pred_members: np.ndarray | None,
    sample_indices: list[int],
    case_names: list[str],
    case_paths: list[str],
    valid_times: list[np.datetime64],
    per_sample_metrics: list[dict],
) -> xr.Dataset:
    input_channels = _make_input_channel_names(conf)
    boundary_channels = _make_boundary_channel_names(conf)
    output_channels = _make_output_channel_names(conf)

    n_samples, _, n_time, n_y, n_x = y_trues.shape

    if x_inputs.shape[1] != len(input_channels):
        input_channels = [f"input_ch_{i}" for i in range(x_inputs.shape[1])]
    if x_boundaries.shape[1] != len(boundary_channels):
        boundary_channels = [f"boundary_ch_{i}" for i in range(x_boundaries.shape[1])]
    if y_trues.shape[1] != len(output_channels):
        output_channels = [f"output_ch_{i}" for i in range(y_trues.shape[1])]

    data_vars = {
            "x_input": (["sample", "input_channel", "time", "y", "x"], x_inputs.astype(np.float32)),
            "x_boundary": (["sample", "boundary_channel", "time", "y", "x"], x_boundaries.astype(np.float32)),
            "y_true": (["sample", "output_channel", "time", "y", "x"], y_trues.astype(np.float32)),
            "y_pred": (["sample", "output_channel", "time", "y", "x"], y_preds.astype(np.float32)),
            "metric_acc": (["sample"], np.array([m["acc"] for m in per_sample_metrics], dtype=np.float32)),
            "metric_rmse": (["sample"], np.array([m["rmse"] for m in per_sample_metrics], dtype=np.float32)),
            "metric_mse": (["sample"], np.array([m["mse"] for m in per_sample_metrics], dtype=np.float32)),
            "metric_mae": (["sample"], np.array([m["mae"] for m in per_sample_metrics], dtype=np.float32)),
            "case_name": (["sample"], _fixed_bytes(case_names, 128)),
            "case_path": (["sample"], _fixed_bytes(case_paths, 512)),
            "global_index": (["sample"], np.array(sample_indices, dtype=np.int64)),
    }
    coords = {
            "sample": np.arange(sample_offset, sample_offset + n_samples, dtype=np.int64),
            "time": np.arange(n_time, dtype=np.int64),
            "y": np.arange(n_y, dtype=np.int64),
            "x": np.arange(n_x, dtype=np.int64),
            "input_channel": np.array(input_channels, dtype=str),
            "boundary_channel": np.array(boundary_channels, dtype=str),
            "output_channel": np.array(output_channels, dtype=str),
            "valid_time": (["sample"], np.array(valid_times, dtype="datetime64[ns]")),
    }
    if y_pred_members is not None:
        data_vars["y_pred_members"] = (
            ["sample", "ensemble_member", "output_channel", "time", "y", "x"],
            y_pred_members.astype(np.float32),
        )
        coords["ensemble_member"] = np.arange(y_pred_members.shape[1], dtype=np.int64)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
    )

    ds.attrs["description"] = "Trainer-like WoFS DA test samples with input, innovation boundary, target increment, and prediction"
    ds.attrs["metric_space"] = "normalized increment space (same space as training metrics)"
    ds.attrs["save_loc"] = str(conf.get("save_loc", ""))
    ds.attrs["variables"] = ",".join(conf["data"].get("variables", []))
    return ds


def _build_batch_physical_dataset_for_zarr(
    conf: dict,
    sample_offset: int,
    y_true_phys: np.ndarray,
    y_pred_phys: np.ndarray,
    sample_indices: list[int],
    case_names: list[str],
    case_paths: list[str],
    valid_times: list[np.datetime64],
) -> xr.Dataset:
    output_channels = _make_output_channel_names(conf)
    n_samples, _, n_time, n_y, n_x = y_true_phys.shape

    if y_true_phys.shape[1] != len(output_channels):
        output_channels = [f"output_ch_{i}" for i in range(y_true_phys.shape[1])]

    ds = xr.Dataset(
        data_vars={
            "y_true_phys": (["sample", "output_channel", "time", "y", "x"], y_true_phys.astype(np.float32)),
            "y_pred_phys": (["sample", "output_channel", "time", "y", "x"], y_pred_phys.astype(np.float32)),
            "case_name": (["sample"], _fixed_bytes(case_names, 128)),
            "case_path": (["sample"], _fixed_bytes(case_paths, 512)),
            "global_index": (["sample"], np.array(sample_indices, dtype=np.int64)),
        },
        coords={
            "sample": np.arange(sample_offset, sample_offset + n_samples, dtype=np.int64),
            "time": np.arange(n_time, dtype=np.int64),
            "y": np.arange(n_y, dtype=np.int64),
            "x": np.arange(n_x, dtype=np.int64),
            "output_channel": np.array(output_channels, dtype=str),
            "valid_time": (["sample"], np.array(valid_times, dtype="datetime64[ns]")),
        },
    )
    ds.attrs["description"] = "Physical-space WoFS DA increments — y_true and y_pred denormalized to original variable units"
    ds.attrs["metric_space"] = "physical space (original variable units, e.g. kg kg-1)"
    ds.attrs["save_loc"] = str(conf.get("save_loc", ""))
    ds.attrs["variables"] = ",".join(conf["data"].get("variables", []))
    return ds


def _append_batch_to_zarr(store_path: str, ds_batch: xr.Dataset, is_first_batch: bool) -> None:
    store_dir = os.path.dirname(store_path)
    if store_dir:
        os.makedirs(store_dir, exist_ok=True)

    sz = ds_batch.sizes
    n_sample = max(1, sz["sample"])
    spatial_chunk = (1, sz.get("time", 1), sz.get("y", 1), sz.get("x", 1))

    encoding: dict = {}
    if "x_input" in ds_batch:
        encoding["x_input"] = {"chunks": (1, sz["input_channel"]) + spatial_chunk[1:]}
    if "x_boundary" in ds_batch:
        encoding["x_boundary"] = {"chunks": (1, sz["boundary_channel"]) + spatial_chunk[1:]}
    if "y_true" in ds_batch:
        encoding["y_true"] = {"chunks": (1, sz["output_channel"]) + spatial_chunk[1:]}
    if "y_pred" in ds_batch:
        encoding["y_pred"] = {"chunks": (1, sz["output_channel"]) + spatial_chunk[1:]}
    if "y_pred_members" in ds_batch:
        encoding["y_pred_members"] = {"chunks": (1, 1, sz["output_channel"]) + spatial_chunk[1:]}
    if "y_true_phys" in ds_batch:
        encoding["y_true_phys"] = {"chunks": (1, sz["output_channel"]) + spatial_chunk[1:]}
    if "y_pred_phys" in ds_batch:
        encoding["y_pred_phys"] = {"chunks": (1, sz["output_channel"]) + spatial_chunk[1:]}
    for scalar_var in ("metric_acc", "metric_rmse", "metric_mse", "metric_mae", "global_index", "case_name", "case_path"):
        if scalar_var in ds_batch:
            encoding[scalar_var] = {"chunks": (n_sample,)}

    if is_first_batch:
        ds_batch.to_zarr(store_path, mode="w", encoding=encoding, zarr_format=2)
    else:
        ds_batch.to_zarr(store_path, mode="a", append_dim="sample", zarr_format=2)


def main() -> None:
    parser = ArgumentParser(description="Trainer-like test evaluation for WoFS DA increment model")
    parser.add_argument("config", help="Path to YAML config used for training")
    parser.add_argument("--test-path", default=None, help="Glob path for test files (overrides eval.test_path)")
    parser.add_argument("--test-dyn-path", default=None, help="Optional glob path for test dynamic forcing files")
    parser.add_argument("--test-date-range", nargs=2, default=None, help="Optional YYYYMMDD YYYYMMDD range for test selection")
    parser.add_argument(
        "--delta-step",
        type=int,
        default=None,
        help="Fixed timestep delta for inference (1, 2, or 3). Overrides eval.delta_step in config. Defaults to 1.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for evaluation (defaults to trainer.valid_batch_size)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max number of samples to evaluate")
    parser.add_argument(
        "--save-inference",
        default=None,
        help="Output sample-store path (overrides eval.save_zarr_path / eval.save_netcdf_path)",
    )
    parser.add_argument(
        "--save-metrics-csv",
        default=None,
        help="Output CSV path (overrides eval.save_metrics_csv_path or eval.save_forecast/output_metrics_name)",
    )
    parser.add_argument(
        "--save-physical",
        default=None,
        help="Output physical-space Zarr store path (overrides eval.save_physical_zarr_path). "
             "Defaults to <save-inference path with _physical suffix>.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        help="Evaluate this many stochastic ensemble members. Defaults to eval.ensemble_size, predict.ensemble_size, then trainer.ensemble_size.",
    )
    parser.add_argument(
        "--save-ensemble-members",
        action="store_true",
        help="Store normalized y_pred_members with an ensemble_member dimension in the inference Zarr.",
    )
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="Do not store ensemble members even when ensemble_size > 1.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    conf = credit_main_parser(conf, parse_training=True, parse_predict=False, print_summary=False)
    _strip_deprecated_concentration_config(conf)
    _validate_eval_transform_config(conf)
    _sync_prognostic_levels(conf)
    _configure_aurora_da_model(conf)
    eval_conf = conf.get("eval", {}) if isinstance(conf.get("eval", {}), dict) else {}
    predict_conf = conf.get("predict", {}) if isinstance(conf.get("predict", {}), dict) else {}

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])
    default_output_dir = os.path.expandvars(
        eval_conf.get("save_forecast", predict_conf.get("save_forecast", conf["save_loc"]))
    )
    os.makedirs(default_output_dir, exist_ok=True)

    output_netcdf_name = str(eval_conf.get("output_inference_name", "trainer_like_test_samples.zarr"))
    output_metrics_name = str(eval_conf.get("output_metrics_name", "trainer_like_test_metrics.csv"))

    save_netcdf = (
        args.save_inference
        if args.save_inference
        else eval_conf.get(
            "save_zarr_path",
            eval_conf.get("save_netcdf_path", os.path.join(default_output_dir, output_netcdf_name)),
        )
    )
    save_metrics_csv = (
        args.save_metrics_csv
        if args.save_metrics_csv
        else eval_conf.get("save_metrics_csv_path", os.path.join(default_output_dir, output_metrics_name))
    )

    if str(save_netcdf).endswith(".nc"):
        save_netcdf = f"{save_netcdf[:-3]}.zarr"
        logger.info("Samples store path ends with .nc; using Zarr store path %s", save_netcdf)

    # Physical-space Zarr: derive default from normalized-store path
    _physical_default = str(save_netcdf).removesuffix(".zarr") + "_physical.zarr"
    save_physical = (
        args.save_physical
        if args.save_physical
        else eval_conf.get("save_physical_zarr_path", _physical_default)
    )
    save_physical_enabled = bool(eval_conf.get("save_physical", True))

    seed_everything(conf["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_path = args.test_path if args.test_path else eval_conf.get("test_path")
    if not test_path:
        raise ValueError("Missing test file glob. Provide --test-path or set eval.test_path in config.")

    test_dyn_path = args.test_dyn_path if args.test_dyn_path is not None else eval_conf.get("test_dyn_path")
    if args.test_date_range is not None:
        test_date_range = tuple(args.test_date_range)
    else:
        configured_range = eval_conf.get("test_date_range", eval_conf.get("custom_date_range", None))
        test_date_range = tuple(configured_range) if configured_range is not None else None

    # Fix the delta step for evaluation (not random — use a single deterministic delta).
    eval_delta = args.delta_step if args.delta_step is not None else int(eval_conf.get("delta_step", 1))
    conf["data"]["delta_steps"] = [eval_delta]
    conf["data"]["delta_probs"] = [1.0]
    logger.info("Evaluation using fixed delta_step=%d", eval_delta)

    dataset = _build_test_dataset(
        conf=conf,
        test_path=os.path.expandvars(test_path),
        test_dyn_path=os.path.expandvars(test_dyn_path) if test_dyn_path else None,
        date_range=test_date_range,
    )

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(eval_conf.get("batch_size", conf["trainer"].get("valid_batch_size", 1)))
    )
    num_workers = int(eval_conf.get("num_workers", 0))
    shuffle = bool(eval_conf.get("shuffle", False))
    drop_last = bool(eval_conf.get("drop_last", False))
    persistent_workers = bool(eval_conf.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(eval_conf.get("prefetch_factor", 2))
    mp_context_name = str(eval_conf.get("dataloader_mp_context", "spawn"))
    loader_seed = eval_conf.get("loader_seed", None)

    loader_generator = None
    if loader_seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(loader_seed))

    if num_workers > 0:
        try:
            mp_context = mp.get_context(mp_context_name)
        except ValueError as exc:
            raise ValueError(
                f"Invalid eval.dataloader_mp_context={mp_context_name}. Choose from fork, spawn, forkserver"
            ) from exc
    else:
        mp_context = None

    max_samples = args.max_samples if args.max_samples is not None else eval_conf.get("max_samples", None)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init_fn,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context=mp_context,
        generator=loader_generator,
    )

    logger.info(
        "DataLoader settings: batch_size=%s num_workers=%s shuffle=%s drop_last=%s mp_context=%s loader_seed=%s",
        batch_size,
        num_workers,
        shuffle,
        drop_last,
        mp_context_name if num_workers > 0 else "none",
        str(loader_seed) if loader_seed is not None else "none",
    )

    model = _load_eval_model(conf, device)
    metrics = LatWeightedMetrics(conf)

    residual_prediction = bool(conf["trainer"].get("residual_prediction", False))
    varnum_diag = len(conf["data"].get("diagnostic_variables", []))
    n_prog_channels = len(conf["data"]["variables"]) * int(conf["model"]["param_interior"]["levels"])
    ensemble_size = (
        int(args.ensemble_size)
        if args.ensemble_size is not None
        else int(eval_conf.get("ensemble_size", predict_conf.get("ensemble_size", conf["trainer"].get("ensemble_size", 1))))
    )
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be >= 1")
    conf.setdefault("trainer", {})["ensemble_size"] = ensemble_size
    conf.setdefault("predict", {})["ensemble_size"] = ensemble_size
    save_ensemble_members = bool(args.save_ensemble_members or eval_conf.get("save_ensemble_members", False))
    if args.mean_only:
        save_ensemble_members = False
    logger.info("Evaluation ensemble_size=%d save_ensemble_members=%s", ensemble_size, save_ensemble_members)

    save_netcdf_enabled = bool(eval_conf.get("save_netcdf", True))
    overwrite_samples_store = bool(eval_conf.get("overwrite_samples_store", True))

    if save_netcdf_enabled and overwrite_samples_store and os.path.exists(save_netcdf):
        if os.path.isdir(save_netcdf):
            shutil.rmtree(save_netcdf)
        else:
            os.remove(save_netcdf)

    if save_physical_enabled and overwrite_samples_store and os.path.exists(save_physical):
        if os.path.isdir(save_physical):
            shutil.rmtree(save_physical)
        else:
            os.remove(save_physical)

    metrics_dir = os.path.dirname(save_metrics_csv)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    metrics_fieldnames = ["sample", "global_index", "case", "case_path", "valid_time", "acc", "rmse", "mse", "mae"]
    metrics_fp = open(save_metrics_csv, "w", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(metrics_fp, fieldnames=metrics_fieldnames)
    metrics_writer.writeheader()

    processed_samples = 0
    sum_acc = 0.0
    sum_rmse = 0.0
    sum_mse = 0.0
    sum_mae = 0.0

    first_zarr_batch = True
    first_physical_batch = True

    total_dataset_samples = len(dataset)
    total_expected_samples = min(total_dataset_samples, int(max_samples)) if max_samples is not None else total_dataset_samples
    total_batches = (total_expected_samples + batch_size - 1) // batch_size
    logger.info("Starting evaluation: %s total dataset samples, expecting ~%s samples in ~%s batches",
                total_dataset_samples, total_expected_samples, total_batches)
    eval_start_time = time.monotonic()

    try:
        with torch.no_grad():
            for batch_id, batch in enumerate(loader):
                batch_start = time.monotonic()
                if max_samples is not None and processed_samples >= int(max_samples):
                    logger.info("Reached max_samples=%s; stopping early", int(max_samples))
                    break

                logger.info(
                    "[Batch %d/%d | %d/%d samples] Starting model inference ...",
                    batch_id + 1, total_batches, processed_samples, total_expected_samples,
                )

                t0 = time.monotonic()
                x, x_boundary, x_time_encode, y_true = _sample_to_model_inputs(batch, device)
                x_model = _repeat_for_ensemble(x, ensemble_size)
                x_boundary_model = _repeat_for_ensemble(x_boundary, ensemble_size)
                x_time_encode_model = _repeat_for_ensemble(x_time_encode, ensemble_size)
                y_pred = model(
                    x_model.float(),
                    x_boundary_model.float(),
                    x_time_encode_model.float(),
                    ensemble_size=ensemble_size,
                )
                y_pred = _apply_residual_prediction(y_pred, x_model, residual_prediction, varnum_diag)
                y_pred_mean = _ensemble_mean(y_pred, y_true)
                y_pred_members_all = _ensemble_members(y_pred, y_true)
                y_pred_members = y_pred_members_all if save_ensemble_members else None
                logger.info("  model inference done in %.2fs", time.monotonic() - t0)

                t0 = time.monotonic()
                x_np = x.detach().cpu().numpy()
                x_boundary_np = x_boundary.detach().cpu().numpy()
                y_true_np = y_true.detach().cpu().numpy()
                y_pred_np = y_pred_mean.detach().cpu().numpy()
                y_pred_members_np = y_pred_members.detach().cpu().numpy() if y_pred_members is not None else None
                logger.info("  GPU->CPU transfer done in %.2fs", time.monotonic() - t0)

                index_values = batch["index"].detach().cpu().numpy().astype(np.int64).tolist()

                if max_samples is None:
                    take_n = len(index_values)
                else:
                    take_n = min(len(index_values), int(max_samples) - processed_samples)

                if take_n <= 0:
                    break

                if take_n < len(index_values):
                    x_np = x_np[:take_n]
                    x_boundary_np = x_boundary_np[:take_n]
                    y_true_np = y_true_np[:take_n]
                    y_pred_np = y_pred_np[:take_n]
                    if y_pred_members_np is not None:
                        y_pred_members_np = y_pred_members_np[:take_n]
                    index_values = index_values[:take_n]

                batch_case_names: list[str] = []
                batch_case_paths: list[str] = []
                batch_valid_times: list[np.datetime64] = []
                batch_metric_rows: list[dict] = []

                for local_idx, global_index in enumerate(index_values):
                    case_name, case_path, valid_time = _collect_sample_metadata(dataset, int(global_index))

                    sample_y_pred = (
                        y_pred_members_all[local_idx]
                        if y_pred_members_all is not None
                        else y_pred_mean[local_idx : local_idx + 1]
                    )
                    sample_y_true = y_true[local_idx : local_idx + 1]
                    sample_metrics = metrics(sample_y_pred.float(), sample_y_true.float(), forecast_datetime=1)

                    row = {
                        "sample": processed_samples + local_idx,
                        "global_index": int(global_index),
                        "case": case_name,
                        "case_path": case_path,
                        "valid_time": str(valid_time),
                        "acc": float(sample_metrics["acc"]),
                        "rmse": float(sample_metrics["rmse"]),
                        "mse": float(sample_metrics["mse"]),
                        "mae": float(sample_metrics["mae"]),
                    }
                    metrics_writer.writerow(row)

                    sum_acc += row["acc"]
                    sum_rmse += row["rmse"]
                    sum_mse += row["mse"]
                    sum_mae += row["mae"]

                    batch_case_names.append(case_name)
                    batch_case_paths.append(case_path)
                    batch_valid_times.append(valid_time)
                    batch_metric_rows.append(row)

                if save_netcdf_enabled:
                    t0 = time.monotonic()
                    ds_batch = _build_batch_dataset_for_zarr(
                        conf=conf,
                        sample_offset=processed_samples,
                        x_inputs=x_np,
                        x_boundaries=x_boundary_np,
                        y_trues=y_true_np,
                        y_preds=y_pred_np,
                        y_pred_members=y_pred_members_np,
                        sample_indices=[int(i) for i in index_values],
                        case_names=batch_case_names,
                        case_paths=batch_case_paths,
                        valid_times=batch_valid_times,
                        per_sample_metrics=batch_metric_rows,
                    )
                    _append_batch_to_zarr(save_netcdf, ds_batch, is_first_batch=first_zarr_batch)
                    first_zarr_batch = False
                    logger.info("  normalized zarr write done in %.2fs", time.monotonic() - t0)

                if save_physical_enabled:
                    x_prog_np = x_np[:, :n_prog_channels, ...]
                    y_true_phys_np = _denormalize_prog_increments(dataset, y_true_np, x_prog_np)
                    y_pred_phys_np = _denormalize_prog_increments(dataset, y_pred_np, x_prog_np)
                    t0 = time.monotonic()
                    ds_phys = _build_batch_physical_dataset_for_zarr(
                        conf=conf,
                        sample_offset=processed_samples,
                        y_true_phys=y_true_phys_np,
                        y_pred_phys=y_pred_phys_np,
                        sample_indices=[int(i) for i in index_values],
                        case_names=batch_case_names,
                        case_paths=batch_case_paths,
                        valid_times=batch_valid_times,
                    )
                    _append_batch_to_zarr(save_physical, ds_phys, is_first_batch=first_physical_batch)
                    first_physical_batch = False
                    logger.info("  physical zarr write done in %.2fs", time.monotonic() - t0)

                processed_samples += take_n
                elapsed = time.monotonic() - eval_start_time
                rate = processed_samples / elapsed if elapsed > 0 else 0.0
                remaining = (total_expected_samples - processed_samples) / rate if rate > 0 else float("inf")
                logger.info(
                    "[Batch %d/%d] batch done in %.2fs | total %d/%d samples | %.2f samples/s | ETA %.0fs",
                    batch_id + 1, total_batches,
                    time.monotonic() - batch_start,
                    processed_samples, total_expected_samples,
                    rate, remaining,
                )

        if processed_samples == 0:
            raise RuntimeError("No samples were evaluated. Check --test-path and --test-date-range.")

        aggregate_row = {
            "sample": "aggregate_mean",
            "global_index": -1,
            "case": "ALL",
            "case_path": "",
            "valid_time": "",
            "acc": float(sum_acc / processed_samples),
            "rmse": float(sum_rmse / processed_samples),
            "mse": float(sum_mse / processed_samples),
            "mae": float(sum_mae / processed_samples),
        }
        metrics_writer.writerow(aggregate_row)
    finally:
        metrics_fp.close()

    if save_netcdf_enabled:
        logger.info("Saved plotting Zarr store to %s", save_netcdf)
    if save_physical_enabled:
        logger.info("Saved physical-space Zarr store to %s", save_physical)
    logger.info("Saved trainer-like test metrics CSV to %s", save_metrics_csv)
    logger.info(
        "Aggregate normalized-space metrics: acc=%.6f rmse=%.6f mse=%.6f mae=%.6f",
        aggregate_row["acc"],
        aggregate_row["rmse"],
        aggregate_row["mse"],
        aggregate_row["mae"],
    )


if __name__ == "__main__":
    main()
