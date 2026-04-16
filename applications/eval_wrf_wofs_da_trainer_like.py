from __future__ import annotations

import logging
import os
import re
import csv
import shutil
import multiprocessing as mp
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
from credit.parser import credit_main_parser
from credit.seed import seed_everything

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
    model = load_model(conf, load_weights=True).to(device)
    model.eval()
    return model


def _sample_to_model_inputs(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = reshape_only(batch["x"]).to(device, non_blocking=True)

    if "x_forcing_static" in batch:
        x_forcing = batch["x_forcing_static"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4)
        x = torch.cat((x, x_forcing), dim=1)

    x_boundary = reshape_only(batch["x_boundary"]).to(device, non_blocking=True)
    x_time_encode = batch["x_time_encode"].to(device, non_blocking=True)
    y_true = reshape_only(batch["y"]).to(device, non_blocking=True)
    return x, x_boundary, x_time_encode, y_true


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

    ds = xr.Dataset(
        data_vars={
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
        },
        coords={
            "sample": np.arange(sample_offset, sample_offset + n_samples, dtype=np.int64),
            "time": np.arange(n_time, dtype=np.int64),
            "y": np.arange(n_y, dtype=np.int64),
            "x": np.arange(n_x, dtype=np.int64),
            "input_channel": np.array(input_channels, dtype=str),
            "boundary_channel": np.array(boundary_channels, dtype=str),
            "output_channel": np.array(output_channels, dtype=str),
            "valid_time": (["sample"], np.array(valid_times, dtype="datetime64[ns]")),
        },
    )

    ds.attrs["description"] = "Trainer-like WoFS DA test samples with input, innovation boundary, target increment, and prediction"
    ds.attrs["metric_space"] = "normalized increment space (same space as training metrics)"
    ds.attrs["save_loc"] = str(conf.get("save_loc", ""))
    ds.attrs["variables"] = ",".join(conf["data"].get("variables", []))
    return ds


def _append_batch_to_zarr(store_path: str, ds_batch: xr.Dataset, is_first_batch: bool) -> None:
    store_dir = os.path.dirname(store_path)
    if store_dir:
        os.makedirs(store_dir, exist_ok=True)
    encoding = {
        "x_input": {"chunks": (1, ds_batch.sizes["input_channel"], ds_batch.sizes["time"], ds_batch.sizes["y"], ds_batch.sizes["x"])},
        "x_boundary": {
            "chunks": (
                1,
                ds_batch.sizes["boundary_channel"],
                ds_batch.sizes["time"],
                ds_batch.sizes["y"],
                ds_batch.sizes["x"],
            )
        },
        "y_true": {"chunks": (1, ds_batch.sizes["output_channel"], ds_batch.sizes["time"], ds_batch.sizes["y"], ds_batch.sizes["x"])},
        "y_pred": {"chunks": (1, ds_batch.sizes["output_channel"], ds_batch.sizes["time"], ds_batch.sizes["y"], ds_batch.sizes["x"])},
        "metric_acc": {"chunks": (max(1, ds_batch.sizes["sample"]),)},
        "metric_rmse": {"chunks": (max(1, ds_batch.sizes["sample"]),)},
        "metric_mse": {"chunks": (max(1, ds_batch.sizes["sample"]),)},
        "metric_mae": {"chunks": (max(1, ds_batch.sizes["sample"]),)},
        "global_index": {"chunks": (max(1, ds_batch.sizes["sample"]),)},
        "case_name": {"chunks": (max(1, ds_batch.sizes["sample"]),)},
        "case_path": {"chunks": (max(1, ds_batch.sizes["sample"]),)},
    }
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    conf = credit_main_parser(conf, parse_training=True, parse_predict=False, print_summary=False)
    _sync_prognostic_levels(conf)
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

    save_netcdf_enabled = bool(eval_conf.get("save_netcdf", True))
    overwrite_samples_store = bool(eval_conf.get("overwrite_samples_store", True))

    if save_netcdf_enabled and overwrite_samples_store and os.path.exists(save_netcdf):
        if os.path.isdir(save_netcdf):
            shutil.rmtree(save_netcdf)
        else:
            os.remove(save_netcdf)

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

    try:
        with torch.no_grad():
            for batch_id, batch in enumerate(loader):
                if max_samples is not None and processed_samples >= int(max_samples):
                    logger.info("Reached max_samples=%s; stopping early", int(max_samples))
                    break

                x, x_boundary, x_time_encode, y_true = _sample_to_model_inputs(batch, device)
                y_pred = model(x.float(), x_boundary.float(), x_time_encode.float())
                y_pred = _apply_residual_prediction(y_pred, x, residual_prediction, varnum_diag)

                x_np = x.detach().cpu().numpy()
                x_boundary_np = x_boundary.detach().cpu().numpy()
                y_true_np = y_true.detach().cpu().numpy()
                y_pred_np = y_pred.detach().cpu().numpy()

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
                    index_values = index_values[:take_n]

                batch_case_names: list[str] = []
                batch_case_paths: list[str] = []
                batch_valid_times: list[np.datetime64] = []
                batch_metric_rows: list[dict] = []

                for local_idx, global_index in enumerate(index_values):
                    case_name, case_path, valid_time = _collect_sample_metadata(dataset, int(global_index))

                    sample_y_pred = y_pred[local_idx : local_idx + 1]
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
                    ds_batch = _build_batch_dataset_for_zarr(
                        conf=conf,
                        sample_offset=processed_samples,
                        x_inputs=x_np,
                        x_boundaries=x_boundary_np,
                        y_trues=y_true_np,
                        y_preds=y_pred_np,
                        sample_indices=[int(i) for i in index_values],
                        case_names=batch_case_names,
                        case_paths=batch_case_paths,
                        valid_times=batch_valid_times,
                        per_sample_metrics=batch_metric_rows,
                    )
                    _append_batch_to_zarr(save_netcdf, ds_batch, is_first_batch=first_zarr_batch)
                    first_zarr_batch = False

                processed_samples += take_n

                if (batch_id + 1) % 20 == 0:
                    logger.info("Processed %s samples", processed_samples)

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
