from __future__ import annotations

import copy
import logging
import os
import re
import warnings
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import xarray as xr
import yaml
from pandas.errors import EmptyDataError

from credit.data import reshape_only
from credit.datasets.wrf_wofs_da_increment import WoFSDAIncrementDataset
from credit.distributed import distributed_model_wrapper, get_rank_info, setup
from credit.metrics import LatWeightedMetrics
from credit.models import load_model
from credit.models.checkpoint import load_model_state, load_state_dict_error_handler
from credit.parser import credit_main_parser
from credit.seed import seed_everything

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _extract_case_date(file_path: str) -> str | None:
    match = re.search(r"wofs_(\d{8})_\d{4}_mem\d+\.zarr$", Path(file_path).name)
    return match.group(1) if match else None


def _select_files(glob_pattern: str, date_range: list[str] | tuple[str, str] | None = None) -> list[str]:
    files = sorted(glob(glob_pattern))
    if date_range is None:
        return files

    start_date, end_date = str(date_range[0]), str(date_range[1])
    return [
        file_path
        for file_path in files
        if (case_date := _extract_case_date(file_path)) is not None and start_date <= case_date <= end_date
    ]


def _build_case_dataset(case_path: str, conf: dict) -> WoFSDAIncrementDataset:
    param_interior = {
        "varname_upper_air": conf["data"]["variables"],
        "varname_context_upper_air": conf["data"]["context_upper_air_variables"],
        "varname_dyn_forcing": conf["data"]["dynamic_forcing_variables"],
        "filenames": [case_path],
        "filename_dyn_forcing": [case_path] if conf["data"].get("save_loc_dynamic_forcing") else None,
        "history_len": conf["data"]["history_len"],
        "forecast_len": conf["data"]["forecast_len"],
    }
    param_outside = {
        "varname_upper_air": conf["data"]["observation_variables"],
    }

    return WoFSDAIncrementDataset(param_interior, param_outside, conf=conf, seed=conf["seed"])


def _distributed_conf(conf: dict) -> dict:
    rollout_conf = copy.deepcopy(conf)
    rollout_conf.setdefault("trainer", {})
    rollout_conf["trainer"]["mode"] = rollout_conf["predict"].get("mode", "none")
    rollout_conf["trainer"].setdefault("activation_checkpoint", False)
    return rollout_conf


def _load_rollout_model(conf: dict, device: torch.device) -> torch.nn.Module:
    mode = conf["predict"].get("mode", "none")
    rollout_conf = _distributed_conf(conf)

    if mode == "none":
        model = load_model(conf, load_weights=True).to(device)
        model.eval()
        return model

    if mode == "ddp":
        model = load_model(conf).to(device)
        model = distributed_model_wrapper(rollout_conf, model, device)

        save_loc = os.path.expandvars(conf["save_loc"])
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device)
        load_msg = model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
        load_state_dict_error_handler(load_msg)
        model.eval()
        return model

    if mode == "fsdp":
        model = load_model(conf, load_weights=True).to(device)
        model = distributed_model_wrapper(rollout_conf, model, device)
        model = load_model_state(rollout_conf, model, device)
        model.eval()
        return model

    raise ValueError(f"Unsupported predict.mode for WoFS DA rollout: {mode}")


def _sample_to_model_inputs(sample: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = reshape_only(sample["x"].unsqueeze(0)).to(device)

    if "x_forcing_static" in sample:
        x_forcing = sample["x_forcing_static"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
        x = torch.cat((x, x_forcing), dim=1)

    x_boundary = reshape_only(sample["x_boundary"].unsqueeze(0)).to(device)
    x_time_encode = sample["x_time_encode"].unsqueeze(0).to(device)
    return x, x_boundary, x_time_encode


def _target_tensor(sample: dict, device: torch.device) -> torch.Tensor:
    return reshape_only(sample["y"].unsqueeze(0)).to(device)


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


def _case_name(case_path: str) -> str:
    return Path(case_path).stem


def _global_ssim(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    channel_scores: list[float] = []

    for channel_index in range(pred.shape[0]):
        x = pred[channel_index].astype(np.float64)
        y = target[channel_index].astype(np.float64)

        dynamic_range = max(float(np.max(x) - np.min(x)), float(np.max(y) - np.min(y)), eps)
        c1 = (0.01 * dynamic_range) ** 2
        c2 = (0.03 * dynamic_range) ** 2

        mu_x = np.mean(x)
        mu_y = np.mean(y)
        sigma_x = np.var(x)
        sigma_y = np.var(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))

        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2) + eps
        channel_scores.append(float(numerator / denominator))

    return float(np.mean(channel_scores))


def _global_psnr(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    mse = float(np.mean((pred - target) ** 2))
    data_range = max(float(np.max(target) - np.min(target)), eps)
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse + eps))


def _flatten_var_level(data_by_var: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(data_by_var, axis=0)


def _make_case_dataset_output(
    case_name: str,
    times: list[np.datetime64],
    level_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    predicted_delta: dict[str, list[np.ndarray]],
    true_delta: dict[str, list[np.ndarray]],
    predicted_state: dict[str, list[np.ndarray]],
    true_state: dict[str, list[np.ndarray]],
) -> xr.Dataset:
    ds = xr.Dataset(coords={"time": np.array(times), "level": level_coords, "y": y_coords, "x": x_coords})

    for var_name in predicted_delta:
        lower_name = var_name.lower()

        ds[f"pred_delta_{lower_name}"] = xr.DataArray(
            np.stack(predicted_delta[var_name], axis=0),
            dims=["time", "level", "y", "x"],
        )
        ds[f"true_delta_{lower_name}"] = xr.DataArray(
            np.stack(true_delta[var_name], axis=0),
            dims=["time", "level", "y", "x"],
        )
        ds[f"pred_state_{lower_name}"] = xr.DataArray(
            np.stack(predicted_state[var_name], axis=0),
            dims=["time", "level", "y", "x"],
        )
        ds[f"true_state_{lower_name}"] = xr.DataArray(
            np.stack(true_state[var_name], axis=0),
            dims=["time", "level", "y", "x"],
        )

    ds.attrs["case"] = case_name
    ds.attrs["description"] = "WoFS DA rollout outputs with predicted/true deltas and autoregressive predicted states"
    return ds


def rollout_case_da(case_path: str, conf: dict, model: torch.nn.Module, device: torch.device) -> tuple[pd.DataFrame, str | None]:
    dataset = _build_case_dataset(case_path, conf)
    metrics = LatWeightedMetrics(conf)

    varnames = conf["data"]["variables"]
    num_levels = int(conf["data"]["levels"])
    varnum_diag = len(conf["data"]["diagnostic_variables"])
    residual_prediction = conf["trainer"].get("residual_prediction", False)
    compute_psnr_ssim = bool(conf.get("predict", {}).get("compute_psnr_ssim", True))

    case_name = _case_name(case_path)
    case_results: list[dict] = []

    predicted_delta_by_var: dict[str, list[np.ndarray]] = {var: [] for var in varnames}
    true_delta_by_var: dict[str, list[np.ndarray]] = {var: [] for var in varnames}
    predicted_state_by_var: dict[str, list[np.ndarray]] = {var: [] for var in varnames}
    true_state_by_var: dict[str, list[np.ndarray]] = {var: [] for var in varnames}

    time_values: list[np.datetime64] = []
    level_coords: np.ndarray | None = None
    y_coords: np.ndarray | None = None
    x_coords: np.ndarray | None = None

    corrected_prev_state: dict[str, np.ndarray] | None = None

    for step in range(len(dataset)):
        sample = dataset[step]

        x_model, x_boundary, x_time_encode = _sample_to_model_inputs(sample, device)
        y_true_norm = _target_tensor(sample, device)

        with torch.no_grad():
            y_pred_norm = model(x_model.float(), x_boundary.float(), x_time_encode.float())
            y_pred_norm = _apply_residual_prediction(y_pred_norm, x_model, residual_prediction, varnum_diag)

        file_idx, ind_in_file = dataset._locate_index(step)
        t0 = ind_in_file
        t1 = ind_in_file + 1
        upper_ds = dataset._get_upper_ds(file_idx)
        chunk = upper_ds.isel(time=slice(t0, t1 + 1)).load()

        first_var = varnames[0]
        first_t0 = chunk[first_var].isel(time=0)
        first_t1 = chunk[first_var].isel(time=1)

        if level_coords is None:
            level_dim, y_dim, x_dim = first_t0.dims
            level_coords = first_t0[level_dim].values
            y_coords = first_t0[y_dim].values
            x_coords = first_t0[x_dim].values

        time_values.append(chunk["time"].values[1])

        if corrected_prev_state is None:
            corrected_prev_state = {var: chunk[var].isel(time=0).values.astype(np.float32) for var in varnames}

        pred_delta_list = []
        true_delta_list = []

        y_pred_norm_np = y_pred_norm.detach().cpu().numpy()

        for var_index, var_name in enumerate(varnames):
            start_idx = var_index * num_levels
            end_idx = start_idx + num_levels

            pred_delta_norm = y_pred_norm_np[0, start_idx:end_idx, 0, :, :]
            base_state_true_t0 = chunk[var_name].isel(time=0).values
            pred_delta_phys = dataset.denormalize_increment(pred_delta_norm, base_state_true_t0, var_name).astype(np.float32)

            true_t0 = chunk[var_name].isel(time=0).values.astype(np.float32)
            true_t1 = chunk[var_name].isel(time=1).values.astype(np.float32)
            true_delta_phys = (true_t1 - true_t0).astype(np.float32)

            corrected_curr = (corrected_prev_state[var_name] + pred_delta_phys).astype(np.float32)

            predicted_delta_by_var[var_name].append(pred_delta_phys)
            true_delta_by_var[var_name].append(true_delta_phys)
            predicted_state_by_var[var_name].append(corrected_curr)
            true_state_by_var[var_name].append(true_t1)

            corrected_prev_state[var_name] = corrected_curr

            pred_delta_list.append(pred_delta_phys)
            true_delta_list.append(true_delta_phys)

        metrics_dict = metrics(y_pred_norm.detach().cpu(), y_true_norm.detach().cpu(), forecast_datetime=step + 1)

        pred_delta_flat = _flatten_var_level(pred_delta_list)
        true_delta_flat = _flatten_var_level(true_delta_list)

        mse_delta_phys = float(np.mean((pred_delta_flat - true_delta_flat) ** 2))
        rmse_delta_phys = float(np.sqrt(mse_delta_phys))
        mae_delta_phys = float(np.mean(np.abs(pred_delta_flat - true_delta_flat)))

        result_row = {
            "case": case_name,
            "forecast_step": step + 1,
            "valid_time": str(chunk["time"].values[1]),
            "acc": float(metrics_dict["acc"]),
            "rmse": float(metrics_dict["rmse"]),
            "mse": float(metrics_dict["mse"]),
            "mae": float(metrics_dict["mae"]),
            "rmse_delta_phys": rmse_delta_phys,
            "mse_delta_phys": mse_delta_phys,
            "mae_delta_phys": mae_delta_phys,
        }

        if compute_psnr_ssim:
            result_row["psnr_delta_phys"] = _global_psnr(pred_delta_flat, true_delta_flat)
            result_row["ssim_delta_phys"] = _global_ssim(pred_delta_flat, true_delta_flat)

        case_results.append(result_row)

    netcdf_path: str | None = None
    save_netcdf = bool(conf.get("predict", {}).get("save_netcdf", True))
    if save_netcdf and time_values and level_coords is not None and y_coords is not None and x_coords is not None:
        output_ds = _make_case_dataset_output(
            case_name=case_name,
            times=time_values,
            level_coords=level_coords,
            y_coords=y_coords,
            x_coords=x_coords,
            predicted_delta=predicted_delta_by_var,
            true_delta=true_delta_by_var,
            predicted_state=predicted_state_by_var,
            true_state=true_state_by_var,
        )

        case_dir = os.path.join(conf["predict"]["save_forecast"], case_name)
        os.makedirs(case_dir, exist_ok=True)
        netcdf_path = os.path.join(case_dir, f"rollout_da_{case_name}.nc")
        output_ds.to_netcdf(netcdf_path)

    return pd.DataFrame(case_results), netcdf_path


def _aggregate_rank_metrics(out_dir: str, world_size: int, output_name: str) -> str:
    rank_paths = [
        os.path.join(out_dir, f"rollout_da_metrics_rank{rank}.csv")
        for rank in range(world_size)
        if os.path.exists(os.path.join(out_dir, f"rollout_da_metrics_rank{rank}.csv"))
    ]

    if rank_paths:
        rank_frames = []
        for path in rank_paths:
            if os.path.getsize(path) == 0:
                continue
            try:
                rank_df = pd.read_csv(path)
            except EmptyDataError:
                continue
            if not rank_df.empty:
                rank_frames.append(rank_df)
        summary_df = pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame()
    else:
        summary_df = pd.DataFrame()

    summary_path = os.path.join(out_dir, output_name)
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    parser = ArgumentParser(description="Evaluate WoFS DA increment model and save rollout-style DA outputs")
    parser.add_argument("config", help="Path to the WoFS DA YAML config")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit on number of cases to process")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["none", "ddp", "fsdp"],
        default=None,
        help="Override predict.mode for evaluation",
    )
    parser.add_argument(
        "--backend",
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="Distributed backend when using ddp/fsdp",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Filename for merged metrics CSV under predict.save_forecast",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    conf = credit_main_parser(conf, parse_training=False, parse_predict=False, print_summary=False)
    conf.setdefault("predict", {})

    if args.mode is not None:
        conf["predict"]["mode"] = args.mode
    conf["predict"].setdefault("mode", "none")
    conf["predict"].setdefault("compute_psnr_ssim", True)
    conf["predict"].setdefault("eval_split", "valid")
    conf["predict"].setdefault("output_metrics_name", "rollout_da_metrics.csv")

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])
    conf["predict"]["save_forecast"] = os.path.expandvars(conf["predict"]["save_forecast"])
    os.makedirs(conf["predict"]["save_forecast"], exist_ok=True)

    seed_everything(conf["seed"])
    local_rank, world_rank, world_size = get_rank_info(conf["predict"]["mode"])

    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]
    if distributed:
        setup(world_rank, world_size, conf["predict"]["mode"], backend=args.backend)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    model = _load_rollout_model(conf, device)

    date_range = conf.get("predict", {}).get("custom_date_range")
    if date_range is None:
        eval_split = str(conf.get("predict", {}).get("eval_split", "valid")).lower()
        if eval_split == "train":
            date_range = conf["data"].get("train_date_range")
        elif eval_split in ["valid", "validation", "test"]:
            date_range = conf["data"].get("valid_date_range")

    case_files = _select_files(conf["data"]["save_loc"], date_range)
    if args.max_cases is not None:
        case_files = case_files[: args.max_cases]
    elif conf.get("predict", {}).get("max_cases") is not None:
        case_files = case_files[: int(conf["predict"]["max_cases"])]

    case_files = case_files[world_rank::world_size]

    if not case_files:
        logger.warning("No WoFS case files matched DA rollout selection on rank %s", world_rank)

    all_results = []
    for case_path in case_files:
        logger.info("Rank %s evaluating DA case %s", world_rank, case_path)
        case_df, netcdf_path = rollout_case_da(case_path, conf, model, device)
        if netcdf_path is not None:
            logger.info("Rank %s saved DA rollout file %s", world_rank, netcdf_path)
        all_results.append(case_df)

    partial_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    partial_path = os.path.join(conf["predict"]["save_forecast"], f"rollout_da_metrics_rank{world_rank}.csv")
    partial_df.to_csv(partial_path, index=False)

    if distributed:
        dist.barrier()

    if world_rank == 0:
        output_name = args.output_name if args.output_name else conf["predict"]["output_metrics_name"]
        summary_path = _aggregate_rank_metrics(conf["predict"]["save_forecast"], world_size, output_name)
        logger.info("Saved DA rollout metrics to %s", summary_path)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
