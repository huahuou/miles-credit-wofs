from __future__ import annotations

import copy
import importlib.util
import logging
import os
import re
import shutil
import warnings
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from pathlib import Path
from types import ModuleType

import pandas as pd
import torch
import torch.distributed as dist
import xarray as xr
import yaml
from pandas.errors import EmptyDataError

from credit.data import concat_and_reshape, drop_var_from_dataset, reshape_only
from credit.metrics import LatWeightedMetrics
from credit.models import load_model
from credit.models.checkpoint import load_model_state, load_state_dict_error_handler
from credit.output import load_metadata, make_xarray, save_netcdf_clean
from credit.parser import credit_main_parser
from credit.seed import seed_everything
from credit.transforms.transforms_wrf import NormalizeWRF, ToTensorWRF

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def _load_wofs_singlestep_module() -> ModuleType:
    datasets_file = Path(__file__).resolve().parents[1] / "credit" / "datasets" / "wrf_wofs_singlestep.py"
    spec = importlib.util.spec_from_file_location("credit_wrf_wofs_singlestep_standalone", datasets_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dataset module from {datasets_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WoFSSingleStepDataset = _load_wofs_singlestep_module().WoFSSingleStepDataset


def _load_distributed_ops():
    from credit.distributed import distributed_model_wrapper, get_rank_info, setup

    return distributed_model_wrapper, get_rank_info, setup


def _calculate_crps_per_channel(ensemble_predictions: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    from credit.ensemble.crps import calculate_crps_per_channel

    return calculate_crps_per_channel(ensemble_predictions, y_true)


def _extract_case_date(file_path: str) -> str | None:
    match = re.search(r"wofs_(\d{8})_\d{4}_mem\d+\.zarr(?:\.zip)?$", Path(file_path).name)
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


def _repeat_to_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return torch.repeat_interleave(tensor, batch_size, dim=0)
    raise ValueError(f"Cannot broadcast batch size {tensor.shape[0]} to {batch_size}")


def _build_case_dataset(
    case_path: str,
    conf: dict,
    transform: NormalizeWRF | None = None,
) -> WoFSSingleStepDataset:
    param_interior = {
        "varname_upper_air": conf["data"]["variables"],
        "varname_surface": conf["data"]["surface_variables"],
        "varname_dyn_forcing": conf["data"]["dynamic_forcing_variables"],
        "varname_forcing": conf["data"]["forcing_variables"],
        "varname_static": conf["data"]["static_variables"],
        "varname_diagnostic": conf["data"]["diagnostic_variables"],
        "filenames": [case_path],
        "filename_surface": [case_path] if conf["data"].get("save_loc_surface") else None,
        "filename_dyn_forcing": [case_path] if conf["data"].get("save_loc_dynamic_forcing") else None,
        "filename_forcing": conf["data"].get("save_loc_forcing"),
        "filename_static": conf["data"].get("save_loc_static"),
        "filename_diagnostic": [case_path] if conf["data"].get("save_loc_diagnostic") else None,
        "history_len": conf["data"]["history_len"],
        "forecast_len": 0,
    }
    param_outside = {
        "varname_upper_air": conf["data"]["boundary"]["variables"],
        "varname_surface": conf["data"]["boundary"].get("surface_variables", []),
        "history_len": conf["data"]["boundary"]["history_len"],
        "forecast_len": conf["data"]["boundary"]["forecast_len"],
    }

    return WoFSSingleStepDataset(param_interior, param_outside, transform=transform, seed=conf["seed"])


def _distributed_conf(conf: dict) -> dict:
    rollout_conf = copy.deepcopy(conf)
    rollout_conf.setdefault("trainer", {})
    rollout_conf["trainer"]["mode"] = rollout_conf["predict"]["mode"]
    rollout_conf["trainer"].setdefault("activation_checkpoint", False)
    return rollout_conf


def _load_rollout_model(conf: dict, device: torch.device) -> torch.nn.Module:
    mode = conf["predict"].get("mode", "none")
    rollout_conf = _distributed_conf(conf)

    if mode == "none":
        model = load_model(conf, load_weights=True).to(device)
        model.eval()
        return model

    distributed_model_wrapper, _, _ = _load_distributed_ops()

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

    raise ValueError(f"Unsupported predict.mode for WoFS rollout: {mode}")


def _sample_to_model_inputs(sample: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "x_surf" in sample:
        x = concat_and_reshape(sample["x"].unsqueeze(0), sample["x_surf"].unsqueeze(0)).to(device)
    else:
        x = reshape_only(sample["x"].unsqueeze(0)).to(device)

    if "x_forcing_static" in sample:
        x_forcing = sample["x_forcing_static"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
        x = torch.cat((x, x_forcing), dim=1)

    if "x_surf_boundary" in sample:
        x_boundary = concat_and_reshape(sample["x_boundary"].unsqueeze(0), sample["x_surf_boundary"].unsqueeze(0)).to(device)
    else:
        x_boundary = reshape_only(sample["x_boundary"].unsqueeze(0)).to(device)

    x_time_encode = sample["x_time_encode"].unsqueeze(0).to(device)
    return x, x_boundary, x_time_encode


def _compose_autoregressive_input(
    x_state: torch.Tensor,
    sample: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x_state.to(device)
    batch_size = x.shape[0]

    if "x_forcing_static" in sample:
        x_forcing = sample["x_forcing_static"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
        x_forcing = _repeat_to_batch(x_forcing, batch_size)
        x = torch.cat((x, x_forcing), dim=1)

    if "x_surf_boundary" in sample:
        x_boundary = concat_and_reshape(sample["x_boundary"].unsqueeze(0), sample["x_surf_boundary"].unsqueeze(0)).to(device)
    else:
        x_boundary = reshape_only(sample["x_boundary"].unsqueeze(0)).to(device)
    x_boundary = _repeat_to_batch(x_boundary, batch_size)

    x_time_encode = sample["x_time_encode"].unsqueeze(0).to(device)
    x_time_encode = _repeat_to_batch(x_time_encode, batch_size)
    return x, x_boundary, x_time_encode


def _target_tensor(sample: dict, device: torch.device) -> torch.Tensor:
    if "y_surf" in sample:
        y = concat_and_reshape(sample["y"].unsqueeze(0), sample["y_surf"].unsqueeze(0)).to(device)
    else:
        y = reshape_only(sample["y"].unsqueeze(0)).to(device)

    if "y_diag" in sample:
        y_diag = sample["y_diag"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
        y = torch.cat((y, y_diag), dim=1)
    return y


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
    name = Path(case_path).name
    if name.endswith(".zarr.zip"):
        return name[:-4]
    if name.endswith(".zarr"):
        return name
    return Path(case_path).stem


def _resolve_rollout_start_index(conf: dict) -> int:
    history_len = int(conf["data"]["history_len"])
    target_start_step = int(conf["data"].get("target_start_step", history_len))
    if target_start_step < 0:
        raise ValueError("data.target_start_step must be >= 0")
    return max(0, target_start_step - history_len)


def _lead_step_from_sample_index(sample_index: int, history_len: int) -> int:
    return sample_index + history_len


def _to_float_metrics(metrics_dict: dict) -> dict[str, float]:
    out = {}
    for key, value in metrics_dict.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu().item())
        else:
            out[key] = float(value)
    return out


def _single_prediction_to_dataset(
    y_pred_denorm: torch.Tensor,
    valid_time: datetime,
    y_coord,
    x_coord,
    conf: dict,
) -> xr.Dataset:
    xarray_outputs = make_xarray(y_pred_denorm, valid_time, y_coord, x_coord, conf)
    if isinstance(xarray_outputs, tuple):
        darray_upper_air, darray_single_level = xarray_outputs
        ds_upper = darray_upper_air.to_dataset(dim="vars")
        ds_single = darray_single_level.to_dataset(dim="vars")
        return xr.merge([ds_upper, ds_single])

    darray_upper_air = xarray_outputs
    return darray_upper_air.to_dataset(dim="vars")


def _prediction_to_dataset(
    y_pred_denorm: torch.Tensor,
    valid_time: datetime,
    y_coord,
    x_coord,
    conf: dict,
    save_ensemble_members: bool,
) -> xr.Dataset:
    if y_pred_denorm.shape[0] == 1:
        return _single_prediction_to_dataset(y_pred_denorm, valid_time, y_coord, x_coord, conf)

    member_datasets = []
    for member_index in range(y_pred_denorm.shape[0]):
        member_pred = y_pred_denorm[member_index : member_index + 1]
        member_ds = _single_prediction_to_dataset(member_pred, valid_time, y_coord, x_coord, conf)
        member_ds = member_ds.expand_dims({"ensemble_member_label": [member_index]})
        member_datasets.append(member_ds)

    members_ds = xr.concat(member_datasets, dim="ensemble_member_label")
    if save_ensemble_members:
        return members_ds
    return members_ds.mean(dim="ensemble_member_label")


def _apply_metadata(ds: xr.Dataset, meta_data: dict | bool) -> xr.Dataset:
    if meta_data is False:
        return ds

    for var in ds.variables:
        if var in meta_data and var != "time":
            ds[var].attrs.update(meta_data[var])
    return ds


def _save_case_zarr(case_name: str, case_ds: xr.Dataset, conf: dict) -> str:
    save_root = conf["predict"]["save_forecast"]
    zarr_name = case_name if case_name.endswith(".zarr") else f"{case_name}.zarr"
    zarr_path = os.path.join(save_root, zarr_name)

    if os.path.isdir(zarr_path):
        shutil.rmtree(zarr_path)

    case_ds.to_zarr(zarr_path, mode="w", consolidated=True)
    return zarr_path


def rollout_case_ensemble(case_path: str, conf: dict, model: torch.nn.Module, device: torch.device) -> pd.DataFrame:
    state_transformer = NormalizeWRF(conf)
    dataset = _build_case_dataset(case_path, conf, transform=state_transformer)
    to_tensor = ToTensorWRF(conf)
    meta_data = load_metadata(conf)

    metrics_conf_ens = copy.deepcopy(conf)
    metrics_conf_ens.setdefault("predict", {})
    ensemble_size = int(conf.get("predict", {}).get("ensemble_size", 1))
    metrics_conf_ens["predict"]["ensemble_size"] = ensemble_size
    metrics_ensemble = LatWeightedMetrics(metrics_conf_ens, training_mode=False)

    metrics_conf_member = copy.deepcopy(conf)
    metrics_conf_member.setdefault("predict", {})
    metrics_conf_member["predict"]["ensemble_size"] = 1
    metrics_member = LatWeightedMetrics(metrics_conf_member, training_mode=False)

    history_len = conf["data"]["history_len"]
    start_index = _resolve_rollout_start_index(conf)
    max_steps = max(0, len(dataset) - start_index)
    varnum_diag = len(conf["data"]["diagnostic_variables"])
    residual_prediction = conf["trainer"].get("residual_prediction", False)
    static_dim_size = (
        len(conf["data"]["dynamic_forcing_variables"])
        + len(conf["data"]["forcing_variables"])
        + len(conf["data"]["static_variables"])
    )
    lead_time_periods = int(conf["data"].get("lead_time_periods", 1))
    save_step_netcdf = bool(conf.get("predict", {}).get("save_step_netcdf", False))
    save_ensemble_members = bool(conf.get("predict", {}).get("save_ensemble_members", True))
    compute_crps = bool(conf.get("predict", {}).get("compute_crps", True))

    case_results: list[dict] = []
    case_pred_steps: list[xr.Dataset] = []
    case_lead_steps: list[int] = []
    case_lead_periods: list[int] = []
    x_state = None
    case_name = _case_name(case_path)
    y_coord = None
    x_coord = None
    warn_netcdf_once = True

    for step in range(max_steps):
        sample_index = start_index + step
        lead_step = _lead_step_from_sample_index(sample_index, history_len)
        lead_period = lead_step * lead_time_periods
        raw_sample = dataset[sample_index]
        if y_coord is None or x_coord is None:
            target_ds = raw_sample["WRF_target"]
            y_coord = target_ds["y"].values if "y" in target_ds.coords else target_ds["south_north"].values
            x_coord = target_ds["x"].values if "x" in target_ds.coords else target_ds["west_east"].values
        sample = to_tensor(raw_sample)

        if step == 0:
            x_model, x_boundary, x_time_encode = _sample_to_model_inputs(sample, device)
            x_model = _repeat_to_batch(x_model, ensemble_size)
            x_boundary = _repeat_to_batch(x_boundary, ensemble_size)
            x_time_encode = _repeat_to_batch(x_time_encode, ensemble_size)

            if "x_surf" in sample:
                x_state = concat_and_reshape(sample["x"].unsqueeze(0), sample["x_surf"].unsqueeze(0))
            else:
                x_state = reshape_only(sample["x"].unsqueeze(0))
            x_state = _repeat_to_batch(x_state, ensemble_size)
        else:
            x_model, x_boundary, x_time_encode = _compose_autoregressive_input(x_state, sample, device)

        y_true = _target_tensor(sample, device)

        with torch.no_grad():
            y_pred = model(
                x_model.float(),
                x_boundary.float(),
                x_time_encode.float(),
                forecast_step=lead_step,
                ensemble_size=ensemble_size,
            )
            y_pred = _apply_residual_prediction(y_pred, x_model, residual_prediction, varnum_diag)

        y_pred_denorm = state_transformer.inverse_transform(y_pred.cpu())
        y_true_denorm = state_transformer.inverse_transform(y_true.cpu())

        result_row = {
            "case": case_name,
            "forecast_step": lead_step,
            "forecast_period": lead_period,
            "rollout_index": step + 1,
            "sample_index": sample_index,
            "ensemble_size": ensemble_size,
        }

        if ensemble_size > 1:
            ensemble_metric_dict = _to_float_metrics(
                metrics_ensemble(y_pred_denorm, y_true_denorm, forecast_datetime=lead_step)
            )
            result_row.update(ensemble_metric_dict)

            member_rows = []
            for member_index in range(ensemble_size):
                member_pred = y_pred_denorm[member_index : member_index + 1]
                member_metric_dict = metrics_member(member_pred, y_true_denorm, forecast_datetime=lead_step)
                member_rows.append(_to_float_metrics(member_metric_dict))
            member_df = pd.DataFrame(member_rows)
            for col in member_df.columns:
                result_row[f"{col}_member_mean"] = float(member_df[col].mean())
                result_row[f"{col}_member_std"] = float(member_df[col].std(ddof=0))

            if compute_crps:
                crps_scores = _calculate_crps_per_channel(y_pred_denorm, y_true_denorm).squeeze(0)
                for i, var in enumerate(metrics_ensemble.vars):
                    result_row[f"crps_{var}"] = float(crps_scores[i].item())
                result_row["crps"] = float(crps_scores.mean().item())

            if "rmse" in ensemble_metric_dict and "std" in ensemble_metric_dict and ensemble_metric_dict["rmse"] != 0.0:
                result_row["spread_skill_ratio"] = float(ensemble_metric_dict["std"] / ensemble_metric_dict["rmse"])

            for key, value in ensemble_metric_dict.items():
                if not key.startswith("rmse_"):
                    continue
                var = key.replace("rmse_", "")
                std_key = f"std_{var}"
                if std_key in ensemble_metric_dict and value != 0.0:
                    result_row[f"spread_skill_ratio_{var}"] = float(ensemble_metric_dict[std_key] / value)
        else:
            metric_dict = _to_float_metrics(metrics_member(y_pred_denorm, y_true_denorm, forecast_datetime=lead_step))
            result_row.update(metric_dict)

        case_results.append(result_row)

        valid_time = datetime.utcfromtimestamp(int(raw_sample["WRF_target"]["time"].values[0].astype("datetime64[s]").astype(int)))
        step_ds = _prediction_to_dataset(
            y_pred_denorm,
            valid_time,
            y_coord,
            x_coord,
            conf,
            save_ensemble_members=save_ensemble_members,
        )
        case_pred_steps.append(step_ds)
        case_lead_steps.append(lead_step)
        case_lead_periods.append(lead_period)

        if save_step_netcdf:
            if ensemble_size > 1 and save_ensemble_members and warn_netcdf_once:
                logger.warning("save_step_netcdf writes ensemble mean only when save_ensemble_members=True")
                warn_netcdf_once = False

            y_pred_netcdf = y_pred_denorm.mean(dim=0, keepdim=True) if ensemble_size > 1 else y_pred_denorm
            xarray_outputs = make_xarray(y_pred_netcdf, valid_time, y_coord, x_coord, conf)
            if isinstance(xarray_outputs, tuple):
                darray_upper_air, darray_single_level = xarray_outputs
            else:
                darray_upper_air = xarray_outputs
                darray_single_level = None
            save_netcdf_clean(darray_upper_air, darray_single_level, case_name, lead_step, meta_data, conf, use_logger=False)

        if history_len == 1:
            x_state = y_pred[:, :-varnum_diag, ...].detach().cpu() if varnum_diag > 0 else y_pred.detach().cpu()
        else:
            if x_state is None:
                raise RuntimeError("Autoregressive state was not initialized before update")

            if static_dim_size == 0:
                x_detach = x_state[:, :, 1:, ...].detach().cpu()
            else:
                x_detach = x_state[:, :-static_dim_size, 1:, ...].detach().cpu()

            new_state = y_pred[:, :-varnum_diag, ...].detach().cpu() if varnum_diag > 0 else y_pred.detach().cpu()
            x_state = torch.cat([x_detach, new_state], dim=2)

    if not case_pred_steps:
        logger.warning("No rollout steps generated for case %s", case_name)
        return pd.DataFrame(case_results)

    case_ds = xr.concat(case_pred_steps, dim="time")
    case_ds = case_ds.assign_coords(
        forecast_step=("time", case_lead_steps),
        forecast_period=("time", case_lead_periods),
    )
    case_ds["forecast_step"] = case_ds["forecast_step"].astype("int32")
    case_ds["forecast_period"] = case_ds["forecast_period"].astype("int32")

    if "save_vars" in conf.get("predict", {}) and len(conf["predict"]["save_vars"]) > 0:
        case_ds = drop_var_from_dataset(case_ds, conf["predict"]["save_vars"])

    case_ds = _apply_metadata(case_ds, meta_data)
    case_ds.attrs["history_len"] = history_len
    case_ds.attrs["target_start_step"] = int(conf["data"].get("target_start_step", history_len))
    case_ds.attrs["rollout_start_sample_index"] = start_index
    case_ds.attrs["lead_time_periods"] = lead_time_periods
    case_ds.attrs["ensemble_size"] = ensemble_size
    case_ds.attrs["save_ensemble_members"] = bool(save_ensemble_members)

    case_zarr_path = _save_case_zarr(case_name, case_ds, conf)
    logger.info("Saved case rollout zarr to %s", case_zarr_path)

    return pd.DataFrame(case_results)


def _aggregate_rank_metrics(out_dir: str, world_size: int, output_name: str) -> str:
    rank_paths = [
        os.path.join(out_dir, f"rollout_metrics_rank{rank}.csv")
        for rank in range(world_size)
        if os.path.exists(os.path.join(out_dir, f"rollout_metrics_rank{rank}.csv"))
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
    parser = ArgumentParser(description="Roll out WoFS WRF model with ensemble metrics and optional distributed inference")
    parser.add_argument("config", help="Path to the WoFS rollout YAML config")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit on number of cases to roll out")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["none", "ddp", "fsdp"],
        default=None,
        help="Override predict.mode for rollout",
    )
    parser.add_argument(
        "--backend",
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="Distributed backend when using ddp/fsdp rollout",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        help="Override ensemble size (default: predict.ensemble_size, then trainer.ensemble_size)",
    )
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="Save only ensemble mean into zarr instead of all ensemble members",
    )
    parser.add_argument(
        "--disable-crps",
        action="store_true",
        help="Disable CRPS computation to reduce rollout cost",
    )
    parser.add_argument(
        "--output-name",
        default="rollout_metrics_ensemble.csv",
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
    conf["predict"].setdefault("mode", "ddp")

    if args.ensemble_size is not None:
        ensemble_size = int(args.ensemble_size)
    else:
        ensemble_size = int(conf.get("predict", {}).get("ensemble_size", conf.get("trainer", {}).get("ensemble_size", 1)))
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be >= 1")

    conf["predict"]["ensemble_size"] = ensemble_size
    conf["predict"]["save_ensemble_members"] = not args.mean_only
    conf["predict"]["compute_crps"] = not args.disable_crps

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])
    conf["predict"]["save_forecast"] = os.path.expandvars(conf["predict"]["save_forecast"])
    os.makedirs(conf["predict"]["save_forecast"], exist_ok=True)

    seed_everything(conf["seed"])
    mode = conf["predict"].get("mode", "none")
    if mode in ["ddp", "fsdp"]:
        _, get_rank_info, setup = _load_distributed_ops()
        local_rank, world_rank, world_size = get_rank_info(mode)
    else:
        local_rank, world_rank, world_size = 0, 0, 1

    distributed = mode in ["ddp", "fsdp"]
    if distributed:
        setup(world_rank, world_size, mode, backend=args.backend)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    model = _load_rollout_model(conf, device)

    date_range = conf.get("predict", {}).get("custom_date_range")
    case_files = _select_files(conf["data"]["save_loc"], date_range)
    if args.max_cases is not None:
        case_files = case_files[: args.max_cases]
    elif conf.get("predict", {}).get("max_cases") is not None:
        case_files = case_files[: int(conf["predict"]["max_cases"])]

    case_files = case_files[world_rank::world_size]

    if not case_files:
        logger.warning("No WoFS case files matched the rollout selection on rank %s", world_rank)
        case_files = []

    all_results = []
    for case_path in case_files:
        logger.info("Rank %s rolling out %s", world_rank, case_path)
        case_df = rollout_case_ensemble(case_path, conf, model, device)
        all_results.append(case_df)

    partial_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    partial_path = os.path.join(conf["predict"]["save_forecast"], f"rollout_metrics_rank{world_rank}.csv")
    partial_df.to_csv(partial_path, index=False)

    if distributed:
        dist.barrier()

    if world_rank == 0:
        summary_path = _aggregate_rank_metrics(conf["predict"]["save_forecast"], world_size, args.output_name)
        logger.info("Saved rollout metrics to %s", summary_path)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
    
## python applications/rollout_wrf_wofs_ensemble.py config/ursa_wofscast_credit_wrf_latest_ensemble.yml --mode ddp --max-cases 1 --ensemble-size 2 --mean-only --output-name rollout_metrics_ensemble_smoke.csv