from __future__ import annotations

import copy
import logging
import os
import re
import warnings
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import yaml
from pandas.errors import EmptyDataError

from credit.data import concat_and_reshape, reshape_only
from credit.datasets.wrf_wofs_singlestep import WoFSSingleStepDataset
from credit.distributed import distributed_model_wrapper, get_rank_info, setup
from credit.metrics import LatWeightedMetrics
from credit.models import load_model
from credit.models.checkpoint import load_model_state, load_state_dict_error_handler
from credit.parser import credit_main_parser
from credit.seed import seed_everything
from credit.transforms.transforms_wrf import NormalizeWRF, ToTensorWRF

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


def _build_case_dataset(case_path: str, conf: dict) -> WoFSSingleStepDataset:
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

    return WoFSSingleStepDataset(param_interior, param_outside, transform=None, seed=conf["seed"])


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
    if "x_forcing_static" in sample:
        x_forcing = sample["x_forcing_static"].unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
        x = torch.cat((x, x_forcing), dim=1)

    if "x_surf_boundary" in sample:
        x_boundary = concat_and_reshape(sample["x_boundary"].unsqueeze(0), sample["x_surf_boundary"].unsqueeze(0)).to(device)
    else:
        x_boundary = reshape_only(sample["x_boundary"].unsqueeze(0)).to(device)

    x_time_encode = sample["x_time_encode"].unsqueeze(0).to(device)
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


def _case_name(case_path: str) -> str:
    return Path(case_path).stem


def rollout_case_metrics(case_path: str, conf: dict, model: torch.nn.Module, device: torch.device) -> pd.DataFrame:
    dataset = _build_case_dataset(case_path, conf)
    state_transformer = NormalizeWRF(conf)
    to_tensor = ToTensorWRF(conf)
    metrics = LatWeightedMetrics(conf)

    history_len = conf["data"]["history_len"]
    max_steps = max(0, len(dataset) - 1)
    varnum_diag = len(conf["data"]["diagnostic_variables"])
    static_dim_size = (
        len(conf["data"]["dynamic_forcing_variables"])
        + len(conf["data"]["forcing_variables"])
        + len(conf["data"]["static_variables"])
    )

    case_results: list[dict] = []
    x_state: torch.Tensor | None = None
    case_name = _case_name(case_path)

    for step in range(max_steps):
        raw_sample = dataset[step]
        sample = to_tensor(state_transformer(raw_sample))

        if step == 0:
            x_model, x_boundary, x_time_encode = _sample_to_model_inputs(sample, device)
            if "x_surf" in sample:
                x_state = concat_and_reshape(sample["x"].unsqueeze(0), sample["x_surf"].unsqueeze(0))
            else:
                x_state = reshape_only(sample["x"].unsqueeze(0))
        else:
            x_model, x_boundary, x_time_encode = _compose_autoregressive_input(x_state, sample, device)

        y_true = _target_tensor(sample, device)

        with torch.no_grad():
            y_pred = model(x_model.float(), x_boundary.float(), x_time_encode.float())

        y_pred_denorm = state_transformer.inverse_transform(y_pred.cpu())
        y_true_denorm = state_transformer.inverse_transform(y_true.cpu())

        metrics_dict = metrics(y_pred_denorm, y_true_denorm, forecast_datetime=step + 1)
        result_row = {"case": case_name, "forecast_hour": step + 1}
        result_row.update({k: float(v) for k, v in metrics_dict.items()})
        case_results.append(result_row)

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
    parser = ArgumentParser(description="Compute rollout metrics for WoFS WRF t0-conditioning runs")
    parser.add_argument("config", help="Path to the WoFS YAML config")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit on number of cases to process")
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
        "--output-name",
        default="rollout_metrics.csv",
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
    case_files = _select_files(conf["data"]["save_loc"], date_range)
    if args.max_cases is not None:
        case_files = case_files[: args.max_cases]
    elif conf.get("predict", {}).get("max_cases") is not None:
        case_files = case_files[: int(conf["predict"]["max_cases"])]

    case_files = case_files[world_rank::world_size]

    if not case_files:
        logger.warning("No WoFS case files matched rollout selection on rank %s", world_rank)

    all_results = []
    for case_path in case_files:
        logger.info("Rank %s computing metrics for %s", world_rank, case_path)
        case_df = rollout_case_metrics(case_path, conf, model, device)
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
