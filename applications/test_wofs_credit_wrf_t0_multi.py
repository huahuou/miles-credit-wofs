from __future__ import annotations

import argparse
import re
from glob import glob
from pathlib import Path

import torch
import yaml

from credit.data import concat_and_reshape, reshape_only
from credit.datasets.wrf_wofs_multistep import WoFSMultiStep
from credit.models import load_model
from credit.parser import credit_main_parser
from credit.transforms import load_transforms


def _unstack_rollout_batch(rollout_batch: dict) -> list[dict]:
    rollout_len = int(rollout_batch["forecast_step"].shape[0])
    rollout = []
    for step_index in range(rollout_len):
        step_sample = {}
        for key, value in rollout_batch.items():
            if torch.is_tensor(value) and value.ndim >= 1:
                step_sample[key] = value[step_index, ...]
            else:
                step_sample[key] = value
        rollout.append(step_sample)
    return rollout


def _extract_case_date(file_path: str) -> str | None:
    match = re.search(r"wofs_(\d{8})_\d{4}_mem\d+\.zarr$", Path(file_path).name)
    return match.group(1) if match else None


def _select_files(
    glob_pattern: str,
    years_range: list[int] | list[str] | None,
    date_range: list[str] | tuple[str, str] | None = None,
) -> list[str]:
    files = sorted(glob(glob_pattern))

    if date_range is not None:
        start_date, end_date = str(date_range[0]), str(date_range[1])
        return [
            file_path
            for file_path in files
            if (case_date := _extract_case_date(file_path)) is not None and start_date <= case_date <= end_date
        ]

    if years_range is None:
        return files

    years = [str(year) for year in range(int(years_range[0]), int(years_range[1]))]
    return [file for file in files if any(year in file for year in years)]


def _build_params(conf: dict, split: str, target_start_step: int | None) -> tuple[dict, dict]:
    years_range = conf["data"]["train_years"] if split == "train" else conf["data"]["valid_years"]
    date_range = conf["data"].get("train_date_range") if split == "train" else conf["data"].get("valid_date_range")
    effective_target_start = target_start_step or conf["data"].get("target_start_step", 1)

    param_interior = {
        "varname_upper_air": conf["data"]["variables"],
        "varname_surface": conf["data"]["surface_variables"],
        "varname_dyn_forcing": conf["data"]["dynamic_forcing_variables"],
        "varname_forcing": conf["data"]["forcing_variables"],
        "varname_static": conf["data"]["static_variables"],
        "varname_diagnostic": conf["data"]["diagnostic_variables"],
        "filenames": _select_files(conf["data"]["save_loc"], years_range, date_range),
        "filename_surface": _select_files(conf["data"]["save_loc_surface"], years_range, date_range)
        if conf["data"].get("save_loc_surface")
        else None,
        "filename_dyn_forcing": _select_files(conf["data"]["save_loc_dynamic_forcing"], years_range, date_range)
        if conf["data"].get("save_loc_dynamic_forcing")
        else None,
        "filename_forcing": conf["data"].get("save_loc_forcing"),
        "filename_static": conf["data"].get("save_loc_static"),
        "filename_diagnostic": _select_files(conf["data"]["save_loc_diagnostic"], years_range, date_range)
        if conf["data"].get("save_loc_diagnostic")
        else None,
        "history_len": conf["data"]["history_len" if split == "train" else "valid_history_len"],
        "forecast_len": conf["data"]["forecast_len" if split == "train" else "valid_forecast_len"],
        "target_start_step": effective_target_start,
    }

    param_outside = {
        "varname_upper_air": conf["data"]["boundary"]["variables"],
        "varname_surface": conf["data"]["boundary"].get("surface_variables", []),
        "history_len": conf["data"]["boundary"]["history_len"],
        "forecast_len": conf["data"]["boundary"]["forecast_len"],
    }

    return param_interior, param_outside


def _print_rollout(rollout: list[dict]) -> None:
    for sample in rollout:
        print(
            f"[rollout] forecast_step={sample['forecast_step']} stop_forecast={sample['stop_forecast']} "
            f"x={tuple(sample['x'].shape)} y={tuple(sample['y'].shape)} x_boundary={tuple(sample['x_boundary'].shape)}"
        )


def _check_rollout(rollout: list[dict]) -> None:
    boundary_constant = all(torch.allclose(rollout[0]["x_boundary"], sample["x_boundary"]) for sample in rollout[1:])
    x_changes = any(not torch.allclose(rollout[0]["x"], sample["x"]) for sample in rollout[1:])
    forecast_steps = [int(sample["forecast_step"]) for sample in rollout]
    stops = [bool(sample["stop_forecast"]) for sample in rollout]
    print(
        f"[check] forecast_steps={forecast_steps} stop_flags={stops} "
        f"boundary_constant={boundary_constant} x_changes={x_changes}"
    )


def _run_forward_rollout(conf: dict, rollout: list[dict]) -> None:
    varnum_diag = len(conf["data"]["diagnostic_variables"])
    static_dim_size = (
        len(conf["data"]["dynamic_forcing_variables"])
        + len(conf["data"]["forcing_variables"])
        + len(conf["data"]["static_variables"])
        + len(conf["data"]["boundary"]["variables"])
        + len(conf["data"]["boundary"]["surface_variables"])
    )

    model = load_model(conf).cpu()
    model.eval()

    x = None
    for sample in rollout:
        if sample["forecast_step"] == 1:
            if "x_surf" in sample:
                x = concat_and_reshape(sample["x"].unsqueeze(0), sample["x_surf"].unsqueeze(0))
            else:
                x = reshape_only(sample["x"].unsqueeze(0))

        if "x_forcing_static" in sample:
            x_forcing_batch = sample["x_forcing_static"].unsqueeze(0).permute(0, 2, 1, 3, 4)
            x_model = torch.cat((x, x_forcing_batch), dim=1)
        else:
            x_model = x

        if "x_surf_boundary" in sample:
            x_boundary = concat_and_reshape(sample["x_boundary"].unsqueeze(0), sample["x_surf_boundary"].unsqueeze(0))
        else:
            x_boundary = reshape_only(sample["x_boundary"].unsqueeze(0))

        x_time_encode = sample["x_time_encode"].unsqueeze(0)

        with torch.no_grad():
            y_pred = model(x_model.float().cpu(), x_boundary.float().cpu(), x_time_encode.float().cpu())

        print(f"[forward-cpu] forecast_step={sample['forecast_step']} y_pred={tuple(y_pred.shape)}")

        if sample["stop_forecast"]:
            break

        if x.shape[2] == 1:
            if "y_diag" in sample:
                x = y_pred[:, :-varnum_diag, ...].detach()
            else:
                x = y_pred.detach()
        else:
            if static_dim_size == 0:
                x_detach = x[:, :, 1:, ...].detach()
            else:
                x_detach = x[:, :-static_dim_size, 1:, ...].detach()

            if "y_diag" in sample:
                x = torch.cat([x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2)
            else:
                x = torch.cat([x_detach, y_pred.detach()], dim=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test WoFS same-file t=0 multistep conditioning")
    parser.add_argument("config", help="Path to the CREDIT YAML config")
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--index", type=int, default=0, help="Rollout start index")
    parser.add_argument("--target-start-step", type=int, default=None, help="Override first learned target lead: 1 means t1, 2 means t2")
    parser.add_argument("--forward", action="store_true", help="Run a CPU forward rollout")
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    conf = credit_main_parser(conf, parse_training=False, parse_predict=False)
    transforms = load_transforms(conf)
    param_interior, param_outside = _build_params(conf, args.split, args.target_start_step)

    dataset = WoFSMultiStep(param_interior, param_outside, transform=transforms, seed=conf["seed"])

    rollout_batch = dataset[args.index]
    rollout = _unstack_rollout_batch(rollout_batch)

    print(
        f"[dataset] length={len(dataset)} split={args.split} forecast_len={param_interior['forecast_len']} "
        f"target_start_step={param_interior['target_start_step']}"
    )
    _print_rollout(rollout)
    _check_rollout(rollout)

    if args.forward:
        _run_forward_rollout(conf, rollout)


if __name__ == "__main__":
    main()