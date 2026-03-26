from __future__ import annotations

import argparse
import re
from glob import glob
from pathlib import Path

import torch
import yaml

from credit.data import concat_and_reshape, find_key_for_number, reshape_only
from credit.datasets.wrf_wofs_singlestep import WoFSSingleStepDataset
from credit.models import load_model
from credit.parser import credit_main_parser
from credit.transforms import load_transforms


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


def _build_params(conf: dict, split: str) -> tuple[dict, dict]:
    years_range = conf["data"]["train_years"] if split == "train" else conf["data"]["valid_years"]
    date_range = conf["data"].get("train_date_range") if split == "train" else conf["data"].get("valid_date_range")

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
    }

    param_outside = {
        "varname_upper_air": conf["data"]["boundary"]["variables"],
        "varname_surface": conf["data"]["boundary"].get("surface_variables", []),
        "history_len": conf["data"]["boundary"]["history_len"],
        "forecast_len": conf["data"]["boundary"]["forecast_len"],
    }

    return param_interior, param_outside


def _print_sample(sample: dict) -> None:
    for key, value in sample.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        print(f"[sample] {key}: {shape}")


def _compare_next_sample(dataset: WoFSSingleStepDataset, index: int) -> None:
    if index + 1 >= len(dataset):
        print("[compare] skipped: no next sample")
        return

    key0 = find_key_for_number(index, dataset.WRF_file_indices)
    key1 = find_key_for_number(index + 1, dataset.WRF_file_indices)
    if key0 != key1:
        print("[compare] skipped: next sample is in a different zarr")
        return

    sample0 = dataset[index]
    sample1 = dataset[index + 1]

    same_boundary = torch.allclose(sample0["x_boundary"], sample1["x_boundary"])
    different_x = not torch.allclose(sample0["x"], sample1["x"])

    print(f"[compare] indices=({index}, {index + 1}) same_boundary={same_boundary} different_x={different_x}")


def _run_forward(conf: dict, sample: dict) -> None:
    if "x_surf" in sample:
        x = concat_and_reshape(sample["x"].unsqueeze(0), sample["x_surf"].unsqueeze(0))
    else:
        x = reshape_only(sample["x"].unsqueeze(0))

    if "x_forcing_static" in sample:
        x = torch.cat((x, sample["x_forcing_static"].unsqueeze(0).permute(0, 2, 1, 3, 4)), dim=1)

    if "x_surf_boundary" in sample:
        xb = concat_and_reshape(sample["x_boundary"].unsqueeze(0), sample["x_surf_boundary"].unsqueeze(0))
    else:
        xb = reshape_only(sample["x_boundary"].unsqueeze(0))

    xt = sample["x_time_encode"].unsqueeze(0)

    model = load_model(conf).cpu()
    model.eval()
    with torch.no_grad():
        y_pred = model(x.float().cpu(), xb.float().cpu(), xt.float().cpu())

    print(f"[forward-cpu] y_pred {tuple(y_pred.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test WoFS same-file t=0 conditioning through a WRF-style dataset")
    parser.add_argument("config", help="Path to the CREDIT YAML config")
    parser.add_argument("--split", choices=("train", "valid"), default="train", help="Dataset split to validate")
    parser.add_argument("--index", type=int, default=0, help="Sample index to load")
    parser.add_argument("--compare-next", action="store_true", help="Compare boundary consistency against the next sample from the same zarr")
    parser.add_argument("--forward", action="store_true", help="Run a CPU forward pass with the current WRF model")
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    conf = credit_main_parser(conf, parse_training=False, parse_predict=False)
    transforms = load_transforms(conf)
    param_interior, param_outside = _build_params(conf, args.split)

    dataset = WoFSSingleStepDataset(param_interior, param_outside, transform=transforms, seed=conf["seed"])
    sample = dataset[args.index]

    print(f"[dataset] length={len(dataset)} split={args.split}")
    _print_sample(sample)

    if args.compare_next:
        _compare_next_sample(dataset, args.index)

    if args.forward:
        _run_forward(conf, sample)


if __name__ == "__main__":
    main()