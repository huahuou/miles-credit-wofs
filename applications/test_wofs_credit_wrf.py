from __future__ import annotations

import argparse
from glob import glob

import yaml

from credit.datasets.wrf_singlestep import WRFDataset
from credit.parser import credit_main_parser
from credit.transforms import load_transforms


def _select_files(glob_pattern: str, years_range: list[int]) -> list[str]:
    years = [str(year) for year in range(years_range[0], years_range[1])]
    return [file for file in sorted(glob(glob_pattern)) if any(year in file for year in years)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test WoFS CREDIT-WRF interior + boundary archives through WRFDataset")
    parser.add_argument("config", help="Path to the CREDIT YAML config")
    parser.add_argument("--split", choices=("train", "valid"), default="train", help="Dataset split to validate")
    parser.add_argument("--index", type=int, default=0, help="Sample index to load")
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    conf = credit_main_parser(conf, parse_training=False, parse_predict=False)
    transforms = load_transforms(conf)
    years_range = conf["data"]["train_years"] if args.split == "train" else conf["data"]["valid_years"]

    param_interior = {
        "varname_upper_air": conf["data"]["variables"],
        "varname_surface": conf["data"]["surface_variables"],
        "varname_dyn_forcing": conf["data"]["dynamic_forcing_variables"],
        "varname_forcing": conf["data"]["forcing_variables"],
        "varname_static": conf["data"]["static_variables"],
        "varname_diagnostic": conf["data"]["diagnostic_variables"],
        "filenames": _select_files(conf["data"]["save_loc"], years_range),
        "filename_surface": _select_files(conf["data"]["save_loc_surface"], years_range)
        if conf["data"].get("save_loc_surface")
        else None,
        "filename_dyn_forcing": _select_files(conf["data"]["save_loc_dynamic_forcing"], years_range)
        if conf["data"].get("save_loc_dynamic_forcing")
        else None,
        "filename_forcing": conf["data"].get("save_loc_forcing"),
        "filename_static": conf["data"].get("save_loc_static"),
        "filename_diagnostic": _select_files(conf["data"]["save_loc_diagnostic"], years_range)
        if conf["data"].get("save_loc_diagnostic")
        else None,
        "history_len": conf["data"]["history_len" if args.split == "train" else "valid_history_len"],
        "forecast_len": conf["data"]["forecast_len" if args.split == "train" else "valid_forecast_len"],
    }

    param_outside = {
        "varname_upper_air": conf["data"]["boundary"]["variables"],
        "varname_surface": conf["data"]["boundary"].get("surface_variables", []),
        "filenames": _select_files(conf["data"]["boundary"]["save_loc"], years_range),
        "filename_surface": _select_files(conf["data"]["boundary"]["save_loc_surface"], years_range)
        if conf["data"]["boundary"].get("save_loc_surface")
        else None,
        "history_len": conf["data"]["boundary"]["history_len"],
        "forecast_len": conf["data"]["boundary"]["forecast_len"],
    }

    dataset = WRFDataset(param_interior, param_outside, transform=transforms, seed=conf["seed"])
    sample = dataset[args.index]

    print(f"[dataset] length={len(dataset)} split={args.split}")
    for key, value in sample.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        print(f"[sample] {key}: {shape}")


if __name__ == "__main__":
    main()