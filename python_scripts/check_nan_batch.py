#!/usr/bin/env python3
"""Check mean/std and dataset samples for NaNs or zero stds.

Usage:
  python scripts/check_nan_batch.py --config path/to/config.yml --indices 33178 24287
"""
import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import xarray as xr
import yaml

from credit.data import find_key_for_number


def load_conf(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def check_mean_std(conf):
    mean_path = conf["data"]["mean_path"]
    std_path = conf["data"]["std_path"]
    print(f"Checking mean: {mean_path}\n      std: {std_path}")
    m = xr.open_dataset(mean_path)
    s = xr.open_dataset(std_path)

    bad = False
    for var in s.data_vars:
        arr = s[var].values
        if np.isnan(arr).any():
            print(f"STD has NaN for var {var}")
            bad = True
        if np.isclose(arr, 0).any():
            print(f"STD has zeros for var {var} (division-by-zero risk)")
            bad = True

    for var in m.data_vars:
        arr = m[var].values
        if np.isnan(arr).any():
            print(f"MEAN has NaN for var {var}")
            bad = True

    if not bad:
        print("mean/std look finite and non-zero where expected")


def build_index_metadata(filenames, history_len, forecast_len, start_index_offset=0):
    wrf_file_indices = {}
    running_start = 0
    for ind_file, filename in enumerate(sorted(filenames)):
        try:
            if filename.endswith((".nc", ".nc4")):
                ds = xr.open_dataset(filename, decode_times=False)
            else:
                ds = xr.open_zarr(filename, decode_times=False)
            n_time = int(ds.sizes["time"])
            available = n_time - (history_len + forecast_len + 1) + 1
            available -= start_index_offset
            if available > 0:
                wrf_file_indices[str(ind_file)] = [available, running_start, running_start + available - 1]
                running_start += available
        except Exception as e:
            print(f"Failed reading {filename}: {e}")
        finally:
            try:
                ds.close()
            except Exception:
                pass
    return wrf_file_indices


def inspect_indices(conf, indices):
    param_interior = {}
    param_interior["filenames"] = sorted(glob(conf["data"]["save_loc"])) if "save_loc" in conf["data"] else []
    # fallback if explicit filenames present in conf
    if isinstance(conf["data"].get("filenames"), list):
        param_interior["filenames"] = conf["data"].get("filenames")

    history_len = int(conf["data"]["history_len"])
    forecast_len = int(conf["data"]["forecast_len"])
    target_start_step = int(conf["data"].get("target_start_step", 1))
    start_index_offset = max(0, target_start_step - history_len)

    print("Building index mapping (may open many small files)...")
    mapping = build_index_metadata(param_interior["filenames"], history_len, forecast_len, start_index_offset)

    for ix in indices:
        key = find_key_for_number(ix, mapping)
        print(f"Index {ix} -> file key {key}")
        if key is None:
            print("  No file mapping found for this index")
            continue
        file_range = mapping[key]
        start = file_range[1]
        ind_in_file = (ix - start) + start_index_offset
        filename = sorted(param_interior["filenames"])[int(key)]
        print(f"  filename: {filename} time index in file: {ind_in_file}")
        try:
            if filename.endswith((".nc", ".nc4")):
                ds = xr.open_dataset(filename)
            else:
                ds = xr.open_zarr(filename, chunks=None)
            # check each data variable at the time slice
            for var in ds.data_vars:
                try:
                    arr = ds[var].isel(time=ind_in_file).values
                    if np.isnan(arr).any():
                        print(f"    VAR {var}: contains NaN at that time slice")
                    else:
                        print(f"    VAR {var}: finite, min={np.nanmin(arr)}, max={np.nanmax(arr)}")
                except Exception as e:
                    print(f"    VAR {var}: cannot isel/time-check: {e}")
        except Exception as e:
            print(f"  Failed to open or inspect {filename}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--indices", nargs="+", type=int, required=True)
    args = p.parse_args()

    conf = load_conf(args.config)
    check_mean_std(conf)
    inspect_indices(conf, args.indices)


if __name__ == "__main__":
    main()
