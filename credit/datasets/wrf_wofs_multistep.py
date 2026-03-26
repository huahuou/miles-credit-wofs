import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr

from credit.data import (
    Sample_WRF,
    drop_var_from_dataset,
    encode_datetime64,
    extract_month_day_hour,
    filter_ds,
    find_common_indices,
    find_key_for_number,
    get_forward_data,
)

logger = logging.getLogger(__name__)


def _stack_rollout_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    stacked: Dict[str, Any] = {}
    sample0 = samples[0]

    for key in sample0.keys():
        values = [sample[key] for sample in samples]
        first_value = values[0]

        if torch.is_tensor(first_value):
            stacked[key] = torch.stack(values, dim=0)
        elif isinstance(first_value, np.ndarray):
            stacked[key] = torch.as_tensor(np.stack(values, axis=0))
        elif isinstance(first_value, (list, tuple)):
            stacked[key] = torch.as_tensor(values)
        elif isinstance(first_value, (int, np.integer, bool, np.bool_)):
            stacked[key] = torch.as_tensor(values)
        else:
            stacked[key] = values

    return stacked


def _expand_anchor_dataset(dataset: xr.Dataset, boundary_seq_len: int) -> xr.Dataset:
    anchor = dataset.isel(time=slice(0, 1)).load()

    if boundary_seq_len == 1:
        return anchor

    anchor_time = anchor["time"].values[0]
    repeated = xr.concat([anchor] * boundary_seq_len, dim="time")
    repeated = repeated.assign_coords(time=np.array([anchor_time] * boundary_seq_len))
    return repeated


def worker(
    tuple_index: Tuple[int, int],
    WRF_file_indices: Dict[str, List[int]],
    list_upper_ds: List[Any],
    list_surf_ds: Optional[List[Any]],
    list_dyn_forcing_ds: Optional[List[Any]],
    list_diag_ds: Optional[List[Any]],
    xarray_forcing: Optional[Any],
    xarray_static: Optional[Any],
    history_len: int,
    list_upper_ds_outside: List[Any],
    list_surf_ds_outside: Optional[List[Any]],
    boundary_seq_len: int,
    start_index_offset: int,
    transform: Optional[Callable],
) -> Dict[str, Any]:
    index, ind_start_current_step = tuple_index

    try:
        ind_file = find_key_for_number(ind_start_current_step, WRF_file_indices)
        file_range = WRF_file_indices[ind_file]

        ind_start_in_file = (ind_start_current_step - file_range[1]) + start_index_offset

        ind_end_in_file = ind_start_in_file + history_len

        WRF_subset = list_upper_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))

        if list_surf_ds:
            surface_subset = list_surf_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))
            WRF_subset = WRF_subset.merge(surface_subset)

        datetime_as_number = WRF_subset.time.values.astype("datetime64[s]").astype(int)

        WRF_input = WRF_subset.isel(time=slice(0, history_len, 1)).load()

        if list_dyn_forcing_ds:
            dyn_forcing_subset = list_dyn_forcing_ds[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )
            dyn_forcing_subset = dyn_forcing_subset.isel(time=slice(0, history_len, 1)).load()
            WRF_input = WRF_input.merge(dyn_forcing_subset)

        if xarray_forcing:
            month_day_forcing = extract_month_day_hour(np.array(xarray_forcing["time"]))
            month_day_inputs = extract_month_day_hour(np.array(WRF_input["time"]))
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = xarray_forcing.isel(time=ind_forcing)
            forcing_subset_input["time"] = WRF_input["time"]
            WRF_input = WRF_input.merge(forcing_subset_input)

        if xarray_static:
            N_time_dims = len(WRF_subset["time"])
            static_subset_input = xarray_static.expand_dims(dim={"time": N_time_dims})
            static_subset_input = static_subset_input.assign_coords({"time": WRF_subset["time"]})
            static_subset_input = static_subset_input.isel(time=slice(0, history_len, 1))
            static_subset_input["time"] = WRF_input["time"]
            WRF_input = WRF_input.merge(static_subset_input)

        WRF_target = WRF_subset.isel(time=slice(history_len, history_len + 1, 1)).load()

        if list_diag_ds:
            diagnostic_subset = list_diag_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))
            diagnostic_subset = diagnostic_subset.isel(time=slice(history_len, history_len + 1, 1)).load()
            WRF_target = WRF_target.merge(diagnostic_subset)

        ds_upper_outside = _expand_anchor_dataset(list_upper_ds_outside[int(ind_file)], boundary_seq_len)

        if list_surf_ds_outside:
            ds_surf_outside = _expand_anchor_dataset(list_surf_ds_outside[int(ind_file)], boundary_seq_len)
            ds_outside = xr.merge([ds_upper_outside, ds_surf_outside])
        else:
            ds_outside = ds_upper_outside

        t0 = WRF_input["time"].values
        t1 = WRF_target["time"].values
        t2 = ds_outside["time"].values
        time_encode = encode_datetime64(np.concatenate([t0, t1, t2]))

        sample = Sample_WRF(
            WRF_input=WRF_input,
            WRF_target=WRF_target,
            boundary_input=ds_outside,
            time_encode=time_encode,
            datetime_index=datetime_as_number,
        )

        if transform:
            sample = transform(sample)

        sample["index"] = index
        sample["datetime"] = [
            int(WRF_input.time.values[0].astype("datetime64[s]").astype(int)),
            int(WRF_target.time.values[0].astype("datetime64[s]").astype(int)),
        ]

    except Exception as e:
        logger.error(f"Error processing index {tuple_index}: {e}")
        raise

    return sample


class WoFSMultiStep(torch.utils.data.Dataset):
    """
    WoFS multi-step dataset with same-file `t=0` conditioning.

    `x_boundary` / `x_surf_boundary` are always derived from `time=0` of the
    same WoFS case/member file and remain fixed across a rollout.

    `target_start_step` controls the first forecast lead included in learning:
    - 1 -> first supervised step is `t0 -> t1`
    - 2 -> first supervised step is `t1 -> t2`
    """

    def __init__(
        self,
        param_interior,
        param_outside,
        transform=None,
        seed=42,
        rank=0,
        world_size=1,
        max_forecast_len=None,
    ):
        varname_upper_air = param_interior["varname_upper_air"]
        varname_surface = param_interior["varname_surface"]
        varname_dyn_forcing = param_interior["varname_dyn_forcing"]
        varname_forcing = param_interior["varname_forcing"]
        varname_static = param_interior["varname_static"]
        varname_diagnostic = param_interior["varname_diagnostic"]
        filenames = sorted(param_interior["filenames"])
        filename_surface = param_interior["filename_surface"]
        filename_dyn_forcing = param_interior["filename_dyn_forcing"]
        filename_forcing = param_interior["filename_forcing"]
        filename_static = param_interior["filename_static"]
        filename_diagnostic = param_interior["filename_diagnostic"]

        all_ds = [get_forward_data(fn) for fn in filenames]

        self.list_upper_ds = [filter_ds(ds, varname_upper_air) for ds in all_ds]

        if filename_surface:
            self.list_surf_ds = [filter_ds(ds, varname_surface) for ds in all_ds]
        else:
            self.list_surf_ds = False

        if filename_dyn_forcing:
            self.list_dyn_forcing_ds = [filter_ds(ds, varname_dyn_forcing) for ds in all_ds]
        else:
            self.list_dyn_forcing_ds = False

        if filename_diagnostic:
            self.list_diag_ds = [filter_ds(ds, varname_diagnostic) for ds in all_ds]
        else:
            self.list_diag_ds = False

        self.history_len = param_interior["history_len"]
        self.forecast_len = param_interior["forecast_len"]
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.max_forecast_len = max_forecast_len
        self.target_start_step = int(param_interior.get("target_start_step", 1))
        if self.target_start_step < 1:
            raise ValueError("target_start_step must be >= 1")
        self.start_index_offset = max(0, self.target_start_step - self.history_len)

        ind_start = 0
        self.WRF_file_indices = {}
        for ind_file, WRF_file_xarray in enumerate(self.list_upper_ds):
            available = len(WRF_file_xarray["time"]) - (self.history_len + self.forecast_len + 1) + 1
            available -= self.start_index_offset
            if available <= 0:
                continue
            self.WRF_file_indices[str(ind_file)] = [available, ind_start, ind_start + available - 1]
            ind_start += available

        self.total_length = ind_start

        self.filename_forcing = filename_forcing
        if self.filename_forcing is not None:
            ds = get_forward_data(filename_forcing)
            self.xarray_forcing = drop_var_from_dataset(ds, varname_forcing).load()
        else:
            self.xarray_forcing = False

        self.filename_static = filename_static
        if self.filename_static is not None:
            ds = get_forward_data(filename_static)
            self.xarray_static = drop_var_from_dataset(ds, varname_static).load()
        else:
            self.xarray_static = False

        self.varname_upper_air_outside = param_outside["varname_upper_air"]
        self.varname_surface_outside = param_outside.get("varname_surface") or []

        self.list_upper_ds_outside = [filter_ds(ds, self.varname_upper_air_outside) for ds in all_ds]
        if self.varname_surface_outside:
            self.list_surf_ds_outside = [filter_ds(ds, self.varname_surface_outside) for ds in all_ds]
        else:
            self.list_surf_ds_outside = False

        self.history_len_outside = param_outside["history_len"]
        self.forecast_len_outside = param_outside["forecast_len"]
        self.boundary_seq_len = max(1, self.history_len_outside + self.forecast_len_outside)

        self.transform = transform
        self.rng = np.random.default_rng(seed=seed)

        self.worker = partial(
            worker,
            WRF_file_indices=self.WRF_file_indices,
            list_upper_ds=self.list_upper_ds,
            list_surf_ds=self.list_surf_ds,
            list_dyn_forcing_ds=self.list_dyn_forcing_ds,
            list_diag_ds=self.list_diag_ds,
            xarray_forcing=self.xarray_forcing,
            xarray_static=self.xarray_static,
            history_len=self.history_len,
            list_upper_ds_outside=self.list_upper_ds_outside,
            list_surf_ds_outside=self.list_surf_ds_outside,
            boundary_seq_len=self.boundary_seq_len,
            start_index_offset=self.start_index_offset,
            transform=self.transform,
        )

        self.current_epoch = None

    def __len__(self):
        return self.total_length

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, index):
        rollout_samples: List[Dict[str, Any]] = []

        for step_offset in range(self.forecast_len + 1):
            current_index = index + step_offset
            sample = self.worker((index, current_index))
            sample["forecast_step"] = step_offset + 1
            sample["index"] = current_index
            sample["stop_forecast"] = step_offset == self.forecast_len
            rollout_samples.append(sample)

        return _stack_rollout_samples(rollout_samples)