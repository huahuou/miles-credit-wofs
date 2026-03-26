import numpy as np
import xarray as xr

import torch
import torch.utils.data

from credit.data import Sample_WRF
from credit.data import (
    drop_var_from_dataset,
    encode_datetime64,
    extract_month_day_hour,
    filter_ds,
    find_common_indices,
    find_key_for_number,
    get_forward_data,
)


class WoFSSingleStepDataset(torch.utils.data.Dataset):
    """
    WoFS single-step dataset compatible with the WRF trainer/model interface.

    This loader mirrors `WRFDataset`, except the conditioning branch is derived
    from `time=0` of the same WoFS case/member zarr instead of a separate
    outside-domain boundary dataset.
    """

    def __init__(
        self,
        param_interior,
        param_outside,
        transform=None,
        seed=42,
    ):
        # ========================================================== #
        # WoFS interior variable and filename info
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

        # 1. Upper‐air
        self.list_upper_ds = [filter_ds(ds, varname_upper_air) for ds in all_ds]

        # 2. Surface
        if filename_surface:
            self.list_surf_ds = [filter_ds(ds, varname_surface) for ds in all_ds]
        else:
            self.list_surf_ds = False

        # 3. Dynamic forcing
        if filename_dyn_forcing:
            self.list_dyn_forcing_ds = [filter_ds(ds, varname_dyn_forcing) for ds in all_ds]
        else:
            self.list_dyn_forcing_ds = False

        # 4. Diagnostics
        if filename_diagnostic:
            self.list_diag_ds = [filter_ds(ds, varname_diagnostic) for ds in all_ds]
        else:
            self.list_diag_ds = False

        self.history_len = param_interior["history_len"]
        self.forecast_len = param_interior["forecast_len"]
        self.total_seq_len = self.history_len + self.forecast_len

        ind_start = 0
        self.WRF_file_indices = {}
        for ind_file, WRF_file_xarray in enumerate(self.list_upper_ds):
            self.WRF_file_indices[str(ind_file)] = [
                len(WRF_file_xarray["time"]),
                ind_start,
                ind_start + len(WRF_file_xarray["time"]),
            ]
            ind_start += len(WRF_file_xarray["time"]) + 1

        # -------------------------------------------------------------------------- #
        # forcing file
        self.filename_forcing = filename_forcing
        if self.filename_forcing is not None:
            ds = get_forward_data(filename_forcing)
            self.xarray_forcing = drop_var_from_dataset(ds, varname_forcing).load()
        else:
            self.xarray_forcing = False

        # -------------------------------------------------------------------------- #
        # static file
        self.filename_static = filename_static
        if self.filename_static is not None:
            ds = get_forward_data(filename_static)
            self.xarray_static = drop_var_from_dataset(ds, varname_static).load()
        else:
            self.xarray_static = False

        # ========================================================== #
        # WoFS t=0 conditioning branch from the same case/member file
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

        # ========================================================== #
        self.transform = transform
        self.rng = np.random.default_rng(seed=seed)

    def __len__(self):
        total_len = 0
        for WRF_file_xarray in self.list_upper_ds:
            total_len += len(WRF_file_xarray["time"]) - self.total_seq_len + 1
        return total_len

    def _expand_anchor_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        anchor = dataset.isel(time=slice(0, 1)).load()

        if self.boundary_seq_len == 1:
            return anchor

        anchor_time = anchor["time"].values[0]
        repeated = xr.concat([anchor] * self.boundary_seq_len, dim="time")
        repeated = repeated.assign_coords(time=np.array([anchor_time] * self.boundary_seq_len))
        return repeated

    def _build_boundary_input(self, ind_file: str) -> xr.Dataset:
        ds_upper_outside = self._expand_anchor_dataset(self.list_upper_ds_outside[int(ind_file)])

        if self.list_surf_ds_outside:
            ds_surf_outside = self._expand_anchor_dataset(self.list_surf_ds_outside[int(ind_file)])
            return xr.merge([ds_upper_outside, ds_surf_outside])

        return ds_upper_outside

    def __getitem__(self, index):
        ind_file = find_key_for_number(index, self.WRF_file_indices)

        ind_start = self.WRF_file_indices[ind_file][1]
        ind_start_in_file = index - ind_start

        ind_largest = len(self.list_upper_ds[int(ind_file)]["time"]) - (self.history_len + self.forecast_len + 1)
        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        ind_end_in_file = ind_start_in_file + self.history_len + self.forecast_len

        WRF_subset = self.list_upper_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))

        if self.list_surf_ds:
            surface_subset = self.list_surf_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))
            WRF_subset = WRF_subset.merge(surface_subset)

        ind_end_time = len(WRF_subset["time"])
        datetime_as_number = WRF_subset.time.values.astype("datetime64[s]").astype(int)

        WRF_input = WRF_subset.isel(time=slice(0, self.history_len, 1)).load()

        if self.list_dyn_forcing_ds:
            dyn_forcing_subset = self.list_dyn_forcing_ds[int(ind_file)].isel(
                time=slice(ind_start_in_file, ind_end_in_file + 1)
            )
            dyn_forcing_subset = dyn_forcing_subset.isel(time=slice(0, self.history_len, 1)).load()
            WRF_input = WRF_input.merge(dyn_forcing_subset)

        if self.xarray_forcing:
            month_day_forcing = extract_month_day_hour(np.array(self.xarray_forcing["time"]))
            month_day_inputs = extract_month_day_hour(np.array(WRF_input["time"]))
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            forcing_subset_input = self.xarray_forcing.isel(time=ind_forcing)
            forcing_subset_input["time"] = WRF_input["time"]
            WRF_input = WRF_input.merge(forcing_subset_input)

        if self.xarray_static:
            N_time_dims = len(WRF_subset["time"])
            static_subset_input = self.xarray_static.expand_dims(dim={"time": N_time_dims})
            static_subset_input = static_subset_input.assign_coords({"time": WRF_subset["time"]})
            static_subset_input = static_subset_input.isel(time=slice(0, self.history_len, 1))
            static_subset_input["time"] = WRF_input["time"]
            WRF_input = WRF_input.merge(static_subset_input)

        WRF_target = WRF_subset.isel(time=slice(self.history_len, ind_end_time, 1)).load()

        if self.list_diag_ds:
            diagnostic_subset = self.list_diag_ds[int(ind_file)].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))
            diagnostic_subset = diagnostic_subset.isel(time=slice(self.history_len, ind_end_time, 1)).load()
            WRF_target = WRF_target.merge(diagnostic_subset)

        ds_outside = self._build_boundary_input(ind_file)

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

        if self.transform:
            sample = self.transform(sample)

        sample["index"] = index
        return sample