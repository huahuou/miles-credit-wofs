import logging
from collections import OrderedDict
import numpy as np
import xarray as xr
import re
from pathlib import Path

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
        # Store configuration but do not eagerly open files here.
        self.varname_upper_air = param_interior["varname_upper_air"]
        self.varname_surface = param_interior["varname_surface"]
        self.varname_dyn_forcing = param_interior["varname_dyn_forcing"]
        self.varname_forcing = param_interior["varname_forcing"]
        self.varname_static = param_interior["varname_static"]
        self.varname_diagnostic = param_interior["varname_diagnostic"]
        # Only include files of the form: wofs_YYYYMMDD_HHMM_memNN.zarr
        pattern = re.compile(r"wofs_\d{8}_\d{4}_mem\d+\.zarr$")
        self.filenames = sorted(
            f for f in param_interior["filenames"] if pattern.search(Path(f).name)
        )
        self.filename_surface = param_interior["filename_surface"]
        self.filename_dyn_forcing = param_interior["filename_dyn_forcing"]
        self.filename_forcing = param_interior["filename_forcing"]
        self.filename_static = param_interior["filename_static"]
        self.filename_diagnostic = param_interior["filename_diagnostic"]

        self.history_len = param_interior["history_len"]
        self.forecast_len = param_interior["forecast_len"]
        self.total_seq_len = self.history_len + self.forecast_len

        # outside branch parameters
        self.varname_upper_air_outside = param_outside["varname_upper_air"]
        self.varname_surface_outside = param_outside.get("varname_surface") or []
        self.history_len_outside = param_outside["history_len"]
        self.forecast_len_outside = param_outside["forecast_len"]
        self.boundary_seq_len = max(1, self.history_len_outside + self.forecast_len_outside)

        self.transform = transform
        self.rng = np.random.default_rng(seed=seed)

        # internal: will be opened in _open_datasets per worker
        self._opened = False
        self.list_upper_ds = None
        self.list_surf_ds = None
        self.list_dyn_forcing_ds = None
        self.list_diag_ds = None
        self.filename_forcing = self.filename_forcing
        self.xarray_forcing = False
        self.filename_static = self.filename_static
        self.xarray_static = False
        self.list_upper_ds_outside = None
        self.list_surf_ds_outside = None
        self.WRF_file_indices = None
        self.total_length = None
        self.file_time_lengths = None
        self._case_cache: "OrderedDict[int, dict]" = OrderedDict()
        self.max_open_files_per_worker = int(param_interior.get("max_open_files_per_worker", 4))

        precomputed_index_metadata = param_interior.get("precomputed_index_metadata")
        if precomputed_index_metadata:
            self.WRF_file_indices = dict(precomputed_index_metadata.get("WRF_file_indices", {}))
            self.file_time_lengths = dict(precomputed_index_metadata.get("file_time_lengths", {}))
            self.total_length = int(precomputed_index_metadata.get("total_length", 0))

        self.logger = logging.getLogger(__name__)

    def _build_index_mapping(self):
        if self.WRF_file_indices is not None and self.total_length is not None:
            return

        ind_start = 0
        self.WRF_file_indices = {}
        self.file_time_lengths = {}

        for ind_file, filename in enumerate(self.filenames):
            if filename.endswith((".nc", ".nc4")):
                ds = xr.open_dataset(filename, decode_times=False)
            else:
                ds = xr.open_zarr(filename, decode_times=False)
            try:
                n_time = int(ds.sizes["time"])
                self.file_time_lengths[str(ind_file)] = n_time

                available = n_time - self.total_seq_len + 1
                if available <= 0:
                    continue

                self.WRF_file_indices[str(ind_file)] = [available, ind_start, ind_start + available - 1]
                ind_start += available
            finally:
                try:
                    ds.close()
                except Exception:
                    pass

        self.total_length = int(ind_start)

    def _load_shared_aux_datasets(self):
        if self.filename_forcing is not None and self.xarray_forcing is False:
            ds = get_forward_data(self.filename_forcing)
            self.xarray_forcing = drop_var_from_dataset(ds, self.varname_forcing).load()
        elif self.filename_forcing is None:
            self.xarray_forcing = False

        if self.filename_static is not None and self.xarray_static is False:
            ds = get_forward_data(self.filename_static)
            self.xarray_static = drop_var_from_dataset(ds, self.varname_static).load()
        elif self.filename_static is None:
            self.xarray_static = False

    def _get_case_views(self, ind_file: str) -> dict:
        key = int(ind_file)
        if key in self._case_cache:
            self._case_cache.move_to_end(key)
            return self._case_cache[key]

        ds = get_forward_data(self.filenames[key])
        views = {
            "ds": ds,
            "upper": filter_ds(ds, self.varname_upper_air),
            "surface": filter_ds(ds, self.varname_surface) if self.filename_surface else False,
            "dyn": filter_ds(ds, self.varname_dyn_forcing) if self.filename_dyn_forcing else False,
            "diag": filter_ds(ds, self.varname_diagnostic) if self.filename_diagnostic else False,
            "upper_out": filter_ds(ds, self.varname_upper_air_outside),
            "surf_out": filter_ds(ds, self.varname_surface_outside) if self.varname_surface_outside else False,
        }

        self._case_cache[key] = views
        self._case_cache.move_to_end(key)

        while len(self._case_cache) > self.max_open_files_per_worker:
            _, old_views = self._case_cache.popitem(last=False)
            try:
                old_views["ds"].close()
            except Exception:
                pass

        return views

    def _open_datasets(self):
        """
        Open zarr/netcdf files and prepare internal structures. Intended to run
        in DataLoader worker processes (or first __getitem__ call).
        """
        if getattr(self, "_opened", False):
            return

        self._build_index_mapping()
        self._load_shared_aux_datasets()

        self._opened = True

    def __len__(self):
        self._build_index_mapping()
        return self.total_length

    def _expand_anchor_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        anchor = dataset.isel(time=slice(0, 1)).load()

        if self.boundary_seq_len == 1:
            return anchor

        anchor_time = anchor["time"].values[0]
        repeated = xr.concat([anchor] * self.boundary_seq_len, dim="time")
        repeated = repeated.assign_coords(time=np.array([anchor_time] * self.boundary_seq_len))
        return repeated

    def _build_boundary_input(self, case: dict) -> xr.Dataset:
        ds_upper_outside = self._expand_anchor_dataset(case["upper_out"])

        if case["surf_out"]:
            ds_surf_outside = self._expand_anchor_dataset(case["surf_out"])
            return xr.merge([ds_upper_outside, ds_surf_outside])

        return ds_upper_outside

    def __getitem__(self, index):
        # ensure datasets are opened in this worker
        if not self._opened:
            self._open_datasets()

        ind_file = find_key_for_number(index, self.WRF_file_indices)
        if ind_file is None:
            raise KeyError(f"No WoFS file mapping found for start index {index}")

        file_range = self.WRF_file_indices[ind_file]
        ind_start_in_file = index - file_range[1]

        case = self._get_case_views(ind_file)

        ind_largest = int(case["upper"].sizes["time"]) - (self.history_len + self.forecast_len + 1)
        if ind_start_in_file > ind_largest:
            ind_start_in_file = ind_largest

        ind_end_in_file = ind_start_in_file + self.history_len + self.forecast_len

        WRF_subset = case["upper"].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))

        if case["surface"]:
            surface_subset = case["surface"].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))
            WRF_subset = WRF_subset.merge(surface_subset)

        ind_end_time = len(WRF_subset["time"])
        datetime_as_number = WRF_subset.time.values.astype("datetime64[s]").astype(int)

        WRF_input = WRF_subset.isel(time=slice(0, self.history_len, 1)).load()

        if case["dyn"]:
            dyn_forcing_subset = case["dyn"].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))
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

        if case["diag"]:
            diagnostic_subset = case["diag"].isel(time=slice(ind_start_in_file, ind_end_in_file + 1))
            diagnostic_subset = diagnostic_subset.isel(time=slice(self.history_len, ind_end_time, 1)).load()
            WRF_target = WRF_target.merge(diagnostic_subset)

        ds_outside = self._build_boundary_input(case)

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