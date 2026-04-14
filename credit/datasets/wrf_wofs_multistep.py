import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
from collections import OrderedDict
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
        # Save config info but do NOT open heavy datasets here. We'll open in workers.
        self.varname_upper_air = param_interior["varname_upper_air"]
        self.varname_surface = param_interior["varname_surface"]
        self.varname_dyn_forcing = param_interior["varname_dyn_forcing"]
        self.varname_forcing = param_interior["varname_forcing"]
        self.varname_static = param_interior["varname_static"]
        self.varname_diagnostic = param_interior["varname_diagnostic"]
        self.filenames = sorted(param_interior["filenames"])
        self.filename_surface = param_interior["filename_surface"]
        self.filename_dyn_forcing = param_interior["filename_dyn_forcing"]
        self.filename_forcing = param_interior["filename_forcing"]
        self.filename_static = param_interior["filename_static"]
        self.filename_diagnostic = param_interior["filename_diagnostic"]

        # workflow parameters
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
        self.zarr_time_chunk = int(param_interior.get("zarr_time_chunk", 0))
        if self.zarr_time_chunk <= 0:
            self.zarr_time_chunk = self._recommended_zarr_time_chunk(self.forecast_len)
        self.zarr_chunks = {"time": self.zarr_time_chunk} if self.zarr_time_chunk > 0 else None
        self.max_open_files = int(param_interior.get("max_open_files_per_worker", 8))

        # outside branch parameters (store them, will be used in _open_datasets)
        self.varname_upper_air_outside = param_outside["varname_upper_air"]
        self.varname_surface_outside = param_outside.get("varname_surface") or []
        self.history_len_outside = param_outside["history_len"]
        self.forecast_len_outside = param_outside["forecast_len"]
        self.boundary_seq_len = max(1, self.history_len_outside + self.forecast_len_outside)

        # transforms & rng
        self.transform = transform
        self.rng = np.random.default_rng(seed=seed)
        self.has_surface = bool(self.filename_surface) and len(self.varname_surface) > 0
        self.has_dyn_forcing = bool(self.filename_dyn_forcing) and len(self.varname_dyn_forcing) > 0
        self.has_diagnostic = bool(self.filename_diagnostic) and len(self.varname_diagnostic) > 0

        # internal state: we'll open datasets per-worker via _open_datasets()
        self._opened = False
        self.case_ds_cache = OrderedDict()
        self.xarray_forcing = False
        self.xarray_static = False
        self.WRF_file_indices = None
        self.total_length = None
        self.file_n_time = None
        self.samples_per_file = None
        self.cumulative_samples = None
        self._all_case_vars = None
        self._forcing_lookup = None
        self._forcing_time_index_cache = {}
        self.worker = None
        self.current_epoch = None
        self.cached_outside_anchors = OrderedDict()
        self.max_cached_outside_anchors = 64
        # self.max_cached_outside_anchors = len(self.filenames)

    def _build_all_case_vars(self) -> List[str]:
        seen = set()
        ordered: List[str] = []
        groups = [
            self.varname_upper_air,
            self.varname_surface if self.has_surface else [],
            self.varname_dyn_forcing if self.has_dyn_forcing else [],
            self.varname_diagnostic if self.has_diagnostic else [],
            self.varname_upper_air_outside,
            self.varname_surface_outside,
        ]
        for group in groups:
            for var in group:
                if var not in seen:
                    ordered.append(var)
                    seen.add(var)
        return ordered

    @staticmethod
    def _safe_close_dataset(ds):
        try:
            ds.close()
        except Exception:
            pass

    def _evict_if_needed(self):
        while len(self.case_ds_cache) > self.max_open_files:
            evicted_file_idx, ds = self.case_ds_cache.popitem(last=False)
            self._safe_close_dataset(ds)
            self.cached_outside_anchors.pop(evicted_file_idx, None)
            self._forcing_time_index_cache.pop(evicted_file_idx, None)

    def _get_case_ds(self, file_index: int) -> xr.Dataset:
        cached = self.case_ds_cache.get(file_index)
        if cached is not None:
            self.case_ds_cache.move_to_end(file_index)
            return cached

        ds = get_forward_data(self.filenames[file_index], zarr_chunks=self.zarr_chunks)
        if self._all_case_vars is None:
            self._all_case_vars = self._build_all_case_vars()
        ds = filter_ds(ds, self._all_case_vars)
        self.case_ds_cache[file_index] = ds
        self._evict_if_needed()
        return ds

    def _ensure_file_time_sizes(self):
        if self.file_n_time is not None:
            return

        self.file_n_time = []
        for fn in self.filenames:
            try:
                if fn.endswith('.zarr'):
                    import zarr
                    z = zarr.open(fn, mode='r')
                    first_var = self.varname_upper_air[0]
                    if first_var in z:
                        n_time = z[first_var].shape[0]
                    else:
                        ds = get_forward_data(fn, zarr_chunks=self.zarr_chunks)
                        n_time = int(ds["time"].size)
                        ds.close()
                elif fn.endswith('.zarr.zip') or fn.endswith('.zarr.zip/'):
                    ds = get_forward_data(fn, zarr_chunks=self.zarr_chunks)
                    n_time = int(ds["time"].size)
                    ds.close()
                else:
                    ds = get_forward_data(fn, zarr_chunks=self.zarr_chunks)
                    n_time = int(ds["time"].size)
                    ds.close()
                    del ds
            except Exception:
                ds = get_forward_data(fn, zarr_chunks=self.zarr_chunks)
                try:
                    n_time = int(filter_ds(ds, self.varname_upper_air)["time"].size)
                finally:
                    try:
                        ds.close()
                    except Exception:
                        pass
                    del ds

            self.file_n_time.append(int(np.asarray(n_time).item()))

    def _build_file_indices(self):
        self._ensure_file_time_sizes()
        self.samples_per_file = []
        for n_time in self.file_n_time:
            available = n_time - (self.history_len + self.forecast_len + 1) + 1
            available -= self.start_index_offset
            self.samples_per_file.append(max(0, int(available)))

        self.cumulative_samples = np.cumsum(self.samples_per_file, dtype=np.int64)
        self.total_length = int(self.cumulative_samples[-1]) if len(self.cumulative_samples) > 0 else 0

        ind_start = 0
        self.WRF_file_indices = {}
        for ind_file, available in enumerate(self.samples_per_file):
            if available <= 0:
                continue
            self.WRF_file_indices[str(ind_file)] = [available, ind_start, ind_start + available - 1]
            ind_start += available

    def file_index_for_global_index(self, index: int) -> int:
        if self.cumulative_samples is None:
            self._build_file_indices()

        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}")
        return int(np.searchsorted(self.cumulative_samples, index, side="right"))

    def _build_forcing_time_index_for_file(self, file_index: int):
        if not self.xarray_forcing or file_index in self._forcing_time_index_cache:
            return

        if self._forcing_lookup is None:
            forcing_times = extract_month_day_hour(np.array(self.xarray_forcing["time"]))
            self._forcing_lookup = {key: i for i, key in enumerate(forcing_times)}

        case_ds = self._get_case_ds(file_index)
        case_times = extract_month_day_hour(np.array(case_ds["time"].values))
        forcing_indices = []
        for key in case_times:
            if key not in self._forcing_lookup:
                raise KeyError(f"Missing forcing timestamp key {key} for file index {file_index}")
            forcing_indices.append(self._forcing_lookup[key])
        self._forcing_time_index_cache[file_index] = np.asarray(forcing_indices, dtype=np.int64)

    @staticmethod
    def _recommended_zarr_time_chunk(forecast_len: int) -> int:
        """Heuristic tuned for WoFS multi-step contiguous rollout reads.

        Empirically:
        - very short rollouts (forecast_len <= 1): smaller chunk around 3
        - short/medium rollouts (<= 4): larger chunk around 8
        - longer rollouts (> 4): moderate chunk around 6
        """
        if forecast_len <= 1:
            return 3
        if forecast_len <= 4:
            return 8
        return 6

    def _get_boundary_anchor(self, file_index: int) -> xr.Dataset:
        """Get pre-normalized boundary anchor with bounded in-worker cache."""
        cached = self.cached_outside_anchors.get(file_index)
        if cached is not None:
            self.cached_outside_anchors.move_to_end(file_index)
            return cached

        case_ds = self._get_case_ds(file_index)
        anchor_upper = filter_ds(case_ds, self.varname_upper_air_outside).isel(time=slice(0, 1)).load()
        if self.boundary_seq_len > 1:
            anchor_time = anchor_upper["time"].values[0]
            anchor_upper = xr.concat([anchor_upper] * self.boundary_seq_len, dim="time")
            anchor_upper = anchor_upper.assign_coords(time=np.array([anchor_time] * self.boundary_seq_len))

        if self.varname_surface_outside:
            anchor_surf = filter_ds(case_ds, self.varname_surface_outside).isel(time=slice(0, 1)).load()
            if self.boundary_seq_len > 1:
                anchor_time = anchor_surf["time"].values[0]
                anchor_surf = xr.concat([anchor_surf] * self.boundary_seq_len, dim="time")
                anchor_surf = anchor_surf.assign_coords(time=np.array([anchor_time] * self.boundary_seq_len))
            anchor = xr.merge([anchor_upper, anchor_surf])
        else:
            anchor = anchor_upper

        if self.transform is not None:
            _normalizer = None
            if hasattr(self.transform, "transforms"):
                for t in self.transform.transforms:
                    if hasattr(t, "mean_ds_outside") and hasattr(t, "std_ds_outside"):
                        _normalizer = t
                        break
            elif hasattr(self.transform, "mean_ds_outside"):
                _normalizer = self.transform

            if _normalizer is not None:
                for varname in anchor.keys():
                    anchor[varname] = (
                        anchor[varname]
                        - _normalizer.mean_ds_outside[varname]
                    ) / _normalizer.std_ds_outside[varname]

        self.cached_outside_anchors[file_index] = anchor
        if len(self.cached_outside_anchors) > self.max_cached_outside_anchors:
            self.cached_outside_anchors.popitem(last=False)
        return anchor

    def _open_datasets(self):
        """
        Called inside each DataLoader worker (or first __getitem__ call) to open
        xarray/zarr datasets and build the worker partial. This avoids
        pickling/opening in main process.
        """
        if getattr(self, "_opened", False):
            return

        if self._all_case_vars is None:
            self._all_case_vars = self._build_all_case_vars()
        self._build_file_indices()

        # forcing and static (load once)
        if self.filename_forcing is not None:
            ds = get_forward_data(self.filename_forcing, zarr_chunks=self.zarr_chunks)
            self.xarray_forcing = drop_var_from_dataset(ds, self.varname_forcing).load()
        else:
            self.xarray_forcing = False

        if self.filename_static is not None:
            ds = get_forward_data(self.filename_static, zarr_chunks=self.zarr_chunks)
            self.xarray_static = drop_var_from_dataset(ds, self.varname_static).load()
        else:
            self.xarray_static = False

        # Boundary anchors are loaded/normalized lazily in _get_boundary_anchor()
        # to avoid high startup latency and per-worker memory spikes.

        # Flag so the transform can skip boundary normalization
        self._boundary_pre_normalized = True

        self._opened = True

    def __del__(self):
        try:
            for ds in self.case_ds_cache.values():
                self._safe_close_dataset(ds)
        except Exception:
            pass

    def __len__(self):
        if getattr(self, "total_length", None) is not None:
            return self.total_length

        self._build_file_indices()
        return self.total_length
    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, index):
        if not self._opened:
            self._open_datasets()

        file_key = find_key_for_number(index, self.WRF_file_indices)
        if file_key is None:
            raise KeyError(f"No WoFS file mapping found for start index {index}")

        file_range = self.WRF_file_indices[file_key]
        file_idx = int(file_key)
        base_start = (index - file_range[1]) + self.start_index_offset
        total_len = self.forecast_len + self.history_len + 1
        case_ds = self._get_case_ds(file_idx)

        # ------------------------------------------------------------------ #
        # 1. Single contiguous disk read for the entire rollout window
        # ------------------------------------------------------------------ #
        upper_chunk = filter_ds(case_ds, self.varname_upper_air).isel(
            time=slice(base_start, base_start + total_len)
        ).load()
        if self.has_surface:
            surf_chunk = filter_ds(case_ds, self.varname_surface).isel(
                time=slice(base_start, base_start + total_len)
            ).load()
            base_chunk = xr.merge([upper_chunk, surf_chunk])
        else:
            base_chunk = upper_chunk

        diag_chunk = None
        if self.has_diagnostic:
            diag_chunk = filter_ds(case_ds, self.varname_diagnostic).isel(
                time=slice(base_start, base_start + total_len)
            ).load()

        # ------------------------------------------------------------------ #
        # 2. Merge auxiliary variables into base_chunk ONCE (not per step)
        # ------------------------------------------------------------------ #
        full_input_chunk = base_chunk
        if self.has_dyn_forcing:
            dyn_chunk = filter_ds(case_ds, self.varname_dyn_forcing).isel(
                time=slice(base_start, base_start + total_len)
            ).load()
            full_input_chunk = xr.merge([full_input_chunk, dyn_chunk])

        if self.xarray_forcing:
            self._build_forcing_time_index_for_file(file_idx)
            forcing_time_index = self._forcing_time_index_cache[file_idx]
            ind_forcing = forcing_time_index[base_start : base_start + total_len]
            forcing_chunk = self.xarray_forcing.isel(time=ind_forcing).copy()
            forcing_chunk["time"] = base_chunk["time"]
            full_input_chunk = xr.merge([full_input_chunk, forcing_chunk])

        if self.xarray_static:
            N_time_dims = len(base_chunk["time"])
            static_chunk = self.xarray_static.expand_dims(dim={"time": N_time_dims}).copy()
            static_chunk = static_chunk.assign_coords({"time": base_chunk["time"]})
            full_input_chunk = xr.merge([full_input_chunk, static_chunk])

        # Pre-normalized, cached boundary anchor
        ds_outside = self._get_boundary_anchor(file_idx)

        # ------------------------------------------------------------------ #
        # 3. Rollout loop — only cheap in-memory isel, no merges
        # ------------------------------------------------------------------ #
        rollout_samples: List[Dict[str, Any]] = []

        for step_offset in range(self.forecast_len + 1):
            rel_start = step_offset
            rel_end = step_offset + self.history_len

            # Input: slice from the pre-merged full_input_chunk
            WRF_input = full_input_chunk.isel(time=slice(rel_start, rel_end))

            # Target: slice from base_chunk (upper + surface only)
            WRF_target = base_chunk.isel(time=slice(rel_end, rel_end + 1))
            if diag_chunk is not None:
                step_diag = diag_chunk.isel(time=slice(rel_end, rel_end + 1))
                WRF_target = xr.merge([WRF_target, step_diag])

            t0 = WRF_input["time"].values
            t1 = WRF_target["time"].values
            t2 = ds_outside["time"].values
            time_encode = encode_datetime64(np.concatenate([t0, t1, t2]))

            datetime_as_number = base_chunk.isel(
                time=slice(rel_start, rel_end + 1)
            ).time.values.astype("datetime64[s]").astype(int)

            sample = Sample_WRF(
                WRF_input=WRF_input,
                WRF_target=WRF_target,
                boundary_input=ds_outside,
                time_encode=time_encode,
                datetime_index=datetime_as_number,
            )

            if self.transform:
                # Tell the transform that boundary data is already normalized
                sample["_boundary_pre_normalized"] = getattr(
                    self, "_boundary_pre_normalized", False
                )
                sample = self.transform(sample)

            sample["index"] = index
            sample["datetime"] = [
                int(t0[0].astype("datetime64[s]").astype(int)),
                int(t1[0].astype("datetime64[s]").astype(int)),
            ]
            sample["forecast_step"] = step_offset + 1
            sample["stop_forecast"] = step_offset == self.forecast_len

            rollout_samples.append(sample)

        return _stack_rollout_samples(rollout_samples)