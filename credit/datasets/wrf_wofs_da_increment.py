"""
WoFS DA Increment Dataset
-------------------------
Constructs (innovation, increment) training pairs from consecutive forecast
timesteps for data assimilation increment learning.

Training signal (Approach 1 — Temporal Self-Supervision):
  - innovation  = normalized(REFL_10CM_t1) - normalized(REFL_10CM_t0)
  - increment   = normalized(QRAIN_t1) - normalized(QRAIN_t0),
                   normalized(QNRAIN_t1) - normalized(QNRAIN_t0)

Returns pre-normalized tensor dicts compatible with the single-step WRF trainer
(trainerWRF.py). Bypasses NormalizeWRF / ToTensorWRF transforms because the
dataset handles normalization and tensor conversion internally.

Tensor dict keys returned
~~~~~~~~~~~~~~~~~~~~~~~~~
  x                : (time=1, num_prog_vars, levels, H, W)   — prognostic vars at t0
  x_forcing_static : (time=1, num_context_chans, H, W)       — context 3D vars (flattened) + dyn forcing
  x_boundary       : (time=1, num_obs_vars, levels, H, W)    — REFL_10CM innovation
  y                : (time=1, num_prog_vars, levels, H, W)   — QRAIN/QNRAIN increment
  x_time_encode    : (time_encode_dim,)                       — temporal encoding
  index            : int                                       — sample index
"""

import logging
from collections import OrderedDict
import numpy as np
import xarray as xr

import torch
import torch.utils.data

from credit.data import (
    encode_datetime64,
    filter_ds,
    get_forward_data,
)

logger = logging.getLogger(__name__)

_CONC_EPS = 1e-4
_CONC_MAX = 2.5
_LOG_EPS = float(np.log(_CONC_EPS))
_NEG_LOG_EPS = float(-np.log(_CONC_EPS))

CONCENTRATION_VARS = {
    "QRAIN",
    "QNRAIN",
    "QSNOW",
    "QNSNOW",
    "QGRAUP",
    "QNGRAUPEL",
    "QVGRAUPEL",
    "QHAIL",
    "QNHAIL",
    "QVHAIL",
    "QICE",
    "QNICE",
}


class WoFSDAIncrementDataset(torch.utils.data.Dataset):
    """
    DA increment dataset for learning QRAIN/QNRAIN corrections from
    REFL_10CM innovations.
    """

    def __init__(self, param_interior, param_outside, conf, seed=42):
        # -- prognostic variables (input + output, 3D with levels) ----------
        self.varname_prognostic = param_interior["varname_upper_air"]

        # -- context 3D variables (input-only, flattened to var*level) ------
        self.varname_context = param_interior["varname_context_upper_air"]

        # -- dynamic forcing variables (input-only, 2D) --------------------
        self.varname_dyn_forcing = param_interior["varname_dyn_forcing"]

        # -- observation variable for innovation (boundary encoder) ---------
        self.varname_observation = param_outside["varname_upper_air"]

        # -- file paths ----------------------------------------------------
        self.filenames = sorted(param_interior["filenames"])
        self.filename_dyn_forcing = param_interior.get("filename_dyn_forcing")
        if self.filename_dyn_forcing:
            self.filename_dyn_forcing = sorted(self.filename_dyn_forcing)

        # -- sequence config -----------------------------------------------
        self.history_len = param_interior["history_len"]
        self.forecast_len = param_interior["forecast_len"]
        self.total_seq_len = self.history_len + self.forecast_len  # 1

        # -- normalization paths -------------------------------------------
        self.mean_path = conf["data"]["mean_path"]
        self.std_path = conf["data"]["std_path"]

        self.levels = conf["data"]["levels"]
        self.rng = np.random.default_rng(seed=seed)
        self.output_dtype = torch.float32
        self.max_open_files = int(conf["data"].get("max_open_files_per_worker", 8))
        self.zarr_time_chunk = int(conf["data"].get("zarr_time_chunk", 2))
        self.zarr_chunks = {"time": self.zarr_time_chunk} if self.zarr_time_chunk > 0 else None

        # -- internal state (lazy-opened in workers) -----------------------
        self._opened = False
        self.total_length = None
        self.file_n_time = None
        self.samples_per_file = None
        self.cumulative_samples = None
        self.upper_ds_cache = OrderedDict()
        self.dyn_ds_cache = OrderedDict()

    @staticmethod
    def _forward_concentration_numpy(x: np.ndarray) -> np.ndarray:
        x64 = np.asarray(x, dtype=np.float64)
        transformed = 0.5 * np.minimum(x64, _CONC_MAX) + 0.5 * (
            np.log(np.maximum(x64, _CONC_EPS)) - _LOG_EPS
        ) / _NEG_LOG_EPS
        return transformed.astype(x.dtype, copy=False)

    @staticmethod
    def _inverse_concentration_numpy(x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64)

        y_low = 0.5 * _CONC_EPS
        y_high = 0.5 * _CONC_MAX + 0.5 * (np.log(_CONC_MAX) - _LOG_EPS) / _NEG_LOG_EPS

        out = np.empty_like(y)

        mask_low = y <= y_low
        if np.any(mask_low):
            out[mask_low] = 2.0 * y[mask_low]

        mask_high = y >= y_high
        if np.any(mask_high):
            out[mask_high] = _CONC_EPS * np.exp(2.0 * _NEG_LOG_EPS * (y[mask_high] - 0.5 * _CONC_MAX))

        mask_mid = (~mask_low) & (~mask_high)
        if np.any(mask_mid):
            target = y[mask_mid]
            lo = np.full_like(target, _CONC_EPS)
            hi = np.full_like(target, _CONC_MAX)
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                f_mid = 0.5 * mid + 0.5 * (np.log(mid) - _LOG_EPS) / _NEG_LOG_EPS
                go_right = f_mid < target
                lo = np.where(go_right, mid, lo)
                hi = np.where(go_right, hi, mid)
            out[mask_mid] = 0.5 * (lo + hi)

        return out.astype(x.dtype, copy=False)

    def _forward_var_transform(self, var_values: np.ndarray, var_name: str) -> np.ndarray:
        if var_name in CONCENTRATION_VARS:
            return self._forward_concentration_numpy(var_values)
        return var_values

    def _inverse_var_transform(self, var_values: np.ndarray, var_name: str) -> np.ndarray:
        if var_name in CONCENTRATION_VARS:
            return self._inverse_concentration_numpy(var_values)
        return var_values

    def denormalize_increment(self, normalized_increment: np.ndarray, base_state_t0: np.ndarray, var_name: str) -> np.ndarray:
        """Convert normalized-space increment (delta z) to physical increment.

        Args:
            normalized_increment: Array of predicted increments in normalized space.
            base_state_t0: Physical state at t0 for the same variable.
            var_name: Variable name.

        Returns:
            Physical increment in original variable units.
        """
        std = self._broadcast_stats_like(self._std_values[var_name], normalized_increment)
        mean = self._broadcast_stats_like(self._mean_values[var_name], normalized_increment)
        z0 = self._normalize_array(base_state_t0, var_name)
        z1 = z0 + normalized_increment
        transformed_1 = z1 * std + mean
        x1 = self._inverse_var_transform(transformed_1, var_name)
        return x1 - base_state_t0

    def _build_index_map(self):
        if self.file_n_time is None:
            self.file_n_time = []
            for fn in self.filenames:
                try:
                    if fn.endswith(".zarr"):
                        import zarr

                        z = zarr.open(fn, mode="r")
                        first_var = self.varname_prognostic[0]
                        n_time = z[first_var].shape[0] if first_var in z else 0
                        if n_time == 0:
                            ds = get_forward_data(fn)
                            try:
                                n_time = int(ds["time"].size)
                            finally:
                                ds.close()
                    else:
                        ds = get_forward_data(fn)
                        try:
                            n_time = int(ds["time"].size)
                        finally:
                            ds.close()
                except Exception:
                    ds = get_forward_data(fn)
                    try:
                        n_time = int(filter_ds(ds, self.varname_prognostic)["time"].size)
                    finally:
                        try:
                            ds.close()
                        except Exception:
                            pass

                self.file_n_time.append(int(np.asarray(n_time).item()))

        self.samples_per_file = [max(0, n_time - self.total_seq_len) for n_time in self.file_n_time]
        self.cumulative_samples = np.cumsum(self.samples_per_file, dtype=np.int64)
        self.total_length = int(self.cumulative_samples[-1]) if len(self.cumulative_samples) > 0 else 0

    def _locate_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}")

        if self.cumulative_samples is None:
            raise RuntimeError("Dataset index map is not initialized")

        file_idx = int(np.searchsorted(self.cumulative_samples, index, side="right"))
        prev_cum = 0 if file_idx == 0 else int(self.cumulative_samples[file_idx - 1])
        ind_in_file = index - prev_cum
        return file_idx, int(ind_in_file)

    @staticmethod
    def _broadcast_stats_like(stats: np.ndarray, values: np.ndarray) -> np.ndarray:
        stats_arr = np.asarray(stats)
        values_arr = np.asarray(values)

        if stats_arr.ndim != 1 or values_arr.ndim == 0:
            return stats_arr

        stat_len = stats_arr.shape[0]
        matching_axes = [axis for axis, size in enumerate(values_arr.shape) if size == stat_len]
        if not matching_axes:
            return stats_arr

        axis = 1 if len(matching_axes) > 1 and 1 in matching_axes else matching_axes[0]
        reshape_dims = [1] * values_arr.ndim
        reshape_dims[axis] = stat_len
        return stats_arr.reshape(reshape_dims)

    def _normalize_array(self, var_values: np.ndarray, var_name: str) -> np.ndarray:
        transformed_values = self._forward_var_transform(var_values, var_name)
        mean = self._broadcast_stats_like(self._mean_values[var_name], transformed_values)
        std = self._broadcast_stats_like(self._std_values[var_name], transformed_values)
        return (transformed_values - mean) / std

    # ------------------------------------------------------------------ #
    # Lazy dataset opening (called per DataLoader worker)
    # ------------------------------------------------------------------ #
    def _open_datasets(self):
        if self._opened:
            return

        if len(self.filenames) == 0:
            raise ValueError("No training files found for WoFSDAIncrementDataset.")

        # Load normalization statistics
        self.mean_ds = xr.open_dataset(self.mean_path).load()
        self.std_ds = xr.open_dataset(self.std_path).load()

        # All upper-air variable names we need from each zarr file
        # (deduplicated, order-preserving)
        seen = set()
        all_upper_vars = []
        for v in self.varname_prognostic + self.varname_context + self.varname_observation:
            if v not in seen:
                all_upper_vars.append(v)
                seen.add(v)
        self._all_upper_vars = all_upper_vars

        for var in all_upper_vars:
            if var not in self.mean_ds or var not in self.std_ds:
                raise KeyError(
                    f"Variable '{var}' is missing from mean/std stats files required for DA increment dataset"
                )

        self._mean_values = {var: self.mean_ds[var].values for var in all_upper_vars}
        self._std_values = {var: self.std_ds[var].values for var in all_upper_vars}

        # Dynamic forcing path checks
        if self.varname_dyn_forcing and self.filename_dyn_forcing:
            if len(self.filename_dyn_forcing) != len(self.filenames):
                raise ValueError(
                    "Mismatch between number of interior files and dynamic forcing files. "
                    f"Got {len(self.filenames)} interior and {len(self.filename_dyn_forcing)} dynamic forcing files."
                )
            self.has_dyn_forcing = True
        else:
            self.has_dyn_forcing = False

        self._build_index_map()

        self._opened = True

    @staticmethod
    def _safe_close_dataset(ds):
        try:
            ds.close()
        except Exception:
            pass

    def _evict_if_needed(self, cache: OrderedDict):
        while len(cache) > self.max_open_files:
            _, ds = cache.popitem(last=False)
            self._safe_close_dataset(ds)

    def _get_upper_ds(self, file_idx: int):
        cached = self.upper_ds_cache.get(file_idx)
        if cached is not None:
            self.upper_ds_cache.move_to_end(file_idx)
            return cached

        ds = get_forward_data(self.filenames[file_idx], zarr_chunks=self.zarr_chunks)
        ds = filter_ds(ds, self._all_upper_vars)
        self.upper_ds_cache[file_idx] = ds
        self._evict_if_needed(self.upper_ds_cache)
        return ds

    def _get_dyn_ds(self, file_idx: int):
        if not self.has_dyn_forcing:
            return None

        cached = self.dyn_ds_cache.get(file_idx)
        if cached is not None:
            self.dyn_ds_cache.move_to_end(file_idx)
            return cached

        ds = get_forward_data(self.filename_dyn_forcing[file_idx], zarr_chunks=self.zarr_chunks)
        ds = filter_ds(ds, self.varname_dyn_forcing)
        self.dyn_ds_cache[file_idx] = ds
        self._evict_if_needed(self.dyn_ds_cache)
        return ds

    def __del__(self):
        try:
            for ds in self.upper_ds_cache.values():
                self._safe_close_dataset(ds)
            for ds in self.dyn_ds_cache.values():
                self._safe_close_dataset(ds)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Length
    # ------------------------------------------------------------------ #
    def __len__(self):
        if self.total_length is not None:
            return self.total_length

        self._build_index_map()
        return self.total_length

    # ------------------------------------------------------------------ #
    # Normalization helper
    # ------------------------------------------------------------------ #
    def _normalize(self, data_array, var_name):
        """Z-score normalise an xr.DataArray using pre-loaded stats.

        Returns a numpy array (same shape as input).
        Uses .values arithmetic for speed (avoids xarray coordinate alignment).
        """
        return self._normalize_array(data_array.values, var_name)

    # ------------------------------------------------------------------ #
    # __getitem__
    # ------------------------------------------------------------------ #
    def __getitem__(self, index):
        if not self._opened:
            self._open_datasets()

        # -- map global index → (file, time_in_file) ----------------------
        file_idx, ind_in_file = self._locate_index(index)

        t0 = ind_in_file
        t1 = ind_in_file + 1

        # -- single contiguous load for both timesteps ---------------------
        upper_ds = self._get_upper_ds(file_idx)
        chunk = upper_ds.isel(
            time=slice(t0, t1 + 1)
        ).load()

        # ================================================================ #
        # 1. Prognostic variables: x (input) and y (increment target)
        # ================================================================ #
        x_prog_list = []
        y_incr_list = []
        for var in self.varname_prognostic:
            norm_t0 = self._normalize(chunk[var].isel(time=0), var)
            norm_t1 = self._normalize(chunk[var].isel(time=1), var)
            x_prog_list.append(norm_t0)          # (level, H, W)
            y_incr_list.append(norm_t1 - norm_t0)  # increment in normalized space

        # (time=1, num_prog_vars, level, H, W)
        x_prog = np.stack(x_prog_list, axis=0)[np.newaxis, ...]
        y_incr = np.stack(y_incr_list, axis=0)[np.newaxis, ...]

        # ================================================================ #
        # 2. Context 3D variables (input-only, flattened)
        # ================================================================ #
        ctx_list = []
        for var in self.varname_context:
            norm_t0 = self._normalize(chunk[var].isel(time=0), var)
            # norm_t0 shape: (level, H, W)  →  contributes `level` channels
            ctx_list.append(norm_t0)

        # Concatenate along axis 0: (num_ctx_vars * level, H, W)
        ctx_flat = np.concatenate(ctx_list, axis=0)

        # ================================================================ #
        # 3. Dynamic forcing (input-only, 2D)
        # ================================================================ #
        if self.has_dyn_forcing:
            dyn_ds = self._get_dyn_ds(file_idx)
            if dyn_ds is None:
                raise RuntimeError("Dynamic forcing dataset is not available for this sample")
            dyn_chunk = dyn_ds.isel(time=t0).load()
            dyn_list = []
            for var in self.varname_dyn_forcing:
                dyn_list.append(dyn_chunk[var].values)  # (H, W) each
            dyn_flat = np.stack(dyn_list, axis=0)  # (num_dyn, H, W)
            # Combine context + dynamic forcing
            forcing_static = np.concatenate([ctx_flat, dyn_flat], axis=0)
        else:
            forcing_static = ctx_flat

        # (time=1, channels, H, W)
        forcing_static = forcing_static[np.newaxis, ...]

        # ================================================================ #
        # 4. Innovation (boundary encoder input)
        # ================================================================ #
        obs_var = self.varname_observation[0]  # 'REFL_10CM'
        norm_obs_t0 = self._normalize(chunk[obs_var].isel(time=0), obs_var)
        norm_obs_t1 = self._normalize(chunk[obs_var].isel(time=1), obs_var)
        innovation = norm_obs_t1 - norm_obs_t0  # (level, H, W)

        # (time=1, num_obs_vars=1, level, H, W)
        x_boundary = innovation[np.newaxis, np.newaxis, ...]

        # ================================================================ #
        # 5. Time encoding
        # ================================================================ #
        t0_time = chunk.time.values[0:1]
        t1_time = chunk.time.values[1:2]
        # 3 time groups: input(t0), target(t1), boundary(t0)
        time_encode = encode_datetime64(
            np.concatenate([t0_time, t1_time, t0_time])
        )

        # ================================================================ #
        # 6. Pack return dict (matches trainer expectations)
        # ================================================================ #
        return {
            "x": torch.as_tensor(x_prog, dtype=self.output_dtype),
            "x_forcing_static": torch.as_tensor(forcing_static, dtype=self.output_dtype),
            "x_boundary": torch.as_tensor(x_boundary, dtype=self.output_dtype),
            "y": torch.as_tensor(y_incr, dtype=self.output_dtype),
            "x_time_encode": torch.as_tensor(time_encode, dtype=self.output_dtype),
            "index": index,
        }
