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
import json
import time
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

_DEFAULT_CONCENTRATION_PARAMS = {
    "c1": 0.5,
    "c2": 0.5,
    "conc_eps": 1e-4,
    "conc_max": 2.5,
    "value_clip_min": None,
    "value_clip_max": None,
}

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
        self.concentration_params_json_path = conf["data"].get("concentration_params_json")

        self.levels = conf["data"]["levels"]
        self.rng = np.random.default_rng(seed=seed)
        self.output_dtype = torch.float32
        self.max_open_files = int(conf["data"].get("max_open_files_per_worker", 8))
        self.zarr_time_chunk = int(conf["data"].get("zarr_time_chunk", 2))
        self.zarr_chunks = {"time": self.zarr_time_chunk} if self.zarr_time_chunk > 0 else None
        self.max_dataset_open_retries = int(conf["data"].get("max_dataset_open_retries", 3))

        # Keep required upper-air variables consistent across startup filtering
        # and training-time sample reads.
        seen = set()
        all_upper_vars = []
        for v in self.varname_prognostic + self.varname_context + self.varname_observation:
            if v not in seen:
                all_upper_vars.append(v)
                seen.add(v)
        self._all_upper_vars = all_upper_vars
        self.has_dyn_forcing = bool(self.varname_dyn_forcing and self.filename_dyn_forcing)
        self._shared_dyn_store = False
        self._upper_open_vars = list(self._all_upper_vars)

        # -- internal state (lazy-opened in workers) -----------------------
        self._opened = False
        self.total_length = None
        self.file_n_time = None
        self.samples_per_file = None
        self.cumulative_samples = None
        self.upper_ds_cache = OrderedDict()
        self.dyn_ds_cache = OrderedDict()
        self._concentration_params = {var: dict(_DEFAULT_CONCENTRATION_PARAMS) for var in CONCENTRATION_VARS}
        self._concentration_level_ops = conf["data"].get("concentration_level_ops", {})
        self._occupancy_delta_threshold = float(conf["data"].get("occupancy_delta_threshold", 1.0e-10))
        if self._occupancy_delta_threshold <= 0.0:
            self._occupancy_delta_threshold = 1.0e-10

        raw_occ_by_var = conf["data"].get("occupancy_delta_threshold_by_var", {})
        self._occupancy_delta_threshold_by_var = {}
        if isinstance(raw_occ_by_var, dict):
            for var_name, value in raw_occ_by_var.items():
                try:
                    value_f = float(value)
                except Exception:
                    continue
                if value_f > 0.0:
                    self._occupancy_delta_threshold_by_var[var_name] = value_f

        raw_occ_vars = conf["data"].get("occupancy_variables", self.varname_prognostic)
        if raw_occ_vars is None:
            raw_occ_vars = []
        self._occupancy_variables = set(raw_occ_vars)

        default_prog_levels = int(
            conf["data"].get(
                "prognostic_levels",
                conf.get("model", {}).get("param_interior", {}).get("levels", self.levels),
            )
        )
        self._prognostic_levels = self._infer_common_prognostic_ceiling_level(default_prog_levels)
        self._refresh_required_open_vars()

    def _refresh_required_open_vars(self):
        self._shared_dyn_store = bool(
            self.has_dyn_forcing
            and self.filename_dyn_forcing is not None
            and len(self.filename_dyn_forcing) == len(self.filenames)
            and all(upper_fn == dyn_fn for upper_fn, dyn_fn in zip(self.filenames, self.filename_dyn_forcing))
        )

        if self._shared_dyn_store:
            seen = set()
            merged = []
            for var_name in self._all_upper_vars + self.varname_dyn_forcing:
                if var_name not in seen:
                    merged.append(var_name)
                    seen.add(var_name)
            self._upper_open_vars = merged
        else:
            self._upper_open_vars = list(self._all_upper_vars)

    def _infer_common_prognostic_ceiling_level(self, fallback_levels: int) -> int:
        level_candidates = []
        if isinstance(self._concentration_level_ops, dict):
            for var_name in self.varname_prognostic:
                cfg = self._concentration_level_ops.get(var_name)
                if not isinstance(cfg, dict) or "ceiling_level" not in cfg:
                    continue
                try:
                    level_candidates.append(int(cfg["ceiling_level"]))
                except Exception:
                    continue

        if len(level_candidates) == 0:
            return max(1, int(fallback_levels))

        unique_levels = sorted(set(level_candidates))
        if len(unique_levels) > 1:
            raise ValueError(
                "All prognostic concentration variables must share the same ceiling_level "
                f"for reduced-level training. Got: {unique_levels}"
            )
        return max(1, int(unique_levels[0]))

    @staticmethod
    def _coerce_concentration_params(params: dict | None) -> dict[str, float]:
        merged = dict(_DEFAULT_CONCENTRATION_PARAMS)
        if isinstance(params, dict):
            for key in ("c1", "c2", "conc_eps", "conc_max"):
                if key in params:
                    merged[key] = float(params[key])
            if "value_clip_min" in params:
                merged["value_clip_min"] = None if params["value_clip_min"] is None else float(params["value_clip_min"])
            if "value_clip_max" in params:
                merged["value_clip_max"] = None if params["value_clip_max"] is None else float(params["value_clip_max"])
        if merged["conc_eps"] <= 0.0:
            merged["conc_eps"] = _DEFAULT_CONCENTRATION_PARAMS["conc_eps"]
        if merged["conc_max"] <= merged["conc_eps"]:
            merged["conc_max"] = max(_DEFAULT_CONCENTRATION_PARAMS["conc_max"], merged["conc_eps"] * 10.0)
        clip_min = merged.get("value_clip_min")
        clip_max = merged.get("value_clip_max")
        if clip_min is not None and clip_max is not None and clip_min >= clip_max:
            merged["value_clip_min"] = None
            merged["value_clip_max"] = None
        return merged

    def _get_concentration_params(self, var_name: str) -> dict[str, float]:
        params = self._concentration_params.get(var_name, _DEFAULT_CONCENTRATION_PARAMS)
        return self._coerce_concentration_params(params)

    def _get_concentration_level_op(self, var_name: str, n_levels: int) -> dict | None:
        if var_name not in CONCENTRATION_VARS or not isinstance(self._concentration_level_ops, dict):
            return None

        raw = self._concentration_level_ops.get(var_name)
        if not isinstance(raw, dict):
            return None

        if "ceiling_level" not in raw:
            return None

        try:
            ceiling_level = int(raw["ceiling_level"])
        except Exception:
            return None

        mode = str(raw.get("mode", "cutoff")).lower().strip()
        if mode not in {"cutoff", "sum", "mean"}:
            mode = "cutoff"

        if n_levels <= 0:
            return None
        ceiling_level = max(1, min(ceiling_level, n_levels))

        return {
            "ceiling_level": ceiling_level,
            "ceiling_index": ceiling_level - 1,
            "mode": mode,
        }

    @staticmethod
    def _pick_axis_by_size(values_arr: np.ndarray, target_size: int) -> int | None:
        matching_axes = [axis for axis, size in enumerate(values_arr.shape) if size == target_size]
        if not matching_axes:
            return None
        return 1 if len(matching_axes) > 1 and 1 in matching_axes else matching_axes[0]

    def _find_level_axis(self, values: np.ndarray, var_name: str) -> int | None:
        stats = np.asarray(self._mean_values[var_name])
        if stats.ndim != 1:
            return None
        level_count = stats.shape[0]
        return self._pick_axis_by_size(np.asarray(values), level_count)

    def reduce_levels_for_var(self, var_values: np.ndarray, var_name: str, is_prognostic: bool = True) -> np.ndarray:
        values_arr = np.asarray(var_values)
        if (not is_prognostic) or (var_name not in CONCENTRATION_VARS):
            return values_arr

        level_axis = self._find_level_axis(values_arr, var_name)
        if level_axis is None:
            return values_arr

        level_first = np.moveaxis(values_arr, level_axis, 0).copy()
        n_levels = level_first.shape[0]
        if self._prognostic_levels >= n_levels:
            return values_arr

        op = self._get_concentration_level_op(var_name, n_levels)
        mode = op["mode"] if op is not None else "cutoff"
        ceiling_idx = self._prognostic_levels - 1

        if mode in {"sum", "mean"} and ceiling_idx + 1 < n_levels:
            upper = level_first[ceiling_idx + 1 :]
            if upper.shape[0] > 0:
                if mode == "sum":
                    level_first[ceiling_idx] = np.sum(upper, axis=0)
                else:
                    level_first[ceiling_idx] = np.mean(upper, axis=0)

        level_first = level_first[: self._prognostic_levels]
        return np.moveaxis(level_first, 0, level_axis)

    def _apply_concentration_level_op_to_normalized(
        self,
        normalized_values: np.ndarray,
        raw_values: np.ndarray,
        var_name: str,
    ) -> np.ndarray:
        level_axis = self._find_level_axis(normalized_values, var_name)
        if level_axis is None:
            return normalized_values

        n_levels = normalized_values.shape[level_axis]
        op = self._get_concentration_level_op(var_name, n_levels)
        if op is None:
            return normalized_values

        ceiling_idx = op["ceiling_index"]
        mode = op["mode"]

        norm_level_first = np.moveaxis(normalized_values, level_axis, 0).copy()
        raw_level_first = np.moveaxis(raw_values, level_axis, 0)

        if mode in {"sum", "mean"}:
            upper = raw_level_first[ceiling_idx + 1 :]
            if upper.shape[0] > 0:
                if mode == "sum":
                    agg_raw = np.sum(upper, axis=0)
                else:
                    agg_raw = np.mean(upper, axis=0)
                agg_trans = self._forward_var_transform(agg_raw, var_name)
                mean_level = np.asarray(self._mean_values[var_name])[ceiling_idx]
                std_level = np.asarray(self._std_values[var_name])[ceiling_idx]
                norm_level_first[ceiling_idx] = (agg_trans - mean_level) / std_level

        if ceiling_idx + 1 < norm_level_first.shape[0]:
            norm_level_first[ceiling_idx + 1 :] = 0.0

        return np.moveaxis(norm_level_first, 0, level_axis)

    @staticmethod
    def _forward_concentration_numpy(x: np.ndarray, params: dict[str, float]) -> np.ndarray:
        x64 = np.asarray(x, dtype=np.float64)
        c1 = float(params["c1"])
        c2 = float(params["c2"])
        conc_eps = float(params["conc_eps"])
        conc_max = float(params["conc_max"])
        clip_min = params.get("value_clip_min")
        clip_max = params.get("value_clip_max")
        if clip_min is not None:
            x64 = np.maximum(x64, float(clip_min))
        if clip_max is not None:
            x64 = np.minimum(x64, float(clip_max))
        log_eps = float(np.log(conc_eps))
        neg_log_eps = float(-log_eps)
        transformed = c1 * np.minimum(x64, conc_max) + c2 * (np.log(np.maximum(x64, conc_eps)) - log_eps) / neg_log_eps
        return transformed.astype(x.dtype, copy=False)

    @staticmethod
    def _inverse_concentration_numpy(x: np.ndarray, params: dict[str, float]) -> np.ndarray:
        target = np.asarray(x, dtype=np.float64)
        target = np.maximum(target, 0.0)

        c1 = float(params["c1"])
        c2 = float(params["c2"])
        conc_eps = float(params["conc_eps"])
        conc_max = float(params["conc_max"])
        clip_min = params.get("value_clip_min")
        clip_max = params.get("value_clip_max")
        log_eps = float(np.log(conc_eps))
        neg_log_eps = float(-log_eps)

        def forward(v: np.ndarray) -> np.ndarray:
            return c1 * np.minimum(v, conc_max) + c2 * (np.log(np.maximum(v, conc_eps)) - log_eps) / neg_log_eps

        lo = np.zeros_like(target)
        hi = np.full_like(target, max(conc_max * 2.0, conc_eps * 2.0))
        hi_val = forward(hi)

        for _ in range(40):
            need_expand = hi_val < target
            if not np.any(need_expand):
                break
            hi = np.where(need_expand, hi * 2.0, hi)
            hi_val = forward(hi)

        for _ in range(48):
            mid = 0.5 * (lo + hi)
            f_mid = forward(mid)
            go_right = f_mid < target
            lo = np.where(go_right, mid, lo)
            hi = np.where(go_right, hi, mid)

        out = 0.5 * (lo + hi)
        if clip_min is not None:
            out = np.maximum(out, float(clip_min))
        if clip_max is not None:
            out = np.minimum(out, float(clip_max))
        return out.astype(x.dtype, copy=False)

    def _forward_var_transform(self, var_values: np.ndarray, var_name: str) -> np.ndarray:
        if var_name in CONCENTRATION_VARS:
            return self._forward_concentration_numpy(var_values, self._get_concentration_params(var_name))
        return var_values

    def _inverse_var_transform(self, var_values: np.ndarray, var_name: str) -> np.ndarray:
        if var_name in CONCENTRATION_VARS:
            return self._inverse_concentration_numpy(var_values, self._get_concentration_params(var_name))
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
        std = self._broadcast_stats_for_var(self._std_values[var_name], normalized_increment, var_name)
        mean = self._broadcast_stats_for_var(self._mean_values[var_name], normalized_increment, var_name)
        z0 = self._normalize_array(base_state_t0, var_name)
        z1 = z0 + normalized_increment
        transformed_1 = z1 * std + mean
        x1 = self._inverse_var_transform(transformed_1, var_name)
        return x1 - base_state_t0

    def _build_index_map(self):
        if self.file_n_time is None:
            has_dyn = bool(self.has_dyn_forcing and self.filename_dyn_forcing)
            valid_filenames = []
            valid_dyn_filenames = [] if has_dyn else None
            file_n_time = []
            skipped = []

            for file_idx, fn in enumerate(self.filenames):
                dyn_fn = self.filename_dyn_forcing[file_idx] if has_dyn else None
                upper_ds = None
                dyn_ds = None
                try:
                    upper_ds = self._open_filtered_dataset(fn, self._upper_open_vars, self.max_dataset_open_retries)
                    n_time = int(upper_ds["time"].size)

                    if dyn_fn is not None and not self._shared_dyn_store:
                        dyn_ds = self._open_filtered_dataset(
                            dyn_fn,
                            self.varname_dyn_forcing,
                            self.max_dataset_open_retries,
                        )
                        dyn_n_time = int(dyn_ds["time"].size)
                        n_time = min(n_time, dyn_n_time)

                    valid_filenames.append(fn)
                    file_n_time.append(int(np.asarray(n_time).item()))
                    if valid_dyn_filenames is not None and dyn_fn is not None:
                        valid_dyn_filenames.append(dyn_fn)
                except Exception as exc:
                    skipped.append((fn, exc))
                    logger.warning("Skipping unreadable training file %s due to %s: %s", fn, type(exc).__name__, exc)
                finally:
                    self._safe_close_dataset(upper_ds)
                    self._safe_close_dataset(dyn_ds)

            if len(valid_filenames) == 0:
                raise ValueError(
                    "No readable training files after filtering corrupted/unavailable datasets. "
                    "Check data path, filesystem stability, and zarr metadata."
                )

            if skipped:
                logger.warning(
                    "Filtered %d unreadable training file(s); using %d file(s).",
                    len(skipped),
                    len(valid_filenames),
                )

            self.filenames = valid_filenames
            if valid_dyn_filenames is not None:
                self.filename_dyn_forcing = valid_dyn_filenames
            self._refresh_required_open_vars()

            self.file_n_time = file_n_time

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

    def file_index_for_global_index(self, index: int) -> int:
        if self.cumulative_samples is None:
            self._build_index_map()

        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}")

        cumulative_samples = self.cumulative_samples
        if cumulative_samples is None:
            raise RuntimeError("Dataset index map is not initialized")

        return int(np.searchsorted(cumulative_samples, index, side="right"))

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

    def _broadcast_stats_for_var(self, stats: np.ndarray, values: np.ndarray, var_name: str) -> np.ndarray:
        stats_arr = np.asarray(stats)
        values_arr = np.asarray(values)

        if stats_arr.ndim != 1 or values_arr.ndim == 0:
            return stats_arr

        stat_len = stats_arr.shape[0]
        axis = self._pick_axis_by_size(values_arr, stat_len)
        stats_used = stats_arr

        if axis is None and var_name in CONCENTRATION_VARS and var_name in self.varname_prognostic:
            reduced_len = int(self._prognostic_levels)
            if reduced_len <= stat_len:
                axis = self._pick_axis_by_size(values_arr, reduced_len)
                if axis is not None:
                    stats_used = stats_arr[:reduced_len]

        if axis is None:
            return stats_used

        reshape_dims = [1] * values_arr.ndim
        reshape_dims[axis] = stats_used.shape[0]
        return stats_used.reshape(reshape_dims)

    def _normalize_array(self, var_values: np.ndarray, var_name: str) -> np.ndarray:
        transformed_values = self._forward_var_transform(var_values, var_name)
        mean = self._broadcast_stats_for_var(self._mean_values[var_name], transformed_values, var_name)
        std = self._broadcast_stats_for_var(self._std_values[var_name], transformed_values, var_name)
        return (transformed_values - mean) / std

    def _delta_threshold_for_var(self, var_name: str) -> float:
        if var_name in self._occupancy_delta_threshold_by_var:
            return float(self._occupancy_delta_threshold_by_var[var_name])
        return float(self._occupancy_delta_threshold)

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

        if self.concentration_params_json_path:
            try:
                with open(self.concentration_params_json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                params_dict = payload.get("variables", payload) if isinstance(payload, dict) else {}
                if isinstance(params_dict, dict):
                    for var, info in params_dict.items():
                        if var not in CONCENTRATION_VARS or not isinstance(info, dict):
                            continue
                        if "recommended" in info and isinstance(info["recommended"], dict):
                            self._concentration_params[var] = self._coerce_concentration_params(info["recommended"])
                        else:
                            self._concentration_params[var] = self._coerce_concentration_params(info)
                logger.info("Loaded concentration params from JSON: %s", self.concentration_params_json_path)
            except Exception as exc:
                logger.warning("Failed to load concentration params JSON %s: %s", self.concentration_params_json_path, exc)

        attr_payload = self.mean_ds.attrs.get("concentration_transform_params_json")
        if isinstance(attr_payload, str) and attr_payload.strip():
            try:
                payload = json.loads(attr_payload)
                params_dict = payload.get("parameters", {}) if isinstance(payload, dict) else {}
                if isinstance(params_dict, dict):
                    for var, params in params_dict.items():
                        if var in CONCENTRATION_VARS and isinstance(params, dict):
                            self._concentration_params[var] = self._coerce_concentration_params(params)
                logger.info("Loaded concentration params from mean/std NetCDF attrs")
            except Exception as exc:
                logger.warning("Failed to parse concentration params from stats attrs: %s", exc)

        for var in CONCENTRATION_VARS:
            if var in self.mean_ds:
                var_attrs = self.mean_ds[var].attrs
                if any(
                    key in var_attrs
                    for key in (
                        "concentration_transform_c1",
                        "concentration_transform_c2",
                        "concentration_transform_conc_eps",
                        "concentration_transform_conc_max",
                        "concentration_transform_clip_min",
                        "concentration_transform_clip_max",
                    )
                ):
                    self._concentration_params[var] = self._coerce_concentration_params(
                        {
                            "c1": var_attrs.get("concentration_transform_c1", self._concentration_params[var]["c1"]),
                            "c2": var_attrs.get("concentration_transform_c2", self._concentration_params[var]["c2"]),
                            "conc_eps": var_attrs.get(
                                "concentration_transform_conc_eps", self._concentration_params[var]["conc_eps"]
                            ),
                            "conc_max": var_attrs.get(
                                "concentration_transform_conc_max", self._concentration_params[var]["conc_max"]
                            ),
                            "value_clip_min": var_attrs.get(
                                "concentration_transform_clip_min", self._concentration_params[var]["value_clip_min"]
                            ),
                            "value_clip_max": var_attrs.get(
                                "concentration_transform_clip_max", self._concentration_params[var]["value_clip_max"]
                            ),
                        }
                    )

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

        self._mean_values = {}
        self._std_values = {}
        for var in all_upper_vars:
            mean_values = self.mean_ds[var].values
            std_values = self.std_ds[var].values
            if var in CONCENTRATION_VARS:
                mean_values = np.where(np.isnan(mean_values), 0.0, mean_values)
                std_values = np.where(np.isnan(std_values), 1.0, std_values)
            self._mean_values[var] = mean_values
            self._std_values[var] = std_values

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

    def _open_filtered_dataset(self, file_path: str, varnames_keep, max_retries: int | None = None):
        attempts = max(1, int(max_retries or self.max_dataset_open_retries))
        last_exc = None

        for attempt in range(1, attempts + 1):
            ds = None
            try:
                ds = get_forward_data(file_path, zarr_chunks=self.zarr_chunks)
                ds = filter_ds(ds, varnames_keep)
                return ds
            except Exception as exc:
                last_exc = exc
                self._safe_close_dataset(ds)
                if attempt < attempts:
                    logger.warning(
                        "Retrying dataset open/filter %d/%d for %s due to %s: %s",
                        attempt,
                        attempts,
                        file_path,
                        type(exc).__name__,
                        exc,
                    )
                    time.sleep(0.1 * attempt)

        raise RuntimeError(
            f"Failed to open/filter dataset after {attempts} attempt(s): {file_path}; "
            f"required_vars={list(varnames_keep)}; last_error={last_exc}"
        ) from last_exc

    def _evict_if_needed(self, cache: OrderedDict):
        while len(cache) > self.max_open_files:
            _, ds = cache.popitem(last=False)
            self._safe_close_dataset(ds)

    def _get_upper_ds(self, file_idx: int):
        cached = self.upper_ds_cache.get(file_idx)
        if cached is not None:
            self.upper_ds_cache.move_to_end(file_idx)
            return cached

        ds = self._open_filtered_dataset(
            self.filenames[file_idx],
            self._upper_open_vars,
            self.max_dataset_open_retries,
        )
        self.upper_ds_cache[file_idx] = ds
        self._evict_if_needed(self.upper_ds_cache)
        return ds

    def _get_dyn_ds(self, file_idx: int):
        if not self.has_dyn_forcing:
            return None

        if self._shared_dyn_store:
            return self._get_upper_ds(file_idx)

        cached = self.dyn_ds_cache.get(file_idx)
        if cached is not None:
            self.dyn_ds_cache.move_to_end(file_idx)
            return cached

        ds = self._open_filtered_dataset(
            self.filename_dyn_forcing[file_idx],
            self.varname_dyn_forcing,
            self.max_dataset_open_retries,
        )
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
        y_occ_list = []
        y_reg_mask_hard_list = []
        y_occ_delta_thr_norm_list = []
        for var in self.varname_prognostic:
            raw_t0 = chunk[var].isel(time=0).values
            raw_t1 = chunk[var].isel(time=1).values
            reduced_t0 = self.reduce_levels_for_var(raw_t0, var_name=var, is_prognostic=True)
            reduced_t1 = self.reduce_levels_for_var(raw_t1, var_name=var, is_prognostic=True)
            norm_t0 = self._normalize_array(reduced_t0, var)
            norm_t1 = self._normalize_array(reduced_t1, var)
            x_prog_list.append(norm_t0)          # (level, H, W)
            y_incr_list.append(norm_t1 - norm_t0)  # increment in normalized space

            delta_raw = reduced_t1 - reduced_t0
            delta_threshold = self._delta_threshold_for_var(var)
            if var in self._occupancy_variables:
                y_occ = (np.abs(delta_raw) >= delta_threshold).astype(np.float32, copy=False)
            else:
                y_occ = np.zeros_like(delta_raw, dtype=np.float32)
            y_occ_list.append(y_occ)
            y_reg_mask_hard_list.append(y_occ.copy())

            std = self._broadcast_stats_for_var(self._std_values[var], reduced_t0, var)
            delta_thr_norm = np.asarray(delta_threshold / std, dtype=np.float32)
            if delta_thr_norm.ndim == 0:
                n_levels = reduced_t0.shape[0] if np.asarray(reduced_t0).ndim > 0 else 1
                delta_thr_norm = np.full((n_levels, 1, 1), float(delta_thr_norm), dtype=np.float32)
            elif delta_thr_norm.ndim == 1:
                delta_thr_norm = delta_thr_norm[:, np.newaxis, np.newaxis]
            y_occ_delta_thr_norm_list.append(delta_thr_norm)

        # (time=1, num_prog_vars, level, H, W)
        x_prog = np.stack(x_prog_list, axis=0)[np.newaxis, ...]
        y_incr = np.stack(y_incr_list, axis=0)[np.newaxis, ...]
        y_occ = np.stack(y_occ_list, axis=0)[np.newaxis, ...]
        y_reg_mask_hard = np.stack(y_reg_mask_hard_list, axis=0)[np.newaxis, ...]
        y_occ_delta_thr_norm = np.stack(y_occ_delta_thr_norm_list, axis=0)[np.newaxis, ...]

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
            if self._shared_dyn_store:
                dyn_chunk = chunk.isel(time=0)
            else:
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
            "y_occupancy": torch.as_tensor(y_occ, dtype=self.output_dtype),
            "y_regression_mask_hard": torch.as_tensor(y_reg_mask_hard, dtype=self.output_dtype),
            "y_occ_delta_threshold_norm": torch.as_tensor(y_occ_delta_thr_norm, dtype=self.output_dtype),
            "x_time_encode": torch.as_tensor(time_encode, dtype=self.output_dtype),
            "index": index,
        }
