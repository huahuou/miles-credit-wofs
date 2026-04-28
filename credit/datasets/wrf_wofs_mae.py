"""
WoFS Multi-Modal MAE Dataset
-----------------------------
Returns per-modality normalized tensors for training WoFSMultiModalMAE.

Unlike WoFSDAIncrementDataset, this dataset does NOT produce target increments.
Instead it provides the raw normalized fields for self-supervised masked
autoencoding.  Each sample is a dict of {modality_key: (C, H, W)} tensors.

Modality layout (channels = levels folded in):
    background   : T, QVAPOR, U, V, W, GEOPOT         → 6  vars × 17 levels = 102 ch
    precip       : QRAIN, QNRAIN, QHAIL, QNHAIL,
                   QGRAUP, QNGRAUPEL, QSNOW, QNSNOW   → 8  vars × 17 levels = 136 ch
    reflectivity : REFL_10CM                           → 1  var  × 17 levels =  17 ch
    surface      : XLAND, HGT                          → 2  channels (no levels)
    forcing      : cos/sin lat, cos/sin lon,
                   cos/sin julian day, cos/sin local
                   time, cos_solar_zenith, insolation  → 10 channels (no levels)

When `mae_include_t1_refl=True` (default), each sample draws two consecutive
timesteps t0 and t1.  The background/precip/surface/forcing are taken from t0,
and the reflectivity modality uses t1 — this lets the model learn to predict
t1 radar from the t0 NWP background, mimicking the assimilation task.

Padding: all 3D fields are spatially padded from (300, 300) → (304, 304) so
that 8×8 patches tile evenly.  Surface/forcing are padded similarly.

Normalization:
    Reuses the same concentration transform + (x-mean)/std pipeline as
    WoFSDAIncrementDataset.  Concentration params are loaded from JSON or
    NetCDF attributes (same logic).

Config keys (conf["data"]):
    background_vars          : list of upper-air var names for background modality
    precip_vars              : list of upper-air var names for precip modality
    reflectivity_vars        : list of upper-air var names for reflectivity modality
    surface_vars             : list of 2D var names for surface modality
    dynamic_forcing_variables: list of 2D var names for forcing modality
    levels                   : int  — number of vertical levels
    mean_path                : str  — path to mean.nc
    std_path                 : str  — path to std.nc
    concentration_params_json: str  — path to concentration_tuning.json
    mae_pad_to               : int  — target spatial size after padding (default 304)
    mae_include_t1_refl      : bool — use t1 reflectivity (default True)
    max_open_files_per_worker: int  — LRU cache size (default 8)
    zarr_time_chunk          : int  — zarr chunk along time dim (default 2)
    max_dataset_open_retries : int  — retry budget for corrupt files (default 3)
"""

import json
import logging
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset

from credit.data import (
    encode_datetime64,
    filter_ds,
    get_forward_data,
)

# Re-use normalization constants from the DA increment dataset
from credit.datasets.wrf_wofs_da_increment import (
    CONCENTRATION_VARS,
    WoFSDAIncrementDataset,  # type: ignore  # used only for class methods via super-dict reuse
    _DEFAULT_CONCENTRATION_PARAMS,
)

logger = logging.getLogger(__name__)


# Default variable groupings (match implementation_MAE_DA_plan.md)
DEFAULT_BACKGROUND_VARS = ["T", "QVAPOR", "U", "V", "W", "GEOPOT"]
DEFAULT_PRECIP_VARS = ["QRAIN", "QNRAIN", "QHAIL", "QNHAIL", "QGRAUP", "QNGRAUPEL", "QSNOW", "QNSNOW"]
DEFAULT_REFLECTIVITY_VARS = ["REFL_10CM"]
DEFAULT_SURFACE_VARS = ["XLAND", "HGT"]
DEFAULT_DYNAMIC_FORCING_VARS = [
    "cos_lat", "sin_lat", "cos_lon", "sin_lon",
    "cos_julian_day", "sin_julian_day",
    "cos_local_time", "sin_local_time",
    "cos_solar_zenith", "insolation",
]


# ---------------------------------------------------------------------------
# Normalization helpers (mirror WoFSDAIncrementDataset)
# ---------------------------------------------------------------------------

def _coerce_concentration_params(params: Optional[dict]) -> dict:
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
    return merged


def _forward_concentration_numpy(arr: np.ndarray, params: dict) -> np.ndarray:
    c1 = float(params["c1"])
    c2 = float(params["c2"])
    eps = float(params["conc_eps"])
    cmax = float(params["conc_max"])
    clip_min = params.get("value_clip_min")
    clip_max = params.get("value_clip_max")

    x = arr.astype(np.float32, copy=True)
    if clip_min is not None or clip_max is not None:
        x = np.clip(x, clip_min, clip_max)
    x = np.clip(x, 0.0, None)
    x = np.arctan(c1 * np.log(x / eps + 1.0)) / np.arctan(c2 * np.log(cmax / eps + 1.0))
    return x


class WoFSMAEDataset(Dataset):
    """
    Dataset for WoFSMultiModalMAE training.

    Returns a dict of modality tensors per sample.
    """

    def __init__(self, filenames: List[str], conf: dict, seed: int = 42):
        super().__init__()

        data_conf = conf["data"]

        # -- variable groupings from config --------------------------------
        self.background_vars = data_conf.get("background_vars", DEFAULT_BACKGROUND_VARS)
        self.precip_vars = data_conf.get("precip_vars", DEFAULT_PRECIP_VARS)
        self.reflectivity_vars = data_conf.get("reflectivity_vars", DEFAULT_REFLECTIVITY_VARS)
        self.surface_vars = data_conf.get("surface_vars", DEFAULT_SURFACE_VARS)
        self.forcing_vars = data_conf.get("dynamic_forcing_variables", DEFAULT_DYNAMIC_FORCING_VARS)

        # upper-air vars union (background + precip + reflectivity)
        seen = set()
        self._upper_air_vars: List[str] = []
        for v in self.background_vars + self.precip_vars + self.reflectivity_vars:
            if v not in seen:
                self._upper_air_vars.append(v)
                seen.add(v)

        # 2D vars union (surface + forcing)
        seen_2d = set()
        self._surface_2d_vars: List[str] = []
        for v in self.surface_vars + self.forcing_vars:
            if v not in seen_2d:
                self._surface_2d_vars.append(v)
                seen_2d.add(v)

        # -- file list -------------------------------------------------------
        self.filenames = sorted(filenames)

        # -- config ----------------------------------------------------------
        self.levels: int = int(data_conf.get("levels", 17))
        self.mean_path: str = data_conf["mean_path"]
        self.std_path: str = data_conf["std_path"]
        self.concentration_params_json_path: Optional[str] = data_conf.get("concentration_params_json")

        self.mae_pad_to: int = int(data_conf.get("mae_pad_to", 304))
        self.mae_include_t1_refl: bool = bool(data_conf.get("mae_include_t1_refl", True))

        self.max_open_files: int = int(data_conf.get("max_open_files_per_worker", 8))
        # Always open zarr WITHOUT Dask (chunks=None) — Dask scheduler accumulates
        # task graphs in CPU RAM across batches, causing a slow memory leak that
        # eventually triggers the system OOM killer.
        self.zarr_chunks = None
        self.max_dataset_open_retries: int = int(data_conf.get("max_dataset_open_retries", 3))

        self.rng = np.random.default_rng(seed=seed)
        self.output_dtype = torch.float32

        # -- lazy-init state -------------------------------------------------
        self._opened: bool = False
        self.total_length: Optional[int] = None
        self.file_n_time: Optional[List[int]] = None   # cached per-file timestep counts; None = not yet scanned
        self.samples_per_file: Optional[List[int]] = None
        self.cumulative_samples: Optional[np.ndarray] = None
        self.upper_ds_cache: OrderedDict = OrderedDict()
        self._concentration_params: dict = {v: dict(_DEFAULT_CONCENTRATION_PARAMS) for v in CONCENTRATION_VARS}
        self._mean_values: dict = {}
        self._std_values: dict = {}
        # Files that failed to open during training are blacklisted so workers
        # don't waste retries on them again.
        self._bad_file_indices: set = set()

        # Determine which vars have levels in the zarr (upper-air = 3D)
        self._upper_air_set = set(self._upper_air_vars)

    # ------------------------------------------------------------------ #
    # Lazy open (called per DataLoader worker)
    # ------------------------------------------------------------------ #
    def _open_datasets(self):
        if self._opened:
            return

        if not self.filenames:
            raise ValueError("WoFSMAEDataset: no training files provided.")

        # Load stats
        mean_ds = xr.open_dataset(self.mean_path).load()
        std_ds = xr.open_dataset(self.std_path).load()

        # Load concentration params from JSON
        if self.concentration_params_json_path:
            try:
                with open(self.concentration_params_json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                params_dict = payload.get("variables", payload) if isinstance(payload, dict) else {}
                for var, info in params_dict.items():
                    if var not in CONCENTRATION_VARS or not isinstance(info, dict):
                        continue
                    if "recommended" in info and isinstance(info["recommended"], dict):
                        self._concentration_params[var] = _coerce_concentration_params(info["recommended"])
                    else:
                        self._concentration_params[var] = _coerce_concentration_params(info)
            except Exception as exc:
                logger.warning("Failed to load concentration params JSON: %s", exc)

        # Load mean/std arrays for all needed vars
        all_vars = self._upper_air_vars + [v for v in self._surface_2d_vars if v in mean_ds]
        for var in all_vars:
            if var in mean_ds and var in std_ds:
                mean_vals = mean_ds[var].values
                std_vals = std_ds[var].values
                if var in CONCENTRATION_VARS:
                    mean_vals = np.where(np.isnan(mean_vals), 0.0, mean_vals)
                    std_vals = np.where(np.isnan(std_vals), 1.0, std_vals)
                # Ensure std > 0
                std_vals = np.where(std_vals <= 0.0, 1.0, std_vals)
                self._mean_values[var] = mean_vals
                self._std_values[var] = std_vals

        self._build_index_map()
        self._opened = True
        logger.info("WoFSMAEDataset opened: %d files, %d samples total",
                    len(self.filenames), self.total_length)

    def _build_index_map(self):
        # Expensive file-scan only runs once (like WoFSDAIncrementDataset).
        # On subsequent calls (e.g., worker re-invocations) we skip the scan
        # and just recompute the cumulative totals from the cached counts.
        # This prevents the main-process and worker counts from diverging,
        # which would cause file_idx-out-of-range errors in __getitem__.
        if self.file_n_time is None:
            valid_filenames: List[str] = []
            file_n_time: List[int] = []
            skipped = []
            for fn in self.filenames:
                ds = None
                try:
                    ds = self._open_filtered_dataset(fn, self._upper_air_vars, max_retries=2)
                    n_time = ds.dims.get("time", ds.sizes.get("time", 0))
                    valid_filenames.append(fn)
                    file_n_time.append(int(n_time))
                except Exception as exc:
                    skipped.append(fn)
                    logger.warning("Skipping file %s due to error: %s", fn, exc)
                finally:
                    self._safe_close_dataset(ds)
            if skipped:
                logger.warning("Filtered %d unreadable file(s); using %d file(s).",
                               len(skipped), len(valid_filenames))
            self.filenames = valid_filenames
            self.file_n_time = file_n_time

        # Cheap: always recompute from cached counts (each sample needs 2 consecutive timesteps)
        self.samples_per_file = [max(0, n - 1) for n in self.file_n_time]
        self.cumulative_samples = np.cumsum(self.samples_per_file, dtype=np.int64)
        self.total_length = int(self.cumulative_samples[-1]) if len(self.cumulative_samples) > 0 else 0

    def _locate_index(self, index: int):
        file_idx = int(np.searchsorted(self.cumulative_samples, index, side="right"))
        prev_cum = 0 if file_idx == 0 else int(self.cumulative_samples[file_idx - 1])
        ind_in_file = index - prev_cum
        return file_idx, int(ind_in_file)

    @staticmethod
    def _safe_close_dataset(ds):
        try:
            ds.close()
        except Exception:
            pass

    def _open_filtered_dataset(self, file_path: str, varnames_keep, max_retries: int = 3):
        attempts = max(1, int(max_retries))
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
                    logger.warning("Retrying open %d/%d for %s: %s", attempt, attempts, file_path, exc)
                    time.sleep(0.1 * attempt)
        raise RuntimeError(f"Failed to open {file_path} after {attempts} attempts: {last_exc}") from last_exc

    def _evict_if_needed(self, cache: OrderedDict):
        while len(cache) > self.max_open_files:
            _, ds = cache.popitem(last=False)
            self._safe_close_dataset(ds)

    def _get_upper_ds(self, file_idx: int):
        if file_idx in self._bad_file_indices:
            raise RuntimeError(
                f"File index {file_idx} ({self.filenames[file_idx]}) is blacklisted "
                "after previous open failure."
            )
        cached = self.upper_ds_cache.get(file_idx)
        if cached is not None:
            self.upper_ds_cache.move_to_end(file_idx)
            return cached
        try:
            ds = self._open_filtered_dataset(
                self.filenames[file_idx], self._upper_air_vars, self.max_dataset_open_retries
            )
        except Exception as exc:
            self._bad_file_indices.add(file_idx)
            logger.warning(
                "Blacklisting file %s after repeated open failures: %s",
                self.filenames[file_idx], exc,
            )
            raise
        self.upper_ds_cache[file_idx] = ds
        self._evict_if_needed(self.upper_ds_cache)
        return ds

    def __del__(self):
        try:
            for ds in self.upper_ds_cache.values():
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
    # Normalization helpers
    # ------------------------------------------------------------------ #
    def _broadcast_stats(self, stats: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Broadcast 1-D stats array to match values shape along the matching axis."""
        stats_arr = np.asarray(stats)
        if stats_arr.ndim != 1:
            return stats_arr
        stat_len = stats_arr.shape[0]
        matching = [ax for ax, sz in enumerate(values.shape) if sz == stat_len]
        if not matching:
            return stats_arr
        axis = 1 if (len(matching) > 1 and 1 in matching) else matching[0]
        shape = [1] * values.ndim
        shape[axis] = stat_len
        return stats_arr.reshape(shape)

    def _normalize_array(self, arr: np.ndarray, var_name: str) -> np.ndarray:
        """Apply concentration transform (if needed) then z-score normalize."""
        x = arr.astype(np.float32, copy=False)
        if var_name in CONCENTRATION_VARS:
            params = self._concentration_params.get(var_name, _DEFAULT_CONCENTRATION_PARAMS)
            x = _forward_concentration_numpy(x, params)
        mean = self._broadcast_stats(self._mean_values[var_name], x)
        std = self._broadcast_stats(self._std_values[var_name], x)
        return (x - mean) / std

    # ------------------------------------------------------------------ #
    # Spatial padding helper
    # ------------------------------------------------------------------ #
    def _pad_to_size(self, arr: np.ndarray, target: int) -> np.ndarray:
        """Pad spatial dims (last two) to target × target.  arr: (..., H, W)."""
        h, w = arr.shape[-2], arr.shape[-1]
        if h == target and w == target:
            return arr
        pad_h = target - h
        pad_w = target - w
        if pad_h < 0 or pad_w < 0:
            raise ValueError(f"_pad_to_size: arr ({h},{w}) larger than target {target}")
        # np.pad with trailing zero-pad on spatial dims
        pad_spec = [(0, 0)] * (arr.ndim - 2) + [(0, pad_h), (0, pad_w)]
        return np.pad(arr, pad_spec, mode="constant", constant_values=0.0)

    # ------------------------------------------------------------------ #
    # Load one modality group from the zarr chunk
    # ------------------------------------------------------------------ #
    def _load_upper_air_group(
        self,
        chunk,
        var_list: List[str],
        time_idx: int,
    ) -> np.ndarray:
        """Load and normalize a list of 3D vars; stack along channel axis → (C, H, W)."""
        parts = []
        for var in var_list:
            raw = chunk[var].isel(time=time_idx).values  # (level, H, W)
            norm = self._normalize_array(raw, var)
            parts.append(norm.astype(np.float32))
        return np.concatenate(parts, axis=0)  # (C=levels*nvars, H, W)

    def _load_surface_group(
        self,
        chunk,
        var_list: List[str],
        time_idx: int,
    ) -> np.ndarray:
        """Load and normalize a list of 2D vars; stack along channel axis → (C, H, W)."""
        parts = []
        for var in var_list:
            if var not in chunk:
                # forcing var not in the zarr (pre-computed externally) — skip with zeros
                logger.debug("Variable %s not found in chunk; filling with zeros.", var)
                # We'll handle forcing fields that are computed on-the-fly below
                continue
            raw = chunk[var].isel(time=time_idx).values  # (H, W)
            if var in self._mean_values:
                norm = self._normalize_array(raw, var)
            else:
                norm = raw.astype(np.float32)
            parts.append(norm.astype(np.float32)[np.newaxis, ...])  # (1, H, W)
        if parts:
            return np.concatenate(parts, axis=0)
        return np.zeros((len(var_list), chunk.dims.get("y", 300), chunk.dims.get("x", 300)), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # __getitem__
    # ------------------------------------------------------------------ #
    def _load_sample(self, index: int) -> Dict[str, torch.Tensor]:
        """Load a single sample by global index.  Raises on any failure."""
        file_idx, ind_in_file = self._locate_index(index)
        t0 = ind_in_file
        t1 = ind_in_file + 1

        # Load two consecutive timesteps and immediately materialise to numpy.
        # We use try/finally to ensure chunk.close() is always called — this
        # prevents xarray from retaining decoded in-memory buffers across batches
        # (the main source of the slow CPU-RAM leak).
        upper_ds = self._get_upper_ds(file_idx)
        chunk = upper_ds.isel(time=slice(t0, t1 + 1)).load()
        try:
            target_size = self.mae_pad_to

            # ---- background (t0) -----------------------------------------
            bg = self._load_upper_air_group(chunk, self.background_vars, time_idx=0)
            bg = self._pad_to_size(bg, target_size)

            # ---- precip (t0) ---------------------------------------------
            pr = self._load_upper_air_group(chunk, self.precip_vars, time_idx=0)
            pr = self._pad_to_size(pr, target_size)

            # ---- reflectivity (t0 or t1) ---------------------------------
            refl_t = 1 if self.mae_include_t1_refl else 0
            rf = self._load_upper_air_group(chunk, self.reflectivity_vars, time_idx=refl_t)
            rf = self._pad_to_size(rf, target_size)

            # ---- surface (t0) --------------------------------------------
            sf = self._load_surface_group(chunk, self.surface_vars, time_idx=0)
            sf = self._pad_to_size(sf, target_size)

            # ---- forcing: encode from datetime ---------------------------
            dt0 = chunk.time.values[0]
            forcing_arr = self._compute_forcing(dt0, chunk, target_size)
        finally:
            # Release the loaded xarray buffer immediately — numpy arrays
            # above already hold copies of the data we need.
            self._safe_close_dataset(chunk)

        return {
            "background":   torch.as_tensor(bg, dtype=self.output_dtype),
            "precip":       torch.as_tensor(pr, dtype=self.output_dtype),
            "reflectivity": torch.as_tensor(rf, dtype=self.output_dtype),
            "surface":      torch.as_tensor(sf, dtype=self.output_dtype),
            "forcing":      torch.as_tensor(forcing_arr, dtype=self.output_dtype),
            "index":        index,
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if not self._opened:
            self._open_datasets()

        _max_retries = 5
        _last_exc: Optional[Exception] = None
        for _attempt in range(_max_retries):
            try:
                return self._load_sample(index)
            except Exception as exc:
                _last_exc = exc
                logger.warning(
                    "Sample index=%d failed (attempt %d/%d): %s — picking a random replacement.",
                    index, _attempt + 1, _max_retries, exc,
                )
                # Pick a different random index to avoid looping on the same bad file.
                index = int(self.rng.integers(0, max(1, self.total_length)))

        raise RuntimeError(
            f"WoFSMAEDataset: all {_max_retries} sample retries exhausted"
        ) from _last_exc

    # ------------------------------------------------------------------ #
    # Forcing channel computation
    # ------------------------------------------------------------------ #
    def _compute_forcing(self, dt64, chunk, target_size: int) -> np.ndarray:
        """
        Compute 10-channel forcing array from datetime and spatial metadata.
        Returns (10, H_pad, W_pad).

        Channels:
            0: cos_lat     1: sin_lat
            2: cos_lon     3: sin_lon
            4: cos_julian_day  5: sin_julian_day
            6: cos_local_time  7: sin_local_time
            8: cos_solar_zenith
            9: insolation
        """
        import datetime

        # Convert numpy datetime64 to Python datetime
        ts = (dt64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
        dt = datetime.datetime.utcfromtimestamp(float(ts))

        # Spatial coordinates — try to get from chunk, else create dummy grids
        H = target_size
        W = target_size

        if "lat" in chunk.coords:
            lat_vals = chunk["lat"].values.astype(np.float32)
        else:
            lat_vals = np.linspace(30.0, 50.0, H, dtype=np.float32)

        if "lon" in chunk.coords:
            lon_vals = chunk["lon"].values.astype(np.float32)
        else:
            lon_vals = np.linspace(-110.0, -80.0, W, dtype=np.float32)

        # Build 2D lat/lon grids (crop / pad to H×W)
        lat_1d = lat_vals.flat if lat_vals.ndim > 1 else lat_vals
        lon_1d = lon_vals.flat if lon_vals.ndim > 1 else lon_vals
        lat_1d = np.asarray(lat_1d, dtype=np.float32).ravel()[:H]
        lon_1d = np.asarray(lon_1d, dtype=np.float32).ravel()[:W]

        lat_2d = np.broadcast_to(lat_1d[:, np.newaxis], (H, W))
        lon_2d = np.broadcast_to(lon_1d[np.newaxis, :], (H, W))

        lat_rad = np.deg2rad(lat_2d)
        lon_rad = np.deg2rad(lon_2d)

        # Julian day / local time
        day_of_year = dt.timetuple().tm_yday
        total_days = 366 if (dt.year % 4 == 0 and (dt.year % 100 != 0 or dt.year % 400 == 0)) else 365
        julian_angle = 2.0 * np.pi * day_of_year / total_days

        utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        local_hour = (utc_hour + lon_2d / 15.0) % 24.0  # approximate LST
        local_angle = 2.0 * np.pi * local_hour / 24.0

        # Solar zenith (simplified)
        # Declination
        decl = 0.409 * np.sin(julian_angle - 1.39)
        # Hour angle (15° per hour)
        h_angle = np.deg2rad((local_hour - 12.0) * 15.0)
        cos_zenith = (
            np.sin(lat_rad) * np.sin(decl)
            + np.cos(lat_rad) * np.cos(decl) * np.cos(h_angle)
        ).astype(np.float32)
        insolation = np.clip(cos_zenith, 0.0, None).astype(np.float32)

        out = np.stack([
            np.cos(lat_rad).astype(np.float32),          # 0
            np.sin(lat_rad).astype(np.float32),          # 1
            np.cos(lon_rad).astype(np.float32),          # 2
            np.sin(lon_rad).astype(np.float32),          # 3
            np.full((H, W), np.cos(julian_angle), dtype=np.float32),  # 4
            np.full((H, W), np.sin(julian_angle), dtype=np.float32),  # 5
            np.cos(local_angle).astype(np.float32),      # 6
            np.sin(local_angle).astype(np.float32),      # 7
            cos_zenith,                                   # 8
            insolation,                                   # 9
        ], axis=0)  # (10, H, W)

        # Pad to target_size if needed
        out = self._pad_to_size(out, target_size)
        return out
