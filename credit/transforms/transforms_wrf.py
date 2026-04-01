"""
normalize_wrf.py
-------------------------------------------------------
Content
    - NormalizeWRF
    - ToTensorWRF
"""

import logging
from typing import Dict

import numpy as np
import xarray as xr

import torch

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


class NormalizeWRF:
    def __init__(self, conf):
        self.mean_ds = xr.open_dataset(conf["data"]["mean_path"]).load()
        self.std_ds = xr.open_dataset(conf["data"]["std_path"]).load()

        varnames_all = conf["data"]["all_varnames"]

        self.mean_tensors = {}
        self.std_tensors = {}

        for var in varnames_all:
            mean_array = self.mean_ds[var].values
            std_array = self.std_ds[var].values
            # convert to tensor
            self.mean_tensors[var] = torch.tensor(mean_array)
            self.std_tensors[var] = torch.tensor(std_array)

        # Get levels and upper air variables
        self.levels = conf["data"]["levels"]  # It was conf['model']['levels']
        self.varname_upper_air = conf["data"]["variables"]
        self.num_upper_air = len(self.varname_upper_air) * self.levels

        # Identify the existence of other variables
        self.flag_surface = ("surface_variables" in conf["data"]) and (len(conf["data"]["surface_variables"]) > 0)
        self.flag_dyn_forcing = ("dynamic_forcing_variables" in conf["data"]) and (
            len(conf["data"]["dynamic_forcing_variables"]) > 0
        )
        self.flag_diagnostic = ("diagnostic_variables" in conf["data"]) and (
            len(conf["data"]["diagnostic_variables"]) > 0
        )
        self.flag_forcing = ("forcing_variables" in conf["data"]) and (len(conf["data"]["forcing_variables"]) > 0)
        self.flag_static = ("static_variables" in conf["data"]) and (len(conf["data"]["static_variables"]) > 0)

        # Get surface varnames
        if self.flag_surface:
            self.varname_surface = conf["data"]["surface_variables"]
            self.num_surface = len(self.varname_surface)

        # Get dynamic forcing varnames
        if self.flag_dyn_forcing:
            self.varname_dyn_forcing = conf["data"]["dynamic_forcing_variables"]
            self.num_dyn_forcing = len(self.varname_dyn_forcing)

        # Get diagnostic varnames
        if self.flag_diagnostic:
            self.varname_diagnostic = conf["data"]["diagnostic_variables"]
            self.num_diagnostic = len(self.varname_diagnostic)

        # Get forcing varnames
        if self.flag_forcing:
            self.varname_forcing = conf["data"]["forcing_variables"]
        else:
            self.varname_forcing = []

        # Get static varnames:
        if self.flag_static:
            self.varname_static = conf["data"]["static_variables"]
        else:
            self.varname_static = []

        if self.flag_forcing or self.flag_static:
            self.has_forcing_static = True
            self.num_static = len(self.varname_static)
            self.num_forcing = len(self.varname_forcing)
            self.num_forcing_static = self.num_static + self.num_forcing
            self.varname_forcing_static = self.varname_forcing + self.varname_static
            self.static_first = conf["data"]["static_first"]
        else:
            self.has_forcing_static = False

        logger.info("WRF domain z-score parameters loaded")

        # ======================================================================= #
        # boundary condition data handling
        # ======================================================================= #
        boundary_conf = conf["data"]["boundary"]
        varnames_all_outside = boundary_conf.get(
            "all_varnames",
            boundary_conf.get("variables", []) + boundary_conf.get("surface_variables", []),
        )

        reuse_interior_boundary_stats = boundary_conf.get("reuse_interior_stats", False)
        has_boundary_stats = ("mean_path" in boundary_conf) and ("std_path" in boundary_conf)

        if has_boundary_stats and not reuse_interior_boundary_stats:
            self.mean_ds_outside = xr.open_dataset(boundary_conf["mean_path"]).load()
            self.std_ds_outside = xr.open_dataset(boundary_conf["std_path"]).load()
            logger.info("Boundary domain z-score parameters loaded")
        else:
            missing_vars = [var for var in varnames_all_outside if var not in self.mean_ds.data_vars or var not in self.std_ds.data_vars]
            if missing_vars:
                raise KeyError(
                    "Boundary stats are configured to reuse interior stats, but these boundary variables are missing "
                    f"from interior mean/std files: {missing_vars}"
                )

            self.mean_ds_outside = self.mean_ds[varnames_all_outside].load()
            self.std_ds_outside = self.std_ds[varnames_all_outside].load()
            logger.info("Boundary domain reuses interior z-score parameters")

        self.mean_tensors_outside = {}
        self.std_tensors_outside = {}

        for var in varnames_all_outside:
            mean_array = self.mean_ds_outside[var].values
            std_array = self.std_ds_outside[var].values
            # convert to tensor
            self.mean_tensors_outside[var] = torch.tensor(mean_array)
            self.std_tensors_outside[var] = torch.tensor(std_array)

        # Get levels and upper air variables
        self.levels_outside = conf["data"]["boundary"]["levels"]
        self.varname_upper_air_outside = conf["data"]["boundary"]["variables"]
        self.num_upper_air_outside = len(self.varname_upper_air_outside) * self.levels_outside

        self.flag_surface_outside = ("surface_variables" in conf["data"]["boundary"]) and (
            len(conf["data"]["boundary"]["surface_variables"]) > 0
        )

        # Get surface varnames
        if self.flag_surface_outside:
            self.varname_surface_outside = conf["data"]["boundary"]["surface_variables"]
            self.num_surface_outside = len(self.varname_surface_outside)

    @staticmethod
    def _forward_concentration_numpy(x: np.ndarray) -> np.ndarray:
        x64 = np.asarray(x, dtype=np.float64)
        transformed = 0.5 * np.minimum(x64, 2.5) + 0.5 * (
            np.log(np.maximum(x64, _CONC_EPS)) - _LOG_EPS
        ) / _NEG_LOG_EPS
        return transformed.astype(x.dtype, copy=False)

    @staticmethod
    def _reshape_stats_for_data(
        stat_values: np.ndarray,
        stat_dims: tuple[object, ...],
        data_values: np.ndarray,
        data_dims: tuple[object, ...],
    ) -> np.ndarray:
        if stat_values.ndim == 0 or stat_values.shape == data_values.shape:
            return stat_values

        reshape = [1] * data_values.ndim
        for dim_name, dim_size in zip(stat_dims, stat_values.shape):
            if dim_name not in data_dims:
                return stat_values
            axis = data_dims.index(dim_name)
            reshape[axis] = dim_size
        return stat_values.reshape(reshape)

    def _normalize_numpy_array(
        self,
        var_values: np.ndarray,
        var_dims: tuple[object, ...],
        mean_da: xr.DataArray,
        std_da: xr.DataArray,
    ) -> np.ndarray:
        mean_values = self._reshape_stats_for_data(mean_da.values, mean_da.dims, var_values, var_dims)
        std_values = self._reshape_stats_for_data(std_da.values, std_da.dims, var_values, var_dims)
        return (var_values - mean_values) / std_values

    @staticmethod
    def _forward_concentration_torch(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.clamp(x, max=_CONC_MAX) + 0.5 * (
            torch.log(torch.clamp(x, min=_CONC_EPS)) - _LOG_EPS
        ) / _NEG_LOG_EPS

    @staticmethod
    def _inverse_concentration_torch(x: torch.Tensor) -> torch.Tensor:
        y = x
        y_low = 0.5 * _CONC_EPS
        y_high = 0.5 * _CONC_MAX + 0.5 * (np.log(_CONC_MAX) - _LOG_EPS) / _NEG_LOG_EPS

        out = torch.empty_like(y)

        mask_low = y <= y_low
        if torch.any(mask_low):
            out[mask_low] = 2.0 * y[mask_low]

        mask_high = y >= y_high
        if torch.any(mask_high):
            out[mask_high] = _CONC_EPS * torch.exp(2.0 * _NEG_LOG_EPS * (y[mask_high] - 0.5 * _CONC_MAX))

        mask_mid = (~mask_low) & (~mask_high)
        if torch.any(mask_mid):
            target = y[mask_mid]
            lo = torch.full_like(target, _CONC_EPS)
            hi = torch.full_like(target, _CONC_MAX)
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                f_mid = 0.5 * mid + 0.5 * (torch.log(mid) - _LOG_EPS) / _NEG_LOG_EPS
                go_right = f_mid < target
                lo = torch.where(go_right, mid, lo)
                hi = torch.where(go_right, hi, mid)
            out[mask_mid] = 0.5 * (lo + hi)

        return out

    def __call__(self, sample, inverse: bool = False):
        if inverse:
            # Inverse transformation
            return self.inverse_transform(sample)
        else:
            # Transformation
            return self.transform(sample)

    def transform_array(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function applies to y_pred, so there won't be boundary input, forcing, and static variables.
        """
        # Get the current device
        device = x.device

        # Subset upper air
        tensor_upper_air = x[:, : self.num_upper_air, :, :]
        transformed_upper_air = tensor_upper_air.clone()

        # Surface variables
        if self.flag_surface:
            tensor_surface = x[:, self.num_upper_air : (self.num_upper_air + self.num_surface), :, :]
            transformed_surface = tensor_surface.clone()

        # y_pred does not have dynamic_forcing, skip this var type

        # Diagnostic variables (the very last of the stack)
        if self.flag_diagnostic:
            tensor_diagnostic = x[:, -self.num_diagnostic :, :, :]
            transformed_diagnostic = tensor_diagnostic.clone()

        # Standardize upper air variables
        # Upper air variable structure: var 1 [all levels] --> var 2 [all levels]
        k = 0
        for name in self.varname_upper_air:
            mean_tensor = self.mean_tensors[name].to(device)
            std_tensor = self.std_tensors[name].to(device)

            for level in range(self.levels):
                var_mean = mean_tensor[level]
                var_std = std_tensor[level]
                upper_air_level = tensor_upper_air[:, k]
                if name in CONCENTRATION_VARS:
                    upper_air_level = self._forward_concentration_torch(upper_air_level)
                transformed_upper_air[:, k] = (upper_air_level - var_mean) / var_std
                k += 1

        # Standardize surface variables
        if self.flag_surface:
            for k, name in enumerate(self.varname_surface):
                var_mean = self.mean_tensors[name].to(device)
                var_std = self.std_tensors[name].to(device)
                surface_slice = tensor_surface[:, k]
                if name in CONCENTRATION_VARS:
                    surface_slice = self._forward_concentration_torch(surface_slice)
                transformed_surface[:, k] = (surface_slice - var_mean) / var_std

        # Standardize diagnostic variables
        if self.flag_diagnostic:
            for k, name in enumerate(self.varname_diagnostic):
                var_mean = self.mean_tensors[name].to(device)
                var_std = self.std_tensors[name].to(device)
                diagnostic_slice = transformed_diagnostic[:, k]
                if name in CONCENTRATION_VARS:
                    diagnostic_slice = self._forward_concentration_torch(diagnostic_slice)
                transformed_diagnostic[:, k] = (diagnostic_slice - var_mean) / var_std

        # Concatenate everything
        if self.flag_surface:
            if self.flag_diagnostic:
                transformed_x = torch.cat(
                    (
                        transformed_upper_air,
                        transformed_surface,
                        transformed_diagnostic,
                    ),
                    dim=1,
                )

            else:
                transformed_x = torch.cat((transformed_upper_air, transformed_surface), dim=1)
        else:
            if self.flag_diagnostic:
                transformed_x = torch.cat((transformed_upper_air, transformed_diagnostic), dim=1)
            else:
                transformed_x = transformed_upper_air

        return transformed_x.to(device)

    def transform(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        This function transforms training batches using .values array arithmetic to avoid xarray slowdowns.
        Boundary inputs are skipped if they were pre-normalized by the dataset (flagged via
        ``sample.get('_boundary_pre_normalized', False)``).
        """
        normalized_sample = {}
        # Check whether boundary anchors have already been normalized by the dataset
        boundary_already_normalized = sample.pop("_boundary_pre_normalized", False)

        if self.has_forcing_static:
            for key, value in sample.items():
                if isinstance(value, xr.Dataset):
                    if key == "WRF_input":
                        varname_inputs = value.keys()
                        for varname_raw in varname_inputs:
                            varname = str(varname_raw)
                            if (varname in self.varname_forcing_static) is False:
                                var_da = value.data_vars[varname]
                                var_values = var_da.values
                                if varname in CONCENTRATION_VARS:
                                    var_values = self._forward_concentration_numpy(var_values)
                                normalized_values = self._normalize_numpy_array(
                                    var_values,
                                    var_da.dims,
                                    self.mean_ds[varname],
                                    self.std_ds[varname],
                                )
                                value[varname] = xr.DataArray(
                                    normalized_values, dims=var_da.dims, coords=var_da.coords
                                )
                        normalized_sample[key] = value
                    elif key == "WRF_target":
                        for varname_raw in value.keys():
                            varname = str(varname_raw)
                            var_da = value.data_vars[varname]
                            var_values = var_da.values
                            if varname in CONCENTRATION_VARS:
                                var_values = self._forward_concentration_numpy(var_values)
                            normalized_values = self._normalize_numpy_array(
                                var_values,
                                var_da.dims,
                                self.mean_ds[varname],
                                self.std_ds[varname],
                            )
                            value[varname] = xr.DataArray(
                                normalized_values, dims=var_da.dims, coords=var_da.coords
                            )
                        normalized_sample[key] = value
                    elif key == "boundary_input":
                        if not boundary_already_normalized:
                            for varname_raw in value.keys():
                                varname = str(varname_raw)
                                var_da = value.data_vars[varname]
                                var_values = var_da.values
                                if varname in CONCENTRATION_VARS:
                                    var_values = self._forward_concentration_numpy(var_values)
                                normalized_values = self._normalize_numpy_array(
                                    var_values,
                                    var_da.dims,
                                    self.mean_ds_outside[varname],
                                    self.std_ds_outside[varname],
                                )
                                value[varname] = xr.DataArray(
                                    normalized_values, dims=var_da.dims, coords=var_da.coords
                                )
                        normalized_sample[key] = value
                elif key == "time_encode":
                    normalized_sample[key] = value
        else:
            for key, value in sample.items():
                if isinstance(value, xr.Dataset):
                    if key == "WRF_input" or key == "WRF_target":
                        for varname_raw in value.keys():
                            varname = str(varname_raw)
                            var_da = value.data_vars[varname]
                            var_values = var_da.values
                            if varname in CONCENTRATION_VARS:
                                var_values = self._forward_concentration_numpy(var_values)
                            normalized_values = self._normalize_numpy_array(
                                var_values,
                                var_da.dims,
                                self.mean_ds[varname],
                                self.std_ds[varname],
                            )
                            value[varname] = xr.DataArray(
                                normalized_values, dims=var_da.dims, coords=var_da.coords
                            )
                        normalized_sample[key] = value
                    elif key == "boundary_input":
                        if not boundary_already_normalized:
                            for varname_raw in value.keys():
                                varname = str(varname_raw)
                                var_da = value.data_vars[varname]
                                var_values = var_da.values
                                if varname in CONCENTRATION_VARS:
                                    var_values = self._forward_concentration_numpy(var_values)
                                normalized_values = self._normalize_numpy_array(
                                    var_values,
                                    var_da.dims,
                                    self.mean_ds_outside[varname],
                                    self.std_ds_outside[varname],
                                )
                                value[varname] = xr.DataArray(
                                    normalized_values, dims=var_da.dims, coords=var_da.coords
                                )
                        normalized_sample[key] = value
                elif key == "time_encode":
                    normalized_sample[key] = value

        return normalized_sample

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function applies to y_pred, so there won't be dynamic forcing, forcing, and static vars
        """
        # Get the current device
        device = x.device

        # Subset upper air
        tensor_upper_air = x[:, : self.num_upper_air, :, :]
        transformed_upper_air = tensor_upper_air.clone()

        # Surface variables
        if self.flag_surface:
            tensor_surface = x[:, self.num_upper_air : (self.num_upper_air + self.num_surface), :, :]
            transformed_surface = tensor_surface.clone()

        # Diagnostic variables (the very last of the stack)
        if self.flag_diagnostic:
            tensor_diagnostic = x[:, -self.num_diagnostic :, :, :]
            transformed_diagnostic = tensor_diagnostic.clone()

        # Reverse upper air variables
        k = 0
        for name in self.varname_upper_air:
            mean_tensor = self.mean_tensors[name].to(device)
            std_tensor = self.std_tensors[name].to(device)
            for level in range(self.levels):
                mean = mean_tensor[level]
                std = std_tensor[level]
                upper_air_level = tensor_upper_air[:, k] * std + mean
                if name in CONCENTRATION_VARS:
                    upper_air_level = self._inverse_concentration_torch(upper_air_level)
                transformed_upper_air[:, k] = upper_air_level
                k += 1

        # Reverse surface variables
        if self.flag_surface:
            for k, name in enumerate(self.varname_surface):
                mean = self.mean_tensors[name].to(device)
                std = self.std_tensors[name].to(device)
                surface_slice = tensor_surface[:, k] * std + mean
                if name in CONCENTRATION_VARS:
                    surface_slice = self._inverse_concentration_torch(surface_slice)
                transformed_surface[:, k] = surface_slice

        # Reverse diagnostic variables
        if self.flag_diagnostic:
            for k, name in enumerate(self.varname_diagnostic):
                mean = self.mean_tensors[name].to(device)
                std = self.std_tensors[name].to(device)
                diagnostic_slice = transformed_diagnostic[:, k] * std + mean
                if name in CONCENTRATION_VARS:
                    diagnostic_slice = self._inverse_concentration_torch(diagnostic_slice)
                transformed_diagnostic[:, k] = diagnostic_slice

        # Concatenate everything
        if self.flag_surface:
            if self.flag_diagnostic:
                transformed_x = torch.cat(
                    (
                        transformed_upper_air,
                        transformed_surface,
                        transformed_diagnostic,
                    ),
                    dim=1,
                )
            else:
                transformed_x = torch.cat((transformed_upper_air, transformed_surface), dim=1)
        else:
            if self.flag_diagnostic:
                transformed_x = torch.cat((transformed_upper_air, transformed_diagnostic), dim=1)
            else:
                transformed_x = transformed_upper_air

        return transformed_x.to(device)


class ToTensorWRF:
    def __init__(self, conf):
        self.conf = conf

        # =============================================== #
        self.output_dtype = torch.float32
        # ============================================== #

        self.hist_len = int(conf["data"]["history_len"])
        self.for_len = int(conf["data"]["forecast_len"])

        # identify the existence of other variables
        self.flag_surface = ("surface_variables" in conf["data"]) and (len(conf["data"]["surface_variables"]) > 0)
        self.flag_dyn_forcing = ("dynamic_forcing_variables" in conf["data"]) and (
            len(conf["data"]["dynamic_forcing_variables"]) > 0
        )
        self.flag_diagnostic = ("diagnostic_variables" in conf["data"]) and (
            len(conf["data"]["diagnostic_variables"]) > 0
        )
        self.flag_forcing = ("forcing_variables" in conf["data"]) and (len(conf["data"]["forcing_variables"]) > 0)
        self.flag_static = ("static_variables" in conf["data"]) and (len(conf["data"]["static_variables"]) > 0)

        self.varname_upper_air = conf["data"]["variables"]

        # get surface varnames
        if self.flag_surface:
            self.varname_surface = conf["data"]["surface_variables"]

        # get dynamic forcing varnames
        self.num_forcing_static = 0
        self.flag_static_first = ("static_first" in conf["data"]) and (conf["data"]["static_first"])

        if self.flag_dyn_forcing:
            self.varname_dyn_forcing = conf["data"]["dynamic_forcing_variables"]
            self.num_forcing_static += len(self.varname_dyn_forcing)
        else:
            self.varname_dyn_forcing = []

        # get diagnostic varnames
        if self.flag_diagnostic:
            self.varname_diagnostic = conf["data"]["diagnostic_variables"]

        # get forcing varnames
        if self.flag_forcing:
            self.varname_forcing = conf["data"]["forcing_variables"]
            self.num_forcing_static += len(self.varname_forcing)
        else:
            self.varname_forcing = []

        # get static varnames:
        if self.flag_static:
            self.varname_static = conf["data"]["static_variables"]
            self.num_forcing_static += len(self.varname_static)
        else:
            self.varname_static = []

        if self.flag_dyn_forcing or self.flag_forcing or self.flag_static:
            self.has_forcing_static = True
        else:
            self.has_forcing_static = False

        # ======================================================================= #
        # boundary condition data handling
        # ======================================================================= #
        self.hist_len_outside = int(conf["data"]["boundary"]["history_len"])
        self.for_len_outside = int(conf["data"]["boundary"]["forecast_len"])

        self.flag_surface_outside = ("surface_variables" in conf["data"]["boundary"]) and (
            len(conf["data"]["boundary"]["surface_variables"]) > 0
        )

        self.varname_upper_air_outside = conf["data"]["boundary"]["variables"]

        # get surface varnames
        if self.flag_surface_outside:
            self.varname_surface_outside = conf["data"]["boundary"]["surface_variables"]

    def __call__(self, sample):
        return_dict = {}

        for key, value in sample.items():
            ## if DataArray
            if isinstance(value, xr.DataArray):
                var_value = value.values

            ## if Dataset
            elif isinstance(value, xr.Dataset):
                # WRF domain ds to numpy conversion
                if key == "WRF_input" or key == "WRF_target":
                    # organize upper-air vars
                    list_vars_upper_air = []
                    for var_name in self.varname_upper_air:
                        var_value = value[var_name].values
                        list_vars_upper_air.append(var_value)

                    # [num_vars, hist_len, num_levels, lat, lon]
                    numpy_vars_upper_air = np.array(list_vars_upper_air)

                    # organize surface vars
                    if self.flag_surface:
                        list_vars_surface = []
                        for var_name in self.varname_surface:
                            var_value = value[var_name].values
                            list_vars_surface.append(var_value)

                        # [num_surf_vars, hist_len, lat, lon]
                        numpy_vars_surface = np.array(list_vars_surface)

                    # organize forcing and static (input only)
                    if self.has_forcing_static or self.flag_dyn_forcing:
                        # enter this scope if one of the (dyn_forcing, folrcing, static) exists
                        if self.flag_static_first:
                            varname_forcing_static = (
                                self.varname_static + self.varname_dyn_forcing + self.varname_forcing
                            )
                        else:
                            varname_forcing_static = (
                                self.varname_dyn_forcing + self.varname_forcing + self.varname_static
                            )

                        if key == "WRF_input":
                            list_vars_forcing_static = []
                            for var_name in varname_forcing_static:
                                var_value = value[var_name].values
                                list_vars_forcing_static.append(var_value)
                            numpy_vars_forcing_static = np.array(list_vars_forcing_static)

                    # organize diagnostic vars (target only)
                    if self.flag_diagnostic:
                        if key == "WRF_target":
                            list_vars_diagnostic = []
                            for var_name in self.varname_diagnostic:
                                var_value = value[var_name].values
                                list_vars_diagnostic.append(var_value)
                            numpy_vars_diagnostic = np.array(list_vars_diagnostic)

                # ================================================================= #
                # boundary domain ds to numpy conversion
                # ================================================================= #
                elif key == "boundary_input":
                    list_vars_upper_air_outside = []
                    for var_name in self.varname_upper_air_outside:
                        var_value = value[var_name].values
                        list_vars_upper_air_outside.append(var_value)

                    # [num_vars, hist_len, num_levels, lat, lon]
                    numpy_vars_upper_air_outside = np.array(list_vars_upper_air_outside)

                    # organize surface vars
                    if self.flag_surface_outside:
                        list_vars_surface_outside = []
                        for var_name in self.varname_surface_outside:
                            var_value = value[var_name].values
                            list_vars_surface_outside.append(var_value)

                        # [num_surf_vars, hist_len, lat, lon]
                        numpy_vars_surface_outside = np.array(list_vars_surface_outside)

            ## if numpy
            else:
                var_value = value

            # WRF domain tensor conversion
            if key == "WRF_input" or key == "WRF_target":
                # ---------------------------------------------------------------------- #
                # ToTensor: upper-air varialbes
                ## produces [time, upper_var, level, lat, lon]
                ## np.hstack concatenates the second dim (axis=1)
                x_upper_air = np.hstack(
                    [np.expand_dims(var_upper_air, axis=1) for var_upper_air in numpy_vars_upper_air]
                )
                x_upper_air = torch.as_tensor(x_upper_air)

                # ---------------------------------------------------------------------- #
                # ToTensor: surface variables
                if self.flag_surface:
                    # this line produces [surface_var, time, lat, lon]
                    x_surf = torch.as_tensor(numpy_vars_surface).squeeze()

                    if len(x_surf.shape) == 4:
                        # permute: [surface_var, time, lat, lon] --> [time, surface_var, lat, lon]
                        x_surf = x_surf.permute(1, 0, 2, 3)

                    # separate single variable vs. single history_len
                    elif len(x_surf.shape) == 3:
                        if len(self.varname_surface) > 1:
                            # single time, multi-vars
                            x_surf = x_surf.unsqueeze(0)
                        else:
                            # multi-time, single vars
                            x_surf = x_surf.unsqueeze(1)

                    else:
                        # num_var=1, time=1, only has lat, lon
                        x_surf = x_surf.unsqueeze(0).unsqueeze(0)

                if key == "WRF_input":
                    # ToTensor: forcing and static
                    if self.has_forcing_static:
                        # this line produces [forcing_var, time, lat, lon]
                        x_static = torch.as_tensor(numpy_vars_forcing_static).squeeze()

                        if len(x_static.shape) == 4:
                            # permute: [forcing_var, time, lat, lon] --> [time, forcing_var, lat, lon]
                            x_static = x_static.permute(1, 0, 2, 3)

                        elif len(x_static.shape) == 3:
                            if self.num_forcing_static > 1:
                                # single time, multi-vars
                                x_static = x_static.unsqueeze(0)
                            else:
                                # multi-time, single vars
                                x_static = x_static.unsqueeze(1)
                        else:
                            # num_var=1, time=1, only has lat, lon
                            x_static = x_static.unsqueeze(0).unsqueeze(0)
                            # x_static = x_static.unsqueeze(1)

                        return_dict["x_forcing_static"] = x_static.type(self.output_dtype)

                    if self.flag_surface:
                        return_dict["x_surf"] = x_surf.type(self.output_dtype)

                    return_dict["x"] = x_upper_air.type(self.output_dtype)

                elif key == "WRF_target":
                    # ---------------------------------------------------------------------- #
                    # ToTensor: diagnostic
                    if self.flag_diagnostic:
                        # this line produces [forcing_var, time, lat, lon]
                        y_diag = torch.as_tensor(numpy_vars_diagnostic).squeeze()

                        if len(y_diag.shape) == 4:
                            # permute: [diag_var, time, lat, lon] --> [time, diag_var, lat, lon]
                            y_diag = y_diag.permute(1, 0, 2, 3)

                        # =============================================== #
                        # separate single variable vs. single history_len
                        elif len(y_diag.shape) == 3:
                            if len(self.varname_diagnostic) > 1:
                                # single time, multi-vars
                                y_diag = y_diag.unsqueeze(0)
                            else:
                                # multi-time, single vars
                                y_diag = y_diag.unsqueeze(1)
                        # =============================================== #

                        else:
                            # num_var=1, time=1, only has lat, lon
                            y_diag = y_diag.unsqueeze(0).unsqueeze(0)

                        return_dict["y_diag"] = y_diag.type(self.output_dtype)

                    if self.flag_surface:
                        return_dict["y_surf"] = x_surf.type(self.output_dtype)

                    return_dict["y"] = x_upper_air.type(self.output_dtype)

            # ================================================================= #
            # boundary domain tensor conversion
            # ================================================================= #
            elif key == "boundary_input":
                # upper air boundary inputs
                x_upper_air_outside = np.hstack(
                    [
                        np.expand_dims(var_upper_air_outside, axis=1)
                        for var_upper_air_outside in numpy_vars_upper_air_outside
                    ]
                )

                x_upper_air_outside = torch.as_tensor(x_upper_air_outside)
                return_dict["x_boundary"] = x_upper_air_outside.type(self.output_dtype)

                # surface boundary inputs
                if self.flag_surface_outside:
                    # this line produces [surface_var, time, lat, lon]
                    x_surf_outside = torch.as_tensor(numpy_vars_surface_outside).squeeze()

                    if len(x_surf_outside.shape) == 4:
                        # permute: [surface_var, time, lat, lon] --> [time, surface_var, lat, lon]
                        x_surf_outside = x_surf_outside.permute(1, 0, 2, 3)

                    # separate single variable vs. single history_len
                    elif len(x_surf_outside.shape) == 3:
                        if len(self.varname_surface_outside) > 1:
                            # single time, multi-vars
                            x_surf_outside = x_surf_outside.unsqueeze(0)
                        else:
                            # multi-time, single vars
                            x_surf_outside = x_surf_outside.unsqueeze(1)
                    else:
                        # num_var=1, time=1, only has lat, lon
                        x_surf_outside = x_surf_outside.unsqueeze(0).unsqueeze(0)

                    return_dict["x_surf_boundary"] = x_surf_outside.type(self.output_dtype)

            elif key == "time_encode":
                return_dict["x_time_encode"] = torch.as_tensor(value).type(self.output_dtype)

        return return_dict
