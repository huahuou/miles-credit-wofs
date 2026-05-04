"""
Physical-space concentration increment loss for WoFS DA increment training.

The model predicts increments in normalized concentration space.  This loss
converts both predicted and target increments back to physical increments using
the batch t0 state:

    dx_pred = inverse(z0 + dz_pred) - inverse(z0)
    dx_true = inverse(z0 + dz_true) - inverse(z0)

It is intended as an auxiliary loss beside normalized-space MSE, not as a full
replacement for the base regression objective.
"""

from typing import Dict, List, Optional

import torch
import xarray as xr

from credit.transforms.concentration import (
    CONCENTRATION_VARS,
    concentration_transform_overrides_stats,
    inverse_concentration_transform_torch,
    load_concentration_transform_json,
)


class PhysicalConcentrationIncrementLoss(torch.nn.Module):
    """Auxiliary physical-increment loss for concentration variables."""

    def __init__(self, conf: dict, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.conf = conf
        self.device = device or torch.device("cpu")

        loss_conf = conf.get("loss", {}).get("physical_concentration_loss", {})
        self.variables: List[str] = list(loss_conf.get("variables", conf.get("data", {}).get("variables", [])))
        self.levels = int(conf.get("model", {}).get("param_interior", {}).get("levels", conf.get("model", {}).get("levels", 1)))
        self.prog_vars: List[str] = list(conf.get("data", {}).get("variables", []))

        self.huber_weight = float(loss_conf.get("huber_weight", 1.0))
        self.log_ratio_weight_mass = float(loss_conf.get("log_ratio_weight_mass", 0.2))
        self.log_ratio_weight_qn = float(loss_conf.get("log_ratio_weight_qn", 1.0))
        self.ratio_threshold = float(loss_conf.get("ratio_threshold", 2.0))
        self.ratio_penalty_weight = float(loss_conf.get("ratio_penalty_weight", 2.0))
        self.ratio_power = float(loss_conf.get("ratio_power", 2.0))
        self.max_log_ratio = float(loss_conf.get("max_log_ratio", 3.0))
        self.max_ratio_weight = float(loss_conf.get("max_ratio_weight", 25.0))
        self.max_scaled_error = float(loss_conf.get("max_scaled_error", 50.0))
        self.positive_mass_underprediction_weight = float(
            loss_conf.get("positive_mass_underprediction_weight", 1.0)
        )
        self.huber_beta = float(loss_conf.get("huber_beta", 1.0))
        self.mask_min = float(loss_conf.get("mask_min", 1.0e-12))
        self.increment_min = float(loss_conf.get("increment_min", 1.0e-12))
        self.eps_mass = float(loss_conf.get("eps_mass", 1.0e-12))
        self.eps_qn = float(loss_conf.get("eps_qn", 1.0e-6))

        self.threshold_by_var = {
            str(k): float(v)
            for k, v in (loss_conf.get("threshold_by_var", {}) or {}).items()
        }
        self.increment_threshold_by_var = {
            str(k): float(v)
            for k, v in (loss_conf.get("increment_threshold_by_var", {}) or {}).items()
        }
        self.eps_by_var = {
            str(k): float(v)
            for k, v in (loss_conf.get("eps_by_var", {}) or {}).items()
        }
        self.scale_by_var = {
            str(k): float(v)
            for k, v in (loss_conf.get("scale_by_var", {}) or {}).items()
        }

        data_conf = conf.get("data", {})
        self._mean: Dict[str, torch.Tensor] = {}
        self._std: Dict[str, torch.Tensor] = {}
        mean_path = data_conf.get("mean_path")
        std_path = data_conf.get("std_path")
        if mean_path and std_path:
            with xr.open_dataset(mean_path) as ds_m, xr.open_dataset(std_path) as ds_s:
                for var in self.prog_vars:
                    if var in ds_m and var in ds_s:
                        self._mean[var] = torch.from_numpy(ds_m[var].values).float()
                        self._std[var] = torch.from_numpy(ds_s[var].values).float()

        log_json = data_conf.get("log_transform_params_json")
        self._concentration_transform_specs: Dict[str, dict] = {}
        if log_json:
            _, self._concentration_transform_specs = load_concentration_transform_json(
                log_json,
                variables=CONCENTRATION_VARS,
            )

        self._batch_ctx = None

    def set_batch_context(self, *, z0_norm: torch.Tensor) -> None:
        """Stash normalized t0 prognostic state with shape ``(B, V, L, H, W)``."""
        self._batch_ctx = {"z0": z0_norm}

    def _stat_view(self, stat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        stat = stat.to(z.device, dtype=z.dtype)
        if stat.ndim == 1:
            stat = stat[:, None, None]
        return stat

    def _inverse_var(self, z: torch.Tensor, var: str) -> torch.Tensor:
        spec = self._concentration_transform_specs.get(var)
        if spec is None:
            mu = self._stat_view(self._mean[var], z)
            sd = self._stat_view(self._std[var], z)
            return z * sd + mu

        level_axis = 1 if z.ndim >= 4 else 0
        if concentration_transform_overrides_stats(spec):
            log_mean = torch.as_tensor(spec["log_mean"], dtype=z.dtype, device=z.device)
            log_std = torch.as_tensor(spec["log_std"], dtype=z.dtype, device=z.device)
            latent = z * log_std + log_mean
        else:
            mu = self._stat_view(self._mean[var], z)
            sd = self._stat_view(self._std[var], z)
            latent = z * sd + mu
        return inverse_concentration_transform_torch(latent, spec, level_axis=level_axis)

    def _reshape_increment(self, tensor: torch.Tensor, z0: torch.Tensor) -> Optional[torch.Tensor]:
        B, V, L, H, W = z0.shape
        if tensor.dim() == 5:
            if tensor.shape[2] == 1:
                tensor = tensor[:, :, 0]
            elif tensor.shape[1] == V and tensor.shape[2] == L:
                return tensor
        if tensor.dim() != 4:
            return None

        C = tensor.shape[1]
        needed = V * L
        if C < needed:
            return None
        return tensor[:, :needed, ...].contiguous().view(B, V, L, H, W)

    @staticmethod
    def _smooth_l1(x: torch.Tensor, beta: float) -> torch.Tensor:
        beta = max(float(beta), 1.0e-12)
        abs_x = x.abs()
        return torch.where(abs_x < beta, 0.5 * abs_x.pow(2) / beta, abs_x - 0.5 * beta)

    def forward(self, pred_incr_norm: torch.Tensor, target_incr_norm: torch.Tensor) -> torch.Tensor:
        if self._batch_ctx is None:
            return torch.tensor(0.0, dtype=torch.float32, device=pred_incr_norm.device)

        # The physical inverse transform is numerically sensitive and its
        # zero-inflated branch mixes masked assignment with inverse-CDF math.
        # Keep this auxiliary loss in fp32 even when the trainer uses AMP.
        z0 = self._batch_ctx["z0"].to(pred_incr_norm.device, dtype=torch.float32)
        pred = self._reshape_increment(pred_incr_norm.float(), z0)
        target = self._reshape_increment(target_incr_norm.float(), z0)
        if pred is None or target is None:
            return torch.tensor(0.0, dtype=torch.float32, device=pred_incr_norm.device)

        total = pred.new_zeros(())
        n_terms = 0

        for var in self.variables:
            if var not in self.prog_vars or var not in CONCENTRATION_VARS:
                continue
            vi = self.prog_vars.index(var)
            if var not in self._concentration_transform_specs and var not in self._mean:
                continue

            z0_v = z0[:, vi]
            x0 = self._inverse_var(z0_v, var)
            x_pred = self._inverse_var(z0_v + pred[:, vi], var)
            x_true = self._inverse_var(z0_v + target[:, vi], var)

            dx_pred = x_pred - x0
            dx_true = x_true - x0

            eps = self.eps_by_var.get(var, self.eps_qn if var.startswith("QN") else self.eps_mass)
            hydro_thr = self.threshold_by_var.get(var, self.mask_min)
            incr_thr = self.increment_threshold_by_var.get(var, self.increment_min)
            mask = (
                (x0.abs() > hydro_thr)
                | (x_true.abs() > hydro_thr)
                | (dx_true.abs() > incr_thr)
            ).to(dx_pred.dtype)
            denom = mask.sum().clamp_min(1.0)

            var_loss = pred.new_zeros(())

            if self.huber_weight > 0.0:
                scale = max(self.scale_by_var.get(var, incr_thr), eps)
                scaled_error = torch.clamp(
                    (dx_pred - dx_true) / scale,
                    min=-self.max_scaled_error,
                    max=self.max_scaled_error,
                )
                huber = self._smooth_l1(scaled_error, self.huber_beta)
                var_loss = var_loss + self.huber_weight * (huber * mask).sum() / denom

            ratio_weight = self.log_ratio_weight_qn if var.startswith("QN") else self.log_ratio_weight_mass
            if ratio_weight > 0.0:
                ratio_mask = mask * (dx_true.abs() > incr_thr).to(dx_pred.dtype)
                ratio_denom = ratio_mask.sum().clamp_min(1.0)
                log_ratio = torch.log((dx_pred.abs() + eps) / (dx_true.abs() + eps))
                log_ratio = torch.clamp(log_ratio, min=-self.max_log_ratio, max=self.max_log_ratio)
                ratio_mag = torch.exp(log_ratio.abs())
                excess = torch.relu(ratio_mag - self.ratio_threshold)
                heavy_weight = 1.0 + self.ratio_penalty_weight * torch.pow(excess, self.ratio_power)
                heavy_weight = torch.clamp(heavy_weight, max=self.max_ratio_weight)
                ratio_loss = log_ratio.pow(2) * heavy_weight
                var_loss = var_loss + ratio_weight * (ratio_loss * ratio_mask).sum() / ratio_denom

            if self.positive_mass_underprediction_weight > 0.0 and not var.startswith("QN"):
                pos_mask = mask * (dx_true > incr_thr).to(dx_pred.dtype)
                pos_denom = pos_mask.sum().clamp_min(1.0)
                scale = max(self.scale_by_var.get(var, incr_thr), eps)
                under = torch.clamp(
                    torch.relu(dx_true - dx_pred) / scale,
                    max=self.max_scaled_error,
                )
                var_loss = var_loss + (
                    self.positive_mass_underprediction_weight
                    * (under.pow(2) * pos_mask).sum()
                    / pos_denom
                )

            total = total + var_loss
            n_terms += 1

        if n_terms == 0:
            return pred.new_zeros(())
        return total / float(n_terms)
