import json
from typing import Dict, List, Optional

import torch
import xarray as xr

from credit.physics.refl_operator import ze_rain, ze_snow, ze_graupel, ze_hail, ze_ice, combine_to_dbz
from credit.transforms.concentration import (
    CONCENTRATION_VARS,
    concentration_transform_overrides_stats,
    inverse_concentration_transform_torch,
    load_concentration_transform_json,
)


class ReflOperatorConstraintLoss(torch.nn.Module):
    """
    H-operator reflectivity constraint.

    - Inverts normalized z-scores (log-zscore for concentrations) to physical (q, N)
    - Applies NSSL 2-moment reflectivity operator H to get dBZ at t0 and t1
    - Compares innovation (or absolute) to dataset reflectivity (dBZ or normalized)
    """

    def __init__(
        self,
        conf: dict,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.conf = conf
        self.device = device or torch.device("cpu")

        loss_conf = conf.get("loss", {}).get("refl_operator_constraint", {})
        self.mode = str(loss_conf.get("mode", "innovation")).lower()
        self.compare_space = str(loss_conf.get("compare_space", "dbz")).lower()
        self.dbz_floor = float(loss_conf.get("dbz_floor", 0.0))
        self.threshold_dbz = float(loss_conf.get("threshold_dbz", 5.0))
        self.threshold_norm = float(loss_conf.get("threshold_norm", 0.05))
        inc_list = loss_conf.get("include_species", ["rain", "snow", "graupel", "hail"]) or []
        self.include_species = [str(s).lower().strip() for s in inc_list]

        data_conf = conf.get("data", {})
        self.levels = int(conf.get("model", {}).get("param_interior", {}).get("levels", data_conf.get("levels", 1)))
        self.prog_vars: List[str] = list(data_conf.get("variables", []))
        self.ctx_vars: List[str] = list(data_conf.get("context_upper_air_variables", []))

        # Load per-level mean/std for every variable that may need inverse normalization.
        self._mean = {}
        self._std = {}
        mean_path = data_conf.get("mean_path")
        std_path = data_conf.get("std_path")
        if mean_path and std_path:
            with xr.open_dataset(mean_path) as ds_m, xr.open_dataset(std_path) as ds_s:
                required_vars = set(self.ctx_vars + self.prog_vars + ["REFL_10CM"])
                for v in required_vars:
                    if v in ds_m and v in ds_s:
                        self._mean[v] = torch.from_numpy(ds_m[v].values).float()
                        self._std[v] = torch.from_numpy(ds_s[v].values).float()

        # Load generalized concentration transform params.
        log_json = data_conf.get("log_transform_params_json")
        self._concentration_transform_specs: Dict[str, dict] = {}
        self._log_params: Dict[str, dict] = {}
        if log_json:
            _, self._concentration_transform_specs = load_concentration_transform_json(
                log_json,
                variables=CONCENTRATION_VARS,
            )
            self._log_params = {
                var: spec
                for var, spec in self._concentration_transform_specs.items()
                if concentration_transform_overrides_stats(spec)
            }

        self._batch_ctx = None

    # ----------------- Context Handling ----------------- #
    def set_batch_context(self, *, z0_norm: torch.Tensor, forcing_static: Optional[torch.Tensor], boundary_innov_norm: Optional[torch.Tensor]) -> None:
        """
        Stash the current-batch normalized background and boundary innovation.

        Args:
          z0_norm: (B, Vp, L, H, W) normalized background at t0 for prognostic variables
          forcing_static: (B, C, H, W) or (B, 1, C, H, W) flattened context+dyn forcing at t0
          boundary_innov_norm: (B, L, H, W) normalized REFL_10CM innovation (t1 - t0)
        """
        self._batch_ctx = {
            "z0": z0_norm,
            "forcing_static": forcing_static,
            "innov_norm": boundary_innov_norm,
        }

    # ----------------- Helpers ----------------- #
    def _inverse_log(self, z: torch.Tensor, var: str) -> torch.Tensor:
        spec = self._concentration_transform_specs.get(var)
        if spec is None:
            # Fallback: treat as standard z-score
            mu = self._mean[var].to(z.device)
            sd = self._std[var].to(z.device)
            if mu.ndim == 1:
                mu = mu[:, None, None]
            if sd.ndim == 1:
                sd = sd[:, None, None]
            return (z * sd + mu)
        level_axis = 1 if z.ndim >= 4 else 0
        if concentration_transform_overrides_stats(spec):
            log_mean = torch.as_tensor(spec["log_mean"], dtype=z.dtype, device=z.device)
            log_std = torch.as_tensor(spec["log_std"], dtype=z.dtype, device=z.device)
            log_x = z * log_std + log_mean
            return inverse_concentration_transform_torch(log_x, spec, level_axis=level_axis)
        mu = self._mean[var].to(z.device)
        sd = self._std[var].to(z.device)
        if mu.ndim == 1:
            mu = mu[:, None, None]
        if sd.ndim == 1:
            sd = sd[:, None, None]
        latent = z * sd + mu
        return inverse_concentration_transform_torch(latent, spec, level_axis=level_axis)

    def _inverse_standard(self, z: torch.Tensor, var: str) -> torch.Tensor:
        mu = self._mean[var].to(z.device)
        sd = self._std[var].to(z.device)
        if mu.ndim == 1:
            mu = mu[:, None, None]
        if sd.ndim == 1:
            sd = sd[:, None, None]
        return z * sd + mu

    def _extract_T_geopot(self, forcing_static: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # forcing_static layout: (B, C, H, W) where first len(ctx_vars)*L are flattened context levels
        if forcing_static is None or len(self.ctx_vars) == 0:
            raise RuntimeError("Context forcing_static is required for T/GEOPOT extraction.")
        fs = forcing_static
        if fs.ndim == 5:  # (B, 1, C, H, W)
            fs = fs[:, 0]
        B, C, H, W = fs.shape
        ctxC = len(self.ctx_vars) * self.levels
        ctx_flat = fs[:, :ctxC]  # (B, ctx_vars*L, H, W)
        # reshape to (B, ctx_vars, L, H, W)
        ctx3d = ctx_flat.view(B, len(self.ctx_vars), self.levels, H, W)
        theta, geopot = None, None
        for i, v in enumerate(self.ctx_vars):
            if v == "T":
                theta = ctx3d[:, i]
            elif v == "GEOPOT":
                geopot = ctx3d[:, i]
        # Inverse-normalize context fields
        if theta is None:
            theta = torch.full((B, self.levels, H, W), 300.0, dtype=fs.dtype, device=fs.device)
        else:
            theta = self._inverse_standard(theta, "T")  # 'T' stored as potential temperature (theta)
        if geopot is None:
            geopot = torch.zeros((B, self.levels, H, W), dtype=fs.dtype, device=fs.device)
        else:
            geopot = self._inverse_standard(geopot, "GEOPOT")
        # Convert theta → actual temperature using Exner function with pressure from barometric formula
        g0 = 9.80665
        Rd = 287.05
        Cp = 1004.5
        P0 = 101325.0
        T0 = 288.15
        L = 0.0065
        z = geopot / g0
        expo = g0 / (Rd * L)
        base = torch.clamp(1.0 - L * torch.clamp(z, min=0.0) / T0, min=1e-6)
        p = P0 * torch.pow(base, expo)
        kappa = Rd / Cp
        T = theta * torch.pow(p / P0, kappa)
        return T, geopot

    @staticmethod
    def _approx_rho_from_T_geopot(T: torch.Tensor, geopot: torch.Tensor) -> torch.Tensor:
        # Use same pressure approximation used to derive T from theta
        g0 = 9.80665
        Rd = 287.05
        P0 = 101325.0
        T0 = 288.15
        L = 0.0065
        z = geopot / g0
        expo = g0 / (Rd * L)
        base = torch.clamp(1.0 - L * torch.clamp(z, min=0.0) / T0, min=1e-6)
        p = P0 * torch.pow(base, expo)
        rho = p / (Rd * torch.clamp(T, min=180.0))
        return rho

    def _diagnose_dbz(self, qdict: Dict[str, torch.Tensor], rho: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        parts = {}
        if "rain" in self.include_species:
            parts["rain"] = ze_rain(qdict.get("QRAIN", 0.0), qdict.get("QNRAIN", 0.0), rho)
        if "snow" in self.include_species:
            parts["snow"] = ze_snow(qdict.get("QSNOW", 0.0), qdict.get("QNSNOW", 0.0), rho, T, qr=qdict.get("QRAIN", 0.0))
        if "graupel" in self.include_species:
            parts["graupel"] = ze_graupel(qdict.get("QGRAUP", 0.0), qdict.get("QNGRAUPEL", 0.0), rho, None)
        if "hail" in self.include_species:
            parts["hail"] = ze_hail(qdict.get("QHAIL", 0.0), qdict.get("QNHAIL", 0.0), rho, None)
        # optional ice not used by default
        if "ice" in self.include_species and ("QICE" in qdict and "QNICE" in qdict):
            parts["ice"] = ze_ice(qdict["QICE"], qdict["QNICE"], rho)
        dbz = combine_to_dbz(parts, self.dbz_floor)
        return dbz

    # ----------------- Forward ----------------- #
    def forward(self, pred_incr_norm: torch.Tensor, target_incr_norm: torch.Tensor) -> torch.Tensor:
        if self._batch_ctx is None:
            return torch.tensor(0.0, dtype=pred_incr_norm.dtype, device=pred_incr_norm.device)

        z0 = self._batch_ctx["z0"].to(pred_incr_norm.device, dtype=pred_incr_norm.dtype)  # (B, Vp, L, H, W)
        forcing_static = self._batch_ctx["forcing_static"]
        innov_norm = self._batch_ctx["innov_norm"]
        if forcing_static is not None:
            forcing_static = forcing_static.to(pred_incr_norm.device, dtype=pred_incr_norm.dtype)
        if innov_norm is not None:
            innov_norm = innov_norm.to(pred_incr_norm.device, dtype=pred_incr_norm.dtype)

        # Map variable names to channels in tensors
        var_list = self.prog_vars
        B, Vp, L, H, W = z0.shape

        # Reshape model output increments to (B, Vp, L, H, W)
        pred = pred_incr_norm
        if pred.dim() == 5:
            # (B, C, T, H, W) or (B, Vp, L, H, W) — detect by channel count
            C = pred.shape[1]
            # Squeeze time if present
            if pred.shape[2] == 1:
                pred = pred[:, :, 0]
            # If channels are flattened across levels, slice prognostic and reshape
            if C != Vp:
                needed = Vp * L
                if C < needed:
                    return torch.tensor(0.0, dtype=pred_incr_norm.dtype, device=pred_incr_norm.device)
                pred = pred[:, :needed, ...].contiguous().view(B, Vp, L, H, W)
        elif pred.dim() == 4:
            # (B, C, H, W)
            C = pred.shape[1]
            if C == Vp * L:
                pred = pred[:, : Vp * L, ...].contiguous().view(B, Vp, L, H, W)
            elif C == Vp and pred.shape[2] == L:
                # Rare path: already (B, Vp, L, H, W) folded as 4D — not expected
                pred = pred.view(B, Vp, L, H, W)
            else:
                # Unknown shape — skip constraint safely
                return torch.tensor(0.0, dtype=pred_incr_norm.dtype, device=pred_incr_norm.device)
        else:
            # Unsupported dims — skip safely
            return torch.tensor(0.0, dtype=pred_incr_norm.dtype, device=pred_incr_norm.device)

        z1 = z0 + pred  # predicted post-increment normalized

        # Invert concentrations (log-zscore) per variable
        # Build dicts of q0/q1 tensors per var name (each B,L,H,W)
        q0: Dict[str, torch.Tensor] = {}
        q1: Dict[str, torch.Tensor] = {}
        for vi, var in enumerate(var_list):
            z0_v = z0[:, vi]
            z1_v = z1[:, vi]
            if var in self._concentration_transform_specs:
                q0[var] = self._inverse_log(z0_v, var)
                q1[var] = self._inverse_log(z1_v, var)
            else:
                # Non-concentration vars (shouldn't occur in prog list), but handle gracefully
                q0[var] = self._inverse_standard(z0_v, var)
                q1[var] = self._inverse_standard(z1_v, var)

        # Extract T and GEOPOT → rho approx
        T, geopot = self._extract_T_geopot(forcing_static)
        rho = self._approx_rho_from_T_geopot(T, geopot)

        # Diagnose dbZ at t0 and t1
        def build_qdict(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            out = {}
            for k in ["QRAIN", "QNRAIN", "QSNOW", "QNSNOW", "QGRAUP", "QNGRAUPEL", "QHAIL", "QNHAIL", "QICE", "QNICE"]:
                if k in d:
                    out[k] = d[k]
            return out

        dbz0 = self._diagnose_dbz(build_qdict(q0), rho, T)
        dbz1 = self._diagnose_dbz(build_qdict(q1), rho, T)

        # Innovation or absolute comparison
        if self.mode == "innovation":
            pred_innov_dbz = dbz1 - dbz0  # (B,L,H,W)
            if self.compare_space == "dbz":
                target_innov = pred_innov_dbz.new_zeros(pred_innov_dbz.shape) if innov_norm is None else (
                    # Convert normalized innovation to dBZ: (z1-z0)*std (mean cancels)
                    innov_norm * self._std["REFL_10CM"].to(pred_innov_dbz.device)[:, None, None]
                )
                diff = pred_innov_dbz - target_innov
                mag = torch.maximum(pred_innov_dbz.abs(), target_innov.abs())
                mask = (mag > self.threshold_dbz).to(diff.dtype)
                denom = mask.sum().clamp_min(1.0)
                return (diff.pow(2) * mask).sum() / denom
            else:
                mu = self._mean["REFL_10CM"].to(pred_innov_dbz.device)[:, None, None]
                sd = self._std["REFL_10CM"].to(pred_innov_dbz.device)[:, None, None]
                norm0 = (dbz0 - mu) / sd
                norm1 = (dbz1 - mu) / sd
                pred_innov_norm = norm1 - norm0
                target_innov = pred_innov_norm.new_zeros(pred_innov_norm.shape) if innov_norm is None else innov_norm
                diff = pred_innov_norm - target_innov
                mag = torch.maximum(pred_innov_norm.abs(), target_innov.abs())
                mask = (mag > self.threshold_norm).to(diff.dtype)
                denom = mask.sum().clamp_min(1.0)
                return (diff.pow(2) * mask).sum() / denom
        else:  # absolute
            if self.compare_space == "dbz":
                # Need observed dBZ at t1 and t0 (not available here) → skip absolute mode by default
                return torch.tensor(0.0, dtype=pred_incr_norm.dtype, device=pred_incr_norm.device)
            else:
                # Similarly requires normalized absolute; skip for now
                return torch.tensor(0.0, dtype=pred_incr_norm.dtype, device=pred_incr_norm.device)
