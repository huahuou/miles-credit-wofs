"""
Test script for the planned H-operator (reflectivity) constraint.

This script:
- Loads a WoFS DA increment sample via WoFSDAIncrementDataset
- Reconstructs physical hydrometeor fields at t0 and t1 from normalized z-scores
  using the closed-form log-zscore inverse transform
- Computes reflectivity dBZ at t0 and t1 with NSSL 2-moment formulas (diag_nssl_refl)
- Forms innovation in either physical dBZ or dataset-normalized space
- Compares to the dataset-provided REFL_10CM innovation used as model input

Usage (example on Ursa):
    srun -A gpu-ai4wp -p u1-compute --mem=128g -N 1 -t 1:20:00 --pty bash -il
    source $MODULESHOME/init/bash
    module load rdhpcs-conda
    module load cuda/12.8
    conda activate credit-wofs
    python python_scripts/test_refl_operator_constraint.py \
      --config config/ursa_wofs_credit_wrf_da_increment.yml \
      --index 0 --compare-space normalized --species rain,snow,graupel,hail
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from glob import glob
from typing import Dict, List, Tuple

import numpy as np

# Ensure project is importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from credit.datasets.wrf_wofs_da_increment import WoFSDAIncrementDataset

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import NSSL reflectivity component functions from the reference implementation
from python_scripts.diag_nssl_refl import (
    ze_rain,
    ze_snow,
    ze_graupel,
    ze_hail,
    ze_ice,
)


def _load_conf(path: str) -> dict:
    try:
        import yaml
    except Exception as e:
        raise SystemExit("PyYAML is required to load the config: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_params_from_conf(conf: dict, split: str = "train") -> Tuple[dict, dict]:
    data = conf["data"]
    years_range = data["train_years"] if split == "train" else data.get("valid_years", data["train_years"]) 
    date_range = data.get("train_date_range") if split == "train" else data.get("valid_date_range")

    pattern = os.path.expandvars(data["save_loc"])
    files = sorted(glob(pattern))

    if date_range is not None and len(date_range) == 2:
        start, end = date_range
        sel = []
        for f in files:
            base = os.path.basename(f)
            # expect wofs_YYYYMMDD_HHMM_memXX.zarr(.zip)
            import re
            m = re.search(r"wofs_(\d{8})_\d{4}_mem\d+\.zarr(?:\.zip)?$", base)
            if m:
                d = m.group(1)
                if start <= d <= end:
                    sel.append(f)
        files = sel

    if len(files) == 0:
        raise SystemExit(f"No case files matched: {pattern}")

    pin = {
        "varname_upper_air": data["variables"],
        "varname_context_upper_air": data.get("context_upper_air_variables", []),
        "varname_dyn_forcing": data.get("dynamic_forcing_variables", []),
        "filenames": files,
        "filename_dyn_forcing": files if data.get("save_loc_dynamic_forcing") else None,
        "history_len": data.get("history_len", 1),
        "forecast_len": data.get("forecast_len", 0),
    }
    pout = {
        "varname_upper_air": data["observation_variables"],
    }
    return pin, pout


def _inverse_concentration_log(
    ds: WoFSDAIncrementDataset, z: np.ndarray, var: str
) -> np.ndarray:
    """Inverse from normalized z to physical using the dataset's log-zscore params.

    Shapes: z: (L,H,W) → returns physical (L,H,W).
    """
    mean = np.asarray(ds._mean_values[var], dtype=np.float64)  # (L,)
    std = np.asarray(ds._std_values[var], dtype=np.float64)   # (L,)
    # broadcast to (L,H,W)
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    log_x = z.astype(np.float64) * std + mean
    params = ds._log_transform_params[var]
    return ds._inverse_log_numpy(log_x, params).astype(np.float32)


def _inverse_standard(
    ds: WoFSDAIncrementDataset, z: np.ndarray, var: str
) -> np.ndarray:
    mean = np.asarray(ds._mean_values[var], dtype=np.float64)
    std = np.asarray(ds._std_values[var], dtype=np.float64)
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    x = z.astype(np.float64) * std + mean
    return x.astype(np.float32)


def _approx_density_from_geopot_T(geopot: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Approximate dry-air density from geopotential and temperature.

    geopot: (L,H,W) GEOPOT [m^2 s^-2], so z ≈ geopot/g0.
    T:      (L,H,W) temperature [K].
    Returns ρ ≈ p / (R_d T) with standard-atmosphere pressure profile.
    """
    g0 = 9.80665
    R_d = 287.05
    z = geopot / g0  # meters
    # Barometric formula with linear lapse to ~11 km
    p0 = 101325.0
    T0 = 288.15
    L = 0.0065
    expo = 9.80665 / (R_d * L)
    # Avoid negative inside when z is very high by clipping
    base = np.clip(1.0 - L * np.maximum(z, 0.0) / T0, 1e-6, None)
    p = p0 * np.power(base, expo)
    rho = p / (R_d * np.maximum(T, 180.0))
    return rho.astype(np.float32)


def _compute_dbz(
    qr: np.ndarray, nr: np.ndarray,
    qs: np.ndarray, ns: np.ndarray,
    qg: np.ndarray, ng: np.ndarray,
    qh: np.ndarray, nh: np.ndarray,
    qi: np.ndarray | None,
    ni: np.ndarray | None,
    rho: np.ndarray,
    T: np.ndarray,
    include_species: List[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    Z_parts: Dict[str, np.ndarray] = {}
    LHW = qr.shape
    zeros = np.zeros(LHW, dtype=np.float64)

    if "rain" in include_species:
        Z_parts["rain"] = ze_rain(qr, nr, rho)
    else:
        Z_parts["rain"] = zeros

    if "snow" in include_species:
        Z_parts["snow"] = ze_snow(qs, ns, rho, T, qr=qr)
    else:
        Z_parts["snow"] = zeros

    if "graupel" in include_species:
        Z_parts["graupel"] = ze_graupel(qg, ng, rho, volg=None)
    else:
        Z_parts["graupel"] = zeros

    if "hail" in include_species:
        Z_parts["hail"] = ze_hail(qh, nh, rho, volh=None)
    else:
        Z_parts["hail"] = zeros

    if qi is not None and ni is not None and "ice" in include_species:
        Z_parts["ice"] = ze_ice(qi, ni, rho)
    else:
        Z_parts["ice"] = zeros

    Z_total = sum(Z_parts.values())
    dbz = np.zeros_like(Z_total, dtype=np.float64)
    pos = Z_total > 0.0
    dbz[pos] = 10.0 * np.log10(Z_total[pos])
    return dbz.astype(np.float32), Z_parts


def main() -> None:
    p = argparse.ArgumentParser(description="Test H-operator reflectivity constraint vs dataset REFL_10CM innovation")
    p.add_argument("--config", required=True, help="Model YAML config (e.g., config/ursa_wofs_credit_wrf_da_increment.yml)")
    p.add_argument("--index", type=int, default=0, help="Sample index to test (default: 0)")
    p.add_argument("--compare-space", choices=["dbz", "normalized"], default="dbz",
                   help="Compare innovation in physical dBZ or dataset-normalized REFL space")
    p.add_argument("--species", type=str, default="rain,snow,graupel,hail",
                   help="Comma-separated subset of species to include: rain,snow,graupel,hail,ice")
    p.add_argument("--outdir", type=str, default="plots/refl_operator",
                   help="Directory to save generated figures")
    p.add_argument("--plot-levels", type=str, default="2,6,10,14",
                   help="Comma-separated 1-based levels to plot (e.g., '2,6,10,14')")
    args = p.parse_args()

    conf = _load_conf(args.config)
    pin, pout = _build_params_from_conf(conf, split="train")

    # Instantiate the dataset (uses normalization and log-transform params)
    ds = WoFSDAIncrementDataset(pin, pout, conf=conf, seed=1000)

    # Materialize one sample; __getitem__ also ensures normalization state is loaded
    sample = ds[args.index]

    # Unpack shapes and needed metadata
    vars_prog: List[str] = pin["varname_upper_air"]
    ctx_vars: List[str] = pin["varname_context_upper_air"]
    levels = int(conf["data"].get("prognostic_levels", conf["model"]["param_interior"]["levels"]))

    x = sample["x"].numpy()[0]      # (num_prog, L, H, W) t0 normalized
    y = sample["y"].numpy()[0]      # (num_prog, L, H, W) delta normalized
    innov_reflect = sample["x_boundary"].numpy()[0, 0]  # (L, H, W) normalized REFL innovation

    H, W = x.shape[-2], x.shape[-1]

    # Reconstruct physical hydrometeors at t0 and t1
    z0_by_var: Dict[str, np.ndarray] = {}
    z1_by_var: Dict[str, np.ndarray] = {}
    q0: Dict[str, np.ndarray] = {}
    q1: Dict[str, np.ndarray] = {}

    for i, var in enumerate(vars_prog):
        z0 = x[i]
        dz = y[i]
        z1 = z0 + dz
        z0_by_var[var] = z0
        z1_by_var[var] = z1
        if var.startswith("QN"):  # number concentration — also log-transform per config
            q0[var] = _inverse_concentration_log(ds, z0, var)
            q1[var] = _inverse_concentration_log(ds, z1, var)
        else:  # mass mixing ratios
            q0[var] = _inverse_concentration_log(ds, z0, var)
            q1[var] = _inverse_concentration_log(ds, z1, var)

    # Extract T and GEOPOT from context (flattened var*level followed by dyn-forcing)
    ctx_flat = sample["x_forcing_static"].numpy()[0]  # (ctx_vars*L + num_dyn, H, W)
    num_ctx = len(ctx_vars) * levels
    ctx_3d = ctx_flat[:num_ctx]  # (ctx_vars*L, H, W)

    def _ctx_slice(varname: str) -> np.ndarray:
        if varname not in ctx_vars:
            raise KeyError(f"Context var '{varname}' not available; configured: {ctx_vars}")
        vi = ctx_vars.index(varname)
        start = vi * levels
        end = start + levels
        return ctx_3d[start:end]

    # Inverse standard z-score for potential temperature (theta) and GEOPOT, then convert theta→T using pressure from z
    try:
        theta_z = _ctx_slice("T")  # stored as potential temperature
        theta = _inverse_standard(ds, theta_z, "T").astype(np.float32)
    except Exception:
        theta = np.full_like(x[0], 300.0, dtype=np.float32)

    try:
        geopot_z = _ctx_slice("GEOPOT")
        geopot = _inverse_standard(ds, geopot_z, "GEOPOT").astype(np.float32)
    except Exception:
        geopot = np.zeros_like(x[0], dtype=np.float32)
    # Derive pressure from geopotential height; convert theta→T via Exner, then rho
    g0 = 9.80665
    Rd = 287.05
    Cp = 1004.5
    P0 = 101325.0
    T0_std = 288.15
    L = 0.0065
    z = geopot / g0
    expo = g0 / (Rd * L)
    base = np.clip(1.0 - L * np.maximum(z, 0.0) / T0_std, 1e-6, None)
    p = P0 * np.power(base, expo)
    kappa = Rd / Cp
    T = theta * np.power(p / P0, kappa)
    rho = (p / (Rd * np.maximum(T, 180.0))).astype(np.float64)

    # Prepare species inputs (missing ones default to zeros)
    def _get(var: str) -> np.ndarray:
        return q0.get(var, np.zeros_like(x[0]))

    # Compute dBZ at t0 and t1 (pred==true since using dataset dz)
    dbz_t0, _ = _compute_dbz(
        qr=_get("QRAIN"), nr=_get("QNRAIN"),
        qs=_get("QSNOW"), ns=_get("QNSNOW"),
        qg=_get("QGRAUP"), ng=_get("QNGRAUPEL"),
        qh=_get("QHAIL"), nh=_get("QNHAIL"),
        qi=None, ni=None, rho=rho, T=T,
        include_species=[s.strip() for s in args.species.split(",") if s.strip()],
    )

    def _get1(var: str) -> np.ndarray:
        return q1.get(var, np.zeros_like(x[0]))

    dbz_t1, _ = _compute_dbz(
        qr=_get1("QRAIN"), nr=_get1("QNRAIN"),
        qs=_get1("QSNOW"), ns=_get1("QNSNOW"),
        qg=_get1("QGRAUP"), ng=_get1("QNGRAUPEL"),
        qh=_get1("QHAIL"), nh=_get1("QNHAIL"),
        qi=None, ni=None, rho=rho, T=T,
        include_species=[s.strip() for s in args.species.split(",") if s.strip()],
    )

    if args.compare_space == "dbz":
        innov_pred = dbz_t1 - dbz_t0  # physical dBZ innovation
        # Inverse-normalize dataset innovation to physical dBZ
        mu = np.asarray(ds._mean_values["REFL_10CM"], dtype=np.float64)[:, None, None]
        sd = np.asarray(ds._std_values["REFL_10CM"], dtype=np.float64)[:, None, None]
        innov_obs_dbz = innov_reflect.astype(np.float64) * sd  # (z1 - z0) normalized → multiply by std
        # Note: mean cancels in the difference; no need to add mu.
        innov_obs = innov_obs_dbz.astype(np.float32)
    else:
        # Normalize diagnosed dBZ with the dataset REFL_10CM stats (per level)
        mu = np.asarray(ds._mean_values["REFL_10CM"], dtype=np.float64)[:, None, None]
        sd = np.asarray(ds._std_values["REFL_10CM"], dtype=np.float64)[:, None, None]
        norm_t0 = (dbz_t0.astype(np.float64) - mu) / sd
        norm_t1 = (dbz_t1.astype(np.float64) - mu) / sd
        innov_pred = (norm_t1 - norm_t0).astype(np.float32)
        innov_obs = innov_reflect.astype(np.float32)

    # Mask tiny magnitudes to avoid clear-air noise
    mag = np.maximum(np.abs(innov_pred), np.abs(innov_obs))
    mask = mag > 0.05 if args.compare_space == "normalized" else mag > 0.25

    def _stats(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> Tuple[float, float, float]:
        if not np.any(m):
            return float("nan"), float("nan"), float("nan")
        aa = a[m].astype(np.float64)
        bb = b[m].astype(np.float64)
        rmse = float(np.sqrt(np.mean((aa - bb) ** 2)))
        mae = float(np.mean(np.abs(aa - bb)))
        corr = float(np.corrcoef(aa.ravel(), bb.ravel())[0, 1]) if aa.size > 10 else float("nan")
        return rmse, mae, corr

    rmse, mae, corr = _stats(innov_pred, innov_obs, mask)

    space = "normalized" if args.compare_space == "normalized" else "dBZ"
    print("=== H-operator vs dataset REFL_10CM innovation ===")
    print(f"Compare space: {space}")
    print(f"Species: {args.species}")
    print(f"Sample index: {args.index}, shape: L={levels}, H={H}, W={W}")
    print(f"RMSE: {rmse:.4f} {space},  MAE: {mae:.4f} {space},  Corr: {corr:.6f}")

    # Per-level summary (optional)
    for k in range(levels):
        rmse_k, mae_k, corr_k = _stats(innov_pred[k], innov_obs[k], mask[k])
        print(f"Level {k+1:02d}: RMSE={rmse_k:.4f}, MAE={mae_k:.4f}, Corr={corr_k:.6f}")

    # -----------------------------
    # Visualization and figure dump
    # -----------------------------
    os.makedirs(args.outdir, exist_ok=True)
    # Parse plot levels (1-based from CLI)
    try:
        plot_levels = [int(x) for x in args.plot_levels.split(",") if x.strip()]
    except Exception:
        plot_levels = [2, 6, 10, 14]
    # Clamp to valid 1..levels
    plot_levels = [i for i in plot_levels if 1 <= i <= levels]
    # Create per-level panels: dBZ_t0, dBZ_t1, ΔdBZ_pred, and optionally ΔdBZ_obs
    vmax = 60.0
    vmin = 0.0
    for lev in plot_levels:
        k = lev - 1
        fig, axs = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
        im0 = axs[0].imshow(dbz_t0[k], origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
        axs[0].set_title(f"dBZ t0 (L{lev})")
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        im1 = axs[1].imshow(dbz_t1[k], origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
        axs[1].set_title(f"dBZ t1 (L{lev})")
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        im2 = axs[2].imshow(innov_pred[k], origin="lower", cmap="RdBu_r", vmin=-20, vmax=20)
        axs[2].set_title(f"ΔdBZ pred (L{lev})")
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        im3 = axs[3].imshow(innov_obs[k], origin="lower", cmap="RdBu_r", vmin=-20, vmax=20)
        axs[3].set_title(f"ΔdBZ obs (L{lev})")
        plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

        for a in axs:
            a.set_xticks([])
            a.set_yticks([])

        out_path = os.path.join(args.outdir, f"refl_panels_L{lev:02d}.png")
        fig.suptitle(f"REFL panels — idx={args.index}, levels={levels}, species={args.species}")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    # Save a composite max reflectivity at t1_pred for a quick sanity check
    comp_t1 = np.nanmax(dbz_t1, axis=0)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    imc = ax2.imshow(comp_t1, origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
    ax2.set_title("Composite dBZ t1 (max over levels)")
    ax2.set_xticks([]); ax2.set_yticks([])
    plt.colorbar(imc, ax=ax2, fraction=0.046, pad=0.04)
    fig2.savefig(os.path.join(args.outdir, "refl_composite_t1.png"), dpi=120)
    plt.close(fig2)


if __name__ == "__main__":
    main()
