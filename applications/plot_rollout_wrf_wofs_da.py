from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


PANEL_CMAP = "RdBu_r"
ERROR_CMAP = "coolwarm"


def _require_var(ds: xr.Dataset, name: str) -> xr.DataArray:
    if name not in ds:
        available = ", ".join(sorted(ds.data_vars))
        raise KeyError(f"Variable '{name}' not found in dataset. Available vars: {available}")
    return ds[name]


def _make_panel_data(
    ds: xr.Dataset,
    var: str,
    time_index: int,
    level_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_delta = _require_var(ds, f"pred_delta_{var}").isel(time=time_index, level=level_index).values
    true_delta = _require_var(ds, f"true_delta_{var}").isel(time=time_index, level=level_index).values
    pred_state = _require_var(ds, f"pred_state_{var}").isel(time=time_index, level=level_index).values
    true_state = _require_var(ds, f"true_state_{var}").isel(time=time_index, level=level_index).values

    prev_state = true_state - true_delta
    pred_next_from_prev = prev_state + pred_delta
    true_next_from_prev = prev_state + true_delta

    state_error = pred_state - true_state
    return pred_delta, true_delta, state_error, prev_state, pred_next_from_prev, true_next_from_prev


def _sym_limits(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    max_abs = float(np.nanmax(np.abs(np.concatenate([a.ravel(), b.ravel()]))))
    if not np.isfinite(max_abs) or max_abs == 0.0:
        max_abs = 1e-8
    return -max_abs, max_abs


def _plot_single(
    ds: xr.Dataset,
    var: str,
    time_index: int,
    level_index: int,
    output_path: Path,
    dpi: int = 140,
) -> None:
    pred_delta, true_delta, state_error, prev_state, pred_next_from_prev, true_next_from_prev = _make_panel_data(
        ds, var, time_index, level_index
    )

    vmin_pred_delta, vmax_pred_delta = _sym_limits(pred_delta, pred_delta)
    vmin_true_delta, vmax_true_delta = _sym_limits(true_delta, true_delta)
    vmin_err, vmax_err = _sym_limits(state_error, state_error)
    vmin_state, vmax_state = _sym_limits(np.concatenate([prev_state, pred_next_from_prev]), true_next_from_prev)

    if "time" in ds.coords:
        time_label = str(ds["time"].values[time_index])
    else:
        time_label = str(time_index)

    if "level" in ds.coords:
        level_label = str(ds["level"].values[level_index])
    else:
        level_label = str(level_index)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    im0 = axes[0, 0].imshow(pred_delta, origin="lower", cmap=PANEL_CMAP, vmin=vmin_pred_delta, vmax=vmax_pred_delta)
    axes[0, 0].set_title(f"pred_delta_{var} (own scale)")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(true_delta, origin="lower", cmap=PANEL_CMAP, vmin=vmin_true_delta, vmax=vmax_true_delta)
    axes[0, 1].set_title(f"true_delta_{var} (own scale)")
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    im2 = axes[0, 2].imshow(state_error, origin="lower", cmap=ERROR_CMAP, vmin=vmin_err, vmax=vmax_err)
    axes[0, 2].set_title(f"pred_state_{var} - true_state_{var}")
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.8)

    im3 = axes[1, 0].imshow(prev_state, origin="lower", cmap=PANEL_CMAP, vmin=vmin_state, vmax=vmax_state)
    axes[1, 0].set_title(f"prev_state_{var} (derived)")
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)

    im4 = axes[1, 1].imshow(pred_next_from_prev, origin="lower", cmap=PANEL_CMAP, vmin=vmin_state, vmax=vmax_state)
    axes[1, 1].set_title(f"prev + pred_delta_{var}")
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)

    im5 = axes[1, 2].imshow(true_next_from_prev, origin="lower", cmap=PANEL_CMAP, vmin=vmin_state, vmax=vmax_state)
    axes[1, 2].set_title(f"prev + true_delta_{var} (= true_state)")
    plt.colorbar(im5, ax=axes[1, 2], shrink=0.8)

    for ax in axes.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(f"DA Rollout | var={var} | time={time_label} | level={level_label}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _infer_vars(ds: xr.Dataset) -> list[str]:
    vars_found = []
    for name in ds.data_vars:
        if name.startswith("pred_delta_"):
            vars_found.append(name.replace("pred_delta_", ""))
    return sorted(vars_found)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DA rollout NetCDF panels for deltas and corrected-state error")
    parser.add_argument("netcdf", help="Path to rollout_da_*.nc")
    parser.add_argument("--var", default=None, help="Variable suffix to plot, e.g. qrain or qnrain")
    parser.add_argument("--time-index", type=int, default=0, help="Time index for single plot mode")
    parser.add_argument("--level-index", type=int, default=0, help="Level index for single plot mode")
    parser.add_argument("--out-dir", default=None, help="Directory to save figures")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Batch mode: plot all times and levels for selected variable(s)",
    )
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    args = parser.parse_args()

    nc_path = Path(args.netcdf)
    ds = xr.open_dataset(nc_path)

    available_vars = _infer_vars(ds)
    if not available_vars:
        raise ValueError("No pred_delta_* variables found in this NetCDF.")

    if args.var is not None:
        selected_vars = [args.var]
    else:
        selected_vars = available_vars

    out_dir = Path(args.out_dir) if args.out_dir else nc_path.with_suffix("")

    n_time = int(ds.sizes.get("time", 1))
    n_level = int(ds.sizes.get("level", 1))

    if args.all:
        for var in selected_vars:
            _require_var(ds, f"pred_delta_{var}")
            _require_var(ds, f"true_delta_{var}")
            _require_var(ds, f"pred_state_{var}")
            _require_var(ds, f"true_state_{var}")
            for t_idx in range(n_time):
                for z_idx in range(n_level):
                    output_file = out_dir / f"{var}_t{t_idx:03d}_z{z_idx:03d}.png"
                    _plot_single(ds, var, t_idx, z_idx, output_file, dpi=args.dpi)
        print(f"Saved batch plots to: {out_dir}")
    else:
        var = selected_vars[0]
        _require_var(ds, f"pred_delta_{var}")
        _require_var(ds, f"true_delta_{var}")
        _require_var(ds, f"pred_state_{var}")
        _require_var(ds, f"true_state_{var}")

        if not (0 <= args.time_index < n_time):
            raise IndexError(f"--time-index must be in [0, {n_time - 1}]")
        if not (0 <= args.level_index < n_level):
            raise IndexError(f"--level-index must be in [0, {n_level - 1}]")

        output_file = out_dir / f"{var}_t{args.time_index:03d}_z{args.level_index:03d}.png"
        _plot_single(ds, var, args.time_index, args.level_index, output_file, dpi=args.dpi)
        print(f"Saved plot: {output_file}")


if __name__ == "__main__":
    main()
