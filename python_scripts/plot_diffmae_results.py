from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
import zarr
import zarr.storage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader

    HAS_CARTOPY = True
except Exception:
    ccrs = None
    cfeature = None
    shpreader = None
    HAS_CARTOPY = False

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning, module="cartopy.io")
        STATE_FEATURE = (
            cfeature.ShapelyFeature(
                shpreader.Reader(
                    shpreader.natural_earth("50m", "cultural", "admin_1_states_provinces_lakes")
                ).geometries(),
                ccrs.PlateCarree(),
                edgecolor="black",
                facecolor="none",
                linewidth=0.45,
            )
            if HAS_CARTOPY
            else None
        )
except Exception:
    STATE_FEATURE = None


# Default inputs. Override with CLI args when running the script.
EXP_PATH = Path(
    "/scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/credit_rollouts/"
    "wofs_diffmae_pretrain_4x4patch_heightmask_v2/repaint_ddim_test2/20210427"
)
FILE_NAME = "wofs_20210416_2000_mem01"
REF_ROOT = Path("/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/cases")
TIME_INDEX = 0
ENSEMBLE_INDEX = 0
LEVEL_INDEX = 0
PRECIP_VARS = [
    "QRAIN",
    "QNRAIN",
    "QHAIL",
    "QNHAIL",
    "QGRAUP",
    "QNGRAUPEL",
    "QSNOW",
    "QNSNOW",
]


def open_wofs_zarr_zip(filepath, max_attempts: int = 3) -> xr.Dataset:
    """Open a WoFS .zarr.zip file produced by the CREDIT pipeline."""
    zip_file = str(filepath).rstrip("/")
    zip_basename = Path(zip_file).stem
    uris = [
        f"zip://{zip_basename}::{zip_file}",
        f"zip://{zip_basename}/::{zip_file}",
    ]

    def _try_open(uri: str) -> xr.Dataset:
        last_exc = None
        for consolidated in (True, False):
            try:
                return xr.open_zarr(
                    uri,
                    consolidated=consolidated,
                    zarr_format=2,
                    decode_coords=False,
                    create_default_indexes=False,
                )
            except Exception as exc:
                last_exc = exc
        raise last_exc

    last_exc = None
    for uri in uris:
        for attempt in range(1, max_attempts + 1):
            try:
                return _try_open(uri)
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts:
                    time.sleep(0.25 * attempt)
    raise last_exc


def _as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D grid after squeezing; got shape {arr.shape}")
    return arr


def get_lat_lon(ds_ref: xr.Dataset) -> tuple[np.ndarray | None, np.ndarray | None]:
    lat_names = ["XLAT", "XLAT_M", "xlat", "latitude", "lat", "LAT"]
    lon_names = ["XLONG", "XLONG_M", "xlong", "longitude", "lon", "LON"]

    lat = next((ds_ref[name] for name in lat_names if name in ds_ref), None)
    lon = next((ds_ref[name] for name in lon_names if name in ds_ref), None)
    if lat is not None and lon is not None:
        return _as_2d(lat.values), _as_2d(lon.values)

    if {"sin_latitude", "cos_latitude", "sin_longitude", "cos_longitude"}.issubset(ds_ref.variables):
        lat = np.rad2deg(np.arctan2(ds_ref["sin_latitude"].values, ds_ref["cos_latitude"].values))
        lon = np.rad2deg(np.arctan2(ds_ref["sin_longitude"].values, ds_ref["cos_longitude"].values))
        return _as_2d(lat), _as_2d(lon)

    return None, None


def _select_grid(
    da: xr.DataArray,
    time_index: int,
    level_index: int | None = None,
    ensemble_index: int | None = None,
) -> np.ndarray:
    indexers = {}
    if "time" in da.dims:
        indexers["time"] = time_index
    if ensemble_index is not None and "ensemble" in da.dims:
        indexers["ensemble"] = ensemble_index
    for level_dim in ("level", "bottom_top", "z"):
        if level_dim in da.dims and level_index is not None:
            indexers[level_dim] = level_index
            break
    return _as_2d(da.isel(**indexers).values)


def _var_is_number_concentration(var_name: str) -> bool:
    return var_name.startswith("QN")


def _positive_boundaries(var_name: str) -> np.ndarray:
    if _var_is_number_concentration(var_name):
        return np.logspace(1, 7, 13)
    return np.logspace(-8, -1, 15)


def _signed_log_boundaries(var_name: str) -> np.ndarray:
    positive = _positive_boundaries(var_name)
    return np.concatenate([-positive[::-1], [0.0], positive])


def _mask_boundaries() -> np.ndarray:
    return np.asarray([-0.5, 0.5, 1.5], dtype=np.float32)


def _norm_boundaries() -> np.ndarray:
    return np.linspace(-4.0, 4.0, 17)


def _trajectory_boundaries(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return _norm_boundaries()
    vmax = max(1.0, float(np.nanpercentile(np.abs(finite), 99.0)))
    return np.linspace(-vmax, vmax, 17)


def _plot_grid(
    ax,
    values: np.ndarray,
    lon: np.ndarray | None,
    lat: np.ndarray | None,
    boundaries: np.ndarray,
    cmap: str | ListedColormap,
    title: str,
    extend: str = "neither",
):
    norm = BoundaryNorm(boundaries, ncolors=plt.get_cmap(cmap).N if isinstance(cmap, str) else cmap.N, clip=False)
    if HAS_CARTOPY and lon is not None and lat is not None:
        mesh = ax.pcolormesh(lon, lat, values, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="auto")
        if STATE_FEATURE is not None:
            ax.add_feature(STATE_FEATURE)
        ax.set_extent([float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))])
    else:
        mesh = ax.imshow(values, cmap=cmap, norm=norm, origin="lower")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    mesh.colorbar_extend = extend
    return mesh


def _add_colorbar(fig, mesh, ax, boundaries: np.ndarray, label: str, ticks: Iterable[float] | None = None):
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02, boundaries=boundaries, ticks=ticks)
    cbar.set_label(label)
    return cbar


def load_mask_dataset(analysis_file: Path, obs_mask_file: Path | None = None) -> xr.Dataset:
    try:
        return xr.open_zarr(analysis_file, group="mask")
    except Exception:
        if obs_mask_file is None or not obs_mask_file.exists():
            raise

    obs_mask = np.load(obs_mask_file)
    patch_masks = obs_mask["patch_mask_grouped"]
    patch_size = int(obs_mask["patch_size"]) if "patch_size" in obs_mask.files else 4
    if "group_channels" in obs_mask.files:
        group_channels = obs_mask["group_channels"]
    else:
        group_channels = np.full((patch_masks.shape[1],), 17, dtype=np.int32)

    if patch_masks.ndim == 5:
        pixel_groups = np.repeat(np.repeat(patch_masks, patch_size, axis=-2), patch_size, axis=-1)
        pixel_channels = []
        for group_index, n_channels in enumerate(group_channels):
            pixel_channels.append(pixel_groups[:, group_index, : int(n_channels)])
        pixel_mask = np.concatenate(pixel_channels, axis=1)
    else:
        pixel_groups = np.repeat(np.repeat(patch_masks, patch_size, axis=-2), patch_size, axis=-1)
        pixel_mask = np.concatenate(
            [np.repeat(pixel_groups[:, i : i + 1], int(n), axis=1) for i, n in enumerate(group_channels)],
            axis=1,
        )

    return xr.Dataset(
        data_vars={
            "pixel_mask_channel": (
                ("time", "channel", "y", "x"),
                pixel_mask[:, :, :300, :300].astype(np.float32),
            )
        },
        coords={
            "time": np.arange(pixel_mask.shape[0], dtype=np.int32),
            "channel": np.arange(pixel_mask.shape[1], dtype=np.int32),
            "y": np.arange(min(300, pixel_mask.shape[-2]), dtype=np.int32),
            "x": np.arange(min(300, pixel_mask.shape[-1]), dtype=np.int32),
        },
    )


def channel_index_for_var_level(ds_phy: xr.Dataset, precip_vars: list[str], var_name: str, level_index: int) -> int:
    channel = 0
    for name in precip_vars:
        if name not in ds_phy:
            continue
        n_levels = int(ds_phy[name].sizes.get("level", ds_phy[name].shape[-3]))
        if name == var_name:
            if level_index >= n_levels:
                raise IndexError(f"level_index={level_index} outside {name} levels={n_levels}")
            return channel + int(level_index)
        channel += n_levels
    raise KeyError(f"{var_name} not found in ds_phy variables")


def get_mask_grid(
    mask_ds: xr.Dataset,
    ds_phy: xr.Dataset,
    precip_vars: list[str],
    var_name: str,
    time_index: int,
    level_index: int,
) -> np.ndarray:
    if "pixel_mask_channel" in mask_ds:
        channel = channel_index_for_var_level(ds_phy, precip_vars, var_name, level_index)
        return _as_2d(mask_ds["pixel_mask_channel"].isel(time=time_index, channel=channel).values)
    if "patch_mask_grouped" in mask_ds:
        group_index = precip_vars.index(var_name)
        mask = mask_ds["patch_mask_grouped"]
        if "level" in mask.dims:
            patch = mask.isel(time=time_index, mask_group=group_index, level=level_index).values
        else:
            patch = mask.isel(time=time_index, mask_group=group_index).values
        return np.repeat(np.repeat(_as_2d(patch), 4, axis=0), 4, axis=1)[:300, :300]
    raise KeyError("Mask dataset must contain pixel_mask_channel or patch_mask_grouped")


def _axes(n_rows: int, n_cols: int, width: float, height_per_row: float, use_map: bool = False):
    projection = ccrs.PlateCarree() if HAS_CARTOPY and use_map else None
    subplot_kw = {"projection": projection} if projection is not None else {}
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width, max(3.0, height_per_row * n_rows)),
        subplot_kw=subplot_kw,
        squeeze=False,
        constrained_layout=True,
    )
    return fig, axes


def plot_physical_comparison(
    ds_phy: xr.Dataset,
    ds_ref: xr.Dataset,
    mask_ds: xr.Dataset,
    precip_vars: list[str],
    time_index: int,
    level_index: int,
    ensemble_index: int,
    lon: np.ndarray | None,
    lat: np.ndarray | None,
    out_path: Path,
) -> None:
    use_map = lon is not None and lat is not None
    fig, axes = _axes(len(precip_vars), 4, width=18.0, height_per_row=3.0, use_map=use_map)
    for row, var_name in enumerate(precip_vars):
        phy = _select_grid(ds_phy[var_name], time_index, level_index, ensemble_index)
        ref = _select_grid(ds_ref[var_name], time_index, level_index, None)
        mask = get_mask_grid(mask_ds, ds_phy, precip_vars, var_name, time_index, level_index)
        mask = mask[: phy.shape[-2], : phy.shape[-1]]
        ref = ref[: phy.shape[-2], : phy.shape[-1]]
        diff = phy - ref

        positive_bounds = _positive_boundaries(var_name)
        signed_bounds = _signed_log_boundaries(var_name)
        clipped_phy = np.clip(phy, positive_bounds[0], positive_bounds[-1])
        clipped_ref = np.clip(ref, positive_bounds[0], positive_bounds[-1])
        clipped_diff = np.clip(diff, signed_bounds[0], signed_bounds[-1])

        mesh0 = _plot_grid(axes[row, 0], clipped_phy, lon, lat, positive_bounds, "viridis", f"{var_name} ds_phy")
        mesh1 = _plot_grid(axes[row, 1], clipped_ref, lon, lat, positive_bounds, "viridis", f"{var_name} ds_ref")
        mesh2 = _plot_grid(axes[row, 2], clipped_diff, lon, lat, signed_bounds, "RdBu_r", f"{var_name} phy-ref")
        mask_cmap = ListedColormap(["white", "black"])
        mesh3 = _plot_grid(axes[row, 3], mask, lon, lat, _mask_boundaries(), mask_cmap, f"{var_name} mask")

        _add_colorbar(fig, mesh0, axes[row, 0], positive_bounds, var_name, ticks=positive_bounds[::2])
        _add_colorbar(fig, mesh1, axes[row, 1], positive_bounds, var_name, ticks=positive_bounds[::2])
        _add_colorbar(fig, mesh2, axes[row, 2], signed_bounds, "phy-ref")
        _add_colorbar(fig, mesh3, axes[row, 3], _mask_boundaries(), "mask", ticks=[0, 1])

    fig.suptitle(f"Physical comparison time={time_index} ensemble={ensemble_index} level={level_index}", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_norm_output(
    ds_norm: xr.Dataset | None,
    precip_vars: list[str],
    time_index: int,
    level_index: int,
    ensemble_index: int,
    lon: np.ndarray | None,
    lat: np.ndarray | None,
    out_path: Path,
) -> None:
    if ds_norm is None:
        return
    use_map = lon is not None and lat is not None
    fig, axes = _axes(len(precip_vars), 1, width=5.0, height_per_row=3.0, use_map=use_map)
    bounds = _norm_boundaries()
    for row, var_name in enumerate(precip_vars):
        values = _select_grid(ds_norm[var_name], time_index, level_index, ensemble_index)
        values = np.clip(values, bounds[0], bounds[-1])
        mesh = _plot_grid(axes[row, 0], values, lon, lat, bounds, "RdBu_r", f"{var_name} ds_norm")
        _add_colorbar(fig, mesh, axes[row, 0], bounds, "normalized")
    fig.suptitle(f"Normalized model output time={time_index} ensemble={ensemble_index} level={level_index}", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_denoise_trajectory(
    ds_trj: xr.Dataset | None,
    time_index: int,
    ensemble_index: int,
    out_path: Path,
) -> None:
    if ds_trj is None or "precip" not in ds_trj:
        return
    arr = ds_trj["precip"].isel(time=time_index)
    if "ensemble" in arr.dims:
        arr = arr.isel(ensemble=ensemble_index)

    values = np.asarray(arr.values)
    if values.ndim != 4:
        raise ValueError(f"Expected denoise trajectory shaped (step, channel, y, x); got {values.shape}")
    n_steps, n_channels = values.shape[:2]
    channel_coord = arr.coords.get("channel", np.arange(n_channels))
    step_coord = arr.coords.get("denoise_step", np.arange(n_steps))
    bounds = _trajectory_boundaries(values)

    fig, axes = _axes(n_channels, n_steps, width=max(8.0, 2.2 * n_steps), height_per_row=2.2)
    for channel_index in range(n_channels):
        for step_index in range(n_steps):
            plot_values = np.clip(values[step_index, channel_index], bounds[0], bounds[-1])
            title = f"ch={int(channel_coord.values[channel_index])} step={int(step_coord.values[step_index])}"
            mesh = _plot_grid(axes[channel_index, step_index], plot_values, None, None, bounds, "RdBu_r", title)
            if step_index == n_steps - 1:
                _add_colorbar(fig, mesh, axes[channel_index, step_index], bounds, "normalized")

    fig.suptitle(f"Denoise trajectory time={time_index} ensemble={ensemble_index}", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def open_optional_zarr(path: Path, group: str) -> xr.Dataset | None:
    try:
        return xr.open_zarr(path, group=group)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DiffMAE WoFS rollout outputs.")
    parser.add_argument("--exp-path", type=Path, default=EXP_PATH)
    parser.add_argument("--file-name", type=str, default=FILE_NAME)
    parser.add_argument("--ref-root", type=Path, default=REF_ROOT)
    parser.add_argument("--time-index", type=int, default=TIME_INDEX)
    parser.add_argument("--ensemble-index", type=int, default=ENSEMBLE_INDEX)
    parser.add_argument("--level-index", type=int, default=LEVEL_INDEX)
    parser.add_argument("--precip-vars", nargs="+", default=PRECIP_VARS)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_file = args.exp_path / f"{args.file_name}_analysis.zarr"
    mask_file = args.exp_path / f"{args.file_name}_mask.npz"
    ref_file = args.ref_root / f"{args.file_name}.zarr.zip"
    out_dir = args.out_dir or args.exp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_ref = open_wofs_zarr_zip(ref_file)
    ds_phy = xr.open_zarr(analysis_file)
    ds_norm = open_optional_zarr(analysis_file, "norm_output")
    # ds_trj = open_optional_zarr(analysis_file, "denoise_trajectory")
    mask_ds = load_mask_dataset(analysis_file, mask_file)
    lat, lon = get_lat_lon(ds_ref)

    stem = f"{args.file_name}_t{args.time_index:02d}_ens{args.ensemble_index:02d}_lev{args.level_index:02d}"
    plot_physical_comparison(
        ds_phy,
        ds_ref,
        mask_ds,
        list(args.precip_vars),
        args.time_index,
        args.level_index,
        args.ensemble_index,
        lon,
        lat,
        out_dir / f"{stem}_physical_ref_diff_mask.png",
    )
    plot_norm_output(
        ds_norm,
        list(args.precip_vars),
        args.time_index,
        args.level_index,
        args.ensemble_index,
        lon,
        lat,
        out_dir / f"{stem}_norm_output.png",
    )
    # plot_denoise_trajectory(
    #     ds_trj,
    #     args.time_index,
    #     args.ensemble_index,
    #     out_dir / f"{stem}_denoise_trajectory.png",
    # )
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
# python python_scripts/plot_diffmae_results.py \
#   --exp-path /scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/credit_rollouts/wofs_diffmae_pretrain_4x4patch_heightmask_v2/ddim_test4/20210427 \
#   --file-name wofs_20210427_0030_mem01 \
#   --time-index 0 \
#   --ensemble-index 0 \
#   --level-index 0 \
#   --precip-vars QRAIN QNRAIN \
#   --out-dir /home/Zhanxiang.Hua/miles-credit-wofs/plots/wofs_diffmae_pretrain_4x4patch/ddim_test4/20210427/plots

# python python_scripts/plot_diffmae_results.py \
#   --exp-path /scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/credit_rollouts/wofs_diffmae_pretrain_latest2-2/ddim-r070-s500-e001/20210416 \
#   --file-name wofs_20210416_0030_mem01 \
#   --time-index 0 \
#   --ensemble-index 0 \
#   --level-index 0 \
#   --precip-vars QRAIN QNRAIN \
#   --out-dir /home/Zhanxiang.Hua/miles-credit-wofs/plots/wofs_diffmae_pretrain_latest2-2/ddim-r070-s500-e001/20210416/plots
