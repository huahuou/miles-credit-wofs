"""
Generate deterministic patch-aligned masks for WoFS DiffMAE rollout.

The output is an .npz bundle consumable by rollout_wrf_wofs_mae_da_metrics.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from credit.wofs_diffmae_mask_utils import (
    build_grouped_patch_masks,
    resolve_precip_group_layout,
    save_mask_bundle,
    token_grid_shape,
)


def primary_main():
    parser = argparse.ArgumentParser(description="Generate a deterministic WoFS DiffMAE precip mask bundle.")
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("--out", required=True, type=str, help="Output .npz path")
    parser.add_argument("--n-times", required=True, type=int, help="Number of rollout timesteps the mask should cover")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--mask-mode",
        default=None,
        type=str,
        help=(
            "Override eval.precip_mask_mode from config. Supported: spatial_patch, "
            "channel_patch, height_patch, mixed, mixed_height."
        ),
    )
    parser.add_argument(
        "--mask-ratio",
        default=None,
        nargs="+",
        type=float,
        help="Override eval.precip_mask_ratio from config. Use one float or two floats for a range.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    eval_conf = conf.setdefault("eval", {})
    if args.mask_mode is not None:
        eval_conf["precip_mask_mode"] = args.mask_mode
    if args.mask_ratio is not None:
        eval_conf["precip_mask_ratio"] = args.mask_ratio[0] if len(args.mask_ratio) == 1 else args.mask_ratio[:2]

    group_names, group_channels = resolve_precip_group_layout(conf)
    token_h, token_w = token_grid_shape(conf)
    bundle = build_grouped_patch_masks(
        n_times=int(args.n_times),
        n_groups=len(group_names),
        token_h=token_h,
        token_w=token_w,
        mask_ratio=eval_conf.get("precip_mask_ratio", 1.0),
        mask_mode=eval_conf.get("precip_mask_mode", "spatial_patch"),
        seed=int(args.seed),
        channel_patch_mask_probability=float(conf.get("trainer", {}).get("channel_patch_mask_probability", 0.5)),
        mixed_height_spatial_probability=float(eval_conf.get("mixed_height_spatial_probability", 1.0)),
        mixed_height_channel_probability=float(eval_conf.get("mixed_height_channel_probability", 0.0)),
        mixed_height_height_probability=float(eval_conf.get("mixed_height_height_probability", 1.0)),
        group_channels=group_channels,
        height_mask_levels=eval_conf.get("height_mask_levels"),
        height_visible_levels=eval_conf.get("height_visible_levels"),
    )
    out_path = Path(args.out)
    save_mask_bundle(
        out_path,
        patch_mask_grouped=bundle["patch_mask_grouped"],
        mask_mode=bundle["mask_mode"],
        requested_mask_ratio=bundle["requested_mask_ratio"],
        actual_group_mask_fraction=bundle["actual_group_mask_fraction"],
        group_names=group_names,
        group_channels=group_channels,
        image_size=tuple(int(v) for v in conf["model"]["image_size"]),
        patch_size=int(conf["model"]["patch_size"]),
        seed=int(args.seed),
        config_path=args.config,
    )
    print(out_path)


if __name__ == "__main__":
    primary_main()
