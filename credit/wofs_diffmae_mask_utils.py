from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


SPATIAL_MASK_MODES = {"spatial_patch"}
CHANNEL_MASK_MODES = {"channel_patch"}
HEIGHT_MASK_MODES = {"height_patch"}
MIXED_MASK_MODES = {"mixed"}
MIXED_HEIGHT_MASK_MODES = {"mixed_height"}


def normalize_mask_mode(mask_mode: str) -> str:
    mode = str(mask_mode).strip().lower()
    if mode in SPATIAL_MASK_MODES:
        return "spatial_patch"
    if mode in CHANNEL_MASK_MODES:
        return "channel_patch"
    if mode in HEIGHT_MASK_MODES:
        return "height_patch"
    if mode in MIXED_MASK_MODES:
        return "mixed"
    if mode in MIXED_HEIGHT_MASK_MODES:
        return "mixed_height"
    raise ValueError(f"Unsupported precip mask mode: {mask_mode!r}")


def resolve_precip_group_layout(conf: dict) -> tuple[list[str], list[int]]:
    model_conf = conf["model"]
    data_conf = conf["data"]
    grouping = str(model_conf.get("precip_grouping", "")).strip().lower()
    if grouping != "level":
        raise ValueError(f"Unsupported precip_grouping {grouping!r}; only 'level' is supported.")

    group_names = list(model_conf.get("precip_group_names") or data_conf.get("precip_vars", []))
    if not group_names:
        raise ValueError("precip_group_names or data.precip_vars must be provided for level tokenization")
    group_channels = model_conf.get("precip_group_channels")
    if not group_channels:
        total_channels = int(model_conf["modality_channels"]["precip"])
        if total_channels % len(group_names) != 0:
            raise ValueError(
                "Could not infer precip_group_channels because precip channels do not divide evenly "
                "across precip_group_names."
            )
        group_channels = [total_channels // len(group_names)] * len(group_names)
    if len(group_names) != len(group_channels):
        raise ValueError("precip_group_names and precip_group_channels must have the same length")
    return group_names, [int(v) for v in group_channels]


def token_grid_shape(conf: dict) -> tuple[int, int]:
    image_h, image_w = conf["model"]["image_size"]
    patch_size = int(conf["model"]["patch_size"])
    return int(image_h) // patch_size, int(image_w) // patch_size


def _sample_ratio(mask_ratio: float | tuple[float, float] | list[float], rng: np.random.Generator) -> float:
    if isinstance(mask_ratio, (list, tuple)):
        lo, hi = float(mask_ratio[0]), float(mask_ratio[1])
        return float(rng.uniform(lo, hi))
    return float(mask_ratio)


def _sample_rank_mask(n_elements: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    noise = rng.random(n_elements)
    ids = np.argsort(noise)
    ranks = np.empty_like(ids)
    ranks[ids] = np.arange(n_elements)
    n_mask = int(np.clip(np.round(ratio * n_elements), 0, n_elements))
    return (ranks < n_mask).astype(np.float32)


def _eligible_levels(
    n_levels: int,
    masked_levels: list[int] | tuple[int, ...] | None,
    visible_levels: list[int] | tuple[int, ...] | None,
) -> np.ndarray:
    eligible = np.ones(n_levels, dtype=bool)
    if masked_levels is not None:
        eligible[:] = False
        for level in masked_levels:
            level = int(level)
            if 0 <= level < n_levels:
                eligible[level] = True
    if visible_levels is not None:
        for level in visible_levels:
            level = int(level)
            if 0 <= level < n_levels:
                eligible[level] = False
    return eligible


def _choose_mixed_height_mode(
    rng: np.random.Generator,
    spatial_probability: float,
    channel_probability: float,
    height_probability: float,
) -> str:
    probs = np.asarray(
        [float(spatial_probability), float(channel_probability), float(height_probability)],
        dtype=np.float64,
    )
    if np.any(probs < 0):
        raise ValueError("Mixed height mask probabilities must be non-negative")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("At least one mixed height mask probability must be positive")
    probs = probs / total
    return str(rng.choice(["spatial_patch", "channel_patch", "height_patch"], p=probs))


def build_grouped_patch_masks(
    n_times: int,
    n_groups: int,
    token_h: int,
    token_w: int,
    mask_ratio: float | tuple[float, float] | list[float],
    mask_mode: str,
    seed: int,
    channel_patch_mask_probability: float = 0.5,
    mixed_height_spatial_probability: float = 1.0,
    mixed_height_channel_probability: float = 1.0,
    mixed_height_height_probability: float = 1.0,
    group_channels: list[int] | tuple[int, ...] | None = None,
    height_mask_levels: list[int] | tuple[int, ...] | None = None,
    height_visible_levels: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    normalized_mode = normalize_mask_mode(mask_mode)
    n_tokens = token_h * token_w

    level_count = 0
    if normalized_mode in {"height_patch", "mixed_height"}:
        if not group_channels:
            raise ValueError("group_channels is required for height_patch or mixed_height mask generation")
        group_channels = [int(v) for v in group_channels]
        if len(set(group_channels)) != 1:
            raise ValueError("height_patch and mixed_height masks require all precip groups to have the same number of levels")
        level_count = group_channels[0]
        masks = np.zeros((n_times, n_groups, level_count, token_h, token_w), dtype=np.float32)
    else:
        masks = np.zeros((n_times, n_groups, token_h, token_w), dtype=np.float32)
    mask_modes: list[str] = []
    requested_ratios = np.zeros(n_times, dtype=np.float32)

    for time_idx in range(n_times):
        sampled_ratio = _sample_ratio(mask_ratio, rng)
        requested_ratios[time_idx] = sampled_ratio

        current_mode = normalized_mode
        if current_mode == "mixed":
            current_mode = "channel_patch" if rng.random() < float(channel_patch_mask_probability) else "spatial_patch"
        elif current_mode == "mixed_height":
            current_mode = _choose_mixed_height_mode(
                rng,
                mixed_height_spatial_probability,
                mixed_height_channel_probability,
                mixed_height_height_probability,
            )

        if current_mode == "spatial_patch":
            base = _sample_rank_mask(n_tokens, sampled_ratio, rng).reshape(token_h, token_w)
            if masks.ndim == 5:
                masks[time_idx] = np.repeat(base[None, None, :, :], n_groups * level_count, axis=0).reshape(
                    n_groups, level_count, token_h, token_w
                )
            else:
                masks[time_idx] = np.repeat(base[None, :, :], n_groups, axis=0)
        elif current_mode == "channel_patch":
            base = _sample_rank_mask(n_groups * n_tokens, sampled_ratio, rng).reshape(n_groups, token_h, token_w)
            if masks.ndim == 5:
                masks[time_idx] = np.repeat(base[:, None, :, :], level_count, axis=1)
            else:
                masks[time_idx] = base
        elif current_mode == "height_patch":
            eligible = _eligible_levels(level_count, height_mask_levels, height_visible_levels)
            n_eligible = int(eligible.sum())
            if n_eligible > 0:
                base = _sample_rank_mask(n_groups * n_eligible * n_tokens, sampled_ratio, rng)
                base = base.reshape(n_groups, n_eligible, token_h, token_w)
                masks[time_idx][:, eligible, :, :] = base
        else:
            raise ValueError(f"Unexpected normalized mask mode {current_mode!r}")

        mask_modes.append(current_mode)

    return {
        "patch_mask_grouped": masks,
        "mask_mode": np.asarray(mask_modes, dtype="<U16"),
        "requested_mask_ratio": requested_ratios,
        "actual_group_mask_fraction": masks.reshape(n_times, n_groups, -1).mean(axis=2).astype(np.float32),
    }


def save_mask_bundle(
    out_path: str | Path,
    *,
    patch_mask_grouped: np.ndarray,
    mask_mode: np.ndarray,
    requested_mask_ratio: np.ndarray,
    actual_group_mask_fraction: np.ndarray,
    group_names: list[str],
    group_channels: list[int],
    image_size: tuple[int, int] | list[int],
    patch_size: int,
    seed: int,
    source_case: str | None = None,
    config_path: str | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        patch_mask_grouped=np.asarray(patch_mask_grouped, dtype=np.float32),
        mask_mode=np.asarray(mask_mode),
        requested_mask_ratio=np.asarray(requested_mask_ratio, dtype=np.float32),
        actual_group_mask_fraction=np.asarray(actual_group_mask_fraction, dtype=np.float32),
        group_names=np.asarray(group_names, dtype="<U64"),
        group_channels=np.asarray(group_channels, dtype=np.int32),
        image_size=np.asarray(image_size, dtype=np.int32),
        patch_size=np.asarray(int(patch_size), dtype=np.int32),
        token_grid=np.asarray(patch_mask_grouped.shape[-2:], dtype=np.int32),
        seed=np.asarray(int(seed), dtype=np.int64),
        source_case=np.asarray("" if source_case is None else str(source_case), dtype="<U512"),
        config_path=np.asarray("" if config_path is None else str(config_path), dtype="<U512"),
    )


def load_mask_bundle(mask_path: str | Path) -> dict[str, Any]:
    mask_path = Path(mask_path)
    with np.load(mask_path, allow_pickle=False) as payload:
        patch_mask_grouped = np.asarray(payload["patch_mask_grouped"], dtype=np.float32)
        mask_mode = np.asarray(payload["mask_mode"])
        requested_mask_ratio = np.asarray(payload["requested_mask_ratio"], dtype=np.float32)
        actual_group_mask_fraction = np.asarray(payload["actual_group_mask_fraction"], dtype=np.float32)
        group_names = [str(v) for v in payload["group_names"].tolist()]
        group_channels = [int(v) for v in payload["group_channels"].tolist()]
        image_size = tuple(int(v) for v in payload["image_size"].tolist())
        patch_size = int(np.asarray(payload["patch_size"]).item())
        token_grid = tuple(int(v) for v in payload["token_grid"].tolist())
        seed = int(np.asarray(payload["seed"]).item())
        source_case = str(np.asarray(payload["source_case"]).item())
        config_path = str(np.asarray(payload["config_path"]).item())

    if patch_mask_grouped.ndim not in (4, 5):
        raise ValueError(
            "Expected patch_mask_grouped to have shape (time, group, token_y, token_x) or "
            f"(time, group, level, token_y, token_x); got {patch_mask_grouped.shape}"
        )
    if patch_mask_grouped.shape[0] != mask_mode.shape[0]:
        raise ValueError("mask_mode length does not match patch_mask_grouped time dimension")
    if patch_mask_grouped.shape[0] != requested_mask_ratio.shape[0]:
        raise ValueError("requested_mask_ratio length does not match patch_mask_grouped time dimension")
    if patch_mask_grouped.shape[:2] != actual_group_mask_fraction.shape[:2]:
        raise ValueError("actual_group_mask_fraction shape does not match patch_mask_grouped leading dimensions")
    if patch_mask_grouped.shape[-2:] != token_grid:
        raise ValueError("token_grid does not match patch_mask_grouped shape")
    if patch_mask_grouped.shape[1] != len(group_names):
        raise ValueError("group_names length does not match grouped mask dimension")
    if patch_mask_grouped.shape[1] != len(group_channels):
        raise ValueError("group_channels length does not match grouped mask dimension")

    return {
        "path": str(mask_path),
        "patch_mask_grouped": patch_mask_grouped,
        "mask_mode": [str(v) for v in mask_mode.tolist()],
        "requested_mask_ratio": requested_mask_ratio,
        "actual_group_mask_fraction": actual_group_mask_fraction,
        "group_names": group_names,
        "group_channels": group_channels,
        "image_size": image_size,
        "patch_size": patch_size,
        "token_grid": token_grid,
        "seed": seed,
        "source_case": source_case or None,
        "config_path": config_path or None,
    }


def grouped_patch_mask_time_slice(mask_bundle: dict[str, Any], time_index: int) -> tuple[np.ndarray, str, float]:
    masks = mask_bundle["patch_mask_grouped"]
    if masks.shape[0] == 1:
        index = 0
    else:
        index = int(time_index)
        if index >= masks.shape[0]:
            raise IndexError(
                f"Requested time index {time_index} but mask bundle only has {masks.shape[0]} timestep entries"
            )
    return (
        np.asarray(masks[index], dtype=np.float32),
        str(mask_bundle["mask_mode"][index]),
        float(mask_bundle["requested_mask_ratio"][index]),
    )
