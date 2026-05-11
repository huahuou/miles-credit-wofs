# WoFS DiffMAE Rollout Metrics Workflow

This document covers the workflow for the two scripts added for WoFS DiffMAE rollout evaluation:

- `applications/generate_wofs_diffmae_mask.py`
- `applications/rollout_wrf_wofs_mae_da_metrics.py`

These scripts are intended to extend the existing `applications/rollout_wrf_wofs_mae_da.py` workflow without changing that script.

## What the new workflow adds

Compared with `rollout_wrf_wofs_mae_da.py`, the new rollout script adds:

1. Metrics in both normalized space and physical space
2. SSIM in normalized space
3. JSON output with per-timestep metrics and summary statistics
4. A copy of the YAML config in the rollout output directory
5. Deterministic custom mask support
6. Mask data stored in the output zarr

## Important mask behavior

The model is patch-token based.

- `patch_size=8` in `config/wofs_diffmae.yml`
- one token corresponds to one `8 x 8` spatial patch

That means:

- the model does **not** support masking a single pixel independently inside a patch
- if a single pixel in an `8 x 8` patch were treated as masked, the effective model behavior would still be patch-level masking for that token

For that reason, the custom mask workflow is patch-aligned, not pixel-aligned.

## Script 1: Generate a deterministic mask

Use `generate_wofs_diffmae_mask.py` when you want repeatable masking across runs.

### Inputs

- config file
- output `.npz` path
- number of rollout timesteps to cover
- random seed
- optional mask mode override
- optional mask ratio override

### Example

```bash
python applications/generate_wofs_diffmae_mask.py \
  -c config/wofs_diffmae.yml \
  --out /path/to/mask.npz \
  --n-times 10 \
  --seed 1000
```

### Optional overrides

```bash
python applications/generate_wofs_diffmae_mask.py \
  -c config/wofs_diffmae.yml \
  --out /path/to/mask_channel.npz \
  --n-times 10 \
  --seed 1000 \
  --mask-mode channel_patch \
  --mask-ratio 0.75
```

For a ratio range:

```bash
--mask-ratio 0.5 1.0
```

### Mask file contents

The `.npz` file stores:

- `patch_mask_grouped`
- `mask_mode`
- `requested_mask_ratio`
- `actual_group_mask_fraction`
- `group_names`
- `group_channels`
- `image_size`
- `patch_size`
- `token_grid`
- `seed`

`patch_mask_grouped` has shape:

```text
(time, mask_group, patch_y, patch_x)
```

## Script 2: Run rollout with metrics

Use `rollout_wrf_wofs_mae_da_metrics.py` to run the actual assimilation rollout and save predictions, masks, and metrics.

### Inputs

- config file
- checkpoint
- output directory
- optional date filter
- optional rollout mode override
- optional custom mask file
- optional mask seed for auto-generated per-case masks

### Example with a saved mask

```bash
python applications/rollout_wrf_wofs_mae_da_metrics.py \
  -c config/wofs_diffmae.yml \
  --checkpoint /path/to/checkpoint.pt \
  --out-dir /path/to/out \
  --mask-file /path/to/mask.npz
```

### Example without a saved mask

If `--mask-file` is omitted, the script generates a deterministic per-case mask bundle and saves it beside the rollout outputs.

```bash
python applications/rollout_wrf_wofs_mae_da_metrics.py \
  -c config/wofs_diffmae.yml \
  --checkpoint /path/to/checkpoint.pt \
  --out-dir /path/to/out \
  --mask-seed 1000
```

### Smoke test example

```bash
python applications/rollout_wrf_wofs_mae_da_metrics.py \
  -c config/wofs_diffmae.yml \
  --checkpoint /path/to/checkpoint.pt \
  --out-dir /path/to/out \
  --mask-seed 1000 \
  --max-files 1 \
  --max-times 2
```

## Output layout

For each case, the rollout script writes:

```text
<out-dir>/
  wofs_diffmae.yml
  <YYYYMMDD>/
    <case>_analysis.zarr
    <case>_metrics.json
    <case>_mask.npz    # if --mask-file was not supplied
```

## What is stored in the zarr

The root zarr contains the physical-space precip analysis for the configured precip variables.

Optional zarr groups:

- `norm_output`
- `denoise_trajectory`
- `mask`

### `mask` group contents

The `mask` zarr group stores:

- `patch_mask_grouped`
- `pixel_mask_channel`
- `requested_mask_ratio`
- `mask_mode`

Shapes:

- `patch_mask_grouped`: `(time, mask_group, patch_y, patch_x)`
- `pixel_mask_channel`: `(time, channel, y, x)`

`pixel_mask_channel` is only an expanded view of the patch mask for inspection and metric calculation. The actual model masking remains patch-token based.

## Metrics written to JSON

Each case gets one JSON file with:

- metadata
- per-timestep metrics
- summary statistics across timesteps

### Per-timestep metrics

Normalized space:

- `normalized_mse_masked`
- `normalized_mae_masked`
- `normalized_ssim_masked`
- `normalized_mse_full`
- `normalized_mae_full`
- `normalized_ssim_full`

Physical space:

- `physical_mse_masked`
- `physical_mae_masked`
- `physical_mse_full`
- `physical_mae_full`

Mask metadata:

- `masked_fraction`
- `mask_mode`
- `requested_mask_ratio`

## Mask mode semantics

The scripts support the same patch-level ideas used by training:

- `spatial_patch`: all precip groups are masked at a patch location
- `channel_patch`: only selected precip groups are masked at a patch location
- `mixed`: each timestep chooses either `spatial_patch` or `channel_patch`

For grouped precip variables in your config, `channel_patch` means:

- one mask decision per precip variable group
- all levels of that variable are masked together at the selected patch

## Required consistency checks for external mask files

When `--mask-file` is supplied, the rollout script checks that the mask bundle matches the config:

- same `patch_size`
- same `image_size`
- same precip group names
- same precip group channel counts

If any of these differ, the rollout exits with an error.

## Recommended workflow

### Option 1: fully repeatable experiment

1. Generate one mask file
2. Reuse that mask file for all rollouts you want to compare

```bash
python applications/generate_wofs_diffmae_mask.py \
  -c config/wofs_diffmae.yml \
  --out /path/to/fixed_mask.npz \
  --n-times 10 \
  --seed 1000

python applications/rollout_wrf_wofs_mae_da_metrics.py \
  -c config/wofs_diffmae.yml \
  --checkpoint /path/to/checkpoint.pt \
  --out-dir /path/to/out_run_a \
  --mask-file /path/to/fixed_mask.npz
```

### Option 2: deterministic per-case masks without precomputing

1. Skip mask generation
2. Provide `--mask-seed`
3. Let the rollout script save the mask bundle it used

```bash
python applications/rollout_wrf_wofs_mae_da_metrics.py \
  -c config/wofs_diffmae.yml \
  --checkpoint /path/to/checkpoint.pt \
  --out-dir /path/to/out_run_b \
  --mask-seed 1000
```

## Cluster environment

Per your local workflow, run under the project environment before testing or production rollout:

```bash
source $MODULESHOME/init/bash
module load cdo/2.4.2
module load ncview
module load rdhpcs-conda
module load cuda/12.8
conda activate credit-wofs
```

If you need an interactive node first:

CPU:

```bash
srun -A gpu-ai4wp -p u1-compute --mem=128g -N 1 -t 00:29:00 --pty bash -il
```

GPU:

```bash
srun -A gpu-ai4wp -p u1-h100 -q gpu --gpus=h100:1 -N 1 -t 00:29:00 --pty bash -il
```

## Notes

- The new rollout script does not modify `rollout_wrf_wofs_mae_da.py`.
- The mask generator exists because `mixed` and `channel_patch` experiments are only reproducible if the exact patch mask is saved and reused.
- The SSIM implementation in the new rollout script is adapted to this workflow and operates on normalized-space precip outputs.
