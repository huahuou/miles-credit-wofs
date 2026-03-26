# WoFS to CREDIT-WRF practical runbook

This is the short operational runbook for producing WoFS interior archives, computing normalization statistics, and validating the preferred WoFS `t=0` conditioning path in CREDIT.

The commands are listed in the order they should be run.

---

## 0. Assumptions

This runbook assumes:

- raw WoFS files are reachable from the `base_dir` used in [../ufs2arco-0.19.0/wrf_preprocess/wofs_credit_wrf_raw.yaml](../ufs2arco-0.19.0/wrf_preprocess/wofs_credit_wrf_raw.yaml)
- the `ufs2arco` conda environment exists
- the `credit` conda environment exists
- output directories below `/work2/zhanxianghua/wofs_preprocess_to_credit` are writable

---

## 1. Convert WoFS raw data to interior CREDIT-WRF case archives

From [../ufs2arco-0.19.0](../ufs2arco-0.19.0):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ufs2arco
cd /home/zhanxiang.hua/ufs2arco-0.19.0

python wrf_preprocess/batch_convert_credit_wrf.py \
  --template wrf_preprocess/wofs_credit_wrf_raw.yaml \
  --output-dir /work2/zhanxianghua/wofs_preprocess_to_credit/cases \
  --years 2019 \
  --member-ids 1 \
  --mpirun-n 16 \
  --overwrite
```

Notes:
- replace `--years` as needed
- add `--dates 20190429,20190501` if you want specific case dates only
- add `--init-times 0000,0100` if you want specific initialization times only
- add more members through `--member-ids`

Output pattern:

- `wofs_YYYYMMDD_HHMM_memNN.zarr`

---

## 2. Compute interior normalization statistics

From [../ufs2arco-0.19.0](../ufs2arco-0.19.0):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ufs2arco
cd /home/zhanxiang.hua/ufs2arco-0.19.0

python wrf_preprocess/compute_credit_stats.py \
  --glob '/work2/zhanxianghua/wofs_preprocess_to_credit/cases/*.zarr' \
  --start-date 20190401 \
  --end-date 20201231 \
  --mean-out /work2/zhanxianghua/wofs_preprocess_to_credit/stats/mean.nc \
  --std-out /work2/zhanxianghua/wofs_preprocess_to_credit/stats/std.nc \
  --latweights-out /work2/zhanxianghua/wofs_preprocess_to_credit/stats/latitude_weights_placeholder.nc \
  --n-workers 4 \
  --threads-per-worker 1 \
  --memory-limit 3GiB \
  --files-per-task 2
```

Notes:
- use only the training date range for stats
- adjust Dask settings to fit node memory

---

## 3. Prepare the CREDIT config

Use [config/wofs_credit_wrf_template.yml](../config/wofs_credit_wrf_template.yml) as the base config.

This template now defaults to the preferred WoFS `t=0` path:

- no `data.boundary.save_loc`
- no `data.boundary.save_loc_surface`
- no `data.boundary.mean_path`
- no `data.boundary.std_path`
- `data.boundary.reuse_interior_stats: True`

At minimum, verify these paths and ranges:

- `data.save_loc`
- `data.save_loc_surface`
- `data.save_loc_dynamic_forcing`
- `data.save_loc_diagnostic`
- `data.mean_path`
- `data.std_path`
- `data.train_years`
- `data.valid_years`

Also verify the model counts match the selected variables:

- `model.param_interior.channels = 6`
- `model.param_interior.surface_channels = 4`
- `model.param_interior.input_only_channels = 10`
- `model.param_interior.output_only_channels = 6`
- `model.param_outside.channels = 6`
- `model.param_outside.surface_channels = 4`

---

## 4. WoFS `t=0` conditioning path without external boundary zarrs

There is now a WoFS-specific path that does **not** read `data.boundary.save_loc` as a dataset source.

Instead:

- the interior WoFS case zarr is still the only trajectory file,
- `t=0` from that same zarr is used as the conditioning tensor,
- interior mean/std are reused by the normalizer.

### Single-step smoke test

From [../miles-credit-wofs](../miles-credit-wofs):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate credit
cd /home/zhanxiang.hua/miles-credit-wofs

CUDA_VISIBLE_DEVICES='' python applications/test_wofs_credit_wrf_t0.py \
    config/wofs_credit_wrf_t0_test_tmp.yml \
    --split train \
    --index 0 \
    --compare-next \
    --forward
```

### Single-step training smoke test

There is now a WoFS-specific single-step training entry point:

- [../applications/train_wrf_wofs.py](../applications/train_wrf_wofs.py)

Minimal smoke test:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate credit
cd /home/zhanxiang.hua/miles-credit-wofs

python applications/train_wrf_wofs.py \
    --config config/wofs_credit_wrf_t0_train_smoke.yml
```

This smoke config is set up to:

- use `WoFSSingleStepDataset`
- run `batches_per_epoch: 1`
- skip validation
- perform one minimal trainer pass through the single-step WRF trainer

### Multi-step smoke test

From [../miles-credit-wofs](../miles-credit-wofs):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate credit
cd /home/zhanxiang.hua/miles-credit-wofs

CUDA_VISIBLE_DEVICES='' python applications/test_wofs_credit_wrf_t0_multi.py \
    config/wofs_credit_wrf_t0_multi_test_tmp.yml \
    --split train \
    --index 0 \
    --forward
```

Expected multi-step checks include:

- forecast steps progress as `[1, 2, ..., forecast_len + 1]`
- `x_boundary` remains constant across the rollout
- `x` changes from step to step
- the model accepts the rollout on CPU

### Optional learning-start offset

The WoFS-specific loaders now support:

- `data.target_start_step: 1` meaning first supervised target is `t1`
- `data.target_start_step: 2` meaning first supervised target is `t2`

With `history_len: 1`, this means:

- `target_start_step: 1` → first pair is `x=t0`, `y=t1`, conditioning=`t0`
- `target_start_step: 2` → first pair is `x=t1`, `y=t2`, conditioning=`t0`

You can override this in the multi-step smoke test with:

```bash
CUDA_VISIBLE_DEVICES='' python applications/test_wofs_credit_wrf_t0_multi.py \
    config/wofs_credit_wrf_t0_multi_test_tmp.yml \
    --split train \
    --index 0 \
    --target-start-step 2 \
    --forward
```

### Is `extract_credit_boundary.py` still needed?

For the current WoFS `t=0` conditioning workflow, [../ufs2arco-0.19.0/wrf_preprocess/extract_credit_boundary.py](../ufs2arco-0.19.0/wrf_preprocess/extract_credit_boundary.py) is **no longer needed as a dataset-input step**.

Reason:

- the WoFS-specific loaders now build `x_boundary` directly from `time=0` of the same case zarr,
- no external boundary zarr is read during training or smoke testing.

So for the current WoFS-specific CREDIT path:

- converting interior case zarrs is still required,
- interior stats are still required,
- external boundary extraction is unnecessary for data loading.

The WoFS-specific path can now reuse interior stats for boundary normalization.

That means for the preferred `t=0` workflow you can set:

- `data.boundary.reuse_interior_stats: True`

and omit:

- `data.boundary.mean_path`
- `data.boundary.std_path`

This has been smoke-tested successfully for:

- single-step WoFS sample loading
- multi-step WoFS sample loading
- single-step WoFS training smoke test
- multi-step WoFS training smoke test

Important note:

- the current `extract_credit_boundary.py` selects 3-hour cadence slices,
- that matches the older proxy-boundary workflow,
- it does **not** match the new WoFS `t=0` conditioning semantics.

So for the current workflow, the script is best treated as a legacy compatibility tool, not part of the preferred WoFS `t=0` path.

In short for the preferred WoFS `t=0` path:

- keep interior conversion
- keep interior stats
- set `data.boundary.reuse_interior_stats: True`
- skip boundary extraction entirely

### Multi-step training smoke test

There is now a WoFS-specific multi-step training entry point that uses same-file `t=0` conditioning directly:

- [../applications/train_wrf_wofs_multi.py](../applications/train_wrf_wofs_multi.py)

Minimal smoke test:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate credit
cd /home/zhanxiang.hua/miles-credit-wofs

python applications/train_wrf_wofs_multi.py \
    --config config/wofs_credit_wrf_t0_multi_train_smoke.yml

torchrun --standalone --nnodes 1 --nproc-per-node=1 applications/train_wrf_wofs_multi.py -c config/wofs_credit_wrf_t0_multi_train_date_range_example.yml
```

This smoke config is set up to:

- use `WoFSMultiStep`
- use `target_start_step: 1`
- run `batches_per_epoch: 1`
- skip validation
- perform one minimal trainer pass through the multi-step WRF trainer

---

## 5. WoFS rollout on the preferred `t=0` path

There is now a WoFS-specific rollout entry point for the preferred same-file `t=0` workflow:

- [../applications/rollout_wrf_wofs.py](../applications/rollout_wrf_wofs.py)

Example config:

- [../config/wofs_credit_wrf_t0_rollout_date_range_example.yml](../config/wofs_credit_wrf_t0_rollout_date_range_example.yml)

This rollout path:

- loads the trained checkpoint from `save_loc`
- selects case zarrs using `predict.custom_date_range`
- uses same-file `t=0` conditioning
- writes forecast NetCDF files under `predict.save_forecast`
- writes summary metrics to `predict.save_forecast/rollout_metrics.csv`

Minimal example:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate credit
cd /home/zhanxiang.hua/miles-credit-wofs

python applications/rollout_wrf_wofs.py \
    config/wofs_credit_wrf_t0_rollout_date_range_example.yml \
    --max-cases 1

torchrun --standalone --nnodes 1 --nproc-per-node=4 applications/train_wrf_wofs_multi.py -c config/wofs_credit_wrf_t0_rollout_date_range_example.yml 
```

Before running it, make sure:

- `save_loc` points to a completed WoFS training run with a checkpoint
- `predict.save_forecast` points to a writable output directory
- `predict.custom_date_range` matches the WoFS case dates you want to roll out

Important:

- `config/wofs_credit_wrf_t0_rollout_date_range_example.yml` is a rollout-only config
- it does not contain a `trainer` section, so it cannot be passed to `train_wrf_wofs.py` or `train_wrf_wofs_multi.py`

For multi-step training with the same date-range style config, use:

- [../config/wofs_credit_wrf_t0_multi_train_date_range_example.yml](../config/wofs_credit_wrf_t0_multi_train_date_range_example.yml)

Single-process example:

```bash
python applications/train_wrf_wofs_multi.py \
    -c config/wofs_credit_wrf_t0_multi_train_date_range_example.yml
```

Distributed example:

1. set `trainer.mode: 'ddp'` in [../config/wofs_credit_wrf_t0_multi_train_date_range_example.yml](../config/wofs_credit_wrf_t0_multi_train_date_range_example.yml)
2. then launch with:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node=4 applications/train_wrf_wofs_multi.py \
    -c config/wofs_credit_wrf_t0_multi_train_date_range_example.yml
```

---

## 6. Legacy compatibility note

[../ufs2arco-0.19.0/wrf_preprocess/extract_credit_boundary.py](../ufs2arco-0.19.0/wrf_preprocess/extract_credit_boundary.py) and boundary-specific normalization statistics are now legacy compatibility tools.

They are not part of the preferred WoFS `t=0` training workflow.

Only keep them if you explicitly need to reproduce the older proxy-boundary CREDIT path.
