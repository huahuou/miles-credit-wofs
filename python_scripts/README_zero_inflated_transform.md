# Zero-Inflated Concentration Transform

This note covers the new script:

- `python_scripts/build_zero_inflated_transform_params.py`

It builds per-variable, per-level transform parameters for sparse WoFS concentration fields using a zero-inflated lognormal-probit transform.

It also supports:

- Dask multiprocessing via a local cluster
- diagnostic plots for each concentration variable

## Environment

Request a node and activate the project environment first.

CPU node:

```bash
srun -A gpu-ai4wp -p u1-compute --mem=128g -N 1 -t 00:29:00 --pty bash -il
module load rdhpcs-conda
module load cuda/12.8
conda activate credit-wofs
cd /home/Zhanxiang.Hua/miles-credit-wofs
```

## 1. Build Transform Parameters

Example:

```bash
python python_scripts/build_zero_inflated_transform_params.py \
  --glob '/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/cases/wofs_*.zarr.zip' \
  --output '/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/zero_inflated_transform_params.json' \
  --plots-dir '/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/zero_inflated_transform_plots' \
  --zero-floor 1e-11 \
  --probit-eps 1e-6 \
  --min-positive-samples-per-level 2000 \
  --n-workers 8 \
  --threads-per-worker 1 \
  --memory-limit 12GiB \
  --files-per-task 8
```

Important options:

- `--glob`: input WoFS zarr files
- `--output`: output JSON for the transform
- `--plots-dir`: directory for diagnostic plots
- `--zero-floor`: values below this are treated as zero
- `--max-files`: optional file subsampling for a faster first pass
- `--variables`: optional comma-separated variable list
- `--n-workers`: number of Dask worker processes
- `--files-per-task`: files grouped into each worker task

## Plot Output

For each variable, the script writes a PNG with:

- raw positive-value distribution
- transformed latent distribution
- per-level zero-mass (`alpha`) summary

The JSON also stores the generated plot path under each variable entry.

## 2. Compute Mean / Std with the New Transform

Use the generated JSON when running `compute_credit_stats.py`.

```bash
python python_scripts/compute_credit_stats.py \
  --glob '/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/cases/wofs_*.zarr.zip' \
  --transform-params-json '/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/zero_inflated_transform_params.json' \
  --mean-out '/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/mean.nc' \
  --std-out '/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats/std.nc'
```

## 3. Train with the Same JSON

In your YAML, point:

- `data.log_transform_params_json` to `zero_inflated_transform_params.json`

The current config already expects that path:

- `config/ursa_wofs_credit_wrf_da_increment.yml`

Then train as usual.

## Recommended Workflow

```bash
python python_scripts/build_zero_inflated_transform_params.py ...
python python_scripts/compute_credit_stats.py --transform-params-json ... 
torchrun --standalone --nnodes=1 --nproc-per-node=2 applications/train_wrf_wofs_da.py -c config/ursa_wofs_credit_wrf_da_increment.yml
```

## Output JSON

The script writes a JSON with:

- global transform metadata
- one entry per concentration variable
- per-level `alpha`, `mu`, `sigma`, and level status

Level statuses:

- `ok`: level fit directly from positive samples
- `borrow_fallback`: level uses pooled variable fit for `mu/sigma`
- `degenerate_zero`: level is effectively all zero
