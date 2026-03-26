# WoFS to CREDIT-WRF implementation summary

This document records what has been implemented so far to move WoFS data through the `ufs2arco` framework into a format that can be consumed by the current CREDIT WRF workflow.

It is written as a technical handoff and progress log. It explains:

1. the original compatibility problem,
2. the design choices that were made,
3. each code modification that was implemented,
4. how WoFS conditioning is handled in the preferred path,
5. how the converted data is adapted into CREDIT,
6. what was validated successfully,
7. what remains incomplete.

---

## 1. Goal of the work

The main goal has been to create a WoFS dataset pipeline that is as compatible as possible with the current CREDIT WRF training path.

The intended flow is:

1. read raw WoFS/WRF files,
2. convert them into per-case/per-member zarr stores with native WoFS grid structure preserved,
3. compute normalization statistics,
4. reuse same-file `t=0` as the conditioning state inside CREDIT,
5. verify that CREDIT can load the resulting tensors and run smoke-test training.

The main repositories involved were:

- [ufs2arco-0.19.0](../ufs2arco-0.19.0)
- [miles-credit-wofs](../miles-credit-wofs)

---

## 2. Why the existing `anemoi` target was not enough

The original `ufs2arco` WoFS preprocessing path already existed through the Anemoi target, but that output format did not match what CREDIT WRF expects.

### Existing Anemoi behavior

The Anemoi-style output effectively:

- flattens the horizontal grid,
- stacks multiple variables into a single `data` array,
- uses a layout closer to `(time, variable, ensemble, cell)`,
- treats geolocation auxiliaries differently from what the CREDIT WRF loader expects.

### CREDIT WRF expectations

The CREDIT WRF path, centered around:

- [credit/datasets/wrf_singlestep.py](../credit/datasets/wrf_singlestep.py)
- [credit/transforms/transforms_wrf.py](../credit/transforms/transforms_wrf.py)
- [applications/train_wrf.py](../applications/train_wrf.py)

expects more ordinary xarray datasets where variables are separate data variables with dimensions like:

- upper-air: `(time, level, y, x)`
- 2-D fields: `(time, y, x)`

That mismatch is why a dedicated WoFS-to-CREDIT target was created.

---

## 3. High-level design choices

### 3.1 Native WoFS grid is preserved

WoFS cases use a native local `300 x 300` grid, but the physical location changes from case to case.

Instead of trying to regrid all cases onto one fixed Earth grid, the chosen approach was:

- keep the native local `y/x` grid,
- preserve case-specific `latitude` and `longitude` as coordinates,
- store geospatial and temporal encoding fields as dynamic forcing variables.

Examples of those forcing variables are:

- `cos_latitude`
- `sin_latitude`
- `cos_longitude`
- `sin_longitude`
- `cos_julian_day`
- `sin_julian_day`
- `cos_local_time`
- `sin_local_time`
- `cos_solar_zenith_angle`
- `insolation`

### 3.2 One case-member per output archive

The current CREDIT WRF path does not naturally consume an ensemble-member dimension in the final input files.

So the chosen workflow was:

- one WoFS case + one member = one zarr archive,
- output naming pattern like `wofs_YYYYMMDD_HHMM_memNN.zarr`.

This keeps training samples simple and avoids having to carry a `member` dimension through CREDIT.

### 3.3 Boundary remains a conditioning branch, but the preferred WoFS path builds it from same-file `t=0`

The current CREDIT WRF trainer and model path still expect boundary information explicitly.

This appears in:

- [credit/datasets/wrf_singlestep.py](../credit/datasets/wrf_singlestep.py)
- [credit/trainers/trainerWRF.py](../credit/trainers/trainerWRF.py)
- [credit/models/swin_wrf.py](../credit/models/swin_wrf.py)

That means even if the interior WoFS dataset is ready, CREDIT still expects another outside-domain or boundary-like input tensor.

The preferred WoFS implementation now satisfies that requirement by deriving the conditioning branch from the same interior zarr:

- interior WoFS case archives,
- interior mean/std,
- same-file `t=0` conditioning in CREDIT,
- optional interior-stat reuse for boundary normalization.

---

## 4. Source-side changes in `ufs2arco`

## 4.1 `WRFRawSource` was extended with explicit `member_ids`

File modified:

- [ufs2arco-0.19.0/ufs2arco/sources/wrf_raw_source.py](../ufs2arco-0.19.0/ufs2arco/sources/wrf_raw_source.py)

### Why

WoFS is stored with ensemble member subdirectories such as `ENS_MEM_1`, `ENS_MEM_2`, etc.

To make per-member conversion clean and reproducible, the source reader needed a way to explicitly request particular members instead of always scanning all of them implicitly.

### What changed

The source now supports:

- `member_ids: [1]`
- `member_ids: [1, 2, 5]`

This affects scanning and sample discovery so that conversion runs can be targeted by member.

### Important behavior retained

This source still treats:

- one `(date, init_time)` pair as one trajectory.

That means trajectory is still a source-side concept used to define a continuous forecast sequence.

---

## 4.2 A new `CreditWRF` target was added

New file:

- [ufs2arco-0.19.0/ufs2arco/targets/credit_wrf.py](../ufs2arco-0.19.0/ufs2arco/targets/credit_wrf.py)

Registered in:

- [ufs2arco-0.19.0/ufs2arco/targets/__init__.py](../ufs2arco-0.19.0/ufs2arco/targets/__init__.py)
- [ufs2arco-0.19.0/ufs2arco/driver.py](../ufs2arco-0.19.0/ufs2arco/driver.py)

### Why

This target was needed because the existing Anemoi target produces the wrong structural layout for CREDIT WRF.

### What `CreditWRF` does

It preserves:

- separate variables instead of stacking them into a single array,
- native `y/x` grid,
- upper-air variables as `(time, level, y, x)`,
- 2-D variables as `(time, y, x)`,
- `latitude` and `longitude` as coordinates.

### Key constraints

The target currently enforces:

- no forecast-hour dimension,
- exactly one member per conversion run.

That matches the current per-case/per-member archive design.

### Important forcing fix

Later, this target was also updated so that computed forcing fields always come out in a CREDIT-friendly shape.

For example, if a forcing was originally:

- `(time,)`, or
- `(y, x)`, or
- `(y, x, time)`

it is now broadcast/transposed into:

- `(time, y, x)`

This was necessary so CREDIT could consume the dynamic forcing fields consistently.

---

## 5. Interior conversion workflow added in `ufs2arco`

## 5.1 Base YAML for interior conversion

New file:

- [ufs2arco-0.19.0/wrf_preprocess/wofs_credit_wrf_raw.yaml](../ufs2arco-0.19.0/wrf_preprocess/wofs_credit_wrf_raw.yaml)

### Purpose

This is the base recipe for converting raw WoFS files into CREDIT-friendly per-case archives.

It defines:

- source = `wrf_raw`
- target = `credit_wrf`
- selected upper-air variables
- selected surface variables
- forcing variables
- one-member conversion pattern

---

## 5.2 Single-run shell scripts

New files:

- [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf.sh)
- [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_smoketest.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_smoketest.sh)

### Purpose

These scripts provide reproducible conversion entry points with the right environment activation and MPI usage.

The smoketest version writes a small test archive into `/tmp` for rapid verification.

---

## 5.3 Batch conversion script

New file:

- [ufs2arco-0.19.0/wrf_preprocess/batch_convert_credit_wrf.py](../ufs2arco-0.19.0/wrf_preprocess/batch_convert_credit_wrf.py)

Helper runner:

- [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_batch.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_batch.sh)

### Purpose

This scans available WoFS cases under the raw source directory and launches one conversion per:

- year
- date
- init time
- member

### Important arguments

It supports filtering by:

- `--years`
- `--dates`
- `--init-times`
- `--member-ids`

### Output pattern

Each result is written like:

- `wofs_20190429_0000_mem01.zarr`

This is the main interior archive format now used for CREDIT adaptation.

---

## 6. Interior statistics workflow added in `ufs2arco`

## 6.1 Stats generator

New file:

- [ufs2arco-0.19.0/wrf_preprocess/compute_credit_stats.py](../ufs2arco-0.19.0/wrf_preprocess/compute_credit_stats.py)

Helper runner:

- [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_stats.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_stats.sh)

### Purpose

This script computes CREDIT-style normalization files:

- `mean.nc`
- `std.nc`

and optionally a placeholder latitude-weights file.

### How statistics are computed

For each variable:

- if it has a `level` dimension, mean/std are computed per level,
- otherwise mean/std are scalar values.

### Date/file selection

The stats script was extended to support:

- `--glob`
- `--start-date`
- `--end-date`

So the statistics can be computed on a chosen training subset only.

### Parallelization and memory handling

The stats script was also upgraded to use a local Dask cluster with options such as:

- `--n-workers`
- `--threads-per-worker`
- `--memory-limit`
- `--files-per-task`

This keeps memory usage manageable by processing small file batches and merging partial sums/counts, instead of loading every case archive at once.

### Important filename support update

The date-filtering logic was extended so it can recognize both:

- `wofs_YYYYMMDD_HHMM_memNN.zarr`
- `wofs_boundary_YYYYMMDD_HHMM_memNN.zarr`

This allows the same stats script to be reused for both interior and boundary products.

---

## 7. What “boundary” means in this work

This needs explicit explanation.

### What CREDIT means by boundary

In the current CREDIT WRF code path, the model expects an additional input called boundary or outside-domain input.

The loader builds a dataset called `boundary_input`, and later the transform path turns it into:

- `x_boundary`
- `x_surf_boundary`

This happens in:

- [credit/datasets/wrf_singlestep.py](../credit/datasets/wrf_singlestep.py)
- [credit/transforms/transforms_wrf.py](../credit/transforms/transforms_wrf.py)
- [credit/trainers/trainerWRF.py](../credit/trainers/trainerWRF.py)

### What the older generic loader assumed

The current loader does not just read arbitrary times. It takes the WRF target time and rounds it to the next 3-hour mark using CREDIT’s existing logic.

That meant the boundary data needed to exist on a time grid compatible with those expected outside-domain lookup times.

### What boundary means in the preferred WoFS implementation

For the preferred WoFS path, boundary is not read from a separate dataset.

Instead:

- `x_boundary` is built from `time=0` of the same interior WoFS case zarr,
- `x_surf_boundary` is built from the same-file surface state at `time=0`,
- the conditioning branch stays fixed while the forecast state evolves.

This keeps the conditioning semantics explicit without requiring separate boundary archives.

### What boundary means in the legacy compatibility implementation

At this stage, boundary is implemented as a **WoFS-derived boundary proxy dataset**.

That means:

- it is extracted from the already converted WoFS case archive,
- it keeps only a subset of variables intended for the outside-domain path,
- it keeps only timestamps aligned with a 3-hour cadence,
- it is written as a separate zarr file per case-member.

### What this is **not**

This is **not yet** a scientifically distinct parent-model or larger-domain lateral boundary dataset.

It is instead a structural compatibility layer that satisfies the current CREDIT WRF interface.

So at the moment, “boundary” means:

- an outside-input dataset in the format CREDIT expects,
- derived from the same WoFS archive,
- usable for software integration and end-to-end testing,
- but still subject to later scientific refinement depending on how true external boundary information should be defined for your experiment.

---

## 8. Legacy boundary extraction workflow kept in `ufs2arco`

## 8.1 Boundary extractor

New file:

- [ufs2arco-0.19.0/wrf_preprocess/extract_credit_boundary.py](../ufs2arco-0.19.0/wrf_preprocess/extract_credit_boundary.py)

Helper runner:

- [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_boundary.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_boundary.sh)

### Purpose

This script takes the interior case archives and writes separate boundary archives for the older proxy-boundary CREDIT path.

### Inputs

It accepts:

- `--glob`
- `--output-dir`
- `--start-date`
- `--end-date`
- `--variables`
- `--surface-variables`
- `--interval-hours`

### Output naming

Boundary archives are written like:

- `wofs_boundary_YYYYMMDD_HHMM_memNN.zarr`

### What it keeps

It selects only requested boundary variables, for example:

- upper-air: `T`, `QVAPOR`, `U`, `V`, `W`, `GEOPOT`
- surface: `T2`, `Q2`, `U10`, `V10`

### What it does in time

It subselects only exact boundary cadence times, currently defaulting to every 3 hours.

This aligns with the current CREDIT WRF boundary lookup behavior.

---

## 8.2 Boundary statistics workflow

New helper script:

- [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_boundary_stats.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_boundary_stats.sh)

### Purpose

This computes:

- boundary `mean.nc`
- boundary `std.nc`

using the same stats generator as the interior dataset.

This was necessary because the older boundary workflow loaded separate normalization files for:

- interior variables
- boundary variables

---

## 9. CREDIT-side configuration and adaptation work

## 9.1 CREDIT config template added

New file:

- [miles-credit-wofs/config/wofs_credit_wrf_template.yml](../config/wofs_credit_wrf_template.yml)

### Purpose

This is the starter WoFS-to-CREDIT config template.

It includes:

- interior variable lists,
- boundary variable lists,
- paths for interior archives,
- paths for interior stats,
- boundary stat reuse via interior mean/std,
- a WRF model configuration.

It now defaults to the preferred WoFS `t=0` conditioning path, so it no longer includes legacy boundary zarr paths or separate boundary mean/std paths.

### Important model structure fix

The WRF model path expects nested model parameters:

- `model.param_interior`
- `model.param_outside`

rather than placing all model shape settings at the top level.

The config template was updated to use that correct structure.

---

## 9.2 CREDIT validation script added

New file:

- [miles-credit-wofs/applications/test_wofs_credit_wrf.py](../applications/test_wofs_credit_wrf.py)

### Purpose

This script loads:

- the config,
- the WoFS interior archives,
- the boundary archives,
- the WRF transforms,
- a sample from `WRFDataset`

and prints the resulting tensor shapes.

This was added to provide a quick compatibility test before trying full training.

---

## 9.3 CREDIT transform bug fixed

File modified:

- [miles-credit-wofs/credit/transforms/transforms_wrf.py](../credit/transforms/transforms_wrf.py)

### Problem

`ToTensorWRF` assumed `flag_static_first` only existed when forcing/static inputs existed in a certain way.

In the WoFS setup, dynamic forcing variables exist even when separate forcing/static datasets do not.

That caused two issues:

1. `flag_static_first` could be missing,
2. dynamic-forcing-only input cases might fail to produce `x_forcing_static` correctly.

### Fix

The transform code was updated so that:

- `flag_static_first` is always defined,
- dynamic forcing alone is enough to make `has_forcing_static = True`.

This is what allowed the WoFS dynamic forcing channels to flow correctly into CREDIT.

---

## 10. Validation results so far

## 10.1 Interior dataset validation

Earlier smoke tests confirmed that the converted interior WoFS archives contain expected dimensions and variables.

Example validated structure:

- `time: 37`
- `y: 300`
- `x: 300`
- `level: 9`

and variables such as:

- `T`
- `QVAPOR`
- `U`
- `V`
- `W`
- `GEOPOT`
- `T2`
- `Q2`
- `COMPOSITE_REFL_10CM`
- forcing variables like `cos_latitude`, `sin_local_time`, etc.

---

## 10.2 Legacy boundary extraction validation

A smoke test boundary archive was generated successfully.

Validated boundary archive properties included:

- dimensions: `time=3`, `level=9`, `y=300`, `x=300`
- times aligned to the 3-hour boundary cadence
- boundary variables included:
  - `GEOPOT`
  - `Q2`
  - `QVAPOR`
  - `T`
  - `T2`
  - `U`
  - `U10`
  - `V`
  - `V10`
  - `W`

---

## 10.3 Legacy boundary stats validation

Boundary `mean.nc` and `std.nc` were computed successfully from the extracted boundary archive.

The resulting stats files had the expected variable set and level-aware statistics for upper-air variables.

---

## 10.4 CREDIT sample loading validation

Using the `credit` conda environment, the adapted WoFS sample was loaded successfully through:

- `WRFDataset`
- `NormalizeWRF`
- `ToTensorWRF`

The resulting tensors included:

- `x_forcing_static: (1, 10, 300, 300)`
- `x_surf: (1, 4, 300, 300)`
- `x: (1, 6, 9, 300, 300)`
- `y_diag: (1, 6, 300, 300)`
- `y_surf: (1, 4, 300, 300)`
- `y: (1, 6, 9, 300, 300)`
- `x_boundary: (1, 6, 9, 300, 300)`
- `x_surf_boundary: (1, 4, 300, 300)`
- `x_time_encode: (12,)`

This was the key early proof that the interior-plus-boundary proxy path could map into the tensor structure CREDIT WRF expects.

---

## 10.5 CREDIT model forward-pass validation

A single forward pass through the WRF model was also tested on CPU.

That forward pass succeeded and produced:

- `y_pred: (1, 64, 1, 300, 300)`

This shows that the adapted WoFS tensors are not only loadable, but also structurally consumable by the current WRF model implementation.

---

## 11. Current workflow summary

At this point, the implemented workflow is:

### Step 1: Convert raw WoFS to interior CREDIT-WRF case archives

Use:

- [ufs2arco-0.19.0/wrf_preprocess/batch_convert_credit_wrf.py](../ufs2arco-0.19.0/wrf_preprocess/batch_convert_credit_wrf.py)
- or [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_batch.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_batch.sh)

Produces:

- `wofs_YYYYMMDD_HHMM_memNN.zarr`

### Step 2: Compute interior stats

Use:

- [ufs2arco-0.19.0/wrf_preprocess/compute_credit_stats.py](../ufs2arco-0.19.0/wrf_preprocess/compute_credit_stats.py)
- or [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_stats.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_stats.sh)

Produces:

- interior `mean.nc`
- interior `std.nc`
- optional latitude-weights placeholder

### Step 3: Point CREDIT to the interior archives and interior stats

Use:

- [miles-credit-wofs/config/wofs_credit_wrf_template.yml](../config/wofs_credit_wrf_template.yml)

with:

- same-file `t=0` conditioning
- `data.boundary.reuse_interior_stats: True`
- no boundary zarr paths
- no boundary mean/std paths

### Step 4: Validate the WoFS-specific `t=0` loaders

Use:

- [miles-credit-wofs/applications/test_wofs_credit_wrf_t0.py](../applications/test_wofs_credit_wrf_t0.py)
- [miles-credit-wofs/applications/test_wofs_credit_wrf_t0_multi.py](../applications/test_wofs_credit_wrf_t0_multi.py)

### Step 5: Run WoFS-specific training smoke tests

Use:

- [miles-credit-wofs/applications/train_wrf_wofs.py](../applications/train_wrf_wofs.py)
- [miles-credit-wofs/applications/train_wrf_wofs_multi.py](../applications/train_wrf_wofs_multi.py)

### Legacy optional step

If the older proxy-boundary interface is ever needed again, the boundary extractor and boundary-stats scripts remain available as compatibility tools, but they are no longer part of the preferred WoFS workflow.

---

## 12. What is still incomplete

Although the software integration path is now working, some important scientific and workflow questions remain.

### 12.1 Preferred conditioning is settled, but the legacy proxy-boundary path remains only a compatibility fallback

The preferred WoFS path is now the same-file `t=0` conditioning design.

If a future experiment requires true parent-model or true outside-domain information, that would be a new design task rather than a continuation of the current preferred path.

### 12.2 Full production training has not yet been run end-to-end

The dataset path, CPU forward path, and one-batch WoFS-specific training smoke tests have been validated, but a larger real training run has not yet been completed.

### 12.3 The microphysics-correction objective is not yet encoded as the final target setup

The current work is focused on dataset and loader compatibility.

The later scientific objective involving quantities like:

- `delta REFL_10`
- `delta QRAIN`
- `delta QNRAIN`

still needs a dedicated target-design step after the data plumbing is fully stable.

---

## 13. Main takeaway

So far, the work has successfully built a working bridge from raw WoFS data through `ufs2arco` into the current CREDIT WRF software path.

That bridge now includes:

- a dedicated WoFS-to-CREDIT target,
- per-case/per-member interior archives,
- scalable stats generation,
- WoFS-specific same-file `t=0` loaders,
- WoFS-specific single-step and multi-step training entry points,
- interior-stat reuse for conditioning normalization,
- a corrected CREDIT transform path,
- validated WoFS-specific dataset loading,
- validated WRF model forward compatibility.

The legacy proxy-boundary product is still available for compatibility, but it is no longer the recommended WoFS workflow.

---

## 13. WoFS-specific `t=0` conditioning path added in CREDIT

After the initial proxy-boundary integration, a WoFS-specific conditioning path was added directly in CREDIT.

### Motivation

The proxy boundary zarrs were structurally useful but scientifically awkward for WoFS.

The new interpretation is:

- `t=0` is the anchor analysis,
- `x_boundary` is that same-file anchor state,
- the model learns forecast evolution conditioned on the anchor.

### New files

Added:

- [miles-credit-wofs/credit/datasets/wrf_wofs_singlestep.py](../credit/datasets/wrf_wofs_singlestep.py)
- [miles-credit-wofs/credit/datasets/wrf_wofs_multistep.py](../credit/datasets/wrf_wofs_multistep.py)
- [miles-credit-wofs/applications/test_wofs_credit_wrf_t0.py](../applications/test_wofs_credit_wrf_t0.py)
- [miles-credit-wofs/applications/test_wofs_credit_wrf_t0_multi.py](../applications/test_wofs_credit_wrf_t0_multi.py)
- [miles-credit-wofs/config/wofs_credit_wrf_t0_test_tmp.yml](../config/wofs_credit_wrf_t0_test_tmp.yml)
- [miles-credit-wofs/config/wofs_credit_wrf_t0_multi_test_tmp.yml](../config/wofs_credit_wrf_t0_multi_test_tmp.yml)

Modified:

- [miles-credit-wofs/credit/datasets/__init__.py](../credit/datasets/__init__.py)

### Single-step behavior

The WoFS single-step loader now:

- reads only the interior case zarrs,
- builds `boundary_input` from `time=0` of the same file,
- keeps the same output structure expected by `NormalizeWRF`, `ToTensorWRF`, and the current WRF model.

Validated outputs included:

- `x_boundary: (1, 6, 9, 300, 300)`
- `x_surf_boundary: (1, 4, 300, 300)`

and a CPU forward pass succeeded.

### Multi-step behavior

The WoFS multi-step loader now:

- keeps `x_boundary` fixed from same-file `t=0` throughout the rollout,
- advances `forecast_step` exactly as the existing multi-step trainer expects,
- preserves trainer/model compatibility without using external boundary zarr files.

### Optional learning-start offset

The WoFS-specific path now supports:

- `data.target_start_step: 1`
- `data.target_start_step: 2`

This controls the first supervised target lead while keeping `t=0` as the conditioning state.

For `history_len: 1`:

- `1` means first pair is `t0 -> t1`
- `2` means first pair is `t1 -> t2`

This gives explicit control over whether very early forecast leads are included in learning.

### Validation

The multi-step smoke test verifies:

- forecast-step progression,
- constant `x_boundary` across the rollout,
- changing `x` across rollout steps,
- CPU forward compatibility with the current WRF model.

### WoFS-specific multi-step training app

To move beyond dataset-only validation, a dedicated WoFS multi-step training application was added:

- [miles-credit-wofs/applications/train_wrf_wofs_multi.py](../applications/train_wrf_wofs_multi.py)

This keeps the generic [applications/train_wrf_multi.py](../applications/train_wrf_multi.py) untouched while replacing only the dataset-loading portion with `WoFSMultiStep`.

A minimal training smoke-test config was also added:

- [miles-credit-wofs/config/wofs_credit_wrf_t0_multi_train_smoke.yml](../config/wofs_credit_wrf_t0_multi_train_smoke.yml)

That config uses:

- `trainer.type: multi-step-wrf`
- `batches_per_epoch: 1`
- `skip_validation: True`

so the first trainer-loop smoke test can be run with minimal risk.

That smoke test was run successfully and completed:

- one training epoch
- one training batch group
- multi-step forecast length `2`
- checkpoint save to the configured `save_loc`

Example observed metrics from the smoke test were:

- `train_loss: 3.315179`
- `train_acc: 0.000776`
- `train_mae: 0.926052`

### WoFS-specific single-step training app

For symmetry with the multi-step path, a dedicated WoFS single-step training application was also added:

- [miles-credit-wofs/applications/train_wrf_wofs.py](../applications/train_wrf_wofs.py)

This mirrors the generic [applications/train_wrf.py](../applications/train_wrf.py) path, but swaps the dataset-loading logic to use:

- [miles-credit-wofs/credit/datasets/wrf_wofs_singlestep.py](../credit/datasets/wrf_wofs_singlestep.py)

A minimal single-step training smoke config was added as well:

- [miles-credit-wofs/config/wofs_credit_wrf_t0_train_smoke.yml](../config/wofs_credit_wrf_t0_train_smoke.yml)

This gives a dedicated single-step trainer entry point for the same-file `t=0` conditioning workflow.

That single-step smoke test was also run successfully and completed:

- one training epoch
- one training batch
- checkpoint save to the configured `save_loc`

Example observed metrics from the smoke test were:

- `train_loss: 1.664369`
- `train_acc: -0.012526`
- `train_mae: 0.999422`

During this smoke test, a small trainer fix was also made in [miles-credit-wofs/credit/trainers/trainerWRF.py](../credit/trainers/trainerWRF.py) so metric/loss aggregation is CPU-safe and does not unconditionally call CUDA.

### Role of `extract_credit_boundary.py` after the `t=0` transition

Under the original proxy-boundary workflow, [ufs2arco-0.19.0/wrf_preprocess/extract_credit_boundary.py](../ufs2arco-0.19.0/wrf_preprocess/extract_credit_boundary.py) was required to create boundary zarr inputs for CREDIT.

Under the current WoFS-specific `t=0` workflow, that is no longer true.

Current status:

- **not needed** for dataset loading,
- **not needed** by the new WoFS-specific single-step or multi-step training apps,
- no longer needed for boundary statistics either when reusing interior stats.

But semantically, the current extractor is tied to the older 3-hour proxy-boundary interpretation, not the newer fixed-`t=0` conditioning interpretation.

So the recommended view is:

- keep it as a legacy compatibility tool,
- do not treat it as part of the preferred WoFS `t=0` training path,
- reuse interior stats for the conditioning branch by setting:
  - `data.boundary.reuse_interior_stats: True`

This behavior is now implemented in [miles-credit-wofs/credit/transforms/transforms_wrf.py](../credit/transforms/transforms_wrf.py), so WoFS-specific configs no longer need:

- `data.boundary.mean_path`
- `data.boundary.std_path`

This interior-stat reuse path was validated successfully with the WoFS-specific configs by re-running:

- single-step dataset smoke test
- multi-step dataset smoke test
- single-step training smoke test
- multi-step training smoke test

Example observed training metrics after enabling reuse were:

- single-step: `train_loss: 1.661551`, `train_acc: -0.012554`, `train_mae: 0.998365`
- multi-step: `train_loss: 3.024040`, `train_acc: 0.024911`, `train_mae: 0.887842`

---

## 14. Files added or modified so far

### In `ufs2arco`

Modified:

- [ufs2arco-0.19.0/ufs2arco/sources/wrf_raw_source.py](../ufs2arco-0.19.0/ufs2arco/sources/wrf_raw_source.py)
- [ufs2arco-0.19.0/ufs2arco/targets/__init__.py](../ufs2arco-0.19.0/ufs2arco/targets/__init__.py)
- [ufs2arco-0.19.0/ufs2arco/driver.py](../ufs2arco-0.19.0/ufs2arco/driver.py)
- [ufs2arco-0.19.0/wrf_preprocess/compute_credit_stats.py](../ufs2arco-0.19.0/wrf_preprocess/compute_credit_stats.py)

Added:

- [ufs2arco-0.19.0/ufs2arco/targets/credit_wrf.py](../ufs2arco-0.19.0/ufs2arco/targets/credit_wrf.py)
- [ufs2arco-0.19.0/wrf_preprocess/wofs_credit_wrf_raw.yaml](../ufs2arco-0.19.0/wrf_preprocess/wofs_credit_wrf_raw.yaml)
- [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf.sh)
- [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_smoketest.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_smoketest.sh)
- [ufs2arco-0.19.0/wrf_preprocess/batch_convert_credit_wrf.py](../ufs2arco-0.19.0/wrf_preprocess/batch_convert_credit_wrf.py)
- [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_batch.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_batch.sh)
- [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_stats.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_stats.sh)
- [ufs2arco-0.19.0/wrf_preprocess/extract_credit_boundary.py](../ufs2arco-0.19.0/wrf_preprocess/extract_credit_boundary.py)
- [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_boundary.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_boundary.sh)
- [ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_boundary_stats.sh](../ufs2arco-0.19.0/wrf_preprocess/run_credit_wrf_boundary_stats.sh)

### In `miles-credit-wofs`

Modified:

- [miles-credit-wofs/credit/transforms/transforms_wrf.py](../credit/transforms/transforms_wrf.py)
- [miles-credit-wofs/config/wofs_credit_wrf_template.yml](../config/wofs_credit_wrf_template.yml)

Added:

- [miles-credit-wofs/applications/test_wofs_credit_wrf.py](../applications/test_wofs_credit_wrf.py)
- [miles-credit-wofs/config/wofs_credit_wrf_test_tmp.yml](../config/wofs_credit_wrf_test_tmp.yml)

---

## 15. Suggested next step after this summary

The next logical implementation slice is:

1. run a larger-scale WoFS-specific training experiment through [applications/train_wrf_wofs.py](../applications/train_wrf_wofs.py) or [applications/train_wrf_wofs_multi.py](../applications/train_wrf_wofs_multi.py),
2. verify stability and throughput on the preferred same-file `t=0` path,
3. then start adapting the actual microphysics-correction target design.
