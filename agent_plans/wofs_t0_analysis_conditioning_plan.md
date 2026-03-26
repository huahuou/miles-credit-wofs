# Executable plan: WoFS `t=0` analysis as sequence-conditioning boundary

This plan converts the current idea into an implementation path that can be executed in small testable slices.

The main idea is:

- for each WoFS forecast zarr, use `time=0` as the sequence anchor,
- treat that `t=0` analysis state as the conditioning tensor currently called `boundary_input` / `x_boundary`,
- learn the forecast dynamics for later times from that fixed anchor.

This keeps the current CREDIT WRF model interface mostly intact, while replacing the current proxy outside-boundary interpretation with a WoFS-specific sequence-conditioning interpretation.

---

## 1. Target behavior

For a single WoFS case file with times:

- `t0` = analysis
- `t1, t2, ..., tN` = forecast states

we want training samples to behave like this.

### Single-step case

For a sample centered at forecast step `tk`:

- `x` = model input history ending at `tk`
- `y` = next target state after `x`
- `x_boundary` = always `t0` analysis from the same zarr
- `x_surf_boundary` = surface variables from `t0`
- `x_time_encode` = encode current input time and target time, plus either:
  - `t0` again as the conditioning time, or
  - a WoFS-specific time encoding layout if we later decide to change it

### Multi-step case

Within one sequence rollout:

- `x_boundary` stays fixed at `t0`
- `x` evolves step by step
- the model learns forecast evolution conditioned on the original analysis

---

## 2. Why this plan is preferable to the current proxy boundary

The current extracted boundary approach is structurally compatible with CREDIT but semantically weak for WoFS.

The new plan is preferable because:

1. WoFS naturally has a sequence anchor at `t=0`
2. no fake outside-domain product is required
3. it matches the idea of forecast drift from an analyzed initial condition
4. the current model can still receive `x_boundary` without a major API rewrite

---

## 3. Recommended implementation strategy

The recommended strategy is **not** to rewrite the generic boundary-aware CREDIT WRF path in place.

Instead, add a WoFS-specific path that reuses as much of the existing WRF machinery as possible.

### Recommendation

Create WoFS-specific dataset code first, and keep the current generic WRF path untouched.

That means new files such as:

- `credit/datasets/wrf_wofs_singlestep.py`
- `credit/datasets/wrf_wofs_multistep.py`
- optionally later a WoFS-specific trainer if needed

The reason is that the current generic WRF path has a strong assumption that the boundary input is an external parallel dataset.

---

## 4. Minimal implementation slices

## Slice 1: WoFS single-step dataset with `t=0` conditioning

### Goal

Add a WoFS-specific single-step dataset that:

- reads only the interior WoFS case zarr files,
- constructs `boundary_input` by taking `isel(time=0)` from the same file,
- keeps the current tensor output structure compatible with existing transforms and trainers.

### Proposed file

- `credit/datasets/wrf_wofs_singlestep.py`

### Concrete logic

For each file:

- open one unified WoFS zarr
- derive upper-air, surface, forcing, diagnostics exactly as before
- replace the old outside-file lookup with:
  - `ds_upper_boundary = upper_air_source.isel(time=slice(0, 1))`
  - `ds_surf_boundary = surf_source.isel(time=slice(0, 1))`
  - `boundary_input = xr.merge([...])`

### Important note

This should use the same variable names as the current `data.boundary.variables` and `data.boundary.surface_variables` config blocks, so downstream transforms do not need a large rewrite.

### Validation target

A smoke test should confirm:

- `x_boundary` exists
- `x_surf_boundary` exists
- shapes match the existing WRF trainer expectations

---

## Slice 2: WoFS multi-step dataset with fixed `t=0` conditioning

### Goal

Clone the same idea into the multi-step path.

### Proposed file

- `credit/datasets/wrf_wofs_multistep.py`

### Concrete logic

This should mirror the current [credit/datasets/wrfmultistep.py](../credit/datasets/wrfmultistep.py), except:

- remove dependence on separate outside files
- always build `boundary_input` from `time=0` of the same case file
- keep it fixed for every forecast step in the rollout

### Validation target

A smoke test should confirm:

- `forecast_step` progression still works
- `x_boundary` is constant across rollout steps from the same sequence

---

## Slice 3: Add loader entry points / application scripts

### Goal

Make the WoFS-specific datasets callable from simple scripts without disturbing the current generic WRF applications.

### Options

#### Option A: separate WoFS test scripts first

Add:

- `applications/test_wofs_credit_wrf_t0.py`

This is the lowest-risk step.

#### Option B: later add WoFS-specific train apps

Add:

- `applications/train_wrf_wofs.py`
- `applications/train_wrf_wofs_multi.py`

This is cleaner than adding conditionals all over the generic WRF apps.

### Recommended choice

Start with Option A, then decide whether a dedicated WoFS train app is needed.

---

## Slice 4: Decide whether to reuse or ignore boundary stats

### Current situation

The current normalization logic in [credit/transforms/transforms_wrf.py](../credit/transforms/transforms_wrf.py) expects separate boundary mean/std files.

### For WoFS `t=0` conditioning

There are two valid options:

#### Option 1: keep separate boundary stats

Create boundary stats from the `t=0` slices only.

Pros:
- least change to normalization code

Cons:
- one more preprocessing artifact

#### Option 2: reuse interior stats for boundary

Use the same mean/std for interior and `t=0` conditioning.

Pros:
- simpler conceptually

Cons:
- requires small transform/config changes

### Recommended first step

Keep separate boundary stats initially, because it avoids touching the normalizer too much.

But boundary stats should then be computed from **`t=0` analysis slices only**, not the old 3-hour proxy boundary.

---

## Slice 5: Replace current proxy-boundary extractor

### Goal

Stop using the current 3-hour cadence boundary extractor as the final WoFS design.

### Current script

- [../ufs2arco-0.19.0/wrf_preprocess/extract_credit_boundary.py](../ufs2arco-0.19.0/wrf_preprocess/extract_credit_boundary.py)

### Replacement idea

Either:

#### Option A: stop using this script for WoFS

and let the WoFS dataset loader derive `t=0` boundary in-memory from the same file.

#### Option B: create a new helper script

For example:

- `extract_credit_t0_boundary.py`

that writes one zarr per case/member containing only `time=0` boundary variables.

### Recommended choice

Option A first.

Reason:
- no extra files needed
- no duplicate storage
- simplest conceptually

---

## 5. Exact code modifications to make first

### Step 1
Create:

- `credit/datasets/wrf_wofs_singlestep.py`

Start from:

- [credit/datasets/wrf_singlestep.py](../credit/datasets/wrf_singlestep.py)

Modify only the boundary section so that instead of loading from `param_outside["filenames"]`, it uses `time=0` from the same interior dataset.

### Step 2
Create:

- `applications/test_wofs_credit_wrf_t0.py`

This should mirror:

- [applications/test_wofs_credit_wrf.py](../applications/test_wofs_credit_wrf.py)

but instantiate the new WoFS dataset class.

### Step 3
Create a temporary WoFS config for testing that still includes:

- `data.boundary.variables`
- `data.boundary.surface_variables`
- `data.boundary.mean_path`
- `data.boundary.std_path`

but does **not** need `data.boundary.save_loc` for the dataset path itself.

### Step 4
Smoke test:

- dataset loading
- transform output shapes
- CPU forward pass through the model

---

## 6. Testing checklist

### Test A: dataset sample test

Expected outputs:

- `x`
- `x_surf`
- `x_forcing_static`
- `y`
- `y_surf`
- `y_diag`
- `x_boundary`
- `x_surf_boundary`
- `x_time_encode`

### Test B: sequence consistency test

For two different sample indices from the same zarr:

- verify `x_boundary` is identical
- verify `x` changes with sample time

### Test C: forward pass

Run one CPU forward pass with the existing WRF model.

Expected result:

- model accepts tensors with no external boundary file lookup
- model produces `y_pred` with expected output shape

---

## 7. What not to change yet

Avoid changing these shared generic files unless necessary:

- `credit/trainers/trainerWRF.py`
- `credit/trainers/trainerWRF_multi.py`
- `credit/models/swin_wrf.py`
- `applications/train_wrf.py`
- `applications/train_wrf_multi.py`

Reason:
- the current model interface already supports a second conditioning tensor
- your WoFS `t=0` analysis can fit into that interface without modifying the shared trainer/model path immediately

---

## 8. Execution order

1. add `wrf_wofs_singlestep.py`
2. add `test_wofs_credit_wrf_t0.py`
3. run sample-shape smoke test
4. run CPU forward-pass smoke test
5. then add `wrf_wofs_multistep.py`
6. test multi-step constant-boundary behavior
7. only after that decide whether new WoFS training application scripts are needed

---

## 9. Success criterion

This plan is successful if:

- WoFS samples load through CREDIT without any external boundary zarr files,
- `x_boundary` corresponds to the same-file `t=0` analysis,
- the existing WRF model forward path works unchanged,
- the meaning of the conditioning branch is now scientifically interpretable for WoFS.
