## Plan: WoFS → CREDIT WRF Dataset

Goal: make WoFS conversion run through `ufs2arco`, but emit files that match the actual expectations of the WRF-specific CREDIT path, not the looser docs alone. The key discovery is that the current WoFS-to-Anemoi route is not compatible because [ufs2arco-0.19.0/ufs2arco/targets/anemoi.py](ufs2arco-0.19.0/ufs2arco/targets/anemoi.py) flattens the grid to `cell`, stacks all channels into a single `data` variable, and drops geolocation auxiliaries, while CREDIT’s WRF path expects ordinary xarray variables loaded from per-file datasets in [miles-credit-wofs/credit/datasets/wrf_singlestep.py](miles-credit-wofs/credit/datasets/wrf_singlestep.py). Because you chose the WRF-specific path, preserved native 300×300 local grids, flattened members to samples, and want boundary support included, the cleanest design is a new `ufs2arco` target dedicated to CREDIT-WRF files, plus a paired boundary product and separate interior/boundary statistics.

**Steps**
1. Define the real compatibility target around `WRFDataset`, `WRFPredict`, `NormalizeWRF`, and `ToTensorWRF` in [miles-credit-wofs/credit/datasets/wrf_singlestep.py](miles-credit-wofs/credit/datasets/wrf_singlestep.py), [miles-credit-wofs/credit/transforms/transforms_wrf.py](miles-credit-wofs/credit/transforms/transforms_wrf.py), and [miles-credit-wofs/applications/train_wrf.py](miles-credit-wofs/applications/train_wrf.py), not around [miles-credit-wofs/docs/source/prepare_new_dataset.md](miles-credit-wofs/docs/source/prepare_new_dataset.md) alone. The current loader behavior means maximal compatibility requires unified per-file datasets containing interior upper-air, surface, dynamic-forcing, and diagnostic variables together, because `WRFDataset` filters `all_ds` from one `filenames` list instead of truly reading separate surface/dynamic/diagnostic file lists.

2. Treat [ufs2arco-0.19.0/ufs2arco/sources/wrf_raw_source.py](ufs2arco-0.19.0/ufs2arco/sources/wrf_raw_source.py) as the correct preprocessing base. It already does the hard WoFS-native work: timestamp recovery, destaggering, `GEOPOT`, `RAIN_AMOUNT`, level subsetting, `y/x/level` renaming, and member-aware trajectory indexing. Plan the new output target to branch from this source output before [ufs2arco-0.19.0/ufs2arco/targets/anemoi.py](ufs2arco-0.19.0/ufs2arco/targets/anemoi.py) flattens and stacks fields.

3. Add a new `ufs2arco` target under [ufs2arco-0.19.0/ufs2arco/targets](ufs2arco-0.19.0/ufs2arco/targets) whose only job is to write CREDIT-WRF-ready stores. The target should preserve separate data variables, keep the unflattened 2-D grid, and write xarray datasets with:
   - interior upper-air vars on `(time, level, y, x)`
   - interior 2-D vars on `(time, y, x)`
   - boundary upper-air vars on `(time, level, y, x)`
   - boundary 2-D vars on `(time, y, x)`
   - auxiliary geolocation retained in a CREDIT-usable form
   - `time` kept continuous within each file only

4. Use per-trajectory, per-member files rather than yearly concatenation. This is a major design choice driven by CREDIT WRF loader behavior in [miles-credit-wofs/credit/datasets/wrf_singlestep.py](miles-credit-wofs/credit/datasets/wrf_singlestep.py): it slices contiguous windows by file-local index, so concatenating multiple WoFS cases into one yearly series risks invalid samples across case boundaries. The recommended file unit is one `(case/init/member)` store, with filenames still carrying the year so [miles-credit-wofs/applications/train_wrf.py](miles-credit-wofs/applications/train_wrf.py) can filter training and validation sets by year substring.

5. Make the interior file schema match current WRF loader quirks exactly:
   - one zarr per trajectory/member containing all interior vars needed by `param_interior`
   - same glob may be reused for `save_loc`, `save_loc_surface`, `save_loc_dynamic_forcing`, and `save_loc_diagnostic` if needed, because the current loader effectively expects a unified file
   - periodic/static inputs remain optional, but for your native-grid choice the better first-class inputs are dynamic geolocation forcings such as `cos_lat`, `sin_lat`, `cos_lon`, `sin_lon`, as you suggested, instead of trying to force a globally fixed lat/lon grid

6. Define the boundary product as a first-class parallel dataset, because the current WRF trainer and model always consume `x_boundary` in [miles-credit-wofs/credit/trainers/trainerWRF.py](miles-credit-wofs/credit/trainers/trainerWRF.py) and [miles-credit-wofs/credit/models/swin_wrf.py](miles-credit-wofs/credit/models/swin_wrf.py). Since the source is still undecided, Plan 1 should lock the interface even if the origin is unresolved:
   - separate boundary zarr files
   - same per-trajectory/per-member file granularity as interior
   - upper-air + boundary surface in the same file
   - separate boundary `mean_path` and `std_path`
   - explicit blocker note: boundary may come from parent/global model or be derived from WoFS context, but the schema must satisfy current CREDIT WRF expectations either way

7. Keep native 300×300 local grids, but stop treating Earth-fixed latitude/longitude as invariant coordinates across all cases. For maximal practical compatibility:
   - keep a stable local grid index basis for storage
   - feed geographic position as forcing-style variables
   - avoid dependence on the standard global latitude-weight machinery in [miles-credit-wofs/credit/losses/weighted_loss.py](miles-credit-wofs/credit/losses/weighted_loss.py) for this plan
   - document that this is a WRF-path dataset, not a drop-in global ERA5-style dataset

8. Define the interior variable mapping now, even if objective-specific targets come later. Recommended bucketing for Plan 1:
   - upper-air prognostics: 3-D WoFS state needed later for correction targets or conditioning
   - surface vars: 2-D near-surface/background fields
   - dynamic forcing vars: case-varying but input-only fields, including your geographic trig forcings and any lead-time-dependent conditioning fields if stored spatially
   - diagnostic vars: output-only quantities that later plans can repurpose for correction labels
   This keeps Plan 1 generic while not blocking Plan 2.

9. Separate statistics generation into two independent products:
   - interior `mean_path` and `std_path` covering all interior prognostic/surface/dynamic-forcing/diagnostic vars used by `NormalizeWRF`
   - boundary `mean_path` and `std_path` under `data.boundary`
   Static and pre-normalized forcing-style variables should stay outside z-score normalization where possible. The statistics writer should operate on the final CREDIT-WRF schema, not on Anemoi output.

10. Add a dedicated YAML config for the new target beside [ufs2arco-0.19.0/wrf_preprocess/wofs_anemoi_v3_raw.yaml](ufs2arco-0.19.0/wrf_preprocess/wofs_anemoi_v3_raw.yaml), likely a CREDIT-WRF-specific variant, and a matching run script beside [ufs2arco-0.19.0/wrf_preprocess/runbash.sh](ufs2arco-0.19.0/wrf_preprocess/runbash.sh). That config should describe:
    - interior variable selection
    - boundary variable selection
    - level selection
    - member flattening to independent outputs
    - output directory layout for interior, boundary, and stats

11. Produce a paired CREDIT training config for the WRF path under [miles-credit-wofs/config](miles-credit-wofs/config) later, with `scaler_type: std-wrf`, explicit `data.boundary`, and globs pointing at the new per-trajectory/member stores. The config should assume `forecast_len: 0` as the near-term default, while documenting that your later objective may open to next-$n$-step targets.

12. Treat two items as explicit handoff blockers for implementation:
    - boundary data origin is undecided
    - objective-specific label timing is not fixed, though current preference is next-step with possible extension to next-$n$ steps
    The plan should proceed with a generic interior/boundary WoFS state archive now so those choices do not block the file-format work.

**Verification**
- Smoke-test one interior store and one boundary store by loading them through `WRFDataset` in [miles-credit-wofs/credit/datasets/wrf_singlestep.py](miles-credit-wofs/credit/datasets/wrf_singlestep.py).
- Confirm `NormalizeWRF` and `ToTensorWRF` in [miles-credit-wofs/credit/transforms/transforms_wrf.py](miles-credit-wofs/credit/transforms/transforms_wrf.py) can produce `x`, `x_surf`, `x_forcing_static`, `x_boundary`, `x_surf_boundary`, `x_time_encode`, `y`, and optional `y_diag` without missing-variable errors.
- Validate that each file contains only one continuous trajectory so no sample window crosses case boundaries.
- Check that interior and boundary stats files cover exactly the variables referenced by `data.all_varnames` and `data.boundary.all_varnames`.
- Run a one-batch dry load through [miles-credit-wofs/applications/train_wrf.py](miles-credit-wofs/applications/train_wrf.py) before any full conversion run.
- Only after this passes should a full MPI conversion config be executed from [ufs2arco-0.19.0/wrf_preprocess](ufs2arco-0.19.0/wrf_preprocess).

**Decisions**
- Chose WRF-specific CREDIT path over the standard ERA5-style path.
- Chose native 300×300 local-grid storage over Earth-fixed regridding.
- Chose member flattening to independent samples/files.
- Chose generic WoFS state archive first; objective-specific labels remain for a later plan.
- Rejected reuse of the current Anemoi target as the main path because [ufs2arco-0.19.0/ufs2arco/targets/anemoi.py](ufs2arco-0.19.0/ufs2arco/targets/anemoi.py) destroys the variable/grid structure that CREDIT WRF loaders expect.
- Marked boundary source as the main unresolved blocker, but kept boundary schema in scope because current CREDIT WRF code requires it.
