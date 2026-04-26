# Ursa Rollout Launchers

This document summarizes rollout launch scripts in [scripts](scripts) for NOAA Ursa.

## 1) Standard rollout / metrics launcher

Script: [scripts/ursa_rollout.sh](scripts/ursa_rollout.sh)

Supported apps:
- APP=rollout -> [applications/rollout_wrf_wofs.py](applications/rollout_wrf_wofs.py)
- APP=metrics -> [applications/rollout_metrics_wrf_wofs.py](applications/rollout_metrics_wrf_wofs.py)

Defaults:
- MODE=ddp
- CONFIG=/home/Zhanxiang.Hua/miles-credit-wofs/config/ursa_wofscast_credit_wrf_latest.yml

Examples:

```bash
sbatch scripts/ursa_rollout.sh
sbatch --export=ALL,APP=metrics,MODE=ddp,NPROC_PER_NODE_OVERRIDE=2 scripts/ursa_rollout.sh
sbatch --export=ALL,APP=rollout,MODE=none,MAX_CASES=1 scripts/ursa_rollout.sh
```

## 2) Ensemble rollout launcher

Script: [scripts/ursa_ens_rollout.sh](scripts/ursa_ens_rollout.sh)

Runs:
- [applications/rollout_wrf_wofs_ensemble.py](applications/rollout_wrf_wofs_ensemble.py)

Defaults:
- MODE=ddp
- CONFIG=/home/Zhanxiang.Hua/miles-credit-wofs/config/ursa_wofscast_credit_wrf_latest_ensemble.yml
- OUTPUT_NAME=rollout_metrics_ensemble.csv

Supported submit-time variables:
- MODE: none | ddp | fsdp
- BACKEND: nccl | gloo | mpi
- NPROC_PER_NODE_OVERRIDE: integer
- MAX_CASES: integer
- ENSEMBLE_SIZE: integer
- OUTPUT_NAME: file name for merged metrics CSV
- MEAN_ONLY: 0/1 (1 adds --mean-only)
- DISABLE_CRPS: 0/1 (1 adds --disable-crps)
- CONFIG: optional full path override

Examples:

```bash
sbatch scripts/ursa_ens_rollout.sh
sbatch --export=ALL,MODE=ddp,ENSEMBLE_SIZE=18,MAX_CASES=30 scripts/ursa_ens_rollout.sh
sbatch --export=ALL,MODE=ddp,ENSEMBLE_SIZE=8,MEAN_ONLY=1,DISABLE_CRPS=1,OUTPUT_NAME=rollout_metrics_ensemble_smoke.csv scripts/ursa_ens_rollout.sh
sbatch --export=ALL,MODE=none,MAX_CASES=1,ENSEMBLE_SIZE=2 scripts/ursa_ens_rollout.sh
```