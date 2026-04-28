#!/bin/bash
#===============================================================================
# Single-Node GPU DA Rollout Script for WoFS MAE on NOAA Ursa
#
# System: Ursa (NESCC, Fairmont WV)
#   - GPU Partition: u1-h100 (58 nodes, 2x NVIDIA H100-NVL per node, 94 GB each)
#   - CPU: AMD Genoa 9654, 192 cores/node
#   - Interconnect: NDR-200 InfiniBand
#   - Scheduler: Slurm
#
# Runs rollout_wrf_wofs_mae_da.py — loads a trained WoFSMultiModalMAE checkpoint
# and produces per-case precip analysis zarr.zip files over a date range.
#
# Usage:
#   sbatch ursa_mae_da_rollout.sh
#
# Edit CHECKPOINT, START_DATE, END_DATE, and OUT_DIR before submitting.
#===============================================================================

#----- Slurm Directives --------------------------------------------------------
#SBATCH --job-name=credit-wofs-mae-rollout
#SBATCH --account=gpu-ai4wp          # <-- Replace with your GPU project ID
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu                             # Use 'gpuwf' if you only have windfall access
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:1                # 1 GPU is sufficient for inference
#SBATCH --cpus-per-task=32                    # Modest CPU count — rollout is mostly GPU-bound
#SBATCH --mem=0                               # Use all available memory on the node
#SBATCH --time=04:00:00                       # Rollout is faster than training
#SBATCH --output=/home/Zhanxiang.Hua/job_log/%x-%j.out
#SBATCH --error=/home/Zhanxiang.Hua/job_log/%x-%j.err
#SBATCH --exclusive

#----- User Configuration (EDIT THESE) ----------------------------------------
CONDA_ENV="credit-wofs"
PROJECT_DIR="/home/Zhanxiang.Hua/miles-credit-wofs"
CONFIG="${PROJECT_DIR}/config/wofs_mae_da.yml"
ROLLOUT_SCRIPT="applications/rollout_wrf_wofs_mae_da.py"

# Path to the trained checkpoint (.pt file)
CHECKPOINT="/scratch5/purged/Zhanxiang.Hua/credit_runs/wofs_mae_pretrain_v1/best_checkpoint.pt"

# Date range to process (YYYYMMDD, inclusive)
START_DATE="20200415"
END_DATE="20200515"

# Directory where analysis zarr.zip files will be written
OUT_DIR="/scratch5/purged/Zhanxiang.Hua/credit_wofs_rollout_example/mae_da_test1"

#----- Load Modules ------------------------------------------------------------
source $MODULESHOME/init/bash

module load rdhpcs-conda
module load cuda/12.8

conda activate ${CONDA_ENV}

#----- Environment Variables ---------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use first available GPU
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "============================================================"
echo "SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME       = ${SLURM_JOB_NAME}"
echo "CONFIG               = ${CONFIG}"
echo "ROLLOUT_SCRIPT       = ${ROLLOUT_SCRIPT}"
echo "CHECKPOINT           = ${CHECKPOINT}"
echo "START_DATE           = ${START_DATE}"
echo "END_DATE             = ${END_DATE}"
echo "OUT_DIR              = ${OUT_DIR}"
echo "CONDA_ENV            = ${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

#----- CUDA preflight ---------------------------------------------------------
python - <<'PY'
import torch
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("No CUDA GPUs visible to PyTorch. Check CUDA_VISIBLE_DEVICES/Slurm GPU allocation.")
PY

#----- Launch DA Rollout -------------------------------------------------------
cd ${PROJECT_DIR}

srun --cpu-bind=none python ${ROLLOUT_SCRIPT} \
    -c ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --out-dir ${OUT_DIR}

echo "MAE DA rollout finished with exit code: $?"
