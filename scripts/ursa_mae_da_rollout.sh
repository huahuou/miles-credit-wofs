#!/bin/bash
#===============================================================================
# Single-Node GPU DA Rollout Script for WoFS DiffMAE on NOAA Ursa
#
# System: Ursa (NESCC, Fairmont WV)
#   - GPU Partition: u1-h100 (58 nodes, 2x NVIDIA H100-NVL per node, 94 GB each)
#   - CPU: AMD Genoa 9654, 192 cores/node
#   - Interconnect: NDR-200 InfiniBand
#   - Scheduler: Slurm
#
# Runs rollout_wrf_wofs_mae_da.py — loads a trained WoFSDiffMAE checkpoint
# and produces per-case precip analysis zarr stores over a date range.
#
# Usage:
#   sbatch ursa_mae_da_rollout.sh
#
# Override CONFIG, CHECKPOINT, START_DATE, END_DATE, OUT_DIR, MAX_FILES, and
# MAX_TIMES with environment variables or edit defaults below.
#===============================================================================

#----- Slurm Directives --------------------------------------------------------
#SBATCH --job-name=credit-wofs-diffmae-rollout
#SBATCH --account=gpu-ai4wp
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/home/Zhanxiang.Hua/job_log/%x-%j.out
#SBATCH --error=/home/Zhanxiang.Hua/job_log/%x-%j.err
#SBATCH --exclusive

#----- User Configuration ------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-credit-wofs}"
PROJECT_DIR="${PROJECT_DIR:-/home/Zhanxiang.Hua/miles-credit-wofs}"
CONFIG="${CONFIG:-/scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/credit_runs/wofs_diffmae_pretrain3/model.yml}"
ROLLOUT_SCRIPT="${ROLLOUT_SCRIPT:-applications/rollout_wrf_wofs_mae_da.py}"

# Path to the trained checkpoint (.pt file). Leave empty to let the rollout
# script load ${save_loc}/checkpoint.pt from the config.
CHECKPOINT="${CHECKPOINT:-/scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/credit_runs/wofs_diffmae_pretrain3/best_checkpoint.pt}"

# Date range to process (YYYYMMDD, inclusive).
START_DATE="${START_DATE:-20210415}"
END_DATE="${END_DATE:-20210530}"

# Directory where analysis .zarr stores will be written.
OUT_DIR="${OUT_DIR:-/scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/credit_rollouts/wofs_diffmae_pretrain3/test1}"

# Optional smoke-test limits. Empty means no explicit limit.
MAX_FILES="${MAX_FILES:-10}"
MAX_TIMES="${MAX_TIMES:-}"

EVAL_MODE="${EVAL_MODE:-none}"
BACKEND="${BACKEND:-nccl}"

#----- Load Modules ------------------------------------------------------------
source $MODULESHOME/init/bash
module load rdhpcs-conda
module load cuda/12.8
conda activate ${CONDA_ENV}

#----- Environment -------------------------------------------------------------
export NCCL_DEBUG=INFO
export LOGLEVEL=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=PBH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=1

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
echo "MAX_FILES            = ${MAX_FILES}"
echo "MAX_TIMES            = ${MAX_TIMES}"
echo "EVAL_MODE            = ${EVAL_MODE}"
echo "BACKEND              = ${BACKEND}"
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

CMD=(python "${ROLLOUT_SCRIPT}" \
    -c "${CONFIG}" \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --out-dir "${OUT_DIR}" \
    --mode "${EVAL_MODE}" \
    --backend "${BACKEND}")

if [ -n "${CHECKPOINT}" ]; then
    CMD+=(--checkpoint "${CHECKPOINT}")
fi
if [ -n "${MAX_FILES}" ]; then
    CMD+=(--max-files "${MAX_FILES}")
fi
if [ -n "${MAX_TIMES}" ]; then
    CMD+=(--max-times "${MAX_TIMES}")
fi

echo "Launching: ${CMD[*]}"

srun --cpu-bind=none "${CMD[@]}"

status=$?
echo "DiffMAE DA rollout finished with exit code: ${status}"
exit ${status}
