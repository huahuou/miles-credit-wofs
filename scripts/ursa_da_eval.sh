#!/bin/bash
#===============================================================================
# Multi-Node Multi-GPU Evaluation Script for miles-credit-wofs on NOAA Ursa
#
# System: Ursa (NESCC, Fairmont WV)
#   - GPU Partition: u1-h100 (58 nodes, 2x NVIDIA H100-NVL per node, 94 GB each)
#   - CPU: AMD Genoa 9654, 192 cores/node
#   - Interconnect: NDR-200 InfiniBand
#   - Scheduler: Slurm (not PBS — Ursa uses Slurm, unlike Derecho which uses PBS)
#
# Usage:
#   sbatch ursa_eval.sh
#
#===============================================================================

#----- Slurm Directives --------------------------------------------------------
#SBATCH --job-name=credit-wofs-da-eval
#SBATCH --account=gpu-ai4wp                    # <-- Replace with your GPU project ID
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu                             # Use 'gpuwf' if you only have windfall access
#SBATCH --nodes=1                             # Number of nodes (adjust as needed)
#SBATCH --ntasks-per-node=1                   # 1 Slurm task per node
#SBATCH --gpus-per-node=h100:2                # 2 H100 GPUs per node
#SBATCH --cpus-per-task=192                    # CPU cores per task (for data loading workers)
#SBATCH --mem=0                               # Use all available memory on the node
#SBATCH --time=02:59:00                       # Wall time limit
#SBATCH --output=/home/Zhanxiang.Hua/job_log/%x-%j.out                   # stdout: <job-name>-<job-id>.out
#SBATCH --error=/home/Zhanxiang.Hua/job_log/%x-%j.err                     # stderr: <job-name>-<job-id>.err
#SBATCH --exclusive                           # Exclusive node access for best GPU performance

#----- User Configuration (EDIT THESE) -----------------------------------------
NUM_NODES=$SLURM_JOB_NUM_NODES               # Automatically set from Slurm allocation
GPUS_PER_NODE=2                               # H100 nodes have 2 GPUs each
AUTO_DETECT_NPROC_PER_NODE=1                  # 1: use visible GPU count, 0: use GPUS_PER_NODE
NPROC_PER_NODE_OVERRIDE=""                    # Optional explicit override, e.g. "2"
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

CONDA_ENV="credit-wofs"                       # Name or path of your conda environment
PROJECT_DIR="/home/Zhanxiang.Hua/miles-credit-wofs"  # <-- Update path
CONFIG="/scratch5/purged/Zhanxiang.Hua/credit_runs/wofs_da_increment_experiment_0429_nophy/model.yml"
EVAL_SCRIPT="applications/eval_wrf_wofs_da_trainer_like.py"   # Evaluation script
SAVE_PHYSICAL="/scratch5/purged/Zhanxiang.Hua/credit_wofs_da_example/wofs_da_increment_experiment_0430/test1/eval_physical.zarr"                              # Path for physical-space Zarr store.
                                              # Leave empty to auto-derive from eval.save_zarr_path
                                              # (appends _physical.zarr suffix). Example:
                                              # SAVE_PHYSICAL="/scratch5/purged/Zhanxiang.Hua/credit_runs/wofs_da_increment_experiment_0423/eval_physical.zarr"

#----- Load Modules ------------------------------------------------------------
# Source the module system so module commands work inside batch scripts
source $MODULESHOME/init/bash

### module purge
module load rdhpcs-conda
module load cuda/12.8

# Activate the conda environment
conda activate ${CONDA_ENV}

#----- Environment Variables ---------------------------------------------------

# --- NCCL Configuration ---
export NCCL_DEBUG=INFO                        # Set to WARN in production to reduce log noise
export LOGLEVEL=INFO

# InfiniBand / network settings for Ursa's NDR-200 IB fabric
export NCCL_IB_DISABLE=0                      # Enable InfiniBand (Ursa has NDR-200 IB)
export NCCL_IB_GID_INDEX=3                    # Typical GID index for RoCE/IB
export NCCL_SOCKET_IFNAME=ib0                 # InfiniBand interface name (verify with `ip link`)
export NCCL_NET_GDR_LEVEL=PBH                 # GPU Direct RDMA level

# --- PyTorch / General ---
export OMP_NUM_THREADS=1                      # Prevent OpenMP thread oversubscription
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=1             # Helps debug hangs

# --- CUDA ---
# Honor scheduler-provided GPU binding when present; otherwise build "0,1,...,N-1".
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
    if [ "${GPUS_PER_NODE}" -lt 1 ]; then
        echo "GPUS_PER_NODE must be >= 1"
        exit 1
    fi
    CUDA_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
fi

echo "============================================================"
echo "SLURM_JOB_ID     = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME   = ${SLURM_JOB_NAME}"
echo "NUM_NODES         = ${NUM_NODES}"
echo "GPUS_PER_NODE     = ${GPUS_PER_NODE}"
echo "CONFIG            = ${CONFIG}"
echo "EVAL_SCRIPT       = ${EVAL_SCRIPT}"
echo "CONDA_ENV         = ${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "SAVE_PHYSICAL         = ${SAVE_PHYSICAL:-<auto-derived>}"
echo "============================================================"

#----- CUDA/NCCL preflight ----------------------------------------------------
python - <<'PY'
import torch
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("No CUDA GPUs visible to PyTorch. Check CUDA_VISIBLE_DEVICES/Slurm GPU allocation.")
PY

#----- Launch Evaluation -------------------------------------------------------
# Strategy:
#   The eval script is not designed with PyTorch DistributedDataParallel (DDP). 
#   It runs on a single node/process natively via the argument parser. 
#   We launch it directly using the python command.

cd ${PROJECT_DIR}

srun --cpu-bind=none python ${EVAL_SCRIPT} ${CONFIG} \
    ${SAVE_PHYSICAL:+--save-physical "${SAVE_PHYSICAL}"}

echo "Evaluation finished with exit code: $?"