#!/bin/bash
#===============================================================================
# Multi-Node Multi-GPU Training Script for miles-credit-wofs on NOAA Ursa
#
# System: Ursa (NESCC, Fairmont WV)
#   - GPU Partition: u1-h100 (58 nodes, 2x NVIDIA H100-NVL per node, 94 GB each)
#   - CPU: AMD Genoa 9654, 192 cores/node
#   - Interconnect: NDR-200 InfiniBand
#   - Scheduler: Slurm (not PBS — Ursa uses Slurm, unlike Derecho which uses PBS)
#
# Usage:
#   sbatch ursa_multinode_training.sh
#
# Adjust NUM_NODES, GPUS_PER_NODE, PROJECT, CONFIG, and CONDA_ENV below.
#===============================================================================

#----- Slurm Directives --------------------------------------------------------
#SBATCH --job-name=credit-wofs-train
#SBATCH --account=gpu-ai4wp          # <-- Replace with your GPU project ID
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu                             # Use 'gpuwf' if you only have windfall access
#SBATCH --nodes=1                             # Number of nodes (adjust as needed)
#SBATCH --ntasks-per-node=1                   # 1 Slurm task per node (torchrun handles GPU processes)
#SBATCH --gpus-per-node=h100:2                # 2 H100 GPUs per node
#SBATCH --cpus-per-task=192                    # CPU cores per task (for data loading workers)
#SBATCH --mem=0                               # Use all available memory on the node
#SBATCH --time=1-08:00:00                       # Wall time limit
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
# CONFIG="${PROJECT_DIR}/config/ursa_wofscast_credit_wrf_latest.yml"
CONFIG="/scratch5/purged/Zhanxiang.Hua/credit_runs/wofs_wrf_experiment_multi_0417_deterministic/model.yml"  # Path to your YAML config file
# TRAINING_SCRIPT="applications/train_wrf_wofs_multi_ensemble.py"   # Multi-step WoFS trainer
TRAINING_SCRIPT="applications/train_wrf_wofs_multi.py"   # Multi-step WoFS trainer
# For single-step training, use: TRAINING_SCRIPT="applications/train_wrf_wofs.py"

#----- Load Modules ------------------------------------------------------------
# Source the module system so module commands work inside batch scripts
source $MODULESHOME/init/bash

### module purge
module load rdhpcs-conda
module load cuda/12.8

# Activate the conda environment
conda activate ${CONDA_ENV}

#----- Environment Variables for Distributed Training --------------------------

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
export TORCH_NCCL_BLOCKING_WAIT=1             # Helps debug hangs in distributed training

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

# Resolve nproc-per-node for torchrun with user override > auto-detect > GPUS_PER_NODE.
if [ -n "${NPROC_PER_NODE_OVERRIDE}" ]; then
    NPROC_PER_NODE=${NPROC_PER_NODE_OVERRIDE}
elif [ "${AUTO_DETECT_NPROC_PER_NODE}" -eq 1 ]; then
    IFS=',' read -r -a _visible_gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE=${#_visible_gpu_list[@]}
else
    NPROC_PER_NODE=${GPUS_PER_NODE}
fi

if [ "${NPROC_PER_NODE}" -lt 1 ]; then
    echo "NPROC_PER_NODE must be >= 1"
    exit 1
fi

TOTAL_GPUS=$((NUM_NODES * NPROC_PER_NODE))

#----- Identify the Head Node for Rendezvous -----------------------------------
# torchrun's c10d rendezvous backend needs a single master address + port.
# We use the first node in the Slurm allocation.

nodes_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address | awk '{print $1}')
MASTER_PORT=$(( RANDOM % 9000 + 10000 ))      # Random port in 10000-18999 range

echo "============================================================"
echo "SLURM_JOB_ID     = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME   = ${SLURM_JOB_NAME}"
echo "NUM_NODES         = ${NUM_NODES}"
echo "GPUS_PER_NODE     = ${GPUS_PER_NODE}"
echo "NPROC_PER_NODE    = ${NPROC_PER_NODE}"
echo "TOTAL_GPUS        = ${TOTAL_GPUS}"
echo "HEAD_NODE         = ${head_node}"
echo "HEAD_NODE_IP      = ${head_node_ip}"
echo "MASTER_PORT       = ${MASTER_PORT}"
echo "CONFIG            = ${CONFIG}"
echo "TRAINING_SCRIPT   = ${TRAINING_SCRIPT}"
echo "CONDA_ENV         = ${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

#----- CUDA/NCCL preflight ----------------------------------------------------
python - <<'PY'
import torch
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("No CUDA GPUs visible to PyTorch. Check CUDA_VISIBLE_DEVICES/Slurm GPU allocation.")
PY

#----- Launch Distributed Training with torchrun via srun ----------------------
# Strategy:
#   srun launches 1 process per node (--ntasks-per-node=1).
#   On each node, torchrun spawns NPROC_PER_NODE worker processes.
#   torchrun handles LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
#
# This is the standard pattern for multi-node PyTorch DDP/FSDP training.

cd ${PROJECT_DIR}

srun --cpu-bind=none torchrun \
    --nnodes=${NUM_NODES} \
    --nproc-per-node=${NPROC_PER_NODE} \
    --rdzv-id=${SLURM_JOB_ID} \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${head_node_ip}:${MASTER_PORT} \
    ${TRAINING_SCRIPT} \
    -c ${CONFIG} \
    --backend nccl

echo "Training finished with exit code: $?"
