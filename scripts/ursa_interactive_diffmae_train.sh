#!/bin/bash
#===============================================================================
# Interactive Multi-Node Multi-GPU Training Script for WoFS Conditional DiffMAE
#
# Trains the WoFS conditional DiffMAE precip inpainting model with CREDIT.
# Designed to run inside an interactive salloc session.
#
# Usage (single-node):
#   salloc -A gpu-ai4wp -p u1-h100 -q gpu --gpus-per-node=h100:2 -N 1 -t 00:50:00
#   bash /home/Zhanxiang.Hua/miles-credit-wofs/scripts/ursa_interactive_diffmae_train.sh
#
# Usage (multi-node):
#   salloc -A gpu-ai4wp -p u1-h100 -q gpu --gpus-per-node=h100:2 -N 2 -t 00:50:00
#   bash /home/Zhanxiang.Hua/miles-credit-wofs/scripts/ursa_interactive_diffmae_train.sh
#
# Edit CONFIG / CONDA_ENV below as needed.
#===============================================================================

set -euo pipefail

#----- User Configuration ------------------------------------------------------
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
GPU_TYPE=${GPU_TYPE:-h100}
AUTO_DETECT_NPROC_PER_NODE=${AUTO_DETECT_NPROC_PER_NODE:-1}
NPROC_PER_NODE_OVERRIDE="${NPROC_PER_NODE_OVERRIDE:-}"

CONDA_ENV="${CONDA_ENV:-credit-wofs}"
PROJECT_DIR="${PROJECT_DIR:-/home/Zhanxiang.Hua/miles-credit-wofs}"
##CONFIG="${PROJECT_DIR}/config/wofs_diffmae.yml"
CONFIG="${CONFIG:-/scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/credit_runs/wofs_diffmae_pretrain3_c1/model.yml}"
TRAINING_SCRIPT="${TRAINING_SCRIPT:-applications/train_wrf_wofs_mae.py}"

#----- Verify salloc environment -----------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Not inside a Slurm allocation. Run salloc first:" >&2
    echo "  salloc -A gpu-ai4wp -p u1-h100 -q gpu --gpus-per-node=h100:2 -N 2 -t 00:50:00" >&2
    exit 1
fi

NUM_NODES=${SLURM_JOB_NUM_NODES:-1}

#----- Load Modules ------------------------------------------------------------
source "${MODULESHOME}/init/bash"
module load rdhpcs-conda
module load cuda/12.8
conda activate "${CONDA_ENV}"

#----- Resolve compute nodes ---------------------------------------------------
nodes_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "${head_node}" hostname --ip-address | awk '{print $1}')
MASTER_PORT=$(( RANDOM % 9000 + 10000 ))

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

#----- Detect GPUs per node ----------------------------------------------------
if [[ -n "${NPROC_PER_NODE_OVERRIDE}" ]]; then
    NPROC_PER_NODE=${NPROC_PER_NODE_OVERRIDE}
elif [[ "${AUTO_DETECT_NPROC_PER_NODE}" -eq 1 ]]; then
    gpu_info=$(srun --nodes="${NUM_NODES}" \
        --ntasks="${NUM_NODES}" \
        --ntasks-per-node=1 \
        --cpu-bind=none \
        python -c 'import os, socket, torch; print("GPU_INFO %s %s %s" % (socket.gethostname(), torch.cuda.device_count(), os.environ.get("CUDA_VISIBLE_DEVICES", "unset")), flush=True)')
    echo "${gpu_info}"
    NPROC_PER_NODE=$(awk '/^GPU_INFO / { if (min == "" || $3 < min) min = $3 } END { print min }' <<< "${gpu_info}")
else
    NPROC_PER_NODE=${GPUS_PER_NODE}
fi

if [[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" -lt 1 ]]; then
    echo "ERROR: No GPUs are visible in this Slurm allocation." >&2
    echo "For two H100s per Ursa node, allocate with:" >&2
    echo "  salloc -A gpu-ai4wp -p u1-h100 -q gpu --gpus-per-node=h100:2 -N ${NUM_NODES} -t 00:50:00" >&2
    exit 1
fi

if [[ "${NPROC_PER_NODE}" -lt "${GPUS_PER_NODE}" ]]; then
    echo "WARNING: Only ${NPROC_PER_NODE} GPU(s) per node are visible, but GPUS_PER_NODE=${GPUS_PER_NODE}." >&2
    echo "If you intended two GPUs on every node, use --gpus-per-node=h100:2 instead of --gpus=h100:2." >&2
fi

TOTAL_GPUS=$((NUM_NODES * NPROC_PER_NODE))

echo "============================================================"
echo "Interactive DiffMAE Training"
echo "============================================================"
echo "SLURM_JOB_ID        = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST  = ${SLURM_JOB_NODELIST}"
echo "NUM_NODES           = ${NUM_NODES}"
echo "GPUS_PER_NODE       = ${GPUS_PER_NODE}"
echo "NPROC_PER_NODE      = ${NPROC_PER_NODE}"
echo "TOTAL_GPUS          = ${TOTAL_GPUS}"
echo "HEAD_NODE           = ${head_node}"
echo "HEAD_NODE_IP        = ${head_node_ip}"
echo "MASTER_PORT         = ${MASTER_PORT}"
echo "CONFIG              = ${CONFIG}"
echo "TRAINING_SCRIPT     = ${TRAINING_SCRIPT}"
echo "============================================================"

#----- Launch Distributed Training with torchrun via srun ----------------------
cd "${PROJECT_DIR}"

srun --nodes="${NUM_NODES}" \
    --ntasks="${NUM_NODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPU_TYPE}:${NPROC_PER_NODE}" \
    --cpu-bind=none \
    --kill-on-bad-exit=1 \
    torchrun \
    --nnodes="${NUM_NODES}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --rdzv-id="${SLURM_JOB_ID}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${head_node_ip}:${MASTER_PORT}" \
    "${TRAINING_SCRIPT}" \
    -c "${CONFIG}" \
    --backend nccl

status=$?
echo "Training finished with exit code: ${status}"
exit "${status}"
