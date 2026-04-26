#!/bin/bash
#===============================================================================
# Multi-Node Multi-GPU Rollout Script for miles-credit-wofs on NOAA Ursa
#
# This single sbatch script can run either:
#   - applications/rollout_wrf_wofs.py
#   - applications/rollout_metrics_wrf_wofs.py
#
# Usage:
#   sbatch scripts/ursa_rollout.sh
#
# Optional override at submit time:
#   sbatch --export=ALL,APP=metrics,MODE=ddp,NPROC_PER_NODE_OVERRIDE=2 scripts/ursa_rollout.sh
#===============================================================================

#----- Slurm Directives --------------------------------------------------------
#SBATCH --job-name=credit-wofs-rollout
#SBATCH --account=gpu-ai4wp
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=11:59:00
#SBATCH --output=/home/Zhanxiang.Hua/job_log/%x-%j.out
#SBATCH --error=/home/Zhanxiang.Hua/job_log/%x-%j.err
#SBATCH --exclusive

#----- User Configuration ------------------------------------------------------
NUM_NODES=${SLURM_JOB_NUM_NODES}
GPUS_PER_NODE=2
AUTO_DETECT_NPROC_PER_NODE=1
NPROC_PER_NODE_OVERRIDE=${NPROC_PER_NODE_OVERRIDE:-""}

# APP: rollout | metrics
APP=${APP:-rollout}
# MODE: none | ddp | fsdp
MODE=${MODE:-ddp}

CONDA_ENV="credit-wofs"
PROJECT_DIR="/home/Zhanxiang.Hua/miles-credit-wofs"
CONFIG="${PROJECT_DIR}/config/ursa_wofscast_credit_wrf_latest.yml"
BACKEND=${BACKEND:-nccl}
MAX_CASES=${MAX_CASES:-""}
OUTPUT_NAME=${OUTPUT_NAME:-rollout_metrics.csv}

#----- Resolve Application Script ---------------------------------------------
if [ "${APP}" = "rollout" ]; then
    APP_SCRIPT="applications/rollout_wrf_wofs.py"
elif [ "${APP}" = "metrics" ]; then
    APP_SCRIPT="applications/rollout_metrics_wrf_wofs.py"
else
    echo "APP must be one of: rollout, metrics"
    exit 1
fi

#----- Module / Conda Setup ---------------------------------------------------
source $MODULESHOME/init/bash
module load rdhpcs-conda
module load cuda/12.8
conda activate ${CONDA_ENV}

#----- Runtime Environment Variables ------------------------------------------
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

nodes_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address | awk '{print $1}')
MASTER_PORT=$(( RANDOM % 9000 + 10000 ))

echo "============================================================"
echo "SLURM_JOB_ID      = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME    = ${SLURM_JOB_NAME}"
echo "APP               = ${APP}"
echo "MODE              = ${MODE}"
echo "NUM_NODES         = ${NUM_NODES}"
echo "NPROC_PER_NODE    = ${NPROC_PER_NODE}"
echo "HEAD_NODE         = ${head_node}"
echo "HEAD_NODE_IP      = ${head_node_ip}"
echo "MASTER_PORT       = ${MASTER_PORT}"
echo "CONFIG            = ${CONFIG}"
echo "APP_SCRIPT        = ${APP_SCRIPT}"
echo "CONDA_ENV         = ${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

python - <<'PY'
import torch
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("No CUDA GPUs visible to PyTorch.")
PY

cd ${PROJECT_DIR} || { echo "Project directory not found"; exit 1; }

EXTRA_ARGS=()
if [ -n "${MAX_CASES}" ]; then
    EXTRA_ARGS+=("--max-cases" "${MAX_CASES}")
fi
if [ "${APP}" = "metrics" ] && [ -n "${OUTPUT_NAME}" ]; then
    EXTRA_ARGS+=("--output-name" "${OUTPUT_NAME}")
fi

if [ "${MODE}" = "none" ]; then
    srun --nodes=1 --ntasks=1 --cpu-bind=none python ${APP_SCRIPT} ${CONFIG} --mode none --backend ${BACKEND} "${EXTRA_ARGS[@]}"
else
    srun --cpu-bind=none torchrun \
        --nnodes=${NUM_NODES} \
        --nproc-per-node=${NPROC_PER_NODE} \
        --rdzv-id=${SLURM_JOB_ID} \
        --rdzv-backend=c10d \
        --rdzv-endpoint=${head_node_ip}:${MASTER_PORT} \
        ${APP_SCRIPT} \
        ${CONFIG} \
        --mode ${MODE} \
        --backend ${BACKEND} \
        "${EXTRA_ARGS[@]}"
fi

echo "${APP} finished with exit code: $?"
