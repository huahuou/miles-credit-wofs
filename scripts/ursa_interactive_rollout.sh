#!/bin/bash
#===============================================================================
# Interactive Multi-GPU Rollout Script for miles-credit-wofs (Ursa-style)
#
# Usage examples:
#   chmod +x scripts/ursa_interactive_rollout.sh
#   scripts/ursa_interactive_rollout.sh
#   scripts/ursa_interactive_rollout.sh metrics ddp 2
#   scripts/ursa_interactive_rollout.sh rollout none
#
# Positional args:
#   1) APP  : rollout | metrics   (default: rollout)
#   2) MODE : none | ddp | fsdp   (default: ddp)
#   3) NPROC_PER_NODE override    (optional)
#===============================================================================

#----- User Configuration ------------------------------------------------------
APP="${1:-rollout}"
MODE="${2:-ddp}"
NPROC_PER_NODE_OVERRIDE="${3:-}"

NUM_NODES=1
GPUS_PER_NODE=1
AUTO_DETECT_NPROC_PER_NODE=1

CONDA_ENV="credit-wofs"
PROJECT_DIR="/home/Zhanxiang.Hua/miles-credit-wofs"
CONFIG="${PROJECT_DIR}/config/ursa_wofscast_credit_wrf_latest.yml"
BACKEND="nccl"
MAX_CASES=""
OUTPUT_NAME="rollout_metrics.csv"

#----- Resolve Application Script ---------------------------------------------
if [ "${APP}" = "rollout" ]; then
    APP_SCRIPT="applications/rollout_wrf_wofs.py"
elif [ "${APP}" = "metrics" ]; then
    APP_SCRIPT="applications/rollout_metrics_wrf_wofs.py"
else
    echo "APP must be one of: rollout, metrics"
    exit 1
fi

#----- Environment Setup -------------------------------------------------------
if [ -n "$MODULESHOME" ]; then
    source $MODULESHOME/init/bash
    module load rdhpcs-conda 2>/dev/null || true
    module load cuda/12.8 2>/dev/null || true
fi

eval "$(conda shell.bash hook)"
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

head_node=$(hostname)
head_node_ip=$(hostname -I | awk '{print $1}')
MASTER_PORT=$(( RANDOM % 9000 + 10000 ))
RUN_ID="local-${APP}-$$"

echo "============================================================"
echo "APP               = ${APP}"
echo "MODE              = ${MODE}"
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
    python ${APP_SCRIPT} ${CONFIG} --mode none --backend ${BACKEND} "${EXTRA_ARGS[@]}"
else
    torchrun \
        --nnodes=${NUM_NODES} \
        --nproc-per-node=${NPROC_PER_NODE} \
        --rdzv-id=${RUN_ID} \
        --rdzv-backend=c10d \
        --rdzv-endpoint=${head_node_ip}:${MASTER_PORT} \
        ${APP_SCRIPT} \
        ${CONFIG} \
        --mode ${MODE} \
        --backend ${BACKEND} \
        "${EXTRA_ARGS[@]}"
fi

echo "${APP} finished with exit code: $?"
