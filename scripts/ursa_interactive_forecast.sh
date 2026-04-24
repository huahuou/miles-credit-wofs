#!/bin/bash
#===============================================================================
# Multi-GPU Training Script for miles-credit-wofs
#
# Usage:
#   chmod +x run_training.sh
#   ./run_training.sh
#===============================================================================

#----- User Configuration ------------------------------------------------------
NUM_NODES=1                                   # Hardcoded to 1 for simple execution
GPUS_PER_NODE=1                               # Set to the number of GPUs you want to use
AUTO_DETECT_NPROC_PER_NODE=1                  # 1: use visible GPU count, 0: use GPUS_PER_NODE
NPROC_PER_NODE_OVERRIDE=""                    # Optional explicit override, e.g. "2"
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

CONDA_ENV="credit-wofs"                       # Name or path of your conda environment
PROJECT_DIR="/home/Zhanxiang.Hua/miles-credit-wofs"  
# CONFIG="${PROJECT_DIR}/config/ursa_wofscast_credit_wrf_latest_ensemble.yml"
CONFIG="/scratch5/purged/Zhanxiang.Hua/credit_runs/wofs_wrf_experiment_multi_0417/model.yml"  # Path to your YAML config file
TRAINING_SCRIPT="applications/train_wrf_wofs_multi_ensemble.py"   

#----- Environment Setup -------------------------------------------------------
# Load modules (if you are still on a system that uses Lmod, otherwise safely ignored)
if [ -n "$MODULESHOME" ]; then
    source $MODULESHOME/init/bash
    module load rdhpcs-conda 2>/dev/null || true
    module load cuda/12.8 2>/dev/null || true
fi

# Initialize conda for the bash script and activate the environment
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

#----- Environment Variables for Distributed Training --------------------------

# --- NCCL Configuration ---
export NCCL_DEBUG=INFO                        # Set to WARN in production to reduce log noise
export LOGLEVEL=INFO

# InfiniBand / network settings (Keep these if you are still on a system with IB)
export NCCL_IB_DISABLE=0                      
export NCCL_IB_GID_INDEX=3                    
export NCCL_SOCKET_IFNAME=ib0                 
export NCCL_NET_GDR_LEVEL=PBH                 

# --- PyTorch / General ---
export OMP_NUM_THREADS=1                      
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=1             

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

#----- Set up Rendezvous for Torchrun ------------------------------------------
# Since this is a simple script running locally, we use localhost/local IP
head_node=$(hostname)
head_node_ip=$(hostname -I | awk '{print $1}')
MASTER_PORT=$(( RANDOM % 9000 + 10000 ))      # Random port in 10000-18999 range
RUN_ID="local-run-$$"                         # Use bash process ID as unique run ID

echo "============================================================"
echo "RUN_ID            = ${RUN_ID}"
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

#----- Launch Distributed Training with torchrun -------------------------------
cd ${PROJECT_DIR} || { echo "Project directory not found!"; exit 1; }

torchrun \
    --nnodes=${NUM_NODES} \
    --nproc-per-node=${NPROC_PER_NODE} \
    --rdzv-id=${RUN_ID} \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${head_node_ip}:${MASTER_PORT} \
    ${TRAINING_SCRIPT} \
    -c ${CONFIG} \
    --backend nccl

echo "Training finished with exit code: $?"