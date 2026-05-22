#!/bin/bash
#===============================================================================
# Multi-Node Multi-GPU Training Script for WoFS Conditional DiffMAE on Ursa
#
# Trains the WoFS conditional DiffMAE precip inpainting model with CREDIT.
#
# Usage:
#   sbatch ursa_diffmae_train.sh
#
# Edit CONFIG / CONDA_ENV / resource directives below as needed.
#===============================================================================

#----- Slurm Directives --------------------------------------------------------
#SBATCH --job-name=credit-wofs-da-train
#SBATCH --account=gpu-ai4wp
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=1-02:49:00
#SBATCH --output=/home/Zhanxiang.Hua/job_log/%x-%j.out
#SBATCH --error=/home/Zhanxiang.Hua/job_log/%x-%j.err
#SBATCH --exclusive

#----- User Configuration ------------------------------------------------------
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
AUTO_DETECT_NPROC_PER_NODE=${AUTO_DETECT_NPROC_PER_NODE:-1}
NPROC_PER_NODE_OVERRIDE="${NPROC_PER_NODE_OVERRIDE:-}"

CONDA_ENV="${CONDA_ENV:-credit-wofs}"
PROJECT_DIR="${PROJECT_DIR:-/home/Zhanxiang.Hua/miles-credit-wofs}"
CONFIG="${PROJECT_DIR}/config/wofs_diffmae_4x4_patch_height_mask.yml"
##CONFIG="${CONFIG:-/scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/credit_runs/wofs_diffmae_pretrain_4x4patch_rev3_1/model.yml}"
TRAINING_SCRIPT="${TRAINING_SCRIPT:-applications/train_wrf_wofs_mae.py}"

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

if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
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

if [ "${NUM_NODES}" -lt 1 ]; then
    echo "NUM_NODES must be >= 1"
    exit 1
fi

if [ "${NPROC_PER_NODE}" -lt 1 ]; then
    echo "NPROC_PER_NODE must be >= 1"
    exit 1
fi

TOTAL_GPUS=$((NUM_NODES * NPROC_PER_NODE))
nodes_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "${head_node}" hostname --ip-address | awk '{print $1}')
MASTER_PORT=$(( RANDOM % 9000 + 10000 ))

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
echo "CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES}"

python - <<'PY'
import torch
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("No CUDA GPUs visible to PyTorch.")
PY

cd ${PROJECT_DIR}

srun --nodes=${NUM_NODES} \
    --ntasks=${NUM_NODES} \
    --ntasks-per-node=1 \
    --cpu-bind=none \
    --kill-on-bad-exit=1 \
    torchrun \
    --nnodes=${NUM_NODES} \
    --nproc-per-node=${NPROC_PER_NODE} \
    --rdzv-id=${SLURM_JOB_ID} \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${head_node_ip}:${MASTER_PORT} \
    ${TRAINING_SCRIPT} \
    -c ${CONFIG} \
    --backend nccl

status=$?
echo "Training finished with exit code: ${status}"
exit ${status}
