#!/bin/bash
#===============================================================================
# Multi-Node Multi-GPU Rollout Script for WoFS DiffMAE Metrics Workflow on Ursa
#
# Runs applications.rollout_wrf_wofs_mae_da_metrics with torchrun so eval.mode=ddp
# can use one process per GPU across one or more nodes.
#
# Usage:
#   sbatch scripts/ursa_mae_da_rollout_metrics.sh
#
# Optional overrides:
#   CONFIG, CHECKPOINT, START_DATE, END_DATE, OUT_DIR, MASK_FILE, MASK_SEED,
#   MAX_FILES, MAX_TIMES, EVAL_MODE, BACKEND, GPUS_PER_NODE,
#   AUTO_DETECT_NPROC_PER_NODE, NPROC_PER_NODE_OVERRIDE
#===============================================================================

#----- Slurm Directives --------------------------------------------------------
#SBATCH --job-name=credit-wofs-da-metrics
#SBATCH --account=gpu-ai4wp
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=01:29:00
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
CONFIG="${CONFIG:-/home/Zhanxiang.Hua/miles-credit-wofs/config/wofs_diffmae_4x4_patch_height_mask.yml}"
ROLLOUT_MODULE="${ROLLOUT_MODULE:-applications.rollout_wrf_wofs_mae_da_metrics}"
CHECKPOINT="${CHECKPOINT:-/scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/credit_runs/wofs_diffmae_pretrain_4x4patch_heightmask_v2/checkpoint.pt}"

START_DATE="${START_DATE:-20210425}"
END_DATE="${END_DATE:-20210530}"
OUT_DIR="${OUT_DIR:-/scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/credit_rollouts/wofs_diffmae_pretrain_4x4patch_heightmask_v2/test1}"

MASK_FILE="${MASK_FILE:-}"
MASK_SEED="${MASK_SEED:-1000}"

MAX_FILES="${MAX_FILES:-32}"
MAX_TIMES="${MAX_TIMES:-1}"

EVAL_MODE="${EVAL_MODE:-ddp}"
BACKEND="${BACKEND:-nccl}"

#----- Load Modules ------------------------------------------------------------
source $MODULESHOME/init/bash
module load cdo/2.4.2
module load ncview
module load rdhpcs-conda
module load cuda/12.8
conda activate "${CONDA_ENV}"

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
mkdir -p "${OUT_DIR}"
mkdir -p /scratch3/NAGAPE/gpu-ai4wp/Zhanxiang.Hua/credit_rollouts/job_log

nodes_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "${head_node}" hostname --ip-address | awk '{print $1}')
MASTER_PORT=$(( RANDOM % 9000 + 10000 ))

echo "SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST   = ${SLURM_JOB_NODELIST}"
echo "NUM_NODES            = ${NUM_NODES}"
echo "GPUS_PER_NODE        = ${GPUS_PER_NODE}"
echo "NPROC_PER_NODE       = ${NPROC_PER_NODE}"
echo "TOTAL_GPUS           = ${TOTAL_GPUS}"
echo "HEAD_NODE            = ${head_node}"
echo "HEAD_NODE_IP         = ${head_node_ip}"
echo "MASTER_PORT          = ${MASTER_PORT}"
echo "CONFIG               = ${CONFIG}"
echo "ROLLOUT_MODULE       = ${ROLLOUT_MODULE}"
echo "CHECKPOINT           = ${CHECKPOINT}"
echo "START_DATE           = ${START_DATE}"
echo "END_DATE             = ${END_DATE}"
echo "OUT_DIR              = ${OUT_DIR}"
echo "MASK_FILE            = ${MASK_FILE}"
echo "MASK_SEED            = ${MASK_SEED}"
echo "MAX_FILES            = ${MAX_FILES}"
echo "MAX_TIMES            = ${MAX_TIMES}"
echo "EVAL_MODE            = ${EVAL_MODE}"
echo "BACKEND              = ${BACKEND}"
echo "CONDA_ENV            = ${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"

python - <<'PY'
import torch
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("No CUDA GPUs visible to PyTorch.")
PY

cd "${PROJECT_DIR}"

ROLLOUT_ARGS=(-c "${CONFIG}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --out-dir "${OUT_DIR}" \
    --mode "${EVAL_MODE}" \
    --backend "${BACKEND}")

if [ -n "${CHECKPOINT}" ]; then
    ROLLOUT_ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [ -n "${MASK_FILE}" ]; then
    ROLLOUT_ARGS+=(--mask-file "${MASK_FILE}")
elif [ -n "${MASK_SEED}" ]; then
    ROLLOUT_ARGS+=(--mask-seed "${MASK_SEED}")
fi
if [ -n "${MAX_FILES}" ]; then
    ROLLOUT_ARGS+=(--max-files "${MAX_FILES}")
fi
if [ -n "${MAX_TIMES}" ]; then
    ROLLOUT_ARGS+=(--max-times "${MAX_TIMES}")
fi

echo "Launching: torchrun --module ${ROLLOUT_MODULE} ${ROLLOUT_ARGS[*]}"

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
    --module "${ROLLOUT_MODULE}" \
    "${ROLLOUT_ARGS[@]}"

status=$?
echo "DiffMAE DA rollout metrics job finished with exit code: ${status}"
exit ${status}
