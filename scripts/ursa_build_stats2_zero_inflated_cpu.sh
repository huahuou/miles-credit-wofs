#!/bin/bash
#===============================================================================
# CPU-only Slurm job to build zero-inflated concentration transform params
# and mean/std stats for WoFS CREDIT preprocessing on Ursa.
#===============================================================================

#SBATCH --job-name=credit-wofs-stats2
#SBATCH --account=gpu-ai4wp
#SBATCH --partition=u1-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256g
#SBATCH --time=07:59:00
#SBATCH --output=/home/Zhanxiang.Hua/job_log/%x-%j.out
#SBATCH --error=/home/Zhanxiang.Hua/job_log/%x-%j.err

set -euo pipefail

PROJECT_DIR="/home/Zhanxiang.Hua/miles-credit-wofs"
CASE_GLOB="/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/cases/wofs_*.zarr.zip"
STATS_DIR="/scratch5/purged/Zhanxiang.Hua/wofs_preprocess_to_credit_0413/stats2"
START_DATE="20190101"
END_DATE="20201231"

TRANSFORM_JSON="${STATS_DIR}/zero_inflated_transform_params.json"
TRANSFORM_PLOTS_DIR="${STATS_DIR}/zero_inflated_transform_plots"
MEAN_OUT="${STATS_DIR}/mean.nc"
STD_OUT="${STATS_DIR}/std.nc"
LATWEIGHTS_OUT="${STATS_DIR}/latitude_weights_placeholder.nc"

N_WORKERS=16
THREADS_PER_WORKER=1
MEMORY_LIMIT="15GiB"
FILES_PER_TASK=8

source $MODULESHOME/init/bash
module load rdhpcs-conda
module load cuda/12.8
conda activate credit-wofs

mkdir -p "${STATS_DIR}"
mkdir -p /home/Zhanxiang.Hua/job_log

cd "${PROJECT_DIR}"

echo "============================================================"
echo "PROJECT_DIR      = ${PROJECT_DIR}"
echo "CASE_GLOB        = ${CASE_GLOB}"
echo "STATS_DIR        = ${STATS_DIR}"
echo "START_DATE       = ${START_DATE}"
echo "END_DATE         = ${END_DATE}"
echo "N_WORKERS        = ${N_WORKERS}"
echo "THREADS_WORKER   = ${THREADS_PER_WORKER}"
echo "MEMORY_LIMIT     = ${MEMORY_LIMIT}"
echo "FILES_PER_TASK   = ${FILES_PER_TASK}"
echo "============================================================"

python python_scripts/build_zero_inflated_transform_params.py \
  --glob "${CASE_GLOB}" \
  --output "${TRANSFORM_JSON}" \
  --plots-dir "${TRANSFORM_PLOTS_DIR}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --zero-floor 1e-11 \
  --probit-eps 1e-6 \
  --min-positive-samples-per-level 2000 \
  --n-workers "${N_WORKERS}" \
  --threads-per-worker "${THREADS_PER_WORKER}" \
  --memory-limit "${MEMORY_LIMIT}" \
  --files-per-task "${FILES_PER_TASK}"

python python_scripts/compute_credit_stats.py \
  --glob "${CASE_GLOB}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --transform-params-json "${TRANSFORM_JSON}" \
  --mean-out "${MEAN_OUT}" \
  --std-out "${STD_OUT}" \
  --latweights-out "${LATWEIGHTS_OUT}" \
  --n-workers "${N_WORKERS}" \
  --threads-per-worker "${THREADS_PER_WORKER}" \
  --memory-limit "${MEMORY_LIMIT}" \
  --files-per-task "${FILES_PER_TASK}"

echo "Finished building stats in ${STATS_DIR}"
