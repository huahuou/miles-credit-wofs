#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-none}"          # none | ddp | fsdp
NPROC="${2:-1}"            # only used for ddp/fsdp
CONFIG="${3:-/home/zhanxiang.hua/scratch/credit_runs/wofs_wrf_experiment_multi_0327/model.yml}"
BACKEND="${4:-gloo}"       # gloo (safe default) | nccl | mpi
MAX_CASES="${5:-}"         # optional integer

SCRIPT="/home/zhanxiang.hua/miles-credit-wofs/applications/rollout_wrf_wofs.py"

if [[ ! -f "$SCRIPT" ]]; then
  echo "Error: rollout script not found at $SCRIPT"
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config not found at $CONFIG"
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module load cuda/12.6
else
  echo "Warning: 'module' command not found; skipping module load cuda/12.6"
fi

if [[ -f "/home/zhanxiang.hua/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "/home/zhanxiang.hua/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Error: conda initialization script not found"
  exit 1
fi

conda activate credit

EXTRA_ARGS=()
if [[ -n "$MAX_CASES" ]]; then
  EXTRA_ARGS+=("--max-cases" "$MAX_CASES")
fi

if [[ "$MODE" == "none" ]]; then
  echo "Running single-process rollout with config: $CONFIG"
  python "$SCRIPT" "$CONFIG" --mode none "${EXTRA_ARGS[@]}"
elif [[ "$MODE" == "ddp" || "$MODE" == "fsdp" ]]; then
  echo "Running distributed rollout: mode=$MODE nproc=$NPROC backend=$BACKEND config=$CONFIG"
  torchrun --nproc_per_node="$NPROC" "$SCRIPT" "$CONFIG" --mode "$MODE" --backend "$BACKEND" "${EXTRA_ARGS[@]}"
else
  echo "Error: MODE must be one of: none, ddp, fsdp"
  echo "Usage: $0 [mode] [nproc] [config] [backend] [max_cases]"
  exit 1
fi
