#!/bin/bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-credit-wofs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${MODULESHOME}/init/bash"
module load rdhpcs-conda
module load cuda/12.8
conda activate "${CONDA_ENV}"

python "${SCRIPT_DIR}/prepare_mpassit_da_case_for_aida.py" "$@"
