#!/usr/bin/env bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source tasks/circle_packing/launchers/common_circle_packing_env.sh
source tasks/circle_packing/launchers/slurm_alloc/staged/profile_env.sh

export CURIO_STAGE_START=sample
export CURIO_STAGE_STOP=generate
export CURIO_STAGE_MAX_EPOCHS=1

python main.py
