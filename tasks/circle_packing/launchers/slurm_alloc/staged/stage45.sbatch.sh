#!/usr/bin/env bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source tasks/circle_packing/launchers/common_circle_packing_env.sh
source tasks/circle_packing/launchers/slurm_alloc/staged/profile_env.sh

export CURIO_STAGE_START=archive_update
export CURIO_STAGE_STOP="${CURIO_STAGE_STOP:-train}"
export CURIO_STAGE_MAX_EPOCHS=1

python main.py
