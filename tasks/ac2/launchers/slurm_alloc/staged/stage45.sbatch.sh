#!/usr/bin/env bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source tasks/ac2/launchers/common_ac2_env.sh
source tasks/ac2/launchers/slurm_alloc/staged/profile_env.sh

export CURIO_STAGE_START=archive_update
export CURIO_STAGE_STOP="${CURIO_STAGE_STOP:-train}"
export CURIO_STAGE_MAX_EPOCHS=1

python main.py
