#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

if [[ -z "${CURIO_RESUME_DIR:-}" ]]; then
  echo "CURIO_RESUME_DIR must point to an existing run directory" >&2
  exit 2
fi
mkdir -p "$CURIO_RESUME_DIR"

source tasks/erdos/launchers/common_erdos_env.sh
source tasks/erdos/launchers/slurm_attached/profile_env.sh

export CURIO_STAGE_START=sample
export CURIO_STAGE_STOP=generate
export CURIO_STAGE_MAX_EPOCHS=1

python main.py
