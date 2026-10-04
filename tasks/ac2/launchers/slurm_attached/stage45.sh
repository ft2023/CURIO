#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

if [[ -z "${CURIO_RESUME_DIR:-}" ]]; then
  echo "CURIO_RESUME_DIR must point to an existing run directory" >&2
  exit 2
fi
mkdir -p "$CURIO_RESUME_DIR"

source tasks/ac2/launchers/common_ac2_env.sh
source tasks/ac2/launchers/slurm_attached/profile_env.sh

export CURIO_STAGE_START=archive_update
export CURIO_STAGE_STOP="${CURIO_STAGE_STOP:-train}"
export CURIO_STAGE_MAX_EPOCHS=1

echo "stage45 start stage_start=${CURIO_STAGE_START} stage_stop=${CURIO_STAGE_STOP} trainer_workers=${CURIO_TRAINER_NUM_WORKERS:-unset} sp_size=${CURIO_SEQUENCE_PARALLEL_SIZE:-unset} trainer_max_tokens=${CURIO_TRAINER_MAX_TOKENS_PER_RANK:-unset} reference_max_tokens=${CURIO_REFERENCE_SCORING_MAX_TOKENS_PER_RANK:-unset}"

python main.py
