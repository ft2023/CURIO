#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

: "${CURIO_LOG_ROOT:=./logs}"
export CURIO_LOG_ROOT
export CURIO_RESUME_DIR="${CURIO_RESUME_DIR:-}"

source tasks/ac2/launchers/common_ac2_env.sh

# Fast smoke overrides aligned with the public Qwen release path.
export CURIO_NUM_EPOCHS=1
export CURIO_SEEDS_PER_EPOCH=1
export CURIO_ROLLOUTS_PER_SEED=2
export CURIO_MAX_ARCHIVE_SIZE=16
export CURIO_GENERATOR_BACKEND=ray_data_llm
export CURIO_GENERATOR_DATA_PARALLEL_SIZE=2
export CURIO_GENERATOR_BATCH_SIZE=2
export CURIO_EVALUATOR_NUM_WORKERS=2
export CURIO_TRAINER_NUM_WORKERS=2
export CURIO_TRAINER_MAX_TOKENS_PER_RANK=16384
export CURIO_SEQUENCE_PARALLEL_SIZE=2
export CURIO_USE_REMOVE_PADDING=1

export CURIO_STAGE_START=sample
export CURIO_STAGE_STOP="${CURIO_STAGE_STOP:-train}"
export CURIO_STAGE_MAX_EPOCHS=0

python main.py
