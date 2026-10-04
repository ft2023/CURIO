#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

: "${CURIO_LOG_ROOT:=./logs}"
export CURIO_LOG_ROOT
export CURIO_RESUME_DIR="${CURIO_RESUME_DIR:-}"

source tasks/hadamard/launchers/common_hadamard_env.sh

export CURIO_STAGE_START=sample
export CURIO_STAGE_STOP="${CURIO_STAGE_STOP:-train}"
export CURIO_STAGE_MAX_EPOCHS=0

python main.py
