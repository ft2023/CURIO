#!/usr/bin/env bash
# shellcheck shell=bash

circle_packing_common_env_fail() {
  echo "$1" >&2
  return 2 2>/dev/null || exit 2
}

if [[ -z "${CURIO_CIRCLE_PACKING_CONFIG:-}" ]]; then
  circle_packing_common_env_fail "CURIO_CIRCLE_PACKING_CONFIG must be set to a Circle Packing config under tasks/circle_packing/launchers/configs/ (for example: qwen3_8b_8xL40S or gpt_oss_120b_8xH100)."
fi

circle_packing_config_path="tasks/circle_packing/launchers/configs/${CURIO_CIRCLE_PACKING_CONFIG}.sh"
if [[ ! -f "${circle_packing_config_path}" ]]; then
  circle_packing_common_env_fail "Unknown Circle Packing launcher config: ${CURIO_CIRCLE_PACKING_CONFIG} (expected ${circle_packing_config_path} to exist)."
fi

export CURIO_CIRCLE_PACKING_CONFIG
source "${circle_packing_config_path}"

# CURIO curiosity weight eta_0 (see the paper's hyperparameter table). Configs may set it explicitly;
# set CURIO_ICM_ETA=0 in the config to run the task-only TTT-Discover control.
export CURIO_ICM_ETA="${CURIO_ICM_ETA:-0.3}"

if [[ -z "${CURIO_EVAL_PYTHON:-}" ]]; then
  circle_packing_common_env_fail "CURIO_EVAL_PYTHON must point to the dedicated math evaluator python for Circle Packing parity."
fi
if [[ ! -x "${CURIO_EVAL_PYTHON}" ]]; then
  circle_packing_common_env_fail "CURIO_EVAL_PYTHON is not executable: ${CURIO_EVAL_PYTHON}"
fi

export CURIO_EVAL_PYTHON

# Safety net: reduce allocator fragmentation risk on long-sequence training workloads.
# Custom configs that do not set this will get the safe default here.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
