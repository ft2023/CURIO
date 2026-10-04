#!/usr/bin/env bash
# shellcheck shell=bash

ac3_common_env_fail() {
  echo "$1" >&2
  return 2 2>/dev/null || exit 2
}

if [[ -z "${CURIO_AC3_CONFIG:-}" ]]; then
  ac3_common_env_fail "CURIO_AC3_CONFIG must be set to an AC3 config under tasks/ac3/launchers/configs/ (for example: qwen3_8b_8xL40S or gpt_oss_120b_8xH100)."
fi

ac3_config_path="tasks/ac3/launchers/configs/${CURIO_AC3_CONFIG}.sh"
if [[ ! -f "${ac3_config_path}" ]]; then
  ac3_common_env_fail "Unknown AC3 launcher config: ${CURIO_AC3_CONFIG} (expected ${ac3_config_path} to exist)."
fi

export CURIO_AC3_CONFIG
source "${ac3_config_path}"

# CURIO curiosity weight eta_0 (see the paper's hyperparameter table). Configs may set it explicitly;
# set CURIO_ICM_ETA=0 in the config to run the task-only TTT-Discover control.
export CURIO_ICM_ETA="${CURIO_ICM_ETA:-0.3}"

if [[ -z "${CURIO_EVAL_PYTHON:-}" ]]; then
  ac3_common_env_fail "CURIO_EVAL_PYTHON must point to the dedicated math evaluator python for AC3 parity."
fi
if [[ ! -x "${CURIO_EVAL_PYTHON}" ]]; then
  ac3_common_env_fail "CURIO_EVAL_PYTHON is not executable: ${CURIO_EVAL_PYTHON}"
fi

export CURIO_EVAL_PYTHON
# Safety net: reduce allocator fragmentation risk on long-sequence training workloads.
# Custom configs that do not set this will get the safe default here.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
