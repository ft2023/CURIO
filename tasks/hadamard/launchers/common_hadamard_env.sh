#!/usr/bin/env bash
# shellcheck shell=bash

hadamard_common_env_fail() {
  echo "$1" >&2
  return 2 2>/dev/null || exit 2
}

if [[ -z "${CURIO_HADAMARD_CONFIG:-}" ]]; then
  hadamard_common_env_fail "CURIO_HADAMARD_CONFIG must be set to a Hadamard config under tasks/hadamard/launchers/configs/ (for example: qwen3_8b_2xH100)."
fi

hadamard_config_path="tasks/hadamard/launchers/configs/${CURIO_HADAMARD_CONFIG}.sh"
if [[ ! -f "${hadamard_config_path}" ]]; then
  hadamard_common_env_fail "Unknown Hadamard launcher config: ${CURIO_HADAMARD_CONFIG} (expected ${hadamard_config_path} to exist)."
fi

export CURIO_HADAMARD_CONFIG
source "${hadamard_config_path}"

# CURIO curiosity weight eta_0 (see the paper's hyperparameter table). Configs may set it explicitly;
# set CURIO_ICM_ETA=0 in the config to run the task-only TTT-Discover control.
export CURIO_ICM_ETA="${CURIO_ICM_ETA:-0.5}"

if [[ -z "${CURIO_EVAL_PYTHON:-}" ]]; then
  hadamard_common_env_fail "CURIO_EVAL_PYTHON must point to the dedicated evaluator python for Hadamard."
fi
if [[ ! -x "${CURIO_EVAL_PYTHON}" ]]; then
  hadamard_common_env_fail "CURIO_EVAL_PYTHON is not executable: ${CURIO_EVAL_PYTHON}"
fi

export CURIO_EVAL_PYTHON
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
