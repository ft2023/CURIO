#!/usr/bin/env bash
# shellcheck shell=bash

denoising_common_env_fail() {
  echo "$1" >&2
  return 2 2>/dev/null || exit 2
}

if [[ -z "${CURIO_DENOISING_CONFIG:-}" ]]; then
  denoising_common_env_fail "CURIO_DENOISING_CONFIG must be set to a denoising config under tasks/denoising/launchers/configs/ (for example: qwen3_8b_2xH100)."
fi

denoising_config_path="tasks/denoising/launchers/configs/${CURIO_DENOISING_CONFIG}.sh"
if [[ ! -f "${denoising_config_path}" ]]; then
  denoising_common_env_fail "Unknown denoising launcher config: ${CURIO_DENOISING_CONFIG} (expected ${denoising_config_path} to exist)."
fi

export CURIO_DENOISING_CONFIG
source "${denoising_config_path}"

# CURIO curiosity weight eta_0 (see the paper's hyperparameter table). Configs may set it explicitly;
# set CURIO_ICM_ETA=0 in the config to run the task-only TTT-Discover control.
export CURIO_ICM_ETA="${CURIO_ICM_ETA:-0.5}"

if [[ -z "${CURIO_EVAL_PYTHON:-}" ]]; then
  denoising_common_env_fail "CURIO_EVAL_PYTHON must point to the dedicated evaluator python for denoising."
fi
if [[ ! -x "${CURIO_EVAL_PYTHON}" ]]; then
  denoising_common_env_fail "CURIO_EVAL_PYTHON is not executable: ${CURIO_EVAL_PYTHON}"
fi

export CURIO_EVAL_PYTHON
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
