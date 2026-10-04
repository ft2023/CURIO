#!/usr/bin/env bash
set -euo pipefail

export CURIO_ALLOC_TASK_NAME=ac1
export CURIO_ALLOC_COMMON_ENV="tasks/ac1/launchers/common_ac1_env.sh"
export CURIO_ALLOC_INNER_MODE=single_node
export CURIO_ALLOC_INNER_ENTRYPOINT=run_one_epoch.sh
export CURIO_ALLOC_JOB_NAME_SUFFIX=alloc-single-node-1ep

if [[ -n "${CURIO_ROOT:-}" ]] && [[ -f "${CURIO_ROOT}/tasks/ac1/launchers/slurm_alloc/common.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CURIO_ROOT}/tasks/ac1/launchers/slurm_alloc/common.sh"
else
  # shellcheck disable=SC1091
  source "$(dirname "$0")/../common.sh"
fi
