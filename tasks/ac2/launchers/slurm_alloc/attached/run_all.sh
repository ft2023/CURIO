#!/usr/bin/env bash
set -euo pipefail

export CURIO_ALLOC_TASK_NAME=ac2
export CURIO_ALLOC_COMMON_ENV="tasks/ac2/launchers/common_ac2_env.sh"
export CURIO_ALLOC_INNER_MODE=slurm_attached
export CURIO_ALLOC_INNER_ENTRYPOINT=run_all.sh
export CURIO_ALLOC_JOB_NAME_SUFFIX=alloc-attached

if [[ -n "${CURIO_ROOT:-}" ]] && [[ -f "${CURIO_ROOT}/tasks/ac2/launchers/slurm_alloc/common.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CURIO_ROOT}/tasks/ac2/launchers/slurm_alloc/common.sh"
else
  # shellcheck disable=SC1091
  source "$(dirname "$0")/../common.sh"
fi
