#!/usr/bin/env bash
set -euo pipefail

export CURIO_ALLOC_TASK_NAME=erdos
export CURIO_ALLOC_COMMON_ENV="tasks/erdos/launchers/common_erdos_env.sh"
export CURIO_ALLOC_INNER_MODE=single_node
export CURIO_ALLOC_INNER_ENTRYPOINT=run_all.sh
export CURIO_ALLOC_JOB_NAME_SUFFIX=alloc-single-node

if [[ -n "${CURIO_ROOT:-}" ]] && [[ -f "${CURIO_ROOT}/tasks/erdos/launchers/slurm_alloc/common.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CURIO_ROOT}/tasks/erdos/launchers/slurm_alloc/common.sh"
else
  # shellcheck disable=SC1091
  source "$(dirname "$0")/../common.sh"
fi
