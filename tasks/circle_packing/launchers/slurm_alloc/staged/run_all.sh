#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

# Set CURIO_RESUME_DIR in the environment to resume an existing run.
# Leave it unset to auto-create a new run directory under CURIO_LOG_ROOT.
: "${CURIO_RESUME_DIR:=}"

: "${CURIO_LOG_ROOT:=./logs}"
if [[ -z "${CURIO_RESUME_DIR}" ]]; then
  ts=$(date +%Y%m%d-%H%M%S)
  export CURIO_RESUME_DIR="${CURIO_LOG_ROOT}/circle_packing-${ts}"
fi
mkdir -p "$CURIO_RESUME_DIR"

source tasks/circle_packing/launchers/common_circle_packing_env.sh

start_epoch=$(python - <<'PY'
from pathlib import Path
import utils
import os
run_dir = Path(os.environ["CURIO_RESUME_DIR"]).resolve()
stage_stop = os.environ.get("CURIO_STAGE_STOP", "train")
print(utils.resume_epoch(run_dir, stage_stop=stage_stop))
PY
)

if (( start_epoch >= CURIO_NUM_EPOCHS )); then
  echo "nothing to submit: start_epoch=$start_epoch num_epochs=$CURIO_NUM_EPOCHS"
  exit 0
fi

prev_dep=""
for (( epoch = start_epoch; epoch < CURIO_NUM_EPOCHS; epoch++ )); do
  echo "submitting epoch=$epoch"
  out=$(CURIO_CHAIN_EPOCH="$epoch" CURIO_CHAIN_DEPENDENCY="$prev_dep" tasks/circle_packing/launchers/slurm_alloc/staged/run_one_epoch.sh)
  echo "$out"
  prev_dep=$(printf '%s\n' "$out" | awk -F= '/^stage45_job=/{print $2}' | tail -n 1)
  if [[ -z "$prev_dep" ]]; then
    echo "failed to parse stage45_job for epoch=$epoch" >&2
    exit 2
  fi
done

echo "submitted full chain run_dir=$CURIO_RESUME_DIR final_job=$prev_dep"
