#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

# Set CURIO_RESUME_DIR in the environment to resume an existing run.
# Leave it unset to auto-create a new run directory under CURIO_LOG_ROOT.
: "${CURIO_RESUME_DIR:=}"

: "${CURIO_LOG_ROOT:=./logs}"
if [[ -z "${CURIO_RESUME_DIR}" ]]; then
  ts=$(date +%Y%m%d-%H%M%S)
  export CURIO_RESUME_DIR="${CURIO_LOG_ROOT}/ac2-${ts}"
fi
mkdir -p "$CURIO_RESUME_DIR"

source tasks/ac2/launchers/common_ac2_env.sh

current_epoch() {
  python - <<'PY'
from pathlib import Path
import utils
import os
run_dir = Path(os.environ["CURIO_RESUME_DIR"]).resolve()
stage_stop = os.environ.get("CURIO_STAGE_STOP", "train")
print(utils.resume_epoch(run_dir, stage_stop=stage_stop))
PY
}

while true; do
  epoch=$(current_epoch)
  if (( epoch >= CURIO_NUM_EPOCHS )); then
    echo "done run_dir=$CURIO_RESUME_DIR epoch=$epoch/$CURIO_NUM_EPOCHS"
    break
  fi
  echo "launching epoch=$epoch run_dir=$CURIO_RESUME_DIR"
  tasks/ac2/launchers/slurm_attached/run_one_epoch.sh
done
