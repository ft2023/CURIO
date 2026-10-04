#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

# Set CURIO_RESUME_DIR in the environment to resume an existing run.
# Leave it unset to auto-create a new run directory under CURIO_LOG_ROOT.
: "${CURIO_RESUME_DIR:=}"

: "${CURIO_LOG_ROOT:=./logs}"
if [[ -z "${CURIO_RESUME_DIR}" ]]; then
  ts=$(date +%Y%m%d-%H%M%S)
  export CURIO_RESUME_DIR="${CURIO_LOG_ROOT}/erdos-${ts}"
fi
mkdir -p "$CURIO_RESUME_DIR"

source tasks/erdos/launchers/common_erdos_env.sh
source tasks/erdos/launchers/slurm_alloc/staged/profile_env.sh

if [[ -z "${CURIO_CHAIN_EPOCH:-}" ]]; then
  CHAIN_EPOCH=$(python - <<'PY'
from pathlib import Path
import utils
import os
run_dir = Path(os.environ["CURIO_RESUME_DIR"]).resolve()
stage_stop = os.environ.get("CURIO_STAGE_STOP", "train")
print(utils.resume_epoch(run_dir, stage_stop=stage_stop))
PY
)
else
  CHAIN_EPOCH="$CURIO_CHAIN_EPOCH"
fi
export CURIO_CHAIN_EPOCH="$CHAIN_EPOCH"
: "${SLURM_ALLOC_STAGED_JOB_PREFIX:=nanodisc-erdos-e$(printf "%03d" "$CHAIN_EPOCH")}"
: "${SLURM_ALLOC_STAGED_STDOUT_PATH:=${CURIO_RESUME_DIR}/slurm/%x-%j.out}"
: "${SLURM_ALLOC_STAGED_STDERR_PATH:=${CURIO_RESUME_DIR}/slurm/%x-%j.err}"
mkdir -p "$(dirname "$SLURM_ALLOC_STAGED_STDOUT_PATH")" "$(dirname "$SLURM_ALLOC_STAGED_STDERR_PATH")"

epoch_status=$(python - <<'PY'
from pathlib import Path
import os
import utils

run_dir = Path(os.environ["CURIO_RESUME_DIR"]).resolve()
epoch = int(os.environ["CURIO_CHAIN_EPOCH"])
stage_stop = os.environ.get("CURIO_STAGE_STOP", "train")
sub = utils.epoch_subdir(run_dir, epoch)

print(f"stage_stop={stage_stop}")
print(f"is_complete={int(utils.stage_stop_reached(sub, stage_stop))}")
print(f"has_sample={int(sub.has_sample())}")
print(f"has_generation={int(sub.has_generation())}")
print(f"has_evaluation={int(sub.has_evaluation())}")
print(f"has_training={int(sub.has_training_result())}")
PY
)
eval "$(printf '%s\n' "$epoch_status" | sed 's/^/export /')"

echo "target_epoch=$CHAIN_EPOCH stage_stop=$stage_stop is_complete=$is_complete has_sample=$has_sample has_generation=$has_generation has_evaluation=$has_evaluation has_training=$has_training"

if [[ "$is_complete" == "1" ]]; then
  echo "epoch $CHAIN_EPOCH already complete for stage_stop=$stage_stop; nothing to submit"
  echo "run_dir=$CURIO_RESUME_DIR"
  echo "epoch=$CHAIN_EPOCH"
  echo "stage12_job=skipped"
  echo "stage3_array_job=skipped"
  echo "stage3_merge_job=skipped"
  echo "stage45_job=skipped"
  exit 0
fi

TOTAL=$(( CURIO_SEEDS_PER_EPOCH * CURIO_ROLLOUTS_PER_SEED ))
mkdir -p "$CURIO_RESUME_DIR/epoch$(printf "%03d" "$CHAIN_EPOCH")/evaluation_shards"

upstream_dep="${CURIO_CHAIN_DEPENDENCY:-}"
job12="skipped"
job3="skipped"
job3m="skipped"

if [[ "$has_generation" != "1" ]]; then
  dep_args=()
  if [[ -n "$upstream_dep" ]]; then
    dep_args=(--dependency "afterok:${upstream_dep}")
  fi
  job12=$(sbatch --parsable \
    "${dep_args[@]}" \
    --partition="$SLURM_ALLOC_STAGED_GPU_PARTITION" \
    --time="$SLURM_ALLOC_STAGED_GPU_TIME" \
    --gpus=8 \
    --cpus-per-task=64 \
    --job-name="${SLURM_ALLOC_STAGED_JOB_PREFIX}-stage12" \
    --output="$SLURM_ALLOC_STAGED_STDOUT_PATH" \
    --error="$SLURM_ALLOC_STAGED_STDERR_PATH" \
    --export=ALL,CURIO_CHAIN_EPOCH="$CHAIN_EPOCH" \
    tasks/erdos/launchers/slurm_alloc/staged/stage12.sbatch.sh)
  upstream_dep="$job12"
fi

if [[ "$has_evaluation" != "1" ]]; then
  dep_args=()
  if [[ -n "$upstream_dep" ]]; then
    dep_args=(--dependency "afterok:${upstream_dep}")
  fi
  job3=$(sbatch --parsable \
    "${dep_args[@]}" \
    --partition="$SLURM_ALLOC_STAGED_CPU_PARTITION" \
    --time="$SLURM_ALLOC_STAGED_EVAL_TIME" \
    --cpus-per-task="$SLURM_ALLOC_STAGED_EVAL_CPUS_PER_TASK" \
    --mem="$SLURM_ALLOC_STAGED_EVAL_MEM_PER_TASK" \
    --array="0-$((TOTAL - 1))" \
    --job-name="${SLURM_ALLOC_STAGED_JOB_PREFIX}-stage3-array" \
    --output="$SLURM_ALLOC_STAGED_STDOUT_PATH" \
    --error="$SLURM_ALLOC_STAGED_STDERR_PATH" \
    --export=ALL,CURIO_CHAIN_EPOCH="$CHAIN_EPOCH" \
    tasks/erdos/launchers/slurm_alloc/staged/stage3_array.sbatch.sh)

  job3m=$(sbatch --parsable \
    --dependency=afterok:"$job3" \
    --partition="$SLURM_ALLOC_STAGED_CPU_PARTITION" \
    --time=00:20:00 \
    --cpus-per-task=2 \
    --mem=2G \
    --job-name="${SLURM_ALLOC_STAGED_JOB_PREFIX}-stage3-merge" \
    --output="$SLURM_ALLOC_STAGED_STDOUT_PATH" \
    --error="$SLURM_ALLOC_STAGED_STDERR_PATH" \
    --export=ALL,CURIO_CHAIN_EPOCH="$CHAIN_EPOCH" \
    tasks/erdos/launchers/slurm_alloc/staged/stage3_merge.sbatch.sh)
  upstream_dep="$job3m"
fi

dep_args=()
if [[ -n "$upstream_dep" ]]; then
  dep_args=(--dependency "afterok:${upstream_dep}")
fi
job45=$(sbatch --parsable \
  "${dep_args[@]}" \
  --partition="$SLURM_ALLOC_STAGED_GPU_PARTITION" \
  --time="$SLURM_ALLOC_STAGED_GPU_TIME" \
  --gpus=8 \
  --cpus-per-task=64 \
  --job-name="${SLURM_ALLOC_STAGED_JOB_PREFIX}-stage45" \
  --output="$SLURM_ALLOC_STAGED_STDOUT_PATH" \
  --error="$SLURM_ALLOC_STAGED_STDERR_PATH" \
  --export=ALL,CURIO_CHAIN_EPOCH="$CHAIN_EPOCH" \
  tasks/erdos/launchers/slurm_alloc/staged/stage45.sbatch.sh)

echo "run_dir=$CURIO_RESUME_DIR"
echo "epoch=$CHAIN_EPOCH"
echo "stage12_job=$job12"
echo "stage3_array_job=$job3"
echo "stage3_merge_job=$job3m"
echo "stage45_job=$job45"
echo "stdout=$SLURM_ALLOC_STAGED_STDOUT_PATH"
echo "stderr=$SLURM_ALLOC_STAGED_STDERR_PATH"
