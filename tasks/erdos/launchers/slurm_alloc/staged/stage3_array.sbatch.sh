#!/usr/bin/env bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${CURIO_CHAIN_EPOCH:-}" ]]; then
  echo "CURIO_CHAIN_EPOCH is required" >&2
  exit 2
fi

idx="$SLURM_ARRAY_TASK_ID"
"${CURIO_EVAL_PYTHON:?CURIO_EVAL_PYTHON must point to the task evaluator python}" -m core.evaluator evaluate-shard \
  --task erdos \
  --run-dir "$CURIO_RESUME_DIR" \
  --epoch "$CURIO_CHAIN_EPOCH" \
  --start "$idx" \
  --stop "$((idx + 1))" \
  --workers 1 \
  --cpu-pack-slot 0 \
  --output "$CURIO_RESUME_DIR/epoch$(printf "%03d" "$CURIO_CHAIN_EPOCH")/evaluation_shards/shard_${idx}.json"
