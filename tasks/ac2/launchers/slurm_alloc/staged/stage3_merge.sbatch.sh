#!/usr/bin/env bash
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

if [[ -z "${CURIO_CHAIN_EPOCH:-}" ]]; then
  echo "CURIO_CHAIN_EPOCH is required" >&2
  exit 2
fi

TOTAL=$(( CURIO_SEEDS_PER_EPOCH * CURIO_ROLLOUTS_PER_SEED ))
python -m core.evaluator merge-shards \
  --run-dir "$CURIO_RESUME_DIR" \
  --epoch "$CURIO_CHAIN_EPOCH" \
  --shard-dir "$CURIO_RESUME_DIR/epoch$(printf "%03d" "$CURIO_CHAIN_EPOCH")/evaluation_shards" \
  --expected-total "$TOTAL"
