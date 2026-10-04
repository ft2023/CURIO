# `single_node/`

This execution mode runs AC2 on one machine with no cluster scheduler. Sampling, generation, evaluation, archive update, and training all run on the same node.

Use this mode if you want the simplest setup or you are testing on a local machine. Be aware that evaluation can be slow: each AC2 rollout needs 2 CPU cores and can evaluate for up to ~18 minutes, and a full epoch has 512 rollouts. If you have many CPU cores available, most of that time can be parallelised.

## Quick start

```bash
cd /path/to/CURIO
source /path/to/curio-runtime-venv/bin/activate

export CURIO_AC2_CONFIG=qwen3_8b_4xL40S
export CURIO_EVAL_PYTHON=/path/to/curio-eval-ac2-venv/bin/python
export CURIO_LOG_ROOT=/path/to/your/log/root

# Fresh run
bash tasks/ac2/launchers/single_node/run_all.sh

# Resume an existing run
export CURIO_RESUME_DIR=/path/to/existing/run
bash tasks/ac2/launchers/single_node/run_all.sh
```

## Entrypoints

- `run_all.sh` — runs the full pipeline until `CURIO_NUM_EPOCHS` is reached. If the run dies, set `CURIO_RESUME_DIR` and call it again.
- `run_one_epoch.sh` — runs exactly one full epoch locally.
- `run_smoke.sh` — minimal one-epoch smoke test for validating that the stack works end-to-end before spending real compute.

## Notes

- This mode has no per-stage scripts. If you need stage-level control, use `slurm_attached/`.
- Common config loading happens through `tasks/ac2/launchers/common_ac2_env.sh`.
