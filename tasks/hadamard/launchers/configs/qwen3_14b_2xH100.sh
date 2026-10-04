#!/usr/bin/env bash
# shellcheck shell=bash

export CURIO_TASK_NAME=hadamard
export CURIO_NUM_EPOCHS=50
export CURIO_SEEDS_PER_EPOCH=8
export CURIO_ROLLOUTS_PER_SEED=64
export CURIO_MAX_ARCHIVE_SIZE=1000
export CURIO_TOPK_CHILDREN=2
export CURIO_PUCT_C=1.0

export CURIO_GENERATOR_DATA_PARALLEL_SIZE=2
export CURIO_GENERATOR_TENSOR_PARALLEL_SIZE=1
export CURIO_GENERATOR_BACKEND=ray_data_llm
export CURIO_MODEL_NAME_OR_PATH="Qwen/Qwen3-14B"
export CURIO_TOKENIZER_NAME_OR_PATH="Qwen/Qwen3-14B"
export CURIO_RENDERER_NAME=qwen_chat
export CURIO_RENDERER_SYSTEM_PROMPT= # Intentionally left empty due to parity with original TTT-Discover code.
export CURIO_RENDERER_STOP_SEQUENCE="<|im_end|>"
export CURIO_TEMPERATURE=1.0
export CURIO_PHASE1_MAX_TOKENS=32718 # Intentionally set to this value to disable phase 2 because original TTT-Discover didn't do it for Qwen3-8B.
export CURIO_CONTEXT_WINDOW=32768
export CURIO_CONTEXT_BUFFER=50
export CURIO_FINAL_ANSWER_MARKER=
export CURIO_FORCED_FINAL_SUFFIX=
export CURIO_PHASE1_END_MARKER=
export CURIO_FORCED_FINAL_SUFFIX_AFTER_PHASE1_END_MARKER=

export CURIO_GENERATOR_BATCH_SIZE=8
export RAY_NUM_CPUS=48 # Prevent ray from spawning ncpus workers
export CURIO_EVALUATOR_NUM_WORKERS=48

export CURIO_TRAIN_BACKEND=deepspeed
export CURIO_LEARNING_RATE=4e-5
export CURIO_ADAM_BETA1=0.9
export CURIO_ADAM_BETA2=0.95
export CURIO_ADAM_EPS=1e-8
export CURIO_WEIGHT_DECAY=0.0
export CURIO_KL_PENALTY_COEF=0.1
export CURIO_REMOVE_CONSTANT_REWARD_GROUPS=1
export CURIO_LORA_RANK=32
export CURIO_LORA_ALPHA=32 # TTT-Discover codebase does not specify it. Tinker's default is 32. https://github.com/thinking-machines-lab/tinker-cookbook/issues/280
export CURIO_LORA_DROPOUT=0.0
export CURIO_LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head" # TTT-Discover codebase does not specify it. Tinker's default is to target all linear modules (train_attn=True, train_mlp=True, train_unembed=True) and TTT-Discover does not touch these values.
export CURIO_NUM_SUBSTEPS=1
export CURIO_TRAINER_NUM_WORKERS=2
export CURIO_TRAINER_MAX_TOKENS_PER_RANK=8192
export CURIO_REFERENCE_SCORING_MAX_TOKENS_PER_RANK=8192
export CURIO_TRAINER_LOGPROB_COMPUTE_DTYPE=float32
export CURIO_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE=4096
export CURIO_REFERENCE_SCORING_MODEL_PARALLEL_SIZE=1
export CURIO_SEQUENCE_PARALLEL_SIZE=2
export CURIO_USE_REMOVE_PADDING=1
export CURIO_GRADIENT_CHECKPOINTING=1
export CURIO_OPTIMIZER_STATE_KEEP_WINDOW=2

# These are fixed DeepSpeed invariants today, not public tuning knobs.
export CURIO_DISTRIBUTED_STRATEGY=ddp

# Reduce allocator fragmentation risk for long-sequence training workloads.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CURIO_STAGE_START=sample
export CURIO_STAGE_STOP=train
export CURIO_STAGE_MAX_EPOCHS=0

# Runtime compatibility and warning-noise controls.
export RAY_USE_UVLOOP=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYDANTIC_DISABLE_PLUGINS=1
if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="$TRANSFORMERS_CACHE"
fi
unset TRANSFORMERS_CACHE
unset NCCL_P2P_DISABLE
