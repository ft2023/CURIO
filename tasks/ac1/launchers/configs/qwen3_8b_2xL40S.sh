#!/usr/bin/env bash
# shellcheck shell=bash

# --- Archive-related settings ---
export CURIO_TASK_NAME=ac1
export CURIO_NUM_EPOCHS=50  # total epochs to run
export CURIO_SEEDS_PER_EPOCH=8  # parent solutions sampled from the archive per epoch
export CURIO_ROLLOUTS_PER_SEED=64  # child solutions generated per parent
export CURIO_MAX_ARCHIVE_SIZE=1000  # max nodes retained in the archive after each flush
export CURIO_TOPK_CHILDREN=2  # per archive flush: keep only the top-k scoring children per parent; rest are pruned
export CURIO_PUCT_C=1.0  # PUCT exploration constant for parent selection

# --- Model and generation ---
export CURIO_GENERATOR_DATA_PARALLEL_SIZE=2
export CURIO_GENERATOR_TENSOR_PARALLEL_SIZE=1  # tensor parallelism >1 is not yet supported
export CURIO_GENERATOR_BACKEND=ray_data_llm
export CURIO_MODEL_NAME_OR_PATH="Qwen/Qwen3-8B"
export CURIO_TOKENIZER_NAME_OR_PATH="Qwen/Qwen3-8B"
export CURIO_RENDERER_NAME=qwen_chat
export CURIO_RENDERER_SYSTEM_PROMPT=  # intentionally empty for parity with TTT-Discover
export CURIO_RENDERER_STOP_SEQUENCE="<|im_end|>"
export CURIO_TEMPERATURE=1.0
# TTT-Discover uses a two-phase decoding strategy: phase 1 generates the
# reasoning/CoT up to a PHASE1_END_MARKER; phase 2 then generates the final
# answer after a FORCED_FINAL_SUFFIX is appended to force the model to
# produce its final solution code. FINAL_ANSWER_MARKER tells phase 2 when to stop.
# For Qwen3-8B, TTT-Discover did not use phase 2. PHASE1_MAX_TOKENS is set to
# (context_window - context_buffer) to consume essentially the whole context in
# phase 1, leaving no budget for phase 2. The phase 2 vars are therefore unused.
# See: https://github.com/test-time-training/discover/issues/11
export CURIO_PHASE1_MAX_TOKENS=32718  # context_window - context_buffer; effectively disables phase 2
export CURIO_CONTEXT_WINDOW=32768  # model's full context window in tokens
export CURIO_CONTEXT_BUFFER=50  # token buffer reserved when computing phase 2 budget to avoid overflowing the context window
export CURIO_FINAL_ANSWER_MARKER=  # not used: two-phase decoding is disabled for this config
export CURIO_FORCED_FINAL_SUFFIX=  # not used: two-phase decoding is disabled for this config
export CURIO_PHASE1_END_MARKER=  # not used: two-phase decoding is disabled for this config
export CURIO_FORCED_FINAL_SUFFIX_AFTER_PHASE1_END_MARKER=  # not used: two-phase decoding is disabled for this config

# --- Evaluation ---
export CURIO_GENERATOR_BATCH_SIZE=16
export RAY_NUM_CPUS=32  # this controls the number of CPUs Ray can use, if you don't set it, Ray will use all available CPUs
export CURIO_EVALUATOR_NUM_WORKERS=32  # local CPU eval workers; additional workers come from Slurm array jobs when using slurm_attached

# --- Training ---
export CURIO_TRAIN_BACKEND=deepspeed
export CURIO_LEARNING_RATE=4e-5
export CURIO_ADAM_BETA1=0.9
export CURIO_ADAM_BETA2=0.95
export CURIO_ADAM_EPS=1e-8
export CURIO_WEIGHT_DECAY=0.0
export CURIO_KL_PENALTY_COEF=0.1  # weight of the KL divergence penalty against the reference policy
export CURIO_REMOVE_CONSTANT_REWARD_GROUPS=1  # drop rollout groups where all rewards are identical (no learning signal)
export CURIO_LORA_RANK=32
export CURIO_LORA_ALPHA=32  # TTT-Discover does not specify this; matches Tinker's default (see https://github.com/thinking-machines-lab/tinker-cookbook/issues/280)
export CURIO_LORA_DROPOUT=0.0
export CURIO_LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head"  # TTT-Discover does not specify this; matches Tinker's default of targeting all linear modules
export CURIO_NUM_SUBSTEPS=1  # splits the RL dataset into this many chunks before packing into microbatches; TTT-Discover always set this to 1
export CURIO_TRAINER_NUM_WORKERS=2  # DeepSpeed processes (one per GPU)
export CURIO_TRAINER_MAX_TOKENS_PER_RANK=8192  # max packed tokens per GPU AFTER Ulysses SP splitting. Multiply this number by the SP size set below to get the pre-split budget. That budget tells you how many sequences can be packed together before SP splits them across the ranks.
export CURIO_REFERENCE_SCORING_MAX_TOKENS_PER_RANK=8192  # per-rank token budget for batching rollouts during reference policy KL scoring; each rank scores its own shard of the rollouts
export CURIO_TRAINER_LOGPROB_COMPUTE_DTYPE=float32
export CURIO_REFERENCE_LOGPROB_VOCAB_CHUNK_SIZE=4096  # vocab dimension chunk size during reference logprob computation; smaller values reduce peak GPU memory
export CURIO_REFERENCE_SCORING_MODEL_PARALLEL_SIZE=1  # GPUs per reference model replica; 1 means each GPU holds a full copy and scores a different subset of rollouts (data parallel)
export CURIO_SEQUENCE_PARALLEL_SIZE=2  # Ulysses sequence parallelism degree; for now, this should equal TRAINER_NUM_WORKERS because we don't support tensor parallelism yet
export CURIO_USE_REMOVE_PADDING=1  # enable sequence packing (remove padding tokens before forward pass)
export CURIO_GRADIENT_CHECKPOINTING=1  # trade compute for memory by recomputing activations during backward
export CURIO_OPTIMIZER_STATE_KEEP_WINDOW=2  # delete optimizer states older than (current_epoch - 2) to save disk space

# These are fixed DeepSpeed invariants today, not public tuning knobs.
export CURIO_DISTRIBUTED_STRATEGY=ddp

# Safety net: reduce allocator fragmentation risk on long-sequence training workloads.
# Custom configs that do not set this will get the safe default here.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Pipeline stage control ---
export CURIO_STAGE_START=sample
export CURIO_STAGE_STOP=train
export CURIO_STAGE_MAX_EPOCHS=0  # 0 means no per-invocation epoch limit; run until NUM_EPOCHS

# --- Runtime compatibility ---
export RAY_USE_UVLOOP=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYDANTIC_DISABLE_PLUGINS=1
if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="$TRANSFORMERS_CACHE"
fi
unset TRANSFORMERS_CACHE
unset NCCL_P2P_DISABLE
