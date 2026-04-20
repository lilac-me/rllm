#!/usr/bin/env bash
set -xeuo pipefail

# ── KernelGYM Fully-Async Training Launch Script ─────────────────────────
# Mirrors the structure of 8b_stale05_rs.sh / verl_fully_async_mega.sh but
# configured for KernelGYM multi-turn kernel optimisation on Ascend NPU.
#
# Prerequisites:
#   1. KernelGYM evaluation server running (kernel_server_url below)
#   2. Dataset registered via DatasetRegistry or train_files provided
#   3. Ray cluster started

# ── Environment ──────────────────────────────────────────────────────────
export HYDRA_FULL_ERROR=1
export RAY_DEBUG_POST_MORTEM=0

# NPU memory allocator – limit segment splitting to reduce fragmentation
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:2048

# NPU / HCCL transport
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HCCL_INTRA_ROCE_ENABLE=1
export VLLM_ASCEND_ENABLE_NZ=0

# vLLM async engine
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# ── Paths ────────────────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-"/home/g00841271/Qwen3-Coder-30B-A3B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"./checkpoints/kernelgym-fully-async"}

# Dataset (registered name or file path)
DATASET_NAME=${DATASET_NAME:-"kernelbench"}
VAL_DATASET_NAME=${VAL_DATASET_NAME:-"kernelbench"}
TRAIN_FILE=${TRAIN_FILE:-""}
TEST_FILE=${TEST_FILE:-""}

# KernelGYM evaluation server
KERNEL_SERVER_URL=${KERNEL_SERVER_URL:-"http://localhost:8000"}
KERNEL_BACKEND=${KERNEL_BACKEND:-"triton"}
KERNEL_MAX_TURNS=${KERNEL_MAX_TURNS:-3}
KERNEL_TIMEOUT=${KERNEL_TIMEOUT:-300}

# ── Cluster topology ─────────────────────────────────────────────────────
n_gpus_rollout=${N_GPUS_ROLLOUT:-4}
n_gpus_training=${N_GPUS_TRAINING:-4}
n_nodes_rollout=${N_NODES_ROLLOUT:-1}
n_nodes_train=${N_NODES_TRAIN:-1}

# ── Rollout / training hyper-params ──────────────────────────────────────
rollout_mode="async"
rollout_name=${ROLLOUT_NAME:-"vllm"}

max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 24))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 8))}

temperature=1.0
top_p=1.0
top_k=-1          # -1 for vLLM rollout, 0 for HF rollout
val_top_p=0.7     # validation uses lower randomness

n_resp_per_prompt=${N_RESP:-4}
train_prompt_mini_bsz=${TRAIN_MINI_BSZ:-4}

total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-$((32 * 100))}
test_freq=${TEST_FREQ:-9999}
save_freq=${SAVE_FREQ:-50}

# Async training control
staleness_threshold=${STALENESS:-0.5}
trigger_parameter_sync_step=${SYNC_STEP:-2}
require_batches=${REQUIRE_BATCHES:-1}
partial_rollout=${PARTIAL_ROLLOUT:-True}
required_samples=${REQUIRED_SAMPLES:-128}

# Algorithm
adv_estimator=grpo
norm_adv_by_std_in_grpo=False      # DAPO-style: no std normalization
loss_agg_mode="token-mean"         # per-token loss averaging (ref: 8b_stale05_rs.sh)

# Performance
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
ref_offload=True
actor_offload=False
enforce_eager=False
nccl_timeout=7200                  # Megatron 长序列任务需要更大超时 (ref: verl_fully_async_mega.sh)

USE_MBRIDGE=${USE_MBRIDGE:-True}
USE_DIST_CKPT=${USE_DIST_CKPT:-False}

# ── Ensure PYTHONPATH ────────────────────────────────────────────────────
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export PYTHONPATH=${PYTHONPATH:-""}:$RLLM_DIR

# ── Data source args ────────────────────────────────────────────────────
DATA_ARGS=()
if [ -n "$TRAIN_FILE" ]; then
    DATA_ARGS+=(data.train_files="$TRAIN_FILE")
else
    DATA_ARGS+=(data.train_dataset_name="$DATASET_NAME")
fi
if [ -n "$TEST_FILE" ]; then
    DATA_ARGS+=(data.val_files="$TEST_FILE")
else
    DATA_ARGS+=(data.val_dataset_name="$VAL_DATASET_NAME")
fi

# ── Launch ───────────────────────────────────────────────────────────────
PYTHONUNBUFFERED=1 python -m examples.kernelgym.train_kernelgym_fully_async \
    "${DATA_ARGS[@]}" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.rollout_correction.bypass_mode=True \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.nccl_timeout=${nccl_timeout} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    \
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=2 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CKPT \
    actor_rollout_ref.actor.megatron.param_offload=${actor_offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${actor_offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=2 \
    \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    \
    trainer.logger=[console] \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.resume_mode=auto \
    trainer.nnodes=${n_nodes_train} \
    trainer.n_gpus_per_node=${n_gpus_training} \
    trainer.total_epochs=100 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.device=npu \
    \
    rollout.nnodes=${n_nodes_rollout} \
    rollout.n_gpus_per_node=${n_gpus_rollout} \
    rollout.total_rollout_steps=${total_rollout_steps} \
    \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.required_samples=${required_samples} \
    async_training.partial_rollout=${partial_rollout} \
    async_training.use_rollout_log_probs=True \
    async_training.compute_prox_log_prob=False \
    \
    ++kernel.server_url=${KERNEL_SERVER_URL} \
    ++kernel.backend=${KERNEL_BACKEND} \
    ++kernel.max_turns=${KERNEL_MAX_TURNS} \
    ++kernel.toolkit=kernelbench \
    ++kernel.timeout=${KERNEL_TIMEOUT} \
    ++kernel.num_correct_trials=5 \
    ++kernel.num_perf_trials=100
