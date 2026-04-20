#!/usr/bin/env bash
set -xeuo pipefail

# ── KernelGYM Mock Test Script ───────────────────────────────────────────
# Tests the full async training pipeline WITHOUT a real KernelGYM eval server.
# vLLM generation is REAL — only the kernel evaluation returns fake results.
#
# This is useful for verifying:
#   - vLLM response generation works
#   - Multi-turn rollout completes max_turns iterations
#   - Fake rewards flow into the trainer
#   - PPO training step executes successfully

# ── Environment ──────────────────────────────────────────────────────────
export HYDRA_FULL_ERROR=1
export RAY_DEBUG_POST_MORTEM=0

# NPU memory allocator
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
CKPTS_DIR=${CKPTS_DIR:-"./checkpoints/kernelgym-mock-test"}

# ── Topology (small for testing) ─────────────────────────────────────────
n_gpus_rollout=${N_GPUS_ROLLOUT:-4}
n_gpus_training=${N_GPUS_TRAINING:-4}
n_nodes_rollout=${N_NODES_ROLLOUT:-1}
n_nodes_train=${N_NODES_TRAIN:-1}

# ── Hyper-params (minimal for smoke test) ────────────────────────────────
rollout_mode="async"
rollout_name=${ROLLOUT_NAME:-"vllm"}

max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 4))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 2))}

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

n_resp_per_prompt=2
train_prompt_mini_bsz=2
total_rollout_steps=32
test_freq=9999
save_freq=9999

staleness_threshold=0.5
trigger_parameter_sync_step=2
require_batches=1
partial_rollout=True
required_samples=16

loss_agg_mode="token-mean"
nccl_timeout=7200

use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
actor_offload=False
enforce_eager=False

USE_MBRIDGE=${USE_MBRIDGE:-True}
USE_DIST_CKPT=${USE_DIST_CKPT:-False}

# ── Ensure PYTHONPATH ────────────────────────────────────────────────────
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export PYTHONPATH=${PYTHONPATH:-""}:$RLLM_DIR

# ── Launch (uses mock dataset registered in-code) ────────────────────────
PYTHONUNBUFFERED=1 python -m examples.kernelgym.train_kernelgym_mock \
    data.train_dataset_name=kernelbench_mock \
    data.val_dataset_name=kernelbench_mock \
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
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
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
    actor_rollout_ref.actor.optim.lr_warmup_steps=2 \
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
    trainer.resume_mode=disable \
    trainer.nnodes=${n_nodes_train} \
    trainer.n_gpus_per_node=${n_gpus_training} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=4 \
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
    ++kernel.max_turns=2
