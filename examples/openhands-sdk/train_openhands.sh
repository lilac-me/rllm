#!/usr/bin/env bash
# ==============================================================================
# Train OpenHands agent with rllm AgentSDKEngine (PPO/GRPO)
#
# Uses sandbox mode (方案 B):
#   - Agent code runs inside Docker containers
#   - worker_server.py inside containers handles LLM routing via metadata slug
#   - LiteLLM Proxy runs in subprocess mode on the host
#
# Usage:
#   bash examples/openhands/train_openhands.sh
#
# Key knobs:
#   MODEL_PATH   — base model (HuggingFace ID or local path)
#   N_GPUS       — number of GPUs per node
#   BATCH_SIZE   — training batch size (number of rollouts per step)
#   PROXY_PORT   — LiteLLM proxy port
# ==============================================================================
set -euo pipefail
set -x

# ------------------------------------------------------------------------------
# vLLM / CUDA environment
# ------------------------------------------------------------------------------
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ------------------------------------------------------------------------------
# OpenHands agent settings — forwarded via environment variables into the sandbox
# (see openhands_agent.py for all supported variables)
# ------------------------------------------------------------------------------
export OPENHANDS_MODEL_NAME="${OPENHANDS_MODEL_NAME:-openhands-model}"
export OPENHANDS_MAX_ITERATIONS="${OPENHANDS_MAX_ITERATIONS:-30}"
# Docker image used by OpenHands' docker runtime for per-rollout code execution
export OPENHANDS_RUNTIME_IMAGE="${OPENHANDS_RUNTIME_IMAGE:-docker.all-hands.dev/all-hands-ai/runtime:0.20-nikolaik}"

# ------------------------------------------------------------------------------
# Configurable parameters
# ------------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
N_GPUS="${N_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROXY_PORT="${PROXY_PORT:-4000}"
TRACE_DB_PATH="${TRACE_DB_PATH:-${HOME}/rllm-openhands-traces.db}"
PROJECT_NAME="${PROJECT_NAME:-rllm-openhands}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-openhands-ppo}"

# Docker image for the rllm sandbox worker (NOT the OpenHands runtime image)
SANDBOX_IMAGE="${SANDBOX_IMAGE:-rllm-openhands-sandbox}"

echo "=== rllm + OpenHands training (Sandbox Mode) ==="
echo "  Model           : ${MODEL_PATH}"
echo "  GPUs            : ${N_GPUS}"
echo "  Batch size      : ${BATCH_SIZE}"
echo "  Proxy port      : ${PROXY_PORT}"
echo "  Trace DB        : ${TRACE_DB_PATH}"
echo "  Sandbox image   : ${SANDBOX_IMAGE}"
echo "  Max iterations  : ${OPENHANDS_MAX_ITERATIONS}"

# ------------------------------------------------------------------------------
# Build Docker sandbox image (if not already built)
# ------------------------------------------------------------------------------
if ! docker image inspect "${SANDBOX_IMAGE}" &>/dev/null; then
    echo "Building Docker sandbox image: ${SANDBOX_IMAGE}"
    docker build -t "${SANDBOX_IMAGE}" -f examples/openhands/Dockerfile .
fi

# ------------------------------------------------------------------------------
# Launch training
# ------------------------------------------------------------------------------
python3 -m examples.openhands.train_openhands \
    algorithm.adv_estimator=grpo \
    \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    rllm.agent.max_steps=30 \
    rllm.stepwise_advantage.enable=True \
    rllm.stepwise_advantage.mode=per_step \
    rllm.compact_filtering.enable=True \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=True \
    rllm.compact_filtering.mask_max_response_length_exceeded=True \
    rllm.compact_filtering.mask_max_turns_exceeded=False \
    rllm.compact_filtering.mask_timeout=True \
    rllm.rejection_sample.enable=False \
    \
    trainer.critic_warmup=0 \
    "trainer.logger=['console','wandb']" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=50 \
    \
    rllm.sdk.proxy.host=127.0.0.1 \
    rllm.sdk.proxy.port=${PROXY_PORT} \
    rllm.sdk.proxy.mode=subprocess \
    rllm.sdk.proxy.enable_result_store=True \
    "rllm.sdk.proxy.model_name=${OPENHANDS_MODEL_NAME}" \
    rllm.sdk.store.path="${TRACE_DB_PATH}" \
    \
    rllm.sdk.sandbox.enabled=True \
    rllm.sdk.sandbox.backend=docker \
    "rllm.sdk.sandbox.image=${SANDBOX_IMAGE}" \
    rllm.sdk.sandbox.agent_dir=examples/openhands \
    rllm.sdk.sandbox.agent_module=openhands_agent \
    rllm.sdk.sandbox.agent_func=rollout \
    rllm.sdk.sandbox.pool_mode=per_task \
    rllm.sdk.sandbox.max_concurrent=16 \
    rllm.sdk.sandbox.execution_timeout=700 \
    \
    "rllm.sdk.sandbox.extra.docker_volumes=[\"/var/run/docker.sock:/var/run/docker.sock\"]"


pkill -9 -f 'ray::WorkerDict' 2>/dev/null || true