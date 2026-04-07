#!/usr/bin/env bash
# ==============================================================================
# Train OpenHands agent with rllm (container-based, no openhands Python lib)
#
# Architecture:
#   Host
#   ├─ vLLM (GPU inference)
#   ├─ LiteLLM proxy  (rllm metadata-slug proxy, port PROXY_PORT)
#   └─ rllm training process (Ray workers)
#        └─ rollout()  in openhands_agent.py
#             ├─ build_proxied_base_url()  → metadata slug embedded in URL
#             └─ docker run ghcr.io/all-hands-ai/openhands
#                  └─ OpenHands headless → LLM_BASE_URL (proxied) → LiteLLM proxy
#
# Key knobs:
#   MODEL_PATH          — base model (HuggingFace ID or local path)
#   N_GPUS              — number of GPUs per node
#   PROXY_PORT          — LiteLLM proxy port
#   OPENHANDS_IMAGE     — OpenHands Docker image to run per rollout
#   OPENHANDS_DATASET   — swe (default) | mock_npu  (算子 bring-up：mock parquet + profiling mock)
# ==============================================================================
set -euo pipefail
set -x

# First time
export RLLM_UI_URL=http://127.0.0.1:3000
export FORCE_BUILD=0
export OPENHANDS_DATASET=mock_npu
export MODEL_PATH=/home/g00841271/Qwen3-8B

# export ASCEND_LAUNCH_BLOCKING=1

# export RAY_DEBUG_POST_MORTEM=0
export RAY_DEBUG_POST_MORTEM=1

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export PYTHONPATH=$PYTHONPATH:$RLLM_DIR

export OPENHANDS_IMAGE=openhands-triton-env:v1

export HYDRA_FULL_ERROR=1

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# ------------------------------------------------------------------------------
# vLLM / CUDA
# ------------------------------------------------------------------------------
export VLLM_ATTENTION_BACKEND="TORCH_SDPA"
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ------------------------------------------------------------------------------
# OpenHands container settings
# (read by openhands_agent.py → forwarded into each OpenHands container)
# ------------------------------------------------------------------------------
# Custom rllm-openhands image (built from workspace/Dockerfile).
# This image extends the official OpenHands image with workspace/entrypoint.py
# which uses the new OpenHands SDK (LLM, Agent, Conversation, Tool).
export OPENHANDS_IMAGE="${OPENHANDS_IMAGE:-openhands-triton-env:v1}"
export OPENHANDS_MODEL_NAME="${OPENHANDS_MODEL_NAME:-/home/g00841271/Qwen3-8B}"
export OPENHANDS_MAX_ITERATIONS="${OPENHANDS_MAX_ITERATIONS:-2}"
export OPENHANDS_CONTAINER_TIMEOUT="${OPENHANDS_CONTAINER_TIMEOUT:-600}"

# ------------------------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
N_GPUS="${N_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROXY_PORT="${PROXY_PORT:-4000}"
TRACE_DB_PATH="${TRACE_DB_PATH:-${HOME}/rllm-openhands-traces.db}"
PROJECT_NAME="${PROJECT_NAME:-rllm-openhands}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-openhands-ppo}"

echo "=== rllm + OpenHands (container-based) ==="
echo "  Model           : ${MODEL_PATH}"
echo "  N_GPUS (trainer) : ${N_GPUS}"
echo "  Proxy port      : ${PROXY_PORT}"
echo "  OpenHands image : ${OPENHANDS_IMAGE}"
echo "  Max iterations  : ${OPENHANDS_MAX_ITERATIONS}"

# ------------------------------------------------------------------------------
# Build the custom rllm-openhands image from workspace/Dockerfile.
# This image extends the official OpenHands base with workspace/entrypoint.py
# (uses new OpenHands SDK: LLM, Agent, Conversation, Tool).
# Rebuild only if image doesn't exist or FORCE_BUILD=1 is set.
# ------------------------------------------------------------------------------
# if [ "${FORCE_BUILD:-0}" = "1" ] || ! docker image inspect "${OPENHANDS_IMAGE}" &>/dev/null; then
#     echo "Building custom OpenHands image: ${OPENHANDS_IMAGE}"
#     docker build \
#         -t "${OPENHANDS_IMAGE}" \
#         -f examples/openhands_sdk/workspace/Dockerfile \
#         examples/openhands_sdk/workspace
# fi


echo "正在重启 Ray 集群清理 NPU 状态..."
ray stop --force || true
rm -rf /tmp/ray/*
sleep 2
ray start --head \
    --port 6379 \
    --dashboard-host 0.0.0.0 \
    --dashboard-port 8265 \
    --disable-usage-stats \
    --node-ip-address ${MASTER_ADDR}

# ------------------------------------------------------------------------------
# Launch training
# rollout() in openhands_agent.py runs directly in rllm Ray workers.
# Each rollout spawns its own OpenHands Docker container via docker run.
# No rllm sandbox (worker_server.py) wrapper is used.
# ------------------------------------------------------------------------------
python3 /home/g00841271/rllm-071/examples/openhands_sdk/train_openhands.py \
    algorithm.adv_estimator=grpo \
    \
    data.train_batch_size=${BATCH_SIZE} \
    data.val_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
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
    rllm.stepwise_advantage.enable=False \
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
    rllm.sdk.proxy.host=0.0.0.0 \
    trainer.device=npu \
    rllm.sdk.proxy.port=${PROXY_PORT} \
    rllm.sdk.proxy.mode=subprocess \
    rllm.sdk.store.path="${TRACE_DB_PATH}" \
    rllm.workflow.n_parallel_tasks=1


# pkill -9 -f 'ray::WorkerDict' 2>/dev/null || true
    # rllm.stepwise_advantage.mode=per_step \