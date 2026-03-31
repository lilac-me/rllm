#!/usr/bin/env bash
# ==============================================================================
# Train OpenHands agent with rllm AgentSDKEngine — NPU Operator Generation
#
# Uses sandbox mode:
#   - Agent code runs inside Docker containers
#   - worker_server.py inside containers handles LLM routing via metadata slug
#   - LiteLLM Proxy runs in subprocess mode on the host
#
# Usage:
#   cd /path/to/verllm
#   bash openhands-sdk/train_openhands.sh
# ==============================================================================
set -euo pipefail
set -x

ulimit -n 65535

# ------------------------------------------------------------------------------
# 1. Ascend/NPU 基础环境加载
# ------------------------------------------------------------------------------
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh 
set -u

# ------------------------------------------------------------------------------
# 2. vLLM / NPU 运行环境变量
# ------------------------------------------------------------------------------
export VLLM_ATTENTION_BACKEND="TORCH_SDPA"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_INTRA_ROCE_ENABLE=1
export HYDRA_FULL_ERROR=1 

# 将 CUDA_VISIBLE_DEVICES 替换为 ASCEND_RT_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export PYTHONPATH=$PYTHONPATH:$RLLM_DIR

# ------------------------------------------------------------------------------
# OpenHands agent settings
# ------------------------------------------------------------------------------
export OPENHANDS_MODEL_NAME="${OPENHANDS_MODEL_NAME:-openhands-model}"
export OPENHANDS_MAX_ITERATIONS="${OPENHANDS_MAX_ITERATIONS:-15}"
export OPENHANDS_RUNTIME_IMAGE="${OPENHANDS_RUNTIME_IMAGE:-docker.all-hands.dev/all-hands-ai/runtime:0.20-nikolaik}"

# ------------------------------------------------------------------------------
# Configurable parameters (N_GPUS 变更为 N_NPUS，语义更准确)
# ------------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/home/g00841271/Qwen3-8B}"
N_NPUS="${N_NPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PROXY_PORT="${PROXY_PORT:-4001}"
TRACE_DB_PATH="${TRACE_DB_PATH:-${HOME}/rllm-openhands-npu-traces.db}"
PROJECT_NAME="${PROJECT_NAME:-rllm-npu-agent}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-openhands-npu-bringup}"

SANDBOX_IMAGE="${SANDBOX_IMAGE:-rllm-openhands-sandbox}"

echo "=== rllm + OpenHands NPU operator training (Sandbox Mode) ==="
echo "  Model           : ${MODEL_PATH}"
echo "  NPUs            : ${N_NPUS}"
echo "  Batch size      : ${BATCH_SIZE}"
echo "  Proxy port      : ${PROXY_PORT}"
echo "  Trace DB        : ${TRACE_DB_PATH}"
echo "  Sandbox image   : ${SANDBOX_IMAGE}"
echo "  Max iterations  : ${OPENHANDS_MAX_ITERATIONS}"

# ------------------------------------------------------------------------------
# Build Docker sandbox image (if not already built)
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! docker image inspect "${SANDBOX_IMAGE}" &>/dev/null; then
    echo "Building Docker sandbox image: ${SANDBOX_IMAGE}"
    docker build -t "${SANDBOX_IMAGE}" -f "${SCRIPT_DIR}/Dockerfile" "${REPO_ROOT}"
fi

# ------------------------------------------------------------------------------
# 3. Ray 集群清理与重启 (参考 NPU 规范，防止 NPU 显存泄露)
# ------------------------------------------------------------------------------
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
# Generate mock parquet data (verl requires real parquet file paths)
# ------------------------------------------------------------------------------
MOCK_DATA_PATH="${SCRIPT_DIR}/mock_npu_data.parquet"
if [ ! -f "${MOCK_DATA_PATH}" ]; then
    echo "Generating mock NPU parquet data..."
    python3 "${SCRIPT_DIR}/create_mock_data.py"
fi

# ------------------------------------------------------------------------------
# 4. Launch training (包含 NPU 特有参数)
# ------------------------------------------------------------------------------
cd "${REPO_ROOT}"

# 此处保留原脚本直接 python3 的启动方式，由于 Ray 已经 start，任务会自动提交给 Ray
# python3 openhands-sdk/train_openhands.py \
#     algorithm.adv_estimator=grpo \
#     \
#     data.train_batch_size=${BATCH_SIZE} \
#     data.val_batch_size=4 \
#     data.max_prompt_length=4096 \
#     data.max_response_length=4096 \
#     data.train_files="[/home/g00841271/rllm-071/examples/openhands-sdk-ops/openhands-sdk/mock_npu_data.parquet]" \
#     data.val_files="[/home/g00841271/rllm-071/examples/openhands-sdk-ops/openhands-sdk/mock_npu_data.parquet]" \
#     actor_rollout_ref.model.path=${MODEL_PATH} \
#     actor_rollout_ref.hybrid_engine=True \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.strategy=fsdp2 \
#     actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \
#     actor_rollout_ref.actor.use_dynamic_bsz=True \
#     actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0.0 \
#     actor_rollout_ref.actor.clip_ratio_low=0.2 \
#     actor_rollout_ref.actor.clip_ratio_high=0.28 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=True \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
#     actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
#     \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.mode=async \
#     ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.temperature=1.0 \
#     actor_rollout_ref.rollout.top_p=1.0 \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
#     actor_rollout_ref.rollout.n=2 \
#     actor_rollout_ref.rollout.val_kwargs.n=1 \
#     actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
#     actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
#     \
#     rllm.agent.max_steps=${OPENHANDS_MAX_ITERATIONS} \
#     rllm.stepwise_advantage.enable=True \
#     rllm.stepwise_advantage.mode=per_step \
#     rllm.compact_filtering.enable=True \
#     rllm.compact_filtering.mask_max_prompt_length_exceeded=True \
#     rllm.compact_filtering.mask_max_response_length_exceeded=True \
#     rllm.compact_filtering.mask_max_turns_exceeded=False \
#     rllm.compact_filtering.mask_timeout=True \
#     rllm.rejection_sample.enable=False \
#     \
#     trainer.critic_warmup=0 \
#     "trainer.logger=['console','wandb']" \
#     trainer.project_name=${PROJECT_NAME} \
#     trainer.experiment_name=${EXPERIMENT_NAME} \
#     trainer.val_before_train=False \
#     trainer.n_gpus_per_node=${N_NPUS} \
#     trainer.nnodes=1 \
#     trainer.device=npu \
#     trainer.save_freq=100 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=10 \
#     \
#     rllm.sdk.proxy.host=127.0.0.1 \
#     rllm.sdk.proxy.port=${PROXY_PORT} \
#     rllm.sdk.proxy.mode=subprocess \
#     rllm.sdk.store.path="${TRACE_DB_PATH}" \
#     \
#     rllm.sdk.sandbox.enabled=True \
#     rllm.sdk.sandbox.backend=docker \
#     "rllm.sdk.sandbox.image=${SANDBOX_IMAGE}" \
#     rllm.sdk.sandbox.agent_dir=openhands-sdk \
#     rllm.sdk.sandbox.agent_module=openhands_agent \
#     rllm.sdk.sandbox.agent_func=rollout \
#     rllm.sdk.sandbox.pool_mode=per_task \
#     rllm.sdk.sandbox.max_concurrent=2 \
#     rllm.sdk.sandbox.execution_timeout=700 \
#     \
#     "++rllm.sdk.sandbox.extra.docker_volumes=[\"/var/run/docker.sock:/var/run/docker.sock\"]"

# # Clean up any stale Ray workers after training

ray job submit --address="http://${MASTER_ADDR}:8265" \
    -- python3 /home/g00841271/rllm-071/examples/openhands-sdk-ops/openhands-sdk/train_openhands.py \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/g00841271/rllm/rllm/data/datasets/SWE_Bench_Verified/test2_verl_mirror.parquet \
    data.val_files=/home/g00841271/rllm/rllm/data/datasets/SWE_Bench_Verified/test2_verl_mirror.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=/home/g00841271/Qwen3-8B \
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
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    rllm.agent.max_steps=${OPENHANDS_MAX_ITERATIONS} \
    rllm.stepwise_advantage.enable=True \
    rllm.stepwise_advantage.mode=per_step \
    rllm.compact_filtering.enable=True \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=True \
    rllm.compact_filtering.mask_max_response_length_exceeded=True \
    rllm.compact_filtering.mask_max_turns_exceeded=False \
    rllm.compact_filtering.mask_timeout=True \
    rllm.rejection_sample.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${N_NPUS} \
    trainer.nnodes=1 \
    trainer.device=npu \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    rllm.sdk.proxy.host=127.0.0.1 \
    rllm.sdk.proxy.port=${PROXY_PORT} \
    rllm.sdk.proxy.mode=subprocess \
    rllm.sdk.store.path="${TRACE_DB_PATH}" \
    rllm.sdk.sandbox.enabled=True \
    rllm.sdk.sandbox.backend=docker \
    rllm.sdk.sandbox.image="${SANDBOX_IMAGE}" \
    rllm.sdk.sandbox.agent_dir=openhands-sdk \
    rllm.sdk.sandbox.agent_module=openhands_agent \
    rllm.sdk.sandbox.agent_func=rollout \
    rllm.sdk.sandbox.pool_mode=per_task \
    rllm.sdk.sandbox.max_concurrent=2 \
    rllm.sdk.sandbox.execution_timeout=700 \
    ++rllm.sdk.sandbox.extra.docker_volumes="['/var/run/docker.sock:/var/run/docker.sock']"

ray stop --force || true
pkill -9 -f 'ray::WorkerDict' 2>/dev/null || true