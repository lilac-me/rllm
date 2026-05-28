
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
pkill -9 python
pkill -9 torchrun
set -euo pipefail
set -x

# -----------------------------------------------------------------------------
# Reap OpenHands containers left over from previous runs.
#   - Normal rollouts already self-clean via the finally block in
#     openhands_agent.py / remote_eval_worker.py (docker rm -f).
#   - This sweep catches orphans from SIGKILLed trainers or Ray worker
#     recycles, plus prunes the dangling-container pool that builds up over
#     long training (each rollout = 1 new overlay2 layer).
#   - Stuck "Created" containers from a dockerd state-machine bug usually
#     ignore "docker rm -f" and need a daemon restart — we don't attempt that
#     here, fail soft and continue.
# -----------------------------------------------------------------------------
docker ps -a --filter "name=rllm-openhands-" -q 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
docker container prune -f 2>/dev/null || true

# -----------------------------------------------------------------------------
# Load machine/path/network config (paths, IPs, ports, image, memory tuning).
# Override which config file is loaded with RLLM_CONFIG_FILE.
# Variables exported from the config: MODEL_PATH, TRAIN_HOST_IP, EVAL_WORKER_IP,
# PROXY_PORT, EVAL_WORKER_PORT, NIC_NAME, OPENHANDS_IMAGE, RLLM_LOG_DIR,
# TRACE_DB_PATH, OPENHANDS_ARTIFACT_DIR, PROFILE_SAVE_PATH,
# OPENHANDS_KERNELBENCH_PARQUET, PYTORCH_NPU_ALLOC_CONF, HCCL_BUFFSIZE.
# See config/.env.example for the full template.
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${RLLM_CONFIG_FILE:-$SCRIPT_DIR/config/qwen30b.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config file not found: $CONFIG_FILE" >&2
    echo "Copy $SCRIPT_DIR/config/.env.example to config/qwen30b.env (or set RLLM_CONFIG_FILE)." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$CONFIG_FILE"

# -----------------------------------------------------------------------------
# Worker health check (fail-fast if remote_eval_worker.py isn't running).
#   - Skipped when OPENHANDS_REMOTE_EVAL_URL is empty (legacy same-machine
#     docker-run path; not the recommended deployment).
#   - Worker exposes GET /health -> 200 (remote_eval_worker.py:190).
#   - The cmd MUST be reachable BEFORE training starts, otherwise the first
#     rollout hangs on a 2100s HTTP timeout — we'd rather die at second 3.
# -----------------------------------------------------------------------------
if [[ -n "${OPENHANDS_REMOTE_EVAL_URL:-}" ]]; then
    if ! curl -sf -m 3 "${OPENHANDS_REMOTE_EVAL_URL}/health" >/dev/null 2>&1; then
        echo "ERROR: remote_eval_worker not reachable at ${OPENHANDS_REMOTE_EVAL_URL}/health" >&2
        echo "Start it first, e.g. on the host:" >&2
        echo "  python3 remote_eval_worker.py --host 0.0.0.0 --port 16881 --work-dir /tmp/openhands_workspace" >&2
        exit 1
    fi
    echo "[debug_oom] worker health OK: ${OPENHANDS_REMOTE_EVAL_URL}"
fi

export LLM_MAX_OUTPUT_TOKENS="${LLM_MAX_OUTPUT_TOKENS:-2048}"

export ASCEND_LAUNCH_BLOCKING=0 # TODO

nic_name="$NIC_NAME"
export HCCL_IF_IP="$TRAIN_HOST_IP"
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_INTRA_ROCE_ENABLE=1 # TODO
export HCCL_INTRA_PCIE_ENABLE=0 # TODO
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
# export HCCL_CONNECT_TIMEOUT=300
# HCCL_BUFFSIZE comes from config

export RAY_DEBUG_POST_MORTEM=0
export RAY_DEDUP_LOGS=0
export VLLM_ASCEND_ENABLE_NZ=0

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export PYTHONPATH=$PYTHONPATH:$RLLM_DIR

# OPENHANDS_IMAGE comes from config

export HYDRA_FULL_ERROR=1

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# ------------------------------------------------------------------------------
# vLLM / CUDA
# ------------------------------------------------------------------------------
export VLLM_ATTENTION_BACKEND="TORCH_SDPA"
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
# PYTORCH_NPU_ALLOC_CONF comes from config (max_split_size_mb:2048 — BS=3 OOM fix)
# DO NOT add expandable_segments:True — conflicts with vLLM sleep mode
#   (vllm-ascend/vllm_ascend/device_allocator/camem.py:151 has an assert)
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# NPU pool, remote-eval URL, and container host alias all come from config:
#   ASCEND_RT_VISIBLE_DEVICES    trainer NPU pool (e.g. 0-7)
#   OPENHANDS_EVAL_DEVICE_IDS    NPUs OpenHands containers may use
#                                (e.g. 0-3 on a separate eval host;
#                                 8-15 in single-host 16-NPU layout)
#   OPENHANDS_REMOTE_EVAL_URL    empty -> same-machine docker-run;
#                                http://host:port -> remote_eval_worker.py
#   OPENHANDS_CONTAINER_HOST_ALIAS  host.docker.internal in single-host;
#                                   train-host IP in multi-host (must reach
#                                   the LiteLLM proxy from inside the
#                                   OpenHands container)

export TOKENIZERS_PARALLELISM=true
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_LEVEL=WARNING
export HYDRA_FULL_ERROR=1

# ------------------------------------------------------------------------------
# OpenHands container settings
# (read by openhands_agent.py → forwarded into each OpenHands container)
# ------------------------------------------------------------------------------
# Custom rllm-openhands image (built from workspace/Dockerfile).
# This image extends the official OpenHands image with workspace/entrypoint.py
# which uses the new OpenHands SDK (LLM, Agent, Conversation, Tool).
# OPENHANDS_IMAGE already exported from config — no override here.
export OPENHANDS_MODEL_NAME=$MODEL_PATH
export OPENHANDS_BASE_URL_PORT=${PROXY_PORT}
if [[ "$MODEL_PATH" == *"Qwen3-Coder"* ]]; then
    export OPENHANDS_MAX_ITERATIONS="${OPENHANDS_MAX_ITERATIONS:-10}"
else
    export OPENHANDS_MAX_ITERATIONS="${OPENHANDS_MAX_ITERATIONS:-20}"
fi
export OPENHANDS_CONTAINER_TIMEOUT="${OPENHANDS_CONTAINER_TIMEOUT:-1800}"
# OPENHANDS_ARTIFACT_DIR already exported from config

# Optional: reserve a separate NPU pool for OpenHands operator validation.
# For strict isolation, remove these ids from ASCEND_RT_VISIBLE_DEVICES above.
export OPENHANDS_EVAL_DEVICE_IDS="${OPENHANDS_EVAL_DEVICE_IDS:-}"
# Empty keeps the original same-machine docker-run path.
# To run OpenHands/evaluation on another machine:
#   1) start: python examples/openhands_sdk/remote_eval_worker.py --host 0.0.0.0 --port 16880
#   2) set OPENHANDS_REMOTE_EVAL_URL=http://<eval-machine-ip>:16880
#   3) set OPENHANDS_CONTAINER_HOST_ALIAS=<train-machine-ip> so remote containers reach the LLM proxy.
export OPENHANDS_REMOTE_EVAL_URL="${OPENHANDS_REMOTE_EVAL_URL:-}"
export OPENHANDS_CONTAINER_HOST_ALIAS="${OPENHANDS_CONTAINER_HOST_ALIAS:-host.docker.internal}"


# ------------------------------------------------------------------------------
# 数据集处理
# export OPENHANDS_DATASET=mock_npu
# export OPENHANDS_DATASET=drkernel
# export OPENHANDS_DRKERNEL_DATASET=/home/p00938733/datasets/HuggingFaceH4/drkernel_rl_data/
# export OPENHANDS_DRKERNEL_PARQUET=/home/p00938733/datasets/HuggingFaceH4/drkernel_rl_operator.parquet
# export OPENHANDS_DRKERNEL_MAX_ROWS=128
# export OPENHANDS_DRKERNEL_ARCH=ascend910b1
# export OPENHANDS_DRKERNEL_OPERATOR_BACKEND=triton

export OPENHANDS_DATASET=kernelbench
export OPENHANDS_KERNELBENCH_LEVELS=level_1
# export OPENHANDS_KERNELBENCH_LEVELS=level_1,level_2
# OPENHANDS_KERNELBENCH_PARQUET comes from config
export OPENHANDS_KERNELBENCH_MAX_ROWS=128
export OPENHANDS_KERNELBENCH_ARCH=ascend910b1
export OPENHANDS_KERNELBENCH_OPERATOR_BACKEND=triton

export OPENHANDS_REWARD_TARGET_SPEEDUP=2.0

# 默认 warmup，会过滤复杂样本
export OPENHANDS_KERNELBENCH_FILTER_MODE=warmup

# 不过滤，使用全部样本
# export OPENHANDS_KERNELBENCH_FILTER_MODE=all

# 只保留包含关键词的样本
# export OPENHANDS_KERNELBENCH_INCLUDE_KEYWORDS=relu,add,mul,sigmoid,tanh,clamp

# 自定义排除关键词，会覆盖 warmup 默认排除列表
# export OPENHANDS_KERNELBENCH_EXCLUDE_KEYWORDS=conv_transpose,attention,lstm

# 过滤过长代码，避免 prompt 太大
# export OPENHANDS_KERNELBENCH_MAX_CODE_CHARS=10000
# export OPENHANDS_KERNELBENCH_MAX_INPUT_ELEMENTS=67108864
# export OPENHANDS_KERNELBENCH_MAX_OUTPUT_ELEMENTS=67108864
export OPENHANDS_PIPELINE_ERROR_PREVIEW_CHARS=1000
# ------------------------------------------------------------------------------

export RLLM_VLLM_MALLOC_TRIM=1
export RLLM_VLLM_TRIM_INTERVAL=1
export RLLM_DISABLE_VLLM_INSTRUMENTATION=1

# ------------------------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------------------------
N_GPUS="${N_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ROLLOUT_N="${ROLLOUT_N:-8}"
PROXY_PORT="${PROXY_PORT:-4000}"
# TRACE_DB_PATH comes from config
PROJECT_NAME="${PROJECT_NAME:-rllm-openhands}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rllm-openhands}"
logs="${RLLM_LOG_DIR}/verl-rllm.log"

# profiling configuration
PROFILE_STEPS="[1]"
PROFILE_RANKS_ALL=False
RANKS="[0]"
DISCRETE=False
# PROFILE_CONTINUOUS_STEPS=True

# profiling NPU options
SAVE_PATH="${PROFILE_SAVE_PATH}"
LEVEL="level0"
CONTENTS=['npu','cpu','memory']
#CONTENTS=['npu','cpu','memory','module','stack']
ANALYSIS=True

# export OOM_SNAPSHOT_ENABLE=1
# export OOM_SNAPSHOT_PATH="/home/p00938733/profile_data"

if [[ -z "${TOOL_PARSER:-}" ]]; then
    if [[ "$MODEL_PATH" == *"Qwen3-Coder"* || "$MODEL_PATH" == *"cszhou_sft_weight"* ]]; then
        TOOL_PARSER=qwen3_coder
        EMP_size=8
        ETP_size=1
    else
        TOOL_PARSER=hermes
        EMP_size=1
        ETP_size=1
    fi
fi

echo "=== rllm + OpenHands (container-based) ==="
echo "  Model           : ${MODEL_PATH}"
echo "  N_GPUS (trainer) : ${N_GPUS}"
echo "  Proxy port      : ${PROXY_PORT}"
echo "  OpenHands image : ${OPENHANDS_IMAGE}"
echo "  Max iterations  : ${OPENHANDS_MAX_ITERATIONS}"
echo "  Tool parser     : ${TOOL_PARSER}"

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

#python prepare_kernelbench_openhands_data.py

echo "正在重启 Ray 集群清理 NPU 状态..."
ray stop --force || true
rm -rf /tmp/ray/*
sleep 2
ray start --head \
    --port 6379 \
    --dashboard-host 0.0.0.0 \
    --dashboard-port 8265 \
    --disable-usage-stats \
    --node-ip-address ${MASTER_ADDR} \
    --object-store-memory=$((32 * 1024 * 1024 * 1024))

# ------------------------------------------------------------------------------
# Launch training
# rollout() in openhands_agent.py runs directly in rllm Ray workers.
# Each rollout spawns its own OpenHands Docker container via docker run.
# No rllm sandbox (worker_server.py) wrapper is used.
# ------------------------------------------------------------------------------

ARGS=(
  # =========================
  # algorithm
  # =========================
  algorithm.adv_estimator=grpo
  algorithm.kl_ctrl.kl_coef=0.001

  # =========================
  # data
  # =========================
  data.train_batch_size=${BATCH_SIZE}
  data.val_batch_size=16
  data.max_prompt_length=16384      # 24K
  data.max_response_length=${LLM_MAX_OUTPUT_TOKENS}     # 18K

  # =========================
  # actor_rollout_ref - common
  # =========================
  actor_rollout_ref.hybrid_engine=True
  # actor_rollout_ref.model.use_shm=True
  actor_rollout_ref.model.path=${MODEL_PATH}
  actor_rollout_ref.model.use_remove_padding=True
  # actor_rollout_ref.model.enable_gradient_checkpointing=True
  # actor_rollout_ref.model.use_fused_kernels=True

  # =========================
  # actor - optimization / PPO
  # =========================
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.ppo_mini_batch_size=4
  # actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 #32768
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28

  # =========================
  # actor.megatron
  # =========================
  actor_rollout_ref.actor.strategy=megatron
  actor_rollout_ref.actor.megatron.use_mbridge=True
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
  # actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH

  actor_rollout_ref.actor.megatron.param_offload=True
  actor_rollout_ref.actor.megatron.grad_offload=True
  actor_rollout_ref.actor.megatron.optimizer_offload=False

  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.actor.megatron.context_parallel_size=2
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EMP_size}
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP_size}
  +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=2
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  +actor_rollout_ref.actor.checkpoint.save_contents="['model']"

  # =========================
  # actor optimizer override
  # =========================
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
  # # +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True

  # =========================
  # ref
  # =========================
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.ref.megatron.context_parallel_size=2
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EMP_size}
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP_size}
  actor_rollout_ref.ref.megatron.param_offload=True
  actor_rollout_ref.ref.megatron.use_mbridge=True
  actor_rollout_ref.ref.megatron.use_dist_checkpointing=False

  # =========================
  # rollout
  # =========================
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
  actor_rollout_ref.rollout.calculate_log_probs=False        # 记录 训推的 log_prob 确认是否存在diff
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.enforce_eager=False # TODO
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.max_model_len=32768
  actor_rollout_ref.rollout.max_num_seqs=1
  actor_rollout_ref.rollout.max_num_batched_tokens=16384
  actor_rollout_ref.rollout.n=${ROLLOUT_N}
  actor_rollout_ref.rollout.val_kwargs.n=4
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True
  +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=${TOOL_PARSER}

  # actor_rollout_ref.rollout.enable_chunked_prefill=False
  # actor_rollout_ref.rollout.enable_prefix_caching=False
  # actor_rollout_ref.rollout.free_cache_engine=True
  # +actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=0
  # +actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb=0
  
  # =========================
  # rllm
  # =========================
  rllm.mask_truncated_samples=False
#   +rllm.agent.engine_args.n_parallel_agents=64  # AgentExecutionEngine
#   rllm.agent.max_steps=5
  rllm.stepwise_advantage.enable=True
  rllm.stepwise_advantage.mode=broadcast
  rllm.compact_filtering.enable=True
  rllm.compact_filtering.mask_max_prompt_length_exceeded=True
  rllm.compact_filtering.mask_max_response_length_exceeded=True
  rllm.compact_filtering.mask_max_turns_exceeded=True
  rllm.compact_filtering.mask_timeout=True
  rllm.rejection_sample.enable=False
  rllm.sdk.proxy.host=0.0.0.0
  rllm.sdk.proxy.port=${PROXY_PORT}
  rllm.sdk.proxy.mode=subprocess
  rllm.sdk.store.path="${TRACE_DB_PATH}"
  rllm.workflow.n_parallel_tasks=16
  
  # =========================
  # trainer
  # =========================
  trainer.critic_warmup=0
  trainer.logger='["console"]'
  trainer.project_name=${PROJECT_NAME}
  trainer.experiment_name=${EXPERIMENT_NAME}
  trainer.val_before_train=False
  trainer.n_gpus_per_node=${N_GPUS}
  trainer.nnodes=1
  trainer.device=npu
  trainer.save_freq=20
  trainer.test_freq=20
  trainer.default_hdfs_dir=null
  trainer.total_epochs=100

  # =========================
  # profiler
  # =========================
  # actor_rollout_ref.rollout.profiler.enable=True
  # actor_rollout_ref.rollout.profiler.all_ranks=$PROFILE_RANKS_ALL
  # actor_rollout_ref.rollout.profiler.ranks=$RANKS
  # actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=True
  # actor_rollout_ref.rollout.profiler.tool_config.npu.contents=$CONTENTS
  # actor_rollout_ref.rollout.profiler.tool_config.npu.level=$LEVEL
  # actor_rollout_ref.rollout.profiler.tool_config.npu.analysis=$ANALYSIS
  # actor_rollout_ref.ref.profiler.enable=True
  # actor_rollout_ref.ref.profiler.all_ranks=$PROFILE_RANKS_ALL
  # actor_rollout_ref.ref.profiler.ranks=$RANKS
  # actor_rollout_ref.ref.profiler.tool_config.npu.discrete=$DISCRETE
  # actor_rollout_ref.ref.profiler.tool_config.npu.contents=$CONTENTS
  # actor_rollout_ref.ref.profiler.tool_config.npu.level=$LEVEL
  # actor_rollout_ref.ref.profiler.tool_config.npu.analysis=$ANALYSIS
  actor_rollout_ref.actor.profiler.enable=True
  actor_rollout_ref.actor.profiler.all_ranks=$PROFILE_RANKS_ALL
  actor_rollout_ref.actor.profiler.ranks=$RANKS
  actor_rollout_ref.actor.profiler.tool_config.npu.discrete=$DISCRETE
  actor_rollout_ref.actor.profiler.tool_config.npu.contents=$CONTENTS
  actor_rollout_ref.actor.profiler.tool_config.npu.level=$LEVEL
  actor_rollout_ref.actor.profiler.tool_config.npu.analysis=$ANALYSIS
  global_profiler.tool=npu
  global_profiler.steps=$PROFILE_STEPS
  global_profiler.save_path=$SAVE_PATH
)

ray job submit --address="http://${MASTER_ADDR}:8265" \
    -- \
    python3 -m examples.openhands_sdk.train_open_megatron "${ARGS[@]}" \
    2>&1 | tee -i $logs

