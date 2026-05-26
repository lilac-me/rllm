#!/usr/bin/env bash
# ==============================================================================
# Stage1 training: Qwen3.6-35B-A3B × OpenHands × Ascend NPU
#
# Per QWEN36_OPENHANDS_AGENT_RL_PLAN.md (v2.4):
#   §0.1  Required NPU env: CUDA_DEVICE_MAX_CONNECTIONS=1,
#         VLLM_USE_V1=1, VLLM_ALLREDUCE_USE_SYMM_MEM=0
#   §0.2  verl-script-authoritative MoE / megatron flags
#         (vanilla_mbridge, moe_aux_loss_coeff, moe_z_loss_coeff,
#         moe_permute_fusion, moe_grouped_gemm)
#   §5.2  max_model_len = 49k (W2.21 L3 unblock; was 32k stage1 baseline)
#   §13.7 Safe stage1 config:
#         - router_replay disabled (no-op on AgentPPOTrainer path anyway,
#           see §13.3; set explicitly for clarity)
#         - use_kl_loss=False, kl_loss_coef=0.0 (rely on PPO clip)
#         - calculate_log_probs=True (required for rollout_probs_diff metric)
#
# Single-node 8-NPU parallelism (user decision 2026-05-25):
#   TP=2, PP=1, CP=1, EP=4, ETP=1  →  TP*EP = 8 ranks per model replica
#
# Architecture (unchanged from train_open_megatron.sh):
#   Host
#   ├─ vLLM (NPU inference)
#   ├─ LiteLLM proxy  (rllm metadata-slug proxy, port PROXY_PORT)
#   └─ rllm training process (Ray workers)
#        └─ rollout()  in openhands_agent.py
#             ├─ build_proxied_base_url()  → metadata slug embedded in URL
#             └─ docker run ${OPENHANDS_IMAGE}
#                  └─ OpenHands headless → LLM_BASE_URL (proxied) → LiteLLM proxy
#
# Key knobs:
#   MODEL_PATH          — base model (HuggingFace ID or local path)
#   N_GPUS              — number of NPU devices per node
#   PROXY_PORT          — LiteLLM proxy port
#   OPENHANDS_IMAGE     — OpenHands Docker image to run per rollout
#   OPENHANDS_DATASET   — mock_npu (default) | swe  (algorithmic bring-up dataset)
# ==============================================================================
# pkill -9 python
# pkill -9 torchrun
set -euo pipefail
set -x

# First time
export FORCE_BUILD=0
export OPENHANDS_DATASET=mock_npu
export MODEL_PATH=/home/docker/Qwen3.6-35B-A3B
export PROXY_PORT=5000

# export ASCEND_LAUNCH_BLOCKING=1 # TODO

# ------------------------------------------------------------------------------
# HCCL / network env — opt-in only (W2.15)
#
# The legacy obs branch hardcoded IP 80.48.5.88 + nic 'ens1f3' for a specific
# dev machine. That broke L2b when run from a different NPU node (HCCL
# HcclGetRootInfo error code 6, because the hardcoded IF_IP doesn't exist on
# the host). verl-native scripts (e.g. run_qwen3_5_35b_megatron.sh) do NOT
# set these and rely on container defaults — Qwen3.6 trains fine that way.
#
# So we leave them unset by default and only export when the user opts in.
# If you DO need overrides on a multi-NIC machine, set the corresponding
# env BEFORE running this script:
#     HCCL_IF_IP_OVERRIDE=10.x.y.z \
#     HCCL_NIC_NAME=enp0s8 \
#     HCCL_FORCE_PORT_RANGE=1 \
#     bash train_openhands_qwen36_npu.sh
# ------------------------------------------------------------------------------
if [[ -n "${HCCL_IF_IP_OVERRIDE:-}" ]]; then
    export HCCL_IF_IP="${HCCL_IF_IP_OVERRIDE}"
    echo "[stage1] HCCL_IF_IP=${HCCL_IF_IP} (override)"
fi
if [[ -n "${HCCL_NIC_NAME:-}" ]]; then
    export GLOO_SOCKET_IFNAME="${HCCL_NIC_NAME}"
    export TP_SOCKET_IFNAME="${HCCL_NIC_NAME}"
    export HCCL_SOCKET_IFNAME="${HCCL_NIC_NAME}"
    echo "[stage1] *_SOCKET_IFNAME=${HCCL_NIC_NAME} (override)"
fi
if [[ "${HCCL_FORCE_PORT_RANGE:-0}" != "0" ]]; then
    # plan §11.3: only enforce when there's port contention (multiple verl
    # instances on the same host); otherwise let HCCL pick freely.
    export HCCL_INTRA_ROCE_ENABLE=1
    export HCCL_INTRA_PCIE_ENABLE=0
    export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
    export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
    echo "[stage1] HCCL_*_SOCKET_PORT_RANGE forced (override)"
fi

export RAY_DEBUG_POST_MORTEM=0
export RAY_DEDUP_LOGS=0
export VLLM_ASCEND_ENABLE_NZ=0

# vLLM timeout settings — NPU inference can be slow, especially during cudagraph warmup
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=86400  # 24 hours
export VLLM_RPC_TIMEOUT=86400000                # 24 hours (ms)

# ------------------------------------------------------------------------------
# Stage1 required env (plan §0.1) — MUST be set, training will be unstable
# or crash without these. New in train_openhands_qwen36_npu.sh vs the
# legacy train_open_megatron.sh:
#   - CUDA_DEVICE_MAX_CONNECTIONS=1   Megatron communication ordering
#   - VLLM_ALLREDUCE_USE_SYMM_MEM=0   vllm-ascend compatibility
# ------------------------------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# plan §0.2 NPU case: verl 脚本 :191 设的 NPU 必须
export CPU_AFFINITY_CONF=1

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export PYTHONPATH=$PYTHONPATH:$RLLM_DIR

# NVIDIA Megatron-Bridge is shipped as a source clone (not pip-installed) on
# this NPU env, so its src/ tree must be on PYTHONPATH for `from megatron.bridge
# import AutoBridge` to resolve (verl/models/mcore/bridge.py:17, reached when
# actor_rollout_ref.actor.megatron.vanilla_mbridge=False — see plan §13.12).
#
# Override MEGATRON_BRIDGE_DIR if the source clone lives elsewhere.
MEGATRON_BRIDGE_DIR="${MEGATRON_BRIDGE_DIR:-/workspace/Megatron-Bridge}"
if [[ -d "${MEGATRON_BRIDGE_DIR}/src/megatron/bridge" ]]; then
    export PYTHONPATH="${MEGATRON_BRIDGE_DIR}/src:${PYTHONPATH}"
    echo "[stage1] Megatron-Bridge: PYTHONPATH += ${MEGATRON_BRIDGE_DIR}/src"
elif python3 -c "import megatron.bridge" 2>/dev/null; then
    echo "[stage1] Megatron-Bridge: already importable (pip-installed)"
else
    echo "[stage1] WARN: ${MEGATRON_BRIDGE_DIR}/src/megatron/bridge not found AND megatron.bridge not pip-installed."
    echo "[stage1] WARN: training will crash at verl _build_tf_config with ModuleNotFoundError."
    echo "[stage1] WARN: set MEGATRON_BRIDGE_DIR=<path-to-NVIDIA-NeMo/Megatron-Bridge clone> and re-run."
fi

export OPENHANDS_IMAGE=openhands-triton-env:v1

export HYDRA_FULL_ERROR=1

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# ------------------------------------------------------------------------------
# vLLM / NPU
# ------------------------------------------------------------------------------
export VLLM_ATTENTION_BACKEND="TORCH_SDPA"
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export TOKENIZERS_PARALLELISM=true
export VLLM_LOGGING_LEVEL=WARN
export HYDRA_FULL_ERROR=1

# ------------------------------------------------------------------------------
# OpenHands container settings
# (read by openhands_agent.py → forwarded into each OpenHands container)
# ------------------------------------------------------------------------------
export OPENHANDS_IMAGE="${OPENHANDS_IMAGE:-openhands-triton-env:v1}"
# LiteLLM proxy 上注册的模型名（必须与 proxy config 中的 model_name 一致）
# proxy 里用的是 actor_rollout_ref.model.path，即完整路径
export OPENHANDS_MODEL_NAME="${MODEL_PATH}"
export OPENHANDS_BASE_URL_PORT=${PROXY_PORT:-4000}
# Stage1 unblock (W2.22): default 1 to lock LLM prompt at first-turn ~30k tokens.
# Each turn adds ~16k of tool-result history → with default 1000 the prompt
# blows max_model_len within 2-3 turns. Override only after OpenHands condenser
# (W3/W4) is in place or max_model_len is raised significantly.
export OPENHANDS_MAX_ITERATIONS="${OPENHANDS_MAX_ITERATIONS:-1}"
export OPENHANDS_CONTAINER_TIMEOUT="${OPENHANDS_CONTAINER_TIMEOUT:-1800}"
export OPENHANDS_ARTIFACT_DIR="${OPENHANDS_ARTIFACT_DIR:-/workspace/results/openhands_results}"

# ------------------------------------------------------------------------------
# DooD runtime dirs precheck (plan §13.23, audit 教训第 12 条)
# ------------------------------------------------------------------------------
# These two dirs must exist + be writable from inside the main container.
# OPENHANDS_WORKSPACE_TEMP_HOST_DIR is the bigger trap: it MUST be a same-path
# bind mount from host (`-v /home/docker/openhands_workspace:/home/docker/openhands_workspace`
# on main container startup), otherwise the sibling OpenHands container's
# `-v {workspace}:/opt/workspace` resolves on host dockerd to an empty dir →
# child container sees empty /opt/workspace → entrypoint.py missing → exit 127.
OPENHANDS_WORKSPACE_TEMP_HOST_DIR="/home/docker/openhands_workspace"  # 同步 openhands_agent.py
for _dir in "${OPENHANDS_WORKSPACE_TEMP_HOST_DIR}" "${OPENHANDS_ARTIFACT_DIR}"; do
    if [ ! -d "${_dir}" ]; then
        echo "[stage1 precheck] ERROR: required dir does not exist: ${_dir}" >&2
        echo "  On host: mkdir -p ${_dir}" >&2
        echo "  Main container startup MUST add: -v ${_dir}:${_dir}" >&2
        exit 1
    fi
    if ! touch "${_dir}/.stage1_precheck" 2>/dev/null; then
        echo "[stage1 precheck] ERROR: dir not writable: ${_dir}" >&2
        exit 1
    fi
    rm -f "${_dir}/.stage1_precheck"
done
# Soft check: workspace_temp dir SHOULD live under a host bind mount, NOT on the
# main container's overlay layer (that defeats DooD — sibling container would see
# empty /opt/workspace). `findmnt -T` walks up the mount tree, so mounting an
# ancestor like `-v /home/docker:/home/docker` is correctly accepted here.
# Warn only when the underlying fs is overlay/tmpfs (i.e. clearly NOT a bind mount).
# findmnt may not be present in all images — skip silently if missing.
if command -v findmnt >/dev/null 2>&1; then
    _fstype=$(findmnt -no FSTYPE -T "${OPENHANDS_WORKSPACE_TEMP_HOST_DIR}" 2>/dev/null || echo "")
    if [ "${_fstype}" = "overlay" ] || [ "${_fstype}" = "tmpfs" ]; then
        echo "[stage1 precheck] WARN: ${OPENHANDS_WORKSPACE_TEMP_HOST_DIR} sits on '${_fstype}' fs (not a bind mount)." >&2
        echo "  Child OpenHands containers will likely see empty /opt/workspace and fail." >&2
        echo "  Fix: restart main container with -v <host_path>:<container_path>; mounting any ancestor (e.g. -v /home/docker:/home/docker) also works." >&2
    fi
    unset _fstype
fi

# ------------------------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------------------------
N_GPUS="${N_GPUS:-8}"
# stage1 batch sizing (must satisfy two constraints):
#   1. train_batch_size * rollout.n >= DP_size   (every DP rank gets >= 1 sample)
#   2. train_batch_size >= ppo_mini_batch_size   (verl actor.py:224 hard check)
#
# DP_size on our parallelism layout = N_GPUS / (TP × PP × CP) = 8 / (2×1×1) = 4
# So defaults below satisfy: BATCH_SIZE=1 × ROLLOUT_N=4 = 4 = DP, and
# PPO_MINI_BATCH_SIZE coupled to BATCH_SIZE keeps train_batch_size >= mini_batch.
# Override any of these on the command line for W3 scaling experiments.
BATCH_SIZE="${BATCH_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${BATCH_SIZE}}"

# Sanity assertions (fail fast with a clear message, not deep in verl validate())
DP_SIZE=$(( N_GPUS / 2 ))   # TP=2, PP=1, CP=1 ⇒ DP = N_GPUS / TP
_total_samples=$(( BATCH_SIZE * ROLLOUT_N ))
if [[ ${_total_samples} -lt ${DP_SIZE} ]]; then
    echo "[stage1] FATAL: BATCH_SIZE($BATCH_SIZE) * ROLLOUT_N($ROLLOUT_N) = ${_total_samples}" \
         "< DP_SIZE($DP_SIZE). Some DP ranks would receive 0 samples." >&2
    echo "[stage1] Either raise BATCH_SIZE or ROLLOUT_N so the product is >= ${DP_SIZE}." >&2
    exit 2
fi
if [[ ${BATCH_SIZE} -lt ${PPO_MINI_BATCH_SIZE} ]]; then
    echo "[stage1] FATAL: BATCH_SIZE($BATCH_SIZE) < PPO_MINI_BATCH_SIZE($PPO_MINI_BATCH_SIZE)." \
         "verl actor.py:224 will reject this." >&2
    exit 2
fi
echo "[stage1] batch sanity OK: BATCH_SIZE=$BATCH_SIZE ROLLOUT_N=$ROLLOUT_N" \
     "PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE total_samples=${_total_samples} DP=$DP_SIZE"
PROXY_PORT="${PROXY_PORT:-4000}"
TRACE_DB_PATH="${TRACE_DB_PATH:-/workspace/results/rllm-openhands-traces.db}"
PROJECT_NAME="${PROJECT_NAME:-rllm-openhands-qwen36}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-stage1-qwen36-npu}"
logs=/workspace/results/verl-rllm-qwen36-npu.log

# profiling configuration
PROFILE_STEPS="[1]"
PROFILE_RANKS_ALL=False
RANKS="[0]"
DISCRETE=False

# profiling NPU options
SAVE_PATH="/workspace/results/profile_data/all"
LEVEL="level1"
CONTENTS=['npu','cpu','memory']
ANALYSIS=True

export OOM_SNAPSHOT_ENABLE=1
export OOM_SNAPSHOT_PATH="/workspace/results/profile_data"

if [[ "$MODEL_PATH" == *"Qwen3-Coder"* ]] || [[ "$MODEL_PATH" == *"Qwen3.6"* ]]; then
    TOOL_PARSER=qwen3_coder
else
    TOOL_PARSER=hermes
fi

echo "=== rllm + OpenHands stage1 (Qwen3.6-35B-A3B × NPU) ==="
echo "  Model           : ${MODEL_PATH}"
echo "  N_GPUS (trainer) : ${N_GPUS}"
echo "  Proxy port      : ${PROXY_PORT}"
echo "  OpenHands image : ${OPENHANDS_IMAGE}"
echo "  Max iterations  : ${OPENHANDS_MAX_ITERATIONS} (W2.22 stage1 unblock; raise after condenser)"
echo "  Tool parser     : ${TOOL_PARSER}"
echo "  Parallelism     : TP=2 PP=1 CP=1 EP=4 ETP=1 (single-node 8-NPU)"
echo "  max_model_len   : 49152 (W2.21 L3 unblock; up from 32k baseline)"
echo "  router_replay   : disabled (see plan §13.3)"
echo "  use_kl_loss     : False (rely on PPO clip; see plan §13.7)"

# ------------------------------------------------------------------------------
# Layered testing hatches (see plan §13.10):
#   PREFLIGHT_ONLY=1     → skip Ray + ray job submit; run preflight only and exit
#   STAGE1_DRY_STEPS=N   → cap training at N global steps (verl total_training_steps)
#   STAGE1_MOCK_ROLLOUT=1 → swap rollout() with mock_rollout (no docker/LLM)
# ------------------------------------------------------------------------------
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
STAGE1_DRY_STEPS="${STAGE1_DRY_STEPS:-0}"
export STAGE1_MOCK_ROLLOUT="${STAGE1_MOCK_ROLLOUT:-0}"   # read by train_openhands_qwen36_npu.py
if [[ "${STAGE1_MOCK_ROLLOUT}" != "0" ]]; then
    echo "[stage1] STAGE1_MOCK_ROLLOUT=${STAGE1_MOCK_ROLLOUT} → mock rollout (no docker/LLM)"
fi

if [[ "${PREFLIGHT_ONLY}" != "0" ]]; then
    echo "[stage1] PREFLIGHT_ONLY=${PREFLIGHT_ONLY} → running preflight, skipping Ray + training"
else
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
        --object-store-memory=$((4 * 1024 * 1024 * 1024))
fi

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
  # plan §13.7: stage1 关 KL，全部归零避免歧义
  algorithm.kl_ctrl.kl_coef=0.0
  # plan §0.2 verl ALGORITHM 段权威：显式设 False（默认应该也是 False，但 stage1 显式更稳）
  algorithm.use_kl_in_reward=False

  # =========================
  # data
  # =========================
  data.train_batch_size=${BATCH_SIZE}
  data.val_batch_size=16
  data.max_prompt_length=32768      # 32K (W2.23 L3 unblock; OpenHands 首轮 prompt 30k+)
  # 上面 8192 → 32768 的根因：agent_sdk_engine.py:563 用 data.max_prompt_length 过滤 step。
  # OpenHands 首轮真 prompt = 30721 tokens, 8K 阈值会把所有 step 过滤掉 → pad_sequence empty。
  # NOTE: agent_sdk_engine.py:624 硬编码 max_prompt_length=16384 做 padding/truncation，
  # 所以 PPO 实际拿到的 prompt 会被 left-truncate 到 16K（保留后 16K tokens），不会爆显存。
  # 那一行是独立 issue（应读 config），stage1 不动。
  data.max_response_length=4096     # 4K (verl 权威默认；OpenHands LLM client 默认 max_tokens=2048 也不超)
  data.truncation='error'                                # plan §0.2 verl 权威：超长 prompt 直接报错
  data.filter_overlong_prompts=True                      # plan §0.2 verl 权威：dataloader 阶段过滤超长 (dataset 端 task instruction 远小于 32k，不受影响)

  # =========================
  # actor_rollout_ref - common
  # =========================
  actor_rollout_ref.hybrid_engine=True
  actor_rollout_ref.model.path=${MODEL_PATH}
  actor_rollout_ref.model.trust_remote_code=True   # plan §0.2 verl MODEL 权威：Qwen3.6 自定义 arch 需要
  # plan §0.1 软约束：use_remove_padding 视实测决定。stage1 起步沿用 True
  # （现有 NPU 训练脚本验证过可跑），W2 若挂或 OOM 再切 False
  actor_rollout_ref.model.use_remove_padding=True

  # =========================
  # actor - optimization / PPO (plan §13.7 + verl NPU 套：use_dynamic_bsz=False)
  # =========================
  # 用户决策 (2026-05-25)：use_dynamic_bsz 走 verl 套 (False + micro_batch=1 + max_token cap)。
  # max_token cap 不直接照搬 verl 的 4096——verl 数据 prompt=1024+resp=2048 = 3072 < 4096；
  # 我们 prompt=8192+resp=4096 = 12288，cap 必须 >= 12288。放大到 16384 留 ~33% headroom。
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  # PPO_MINI_BATCH_SIZE coupled to BATCH_SIZE by default (see header sanity check)
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1   # plan §0.2 verl 套必需
  actor_rollout_ref.actor.use_dynamic_bsz=False            # plan §0.2 verl 套
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384  # cap >= data.max_prompt+max_response
  # plan §13.7: 关 KL loss，靠 PPO clip 控制 off-policy
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28

  # =========================
  # actor.megatron — Qwen3.6-A3B MoE, single-node TP=2 EP=4
  # =========================
  actor_rollout_ref.actor.strategy=megatron
  actor_rollout_ref.actor.megatron.use_mbridge=True
  # plan §0.2 NPU override: 走 NVIDIA Megatron-Bridge (`from megatron.bridge import AutoBridge`)
  # 而非 pypi mbridge (vanilla_mbridge=True)。原因：当前 pypi mbridge 0.15.1 不注册
  # qwen3_5_moe model_type，会 raise "Unregistered model type"。NPU 上必须 False，
  # 与 verl/examples/grpo_trainer/run_qwen3_5_35b_megatron.sh NPU case override 一致。
  actor_rollout_ref.actor.megatron.vanilla_mbridge=False
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
  actor_rollout_ref.actor.megatron.use_remove_padding=True   # 与 model.use_remove_padding 同步；plan §0.1 软约束
  actor_rollout_ref.actor.megatron.dtype=bfloat16            # plan §0.2 verl ACTOR 权威：显式 bf16
  actor_rollout_ref.actor.checkpoint.strict=False            # plan §0.2 verl NPU case：ckpt key 不严格匹配

  actor_rollout_ref.actor.megatron.param_offload=True
  actor_rollout_ref.actor.megatron.grad_offload=True
  # 表面 False，但下面 +override_optimizer_config.optimizer_cpu_offload=True
  # 等价于开启 CPU offload（先精确控制 fraction）
  actor_rollout_ref.actor.megatron.optimizer_offload=False

  # Stage1 并行：单节点 8 NPU，TP=2 PP=1 CP=1 EP=4 ETP=1
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.actor.megatron.context_parallel_size=1
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=4
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1

  ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto   # plan §0.2 verl 权威
  +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=1
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1

  # plan §0.2: MoE 关键配置（A3B 训练稳定性必需）
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True

  # plan §0.2 NPU case override 必需（verl 脚本 :198-200）
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True

  +actor_rollout_ref.actor.checkpoint.save_contents="['model']"

  # =========================
  # actor optimizer override (与 verl 脚本 NPU 分支一致)
  # =========================
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
  # plan §0.2 verl 权威启用（之前误注释）：D2H/H2D 与 optimizer step 重叠，影响吞吐
  +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True

  # =========================
  # ref (与 actor 并行配置一致)
  # =========================
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False         # plan §0.2 与 actor 套一致
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384   # cap >= prompt+response（同 actor）
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.ref.megatron.context_parallel_size=1
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=4
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.ref.megatron.param_offload=True
  actor_rollout_ref.ref.megatron.use_mbridge=True
  actor_rollout_ref.ref.megatron.vanilla_mbridge=False
  actor_rollout_ref.ref.megatron.use_dist_checkpointing=False

  # =========================
  # rollout (vllm-ascend, single-node TP=8 across all 8 NPUs)
  # =========================
  actor_rollout_ref.rollout.tensor_model_parallel_size=8
  # plan §13.7: 必须 True，才能上报 rollout_probs_diff 监控指标
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.dtype=bfloat16                            # plan §0.2 verl ROLLOUT 权威：显式 bf16
  # W2.16: verl 写死 cudagraph_mode='FULL_AND_PIECEWISE'（vllm_async_server.py:237
  # `compilation_config.setdefault`），enforce_eager=True 与 cudagraph 矛盾，
  # vllm-ascend NPU 上启动 worker 直接 exit 2。verl 自己脚本不设这条 (yaml 默认
  # false)，Qwen3.6 能跑通。所以我们也对齐 verl 默认 False，让 cudagraph 真生效。
  # 如 NPU cudagraph warmup 时间过长可 env override: ROLLOUT_ENFORCE_EAGER=True
  actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER:-False}
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  # plan §5.2: stage1 起步 32k → W2.21 L3 unblock 升 49152。第一次 LLM call 已经 30721 tokens
  # (OpenHands system prompt + tools + AGENTS.md heavy)，加 2048 output = 32769 越界 1 tok。
  # 49152 给 prompt + 后续 turn 留 16k buffer；Qwen3.5/3.6 native 支持 256K，模型侧无压力。
  # W3 还不够就升 65536。NPU KV cache 多吃一点但 batch=1 dry-step 扛得住。
  actor_rollout_ref.rollout.max_model_len=49152
  actor_rollout_ref.rollout.max_num_seqs=4
  actor_rollout_ref.rollout.max_num_batched_tokens=8192
  actor_rollout_ref.rollout.n=${ROLLOUT_N}
  actor_rollout_ref.rollout.val_kwargs.n=2
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False            # plan §0.2 与 actor 套一致
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384      # cap >= prompt+response（同 actor）
  ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True
  +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=${TOOL_PARSER}

  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.free_cache_engine=True
  # W2.17: obs 历史这三个 engine_kwargs 透传给 vllm CLI，但 vllm-ascend 不识别
  # （ray log: "default_worker.py: error: unrecognized arguments: --swap-space 0"）。
  # verl 自己脚本不设这三个，靠 vllm 默认能跑通 → 直接删，对齐 verl 默认。
  # 如确实需要 disable swap / prefix_caching 等，得改去找 vllm-ascend 对应参数名。
  # +actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=0
  # +actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb=0
  # +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=False

  # =========================
  # rllm
  # =========================
  # plan §13.7 防御性显式 disabled（避免上游默认值变动；当前 AgentPPOTrainer
  # 路径上 router_replay 入口不生效，见 §13.3）
  +rllm.algorithm.router_replay=disabled

  rllm.mask_truncated_samples=False
  rllm.stepwise_advantage.enable=False
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
  rllm.workflow.n_parallel_tasks=8

  # =========================
  # trainer
  # =========================
  trainer.critic_warmup=0
  trainer.logger=[console]
  trainer.project_name=${PROJECT_NAME}
  trainer.experiment_name=${EXPERIMENT_NAME}
  trainer.val_before_train=False
  trainer.n_gpus_per_node=8
  trainer.nnodes=1
  trainer.device=npu
  trainer.save_freq=20
  trainer.test_freq=20
  trainer.default_hdfs_dir=null
  trainer.total_epochs=100

  # =========================
  # profiler (stage1 默认关闭；需要时取消注释)
  # =========================
  # actor_rollout_ref.rollout.profiler.enable=True
  # actor_rollout_ref.rollout.profiler.all_ranks=$PROFILE_RANKS_ALL
  # actor_rollout_ref.rollout.profiler.ranks=$RANKS
  # actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=True
  # actor_rollout_ref.rollout.profiler.tool_config.npu.contents=$CONTENTS
  # actor_rollout_ref.rollout.profiler.tool_config.npu.level=$LEVEL
  # actor_rollout_ref.rollout.profiler.tool_config.npu.analysis=$ANALYSIS
  # global_profiler.tool=npu
  # global_profiler.steps=$PROFILE_STEPS
  # global_profiler.save_path=$SAVE_PATH
)

# STAGE1_DRY_STEPS=N → cap training at N global steps for layered smoke
if [[ "${STAGE1_DRY_STEPS}" != "0" ]]; then
    ARGS+=(
      "+trainer.total_training_steps=${STAGE1_DRY_STEPS}"
    )
    echo "[stage1] STAGE1_DRY_STEPS=${STAGE1_DRY_STEPS} → capping global_steps via trainer.total_training_steps"
fi

if [[ "${PREFLIGHT_ONLY}" != "0" ]]; then
    # Layer 1 isolation: no Ray, no NPU, no docker — just the cheap config/import checks.
    python3 -m examples.openhands_sdk.preflight_qwen36_npu "${ARGS[@]}" 2>&1 | tee -i "${logs}.preflight"
    exit ${PIPESTATUS[0]}
fi

ray job submit --address="http://${MASTER_ADDR}:8265" \
    -- \
    python3 -m examples.openhands_sdk.train_openhands_qwen36_npu "${ARGS[@]}" \
    2>&1 | tee -i $logs
