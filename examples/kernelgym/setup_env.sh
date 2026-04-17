#!/bin/bash

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export PYTHONPATH=$PYTHONPATH:$RLLM_DIR

# ── vLLM / PyTorch 环境变量 ───────────────────────────────────────────────────
export FORCE_BUILD=0
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND="TORCH_SDPA"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ASCEND_ENABLE_NZ=0
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

export RAY_DEBUG_POST_MORTEM=0

export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512
export TASK_QUEUE_ENABLE=2 # 下发优化，图模式设置为1，非图模式设置为2

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

export TOKENIZERS_PARALLELISM=true
export VLLM_LOGGING_LEVEL=WARN
export RAY_DEDUP_LOGS=1
export HYDRA_FULL_ERROR=1


# HCCL 相关配置
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_ASYNC_ERROR_HANDLING=0
export P2P_HCCL_BUFFSIZE=30
export HCCL_BUFFSIZE=300
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HCCL_INTRA_ROCE_ENABLE=1

export GLOO_SOCKET_TIMEOUT=7200

export NPUS_PER_NODE=16

# 获取IP / 指定hostname
export SOCKET_IFNAME=ens1f3
export RAY_MASTER_HOSTNAME="rllm-node-61"
export RAY_MASTER_PORT=6766
# export CURRENT_HOSTNAME=$(hostname)
export CURRENT_IP=$(ifconfig "$SOCKET_IFNAME" | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')