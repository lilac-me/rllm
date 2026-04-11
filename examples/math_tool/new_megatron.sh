#!/bin/bash

# 1. 基础环境配置 (打印执行过程)
set -x

# 昇腾 & vLLM 核心环境变量
#   Value error, Invalid value '' for VLLM_ATTENTION_BACKEND. Valid options: ['FLASH_ATTN', 'TRITON_ATTN', 'ROCM_ATTN', 'ROCM_AITER_MLA', 'ROCM_AITER_TRITON_MLA', 'ROCM_AITER_FA', 'ROCM_AITER_MLA_SPARSE', 'TORCH_SDPA', 'FLASHINFER', 'FLASHINFER_MLA', 'TRITON_MLA', 'CUTLASS_MLA', 'FLASHMLA', 'FLASHMLA_SPARSE', 'FLASH_ATTN_MLA', 'PALLAS', 'IPEX', 'NO_ATTENTION', 'FLEX_ATTENTION', 'TREE_ATTN', 'ROCM_AITER_UNIFIED_ATTN', 'CPU_ATTN', 'CUSTOM']. [type=value_error, input_value=ArgsKwargs(()), input_type=ArgsKwargs]
export VLLM_ATTENTION_BACKEND="TORCH_SDPA"
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export RAY_DEBUG_POST_MORTEM=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
# ValueError: FRACTAL_NZ mode is enabled. This may cause model parameter precision issues in the RL scenarios. Please set VLLM_ASCEND_ENABLE_NZ=0.
export VLLM_ASCEND_ENABLE_NZ=0

# 修复 ModuleNotFoundError: 自动获取当前目录并加入 PYTHONPATH
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export PYTHONPATH=$PYTHONPATH:$RLLM_DIR

# 设置 Master 节点地址
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# 2. 重启 Ray 服务 (解决之前的 Connection Refused 问题)
echo "正在重启 Ray 集群..."
ray stop --force
rm -rf /tmp/ray/*
sleep 2
ray start --head \
    --port 6379 \
    --dashboard-host 0.0.0.0 \
    --dashboard-port 8265 \
    --disable-usage-stats \
    --node-ip-address ${MASTER_ADDR}

# 3. 定义 Runtime Env (Ray Job 提交时需要的环境变量)
# 注意：这里要把 NPU 相关的路径传给 Ray 的 Worker 进程
# CURRENT_ENV="$(
# python - <<'PY'
# import os, json, rllm
# print(json.dumps({
# "env_vars":{
#   **os.environ, 
#   "PYTHONPATH": os.path.dirname(os.path.dirname(rllm.__file__)),
#   "ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15"}
# }))
# PY
# )"

# RUNTIME_ENV_JSON='{
#   "env_vars": {
#     "PYTHONPATH": "'$PYTHONPATH'",
#     "ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
#   }
# }'

# 4. 使用 ray job submit 提交任务
# 这样即便终端断开了，训练任务也会在 Ray 后台持续运行
echo "正在提交 rllm 训练任务..."

# --runtime-env-json="${RUNTIME_ENV_JSON}" \
    # actor_rollout_ref.actor.megatron.optimizer_offload=True \

ray job submit --address="http://127.0.0.1:8265" \
    -- \
    python3 -m examples.math_tool.train_math_with_tool_megatron \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=2 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=/home/g00841271/Qwen3-Coder-30B-A3B-Instruct \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.context_parallel_size=2 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=2 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.context_parallel_size=2 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=8 \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    critic.megatron.tensor_model_parallel_size=4 \
    critic.megatron.pipeline_model_parallel_size=1 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='4b-math-tool' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.device=npu \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=2 \
    rllm.stepwise_advantage.enable=False \
    trainer.total_epochs=100