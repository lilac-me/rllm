#!/bin/bash

# 1. 基础环境配置
set -x
logs=/home/p00938733/verl-rllm.log

# 昇腾 & vLLM 环境变量
export VLLM_ATTENTION_BACKEND="TORCH_SDPA"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_ASCEND_ENABLE_NZ=0
# Find the directory where rllm package is located
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export PYTHONPATH=$PYTHONPATH:$RLLM_DIR

# 设置 Master 节点地址
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# export ASCEND_RT_VISIBLE_DEVICES="8,9,10,11,12,13,14,15"
# export RAY_DEBUG_POST_MORTEM=1

# 2. 重启 Ray 服务
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

# 3. 定义 Runtime Env
# 注意：这里要把 NPU 相关的路径传给 Ray 的 Worker 进程

# RUNTIME_ENV_JSON='{
#   "env_vars": {
#     "PYTHONPATH": "'$PYTHONPATH'",
#     "ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
#   }
# }'

# 4. profiling config
# profiling configuration
PROFILE_STEPS="[1,2]"
PROFILE_RANKS_ALL=False
RANKS="[0,1]"
DISCRETE=False
# PROFILE_CONTINUOUS_STEPS=True

# profiling NPU options
SAVE_PATH="/home/p00938733/profile_data/"
LEVEL="level1"
CONTENTS=['npu','cpu','memory']
ANALYSIS=True

# 5. 使用 ray job submit 提交任务
# 这样即便终端断开了，训练任务也会在 Ray 后台持续运行
echo "正在提交 rllm 训练任务..."

# --runtime-env-json="${RUNTIME_ENV_JSON}" \

ray job submit --address="http://127.0.0.1:8265" \
    -- \
    python3 -m examples.math_tool.train_math_with_tool \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=32 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=/home/g00841271/Qwen3-8B \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name='4b-math-tool' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.device=npu \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=2 \
    rllm.stepwise_advantage.enable=False \
    trainer.total_epochs=100 \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.all_ranks=$PROFILE_RANKS_ALL \
    actor_rollout_ref.actor.profiler.ranks=$RANKS \
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete=$DISCRETE \
    actor_rollout_ref.actor.profiler.tool_config.npu.contents=$CONTENTS \
    actor_rollout_ref.actor.profiler.tool_config.npu.level=$LEVEL \
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis=$ANALYSIS \
    global_profiler.tool=npu \
    global_profiler.steps=$PROFILE_STEPS \
    global_profiler.save_path=$SAVE_PATH  2>&1 | tee -i $logs