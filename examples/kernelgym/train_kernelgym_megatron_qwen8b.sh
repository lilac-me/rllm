#!/bin/bash

pkill -9 python
pkill -9 torchrun
set -euo pipefail
set -x

DETERMINISTIC_SEED=${DETERMINISTIC_SEED:-1234}
ROLLOUT_DO_SAMPLE=${ROLLOUT_DO_SAMPLE:-false}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.0}
ROLLOUT_N=${ROLLOUT_N:-1}
VAL_N=${VAL_N:-1}
N_PARALLEL_AGENTS=${N_PARALLEL_AGENTS:-1}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
DEBUG_ROLLOUT_DIR=${DEBUG_ROLLOUT_DIR:-"${PWD}/debug_rollouts_npu_seed${DETERMINISTIC_SEED}"}
DEBUG_ROLLOUT_LOAD_PATH=${DEBUG_ROLLOUT_LOAD_PATH:-}
ENABLE_PROFILING=${ENABLE_PROFILING:-false}
NUM_PERF_TRIALS=${NUM_PERF_TRIALS:-100}
INIT_CORRECT_WEIGHT=${INIT_CORRECT_WEIGHT:-0.5}
INIT_PERFORMANCE_WEIGHT=${INIT_PERFORMANCE_WEIGHT:-0.5}
TASK_NAMESPACE=${TASK_NAMESPACE:-npu}
KERNELGYM_FORCE_REFRESH=${KERNELGYM_FORCE_REFRESH:-true}
export KERNELGYM_TASK_NAMESPACE=${KERNELGYM_TASK_NAMESPACE:-$TASK_NAMESPACE}
export KERNELGYM_FORCE_REFRESH=${KERNELGYM_FORCE_REFRESH}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $ROOT_DIR/common_env.sh

export PYTHONHASHSEED=${PYTHONHASHSEED:-$DETERMINISTIC_SEED}
export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}
export HCCL_DETERMINISTIC=${HCCL_DETERMINISTIC:-true}
export LCCL_DETERMINISTIC=${LCCL_DETERMINISTIC:-1}
export CLOSE_MATMUL_K_SHIFT=${CLOSE_MATMUL_K_SHIFT:-1}

RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")
export PYTHONPATH=${PYTHONPATH:-}:$RLLM_DIR

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

MODEL_PATH=/home/docker/drkernel-8b-codestart
# MODEL_PATH=/home/g00841271/models/drkernel-8b

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

ARGS=(
  # =========================
  # algorithm
  # =========================
  algorithm.adv_estimator=grpo
  algorithm.kl_ctrl.kl_coef=0.001

  # =========================
  # data
  # =========================
  data.train_batch_size=16
  data.val_batch_size=16
  data.shuffle=false
  data.seed=${DETERMINISTIC_SEED}
  data.dataloader_num_workers=0
  data.max_prompt_length=24576      # 24K
  data.max_response_length=8192    # 18K

  # =========================
  # actor_rollout_ref - common
  # =========================
  actor_rollout_ref.hybrid_engine=True
  # actor_rollout_ref.model.use_shm=True
  actor_rollout_ref.model.path=$MODEL_PATH
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True

  # =========================
  # actor - optimization / PPO
  # =========================
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.ppo_mini_batch_size=8
  actor_rollout_ref.actor.shuffle=false
  actor_rollout_ref.actor.data_loader_seed=${DETERMINISTIC_SEED}
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
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
  actor_rollout_ref.actor.megatron.seed=${DETERMINISTIC_SEED}
  actor_rollout_ref.actor.megatron.use_mbridge=True
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=False

  actor_rollout_ref.actor.megatron.param_offload=True
  actor_rollout_ref.actor.megatron.grad_offload=True
  # actor_rollout_ref.actor.megatron.optimizer_offload=True

  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.actor.megatron.context_parallel_size=4
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=1
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
  +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=4
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  +actor_rollout_ref.actor.checkpoint.save_contents="['model']"

  # =========================
  # actor optimizer override
  # =========================
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
#   +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True

  # =========================
  # ref
  # =========================
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.ref.megatron.context_parallel_size=4
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=1
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.ref.megatron.param_offload=True
  actor_rollout_ref.ref.megatron.use_mbridge=True
  actor_rollout_ref.ref.megatron.use_dist_checkpointing=False

  # =========================
  # rollout
  # =========================
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.calculate_log_probs=True        # 记录 训推的 log_prob 确认是否存在diff
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.do_sample=${ROLLOUT_DO_SAMPLE}
  actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75
  actor_rollout_ref.rollout.max_model_len=32768
  actor_rollout_ref.rollout.n=${ROLLOUT_N}
  actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
  actor_rollout_ref.rollout.val_kwargs.do_sample=false
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  actor_rollout_ref.rollout.enable_chunked_prefill=false
  actor_rollout_ref.rollout.enable_prefix_caching=false
  actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}
  ++actor_rollout_ref.rollout.engine_kwargs.vllm.seed=${DETERMINISTIC_SEED}
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096

  rllm.rejection_sample.enable=True
  rllm.deterministic_rollout.enable=True
  rllm.deterministic_rollout.seed=${DETERMINISTIC_SEED}
  rllm.deterministic_rollout.request_seed=True
  rllm.deterministic_rollout.task_ids=True
  rllm.debug_rollout.save=True
  rllm.debug_rollout.dir=${DEBUG_ROLLOUT_DIR}

  # =========================
  # rllm
  # =========================
  rllm.mask_truncated_samples=False
  +rllm.agent.engine_args.n_parallel_agents=${N_PARALLEL_AGENTS}
  rllm.agent.max_steps=3
  rllm.stepwise_advantage.enable=True
  rllm.stepwise_advantage.mode=broadcast

  # =========================
  # trainer
  # =========================
  trainer.critic_warmup=0
  trainer.logger="[console,wandb]"
  trainer.project_name=rllm-agent
  trainer.experiment_name=kernelgym-dr8b
  trainer.val_before_train=False
  trainer.n_gpus_per_node=16
  trainer.nnodes=1
  trainer.device=npu
  trainer.save_freq=20
  trainer.test_freq=20
  trainer.default_hdfs_dir=null
  trainer.balance_batch=false
  trainer.total_epochs=100

  # # =========================
  # # kernel
  # # =========================

  reward_model.max_turns=3
  reward_model.reference_backend=triton
  reward_model.server_url="http://127.0.0.1:8002"
  reward_model.reward_func_name=calculate_reward_weighted
  reward_model.init_correct_weight=${INIT_CORRECT_WEIGHT}
  reward_model.init_performance_weight=${INIT_PERFORMANCE_WEIGHT}

  reward_model.speedup_reward_upper_bound=3.0
  reward_model.speedup_reward_lower_bound=0.0

  reward_model.reward_policy.penalties.penalty_score=0.0
  reward_model.reward_policy.penalties.compilation_fail=-0.5
  reward_model.reward_policy.penalties.correctness_fail=-0.3
  reward_model.reward_policy.penalties.perf_degrade=-0.1

  reward_model.timeout=120
  reward_model.max_retries=2
  reward_model.task_timeout=600
  reward_model.task_timeout_in_client=150
  reward_model.num_perf_trials=${NUM_PERF_TRIALS}
  reward_model.num_correct_trials=5
  reward_model.enable_profiling=${ENABLE_PROFILING}
  reward_model.verbose_errors=true

  reward_model.reward_weights.compilation=0.3
  reward_model.reward_weights.correctness=0.4
  reward_model.reward_weights.performance=0.3

  reward_model.coverage_reward.reward_type=time_coverage
  reward_model.coverage_reward.enable=false
  reward_model.coverage_reward.weight=0.25
)

if [[ -n "${DEBUG_ROLLOUT_LOAD_PATH}" ]]; then
  ARGS+=(rllm.debug_rollout.load_path=${DEBUG_ROLLOUT_LOAD_PATH})
fi

# python3 -m examples.kernelgym.train_kernelgym "${ARGS[@]}"

# ray job submit --address="http://${MASTER_ADDR}:8265" \
#     -- \
python3 -m examples.kernelgym.train_kernelgym "${ARGS[@]}" 2>&1| tee "gpu_training_$(date +%Y%m%d_%H%M%S).log"
