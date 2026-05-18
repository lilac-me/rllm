# rLLM + KernelGYM 去随机性验证总结（GPU / NPU）

> **日期**：2026-05-17  
> **目标**：先在同平台（GPU）实现多次启动训练流程的 rollout 可复现，再推进 GPU/NPU 对齐；若无法做到 token 一致，至少保证样本 reward 一致，并能解释原因。

---

## 1. 思路总览

这次方案分两阶段推进：

1. **第一阶段：推理/rollout 侧确定性**
   - 先关闭采样（`do_sample=false`, `temperature=0`, `n=1`），把随机性压到最低。
   - 固定数据顺序、请求 seed、task/session id、vLLM 批处理行为。
   - 先验证 token 一致；如果 token 一致但 reward 不一致，说明差异主要来自 KernelGYM 评测/计时链路。

2. **第二阶段：训练侧确定性**
   - 在 rollout 稳定后，用同一份 debug rollout dump 回放训练，隔离“推理随机性”。
   - 对齐 loss/logprob/梯度/参数更新，定位训练侧精度差异。

核心原则：**先拆分问题，再端到端复核**。先把 “模型生成” 和 “reward 评测” 分开验证，再看整体曲线。

---

## 2. 已实现内容（Debug Patch）

### 2.1 启动脚本默认进入确定性验证模式

文件：[train_kernelgym_megatron_qwen8b_gpu.sh](/Users/yeji/Documents/Code/Python/source_code/rllm/examples/kernelgym/train_kernelgym_megatron_qwen8b_gpu.sh)

主要改动：

- 新增可控变量：
  - `DETERMINISTIC_SEED`（默认 `1234`）
  - `ROLLOUT_DO_SAMPLE=false`
  - `ROLLOUT_TEMPERATURE=0.0`
  - `ROLLOUT_N=1`
  - `VAL_N=1`
  - `N_PARALLEL_AGENTS=1`
  - `MAX_NUM_SEQS=1`
  - `DEBUG_ROLLOUT_LOAD_PATH`（默认空；非空时会直接启用 debug rollout 回放）
  - `ENABLE_PROFILING=false`
  - `NUM_PERF_TRIALS=100`
  - `INIT_CORRECT_WEIGHT=0.5`
  - `INIT_PERFORMANCE_WEIGHT=0.5`
- 固定环境与数据顺序：
  - `data.shuffle=false`
  - `data.seed=${DETERMINISTIC_SEED}`
  - `data.dataloader_num_workers=0`
  - `actor_rollout_ref.actor.shuffle=false`
  - `actor_rollout_ref.actor.data_loader_seed=${DETERMINISTIC_SEED}`
  - `actor_rollout_ref.actor.megatron.seed=${DETERMINISTIC_SEED}`
- 固定 vLLM/算子路径确定性开关：
  - `VLLM_BATCH_INVARIANT=1`
  - `NVIDIA_TF32_OVERRIDE=0`
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  - `CUDA_DEVICE_MAX_CONNECTIONS=1`
  - `++actor_rollout_ref.rollout.engine_kwargs.vllm.seed=${DETERMINISTIC_SEED}`
- 关闭会改变 batch 排布的训练项：
  - `trainer.balance_batch=false`
- 默认保存 rollout 调试快照：
  - `rllm.debug_rollout.save=True`
  - `rllm.debug_rollout.dir=${DEBUG_ROLLOUT_DIR}`

### 2.2 rLLM rollout 链路补全“确定性 seed + 确定性 id”

文件：[agent_execution_engine.py](/Users/yeji/Documents/Code/Python/source_code/rllm/rllm/engine/agent_execution_engine.py)

主要改动：

- 新增 `rllm.deterministic_rollout.*` 配置读取。
- 新增稳定 seed 生成逻辑：
  - trajectory seed：按 `global_steps/env_idx` 派生
  - request seed：按 `global_steps/env_idx/step_idx` 派生
- 每步 generate 时把 `seed` 写入 sampling params（传到 vLLM）。
- `env.reset` 支持传 seed（带 fallback）。
- deterministic 模式下，`application_id` 改为稳定字符串（非随机 uuid）。

### 2.3 PPO Trainer 侧 UID 固定化

文件：[agent_ppo_trainer.py](/Users/yeji/Documents/Code/Python/source_code/rllm/rllm/trainer/verl/agent_ppo_trainer.py)

主要改动：

- 训练集与验证集的 `uid` 由 `uuid4` 改为基于 `seed + step + sample + extra_info` 的 hash。
- 调用 `trajectory_generator` 时显式传入 deterministic seed。

### 2.4 KernelGym 环境 task/session 稳定化

文件：[kernelgym_env.py](/Users/yeji/Documents/Code/Python/source_code/rllm/rllm/environments/kernelgym/kernelgym_env.py)

主要改动：

- `reset(seed=...)` 时，`session_uuid` 使用 `problem_id|seed` 的 hash，而不是 `uuid4`。
- 目标是稳定 `task_id` 组成，避免同样输入在服务侧出现不同 task 身份。

### 2.5 verl Server Manager 内层 request id 稳定化

文件：[agent_loop.py](/Users/yeji/Documents/Code/Python/source_code/verl/verl/experimental/agent_loop/agent_loop.py)

主要改动：

- deterministic 模式下，若 sampling params 带 seed，则 vLLM 内层 `request_id` 也稳定生成。

### 2.6 KernelGYM / kernelGym-NPU 补齐种子设置

文件：
- [KernelGYM exec_types.py](/Users/yeji/Documents/Code/Python/source_code/KernelGYM/kernelgym/toolkit/kernelbench/exec_types.py)
- [kernelGym-NPU exec_types.py](/Users/yeji/Documents/Code/Python/source_code/kernelGym-NPU/kernelgym/toolkit/kernelbench/exec_types.py)

主要改动：

- 增加 `PYTHONHASHSEED`、`random.seed`、`numpy.seed`。
- GPU 侧增加 `torch.cuda.manual_seed_all`、`cudnn.deterministic=True`、`cudnn.benchmark=False`。
- NPU 侧增加 `manual_seed_all`（若可用）。

### 2.7 新增 rollout dump 对比脚本

文件：[compare_debug_rollouts.py](/Users/yeji/Documents/Code/Python/source_code/rllm/examples/kernelgym/compare_debug_rollouts.py)

功能：

- 对比两次 debug rollout dump（`step` / `trajectory` 模式）。
- 逐条检查：
  - `prompt_ids` / `completion_ids`
  - `logprobs`
  - `trajectory_reward`
- 首次发现差异时输出定位位置（第几条 trajectory、第几步、第几个 token）。

---

## 3. 使用指南

## 3.1 第一阶段：GPU 同平台 rollout 复现

运行两次（只改 dump 目录）：

```bash
DEBUG_ROLLOUT_DIR=/tmp/kg_gpu_run1 \
bash /Users/yeji/Documents/Code/Python/source_code/rllm/examples/kernelgym/train_kernelgym_megatron_qwen8b_gpu.sh

DEBUG_ROLLOUT_DIR=/tmp/kg_gpu_run2 \
bash /Users/yeji/Documents/Code/Python/source_code/rllm/examples/kernelgym/train_kernelgym_megatron_qwen8b_gpu.sh
```

对比两次 rollout：

```bash
python3 /Users/yeji/Documents/Code/Python/source_code/rllm/examples/kernelgym/compare_debug_rollouts.py \
  /tmp/kg_gpu_run1/rollout_step_1.pt \
  /tmp/kg_gpu_run2/rollout_step_1.pt
```

判定：

- 输出 `MATCH`：同平台 rollout 已复现。
- 输出 `MISMATCH`：按脚本给的首个分叉点继续定位。

## 3.2 若只想先验证 correctness reward 一致

```bash
INIT_CORRECT_WEIGHT=1.0 \
INIT_PERFORMANCE_WEIGHT=0.0 \
ENABLE_PROFILING=false \
DEBUG_ROLLOUT_DIR=/tmp/kg_gpu_correct_only \
bash /Users/yeji/Documents/Code/Python/source_code/rllm/examples/kernelgym/train_kernelgym_megatron_qwen8b_gpu.sh
```

说明：

- 这一步可以把“性能计时噪声”从 reward 中剥离，只看 correctness 路径。

## 3.3 第二阶段：训练侧隔离验证

用第一阶段生成的固定 rollout dump 回放训练：

```bash
DEBUG_ROLLOUT_LOAD_PATH=/tmp/kg_gpu_run1/rollout_step_1.pt \
bash /Users/yeji/Documents/Code/Python/source_code/rllm/examples/kernelgym/train_kernelgym_megatron_qwen8b_gpu.sh
```

关注指标：

- `training/rollout_probs_diff_*`
- `old_log_probs` / `ref_log_prob`
- actor loss、entropy、梯度范数、参数更新量

目标：

- 同平台重复训练时，上述指标应一致或在极小数值误差内一致。

## 3.4 GPU/NPU 跨平台建议顺序

1. 先在各自平台内完成“同平台复现”。
2. 再做跨平台对比：优先比 prompt/token/logprob 分叉位置，再比 reward。
3. 若跨平台 token 难以逐 token 一致，至少要保证：
   - 同一平台内部稳定
   - reward 统计分布可解释
   - correctness 一致

---

## 4. 做不到一致时的常见原因与判因

1. **token 不一致但 reward 一致**
   - 采样路径残留随机项或并发调度差异仍存在。
   - 优先检查：`do_sample`、request seed 是否生效、`n_parallel_agents` 与 `max_num_seqs`。

2. **token 一致但 reward 不一致**
   - KernelGYM 评测侧噪声（性能计时、profiling、服务负载、retry）导致。
   - 优先切 correctness-only 验证，确认不是生成侧问题。

3. **同平台可复现，跨平台不一致**
   - 可能是 CUDA/NPU 内核实现、数值路径、算子调度差异导致。
   - 这时目标应转为“统计一致与行为可解释”，而非强求 bitwise 一致。

---

## 5. 备注

- 本文档对应的 patch 已落地到 `rllm` / `verl` / `KernelGYM` / `kernelGym-NPU` 本地代码。
- 尚未替你实际跑完整训练与长周期统计；当前是代码层和流程层可执行方案。
