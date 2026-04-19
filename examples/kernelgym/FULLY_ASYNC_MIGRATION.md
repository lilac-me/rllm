# KernelGYM Fully-Async 迁移开发总结

## 背景

将 KernelGYM 算子生成 Agent 从原有的 `AgentExecutionEngine`（共卡/半异步模式）迁移至 RLLM 的 **Fully Async** 全异步训练架构。迁移后 rollout 与 PPO 训练完全解耦，通过 MessageQueue 交接，支持异步参数同步和 staleness 控制。

## 约束

- **非侵入性**：不修改 `rllm` 或 `verl` 核心框架代码，所有改动限于 `examples/kernelgym` 和业务插件模块（`rllm/agents`、`rllm/environments/kernelgym`）
- **Reward 并行开发**：KernelGYM 的 reward 计算逻辑正在同步修改中，本次实现保持薄适配层以便后续对齐
- **主线缺失文件**：`KernelAgent` 和 `KernelGymEnv` 源文件尚未合入上游主线，需从 `rllm-071_old` 分支手动合入

---

## 改动清单

### Phase 0：合入 Agent/Env

| 操作 | 文件 | 说明 |
|------|------|------|
| 新增 | `rllm/agents/kernelgym_agent.py` | 从 `rllm-071_old` 复制，多轮对话 Agent |
| 新增 | `rllm/environments/kernelgym/kernelgym_env.py` | 从 `rllm-071_old` 复制，Kernel 评估环境 |
| 新增 | `rllm/environments/kernelgym/__init__.py` | 包入口 |
| 修改 | `rllm/agents/__init__.py` | 注册 `KernelAgent` 到 `_LAZY_IMPORTS` |
| 修改 | `rllm/environments/__init__.py` | 注册 `KernelGymEnv` 到 `ENVIRONMENT_IMPORTS` |

### Phase 1-3：Fully Async 实现

| 操作 | 文件 | 说明 |
|------|------|------|
| 新增 | `examples/kernelgym/kernelgym_rollout.py` | 核心 rollout 逻辑 |
| 新增 | `examples/kernelgym/train_kernelgym_fully_async.py` | 训练入口 |
| 新增 | `examples/kernelgym/train_kernelgym_fully_async.sh` | 启动脚本 |
| 新增 | `examples/__init__.py` | Python 包标记（支持相对 import） |
| 新增 | `examples/kernelgym/__init__.py` | Python 包标记（支持相对 import） |

---

## 架构设计

### 整体数据流

```
Dataset ──→ RolloutExecutor ──→ rollout_fn() ──→ MessageQueue ──→ FullyAsyncTrainer (PPO)
                 │                    │
                 │              ┌─────┴──────┐
                 │              │ LLM 生成    │  ← RolloutClient.chat_completion()
                 │              │ Kernel 评估 │  ← httpx.AsyncClient → KernelGYM Server
                 │              │ Feedback 拼接│
                 │              │ Trajectory  │  → Sequence[] + reward + metadata
                 │              └─────────────┘
                 │
          ParameterSynchronizer ← ─ ─ ─ ─ ─ ─ ─ ─ Trainer 参数更新
```

### 关键设计决策

#### 1. 闭包工厂模式

`RolloutExecutor.generate_trajectory()` 只传 dataset row 字段作为 `**kwargs`，不传 Hydra config。因此使用 `make_rollout_fn(kernel_cfg)` 工厂函数，通过闭包捕获 kernel 配置参数：

```python
def make_rollout_fn(kernel_cfg: dict):
    kernel_server_url = kernel_cfg.get("server_url", ...)
    max_turns = kernel_cfg.get("max_turns", 3)
    # ... 其他参数

    async def rollout_fn(client, tokenizer, **kwargs):
        task = dict(kwargs)  # dataset row fields
        result = await kernelgym_rollout(client=client, task=task, ...)
        return trajectory

    return rollout_fn
```

#### 2. 异步 Kernel 评估

原 `KernelGymEnv` 使用同步 `httpx.Client`，在异步 event loop 中会阻塞。替换为 `httpx.AsyncClient`：

```python
async def _evaluate_kernel_async(task, kernel_code, server_url, ...):
    async with httpx.AsyncClient(timeout=timeout + 60) as http_client:
        resp = await http_client.post(f"{server_url}/evaluate", json=payload)
        return resp.json()
```

#### 3. Thinking 剥离

对齐 `KernelAgent.chat_completions` 的行为：多轮交互中，发送给 LLM 的消息会去掉前序轮次 assistant 回复中的 `<think>...</think>` 块，避免占用上下文窗口。

```python
def _strip_thinking(messages):
    out = copy.deepcopy(messages)
    for msg in out[:-1]:
        if msg["role"] == "assistant":
            _, sep, after = msg["content"].partition("</think>")
            if sep:
                msg["content"] = after
    return out
```

#### 4. Overlong 保护

生成或评估过程异常时，mask 掉所有 response token，使该 trajectory 不参与 loss 计算：

```python
if overlong:
    for seq in trajectory.sequences:
        seq.response_masks = [0] * len(seq.response_masks)
```

---

## 对齐验证

| 对齐项 | 原实现位置 | Fully Async 位置 | 状态 |
|--------|-----------|-----------------|------|
| System Prompt | `kernelgym_agent.py` L21-32 | `kernelgym_rollout.py` L28-39 | 一致 |
| Initial User Template | `kernelgym_agent.py` L35-39 | `kernelgym_rollout.py` L41-45 | 一致 |
| Revision User Template | `kernelgym_agent.py` L42-49 | `kernelgym_rollout.py` L47-54 | 一致 |
| Kernel Code 提取 | `kernelgym_env.py` L58-77 | `kernelgym_rollout.py` L79-97 | 一致 |
| Feedback 格式 | `kernelgym_env.py` L556-582 | `kernelgym_rollout.py` L104-128 | 一致 |
| Reward 计算 | `kernelgym_env.py` L226-238 | `kernelgym_rollout.py` L141-152 | 一致 |
| Thinking 剥离 | `kernelgym_agent.py` L174-183 | `kernelgym_rollout.py` L61-76 | 一致 |
| 多轮消息交替 | `AgentExecutionEngine` | `kernelgym_rollout` for loop | 一致 |

---

## 使用方式

### 启动命令

```bash
# 方式 1：使用启动脚本（推荐）
cd rllm-071/examples/kernelgym
KERNEL_SERVER_URL=http://your-server:8000 \
MODEL_PATH=/path/to/model \
bash train_kernelgym_fully_async.sh

# 方式 2：直接 python -m
cd rllm-071
python -m examples.kernelgym.train_kernelgym_fully_async \
    ++kernel.server_url=http://your-server:8000 \
    ++kernel.backend=triton \
    ++kernel.max_turns=3 \
    actor_rollout_ref.model.path=/path/to/model \
    ...
```

### 关键环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `KERNEL_SERVER_URL` | `http://localhost:8000` | KernelGYM 评估服务器地址 |
| `KERNEL_BACKEND` | `triton` | 后端类型 (cuda/triton) |
| `KERNEL_MAX_TURNS` | `3` | 最大多轮交互轮数 |
| `MODEL_PATH` | - | 模型路径 |
| `N_GPUS_ROLLOUT` | `4` | Rollout 使用的 GPU 数 |
| `N_GPUS_TRAINING` | `4` | Training 使用的 GPU 数 |
| `STALENESS` | `0.5` | 异步 staleness 阈值 |

### Hydra Config 覆盖

Kernel 相关参数通过 `++kernel.*` 注入：

```
++kernel.server_url=http://...
++kernel.backend=triton
++kernel.max_turns=5
++kernel.toolkit=kernelbench
++kernel.timeout=300
++kernel.num_correct_trials=5
++kernel.num_perf_trials=100
```

---

## Metrics 输出

每个 trajectory 的 metadata 包含以下字段，可在 wandb/console 日志中查看：

| 字段 | 类型 | 说明 |
|------|------|------|
| `num_turns` | int | 实际交互轮数 |
| `total_generation_time` | float | LLM 生成总耗时 (秒) |
| `total_eval_time` | float | Kernel 评估总耗时 (秒) |
| `total_completion_tokens` | int | 生成的总 token 数 |
| `last_reward` | float | 最后一轮的 reward |
| `best_reward` | float | 所有轮次中的最高 reward |
| `compiled` | float | 最后一轮是否编译成功 (0/1) |
| `correctness` | float | 最后一轮是否正确性通过 (0/1) |
| `speedup` | float | 最后一轮的加速比 |
| `overlong` | float | 是否因异常被 mask (0/1) |
| `merged_step` | int | merge 后的 sequence 数 |
| `param_version_start` | int | rollout 开始时的参数版本 |
| `param_version_end` | int | rollout 结束时的参数版本 |
| `is_partial` | bool | 是否跨越了参数更新 |
| `processing_time` | float | 总处理耗时 (秒) |

---

## 后续 TODO

1. **Reward 对齐**：reward 模块并行修改完成后，更新 `kernelgym_rollout.py` 中的 `_compute_reward_simple` 或 `_evaluate_kernel_async` 的返回值处理
2. **`KernelGymEnv` 硬编码路径**：`kernelgym_env.py` L28 的 `_KERNELGYM_ROOT` 路径需要适配实际部署环境
3. **端到端集成测试**：在实际 NPU 集群上验证全流程
4. **上游合入**：待 `KernelAgent`/`KernelGymEnv` 的上游改动上线后，删除本地合入的副本并切换到上游版本
