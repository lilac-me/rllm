# KernelGYM + rllm Integration

## KernelGYM 接入 rllm 训练框架

本目录为 **rllm** 框架集成 **KernelGYM** 的示例，实现 CUDA/Triton kernel 生成的 RL 训练。

---

## 架构概览

```
rllm AgentTrainer
  ├── KernelAgent          # BaseAgent 子类，管理多轮 kernel 编写对话
  └── KernelGymEnv         # MultiTurnEnvironment 子类，调用 KernelGYM HTTP API
        └── KernelGYM Server  → compiled / correctness / speedup 评估结果
```

**与 DrKernel 的关系：**

| 组件 | DrKernel | rllm (本集成) |
|------|----------|--------------|
| 训练引擎 | verl `RayKernelTrainer` | rllm `AgentTrainer` |
| 奖励方式 | `AsyncKernelRewardManager` 轮询 REST | `KernelGymEnv.step()` 同步调用 |
| 奖励公式 | `compiled + correct + speedup` | 相同 (0.1 / 0.3 / 0.6) |
| 多轮修改 | ❌ 单轮 | ✅ 最多 `max_turns` 轮 |

---

## 快速开始

### 1. 启动 KernelGYM 服务

```bash
cd /path/to/KernelGYM
bash start_worker_node.sh   # 启动 GPU worker
# 等待 API 服务就绪（默认 http://localhost:8000）
```

### 2. 准备数据集

每条训练记录（JSONL 格式）需要以下字段：

```json
{
  "problem_id": "relu",
  "reference_code": "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)\n",
  "description": "Optimise the ReLU operation.",
  "entry_point": "Model",
  "backend": "cuda"
}
```

### 3. 启动训练

```bash
cd /path/to/rllm

python examples/kernelgym/train_kernelgym.py \
    --kernel_server_url http://localhost:8000 \
    --train_data data/kernelbench_train.jsonl \
    --val_data   data/kernelbench_val.jsonl \
    --max_turns  3 \
    --backend    cuda
```

---

## 奖励函数

```python
reward = 0.1 * compiled         # 能成功编译
       + 0.3 * correctness       # 输出与参考实现一致
       + 0.6 * clip(speedup, 0, 10) / 10   # 加速比（最高上限 10×）
```

最大奖励 = 1.0（编译通过 + 正确 + 达到或超过 10× 加速）。

---

## 多轮修改流程

```
Turn 0: 初始 prompt（参考代码 + 问题描述）
         ↓ LLM 生成 <kernel>...</kernel>
Turn 1: KernelGYM 返回编译/正确性/加速比 → 错误信息反馈给 LLM
         ↓ LLM 修改 kernel
Turn 2: … 最多 max_turns 轮后强制结束
```

---

## 单元测试

```bash
cd /path/to/rllm
python -m pytest tests/envs/test_kernelgym_env.py tests/agents/test_kernelgym_agent.py -v
```