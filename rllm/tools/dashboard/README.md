# rllm Training Dashboard

这是一个针对 **rllm SDK** 的轻量级、无依赖的前端页面看版，提供针对 LLM agent 调用的实时监控面板，以辅助分析 SDK 模式下 `AgentSdkTrainer` 和 `AgentSdkEngine` 在训练和推断时生成的数据。

面板不依赖于任何复杂的构建系统，直接利用 Python、FastAPI 及 SQLite 提供纯只读数据的可视化面板。前端界面完全通过一个 `index.html` 实现，后端读取本地的 `~/.rllm/traces.db` 以及 `~/.rllm/datasets` 进行信息汇总展示。

## 启动 Dashboard

本模块被封装在 `/home/robomaster/Research/rllm/rllm/tools/dashboard/` 中。作为一个独立的模块，你可以以独立进程直接启动：

```bash
python -m rllm.tools.dashboard
```

**可用参数：**
* `--db-path`: Tracer SQLite 的存储地址（默认 `~/.rllm/traces.db`）
* `--dataset-dir`: 数据集的存储地址（默认 `~/.rllm/datasets`）
* `--metrics-file`: Trainer 使用的 PPO 训练指标接收地址（默认 `~/.rllm/metrics.jsonl`）
* `--host`: 绑定地址 (默认 `0.0.0.0`)
* `--port`: 服务端口 (默认 `8080`)
* `--log-level`: 终端输出日志等级 (默认 `info`)

启动后，可通过浏览器访问 `http://localhost:8080` 查看看板。

## 看板功能模块介绍

1. **Overview (总览)**：
   显示库中总 Trace 数量、使用量最多的 Namespace 和最新的 Finish Reason 总结分布图，监控系统的大致运行数据。

2. **Traces (轨迹列表与查询)**：
   支持按不同的时间和名称条件查询单个对话 Trace 记录。点开各个 Session 可以看到逐步骤的 prompt 回复气泡以及系统对齐推理的 Reasoning 等详细响应内容。

3. **Datasets (预训练/验证数据集浏览器)**：
   检查注册至 `registry.json` 中各个数据的详情并渲染 `.parquet` 和 `.arrow` 中的抽样行展示字段概况。

4. **Episodes (样本回合聚合追踪)**：
   按 `task_id` 聚合同属一组 Episode 的 `task` 数据，分析每次 Episode 中的步骤重试率或终止/成功原因分布，诊断生成模型执行表现。

5. **Sampling / PPO Metrics (策略生成采样监控 & PPO 训练曲线)**：
   根据生成的 Trace 收集每次通信请求延迟或回复 Token 分组；训练指标读取 `metrics.jsonl` 以渲染强化学习期间如模型奖励、Actor Entropy (熵) 和通过率等。

---

## 额外辅助：结合 Trainer 实现训练曲线展示

Dashboard 中的 **"PPO Metrics"** 标签页需要读取 PPO 训练指标。因为当前 `AgentSdkTrainer` 通常将指标发往 Wandb，而不在本地。要点亮此功能，你可以直接在您的 Trainer 循环中追加类似下面的代码，将其直接输出到了 Dashboard 监听的 `~/.rllm/metrics.jsonl` 文件。

你可以参考以下简易逻辑插入你的训练器：

```python
import json
import asyncio
import os
from aiohttp import ClientSession

# --- 方法 1: 直接追加写本地文件 ---
def log_metrics_to_file(metrics: dict, step: int, filepath="~/.rllm/metrics.jsonl"):
    metrics['global_step'] = step
    file_path = os.path.expanduser(filepath)
    with open(file_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


# --- 方法 2: 通过 POST HTTP 发送到运行中的看版中 ---
async def post_metrics(metrics: dict, step: int):
    metrics['global_step'] = step
    async with ClientSession() as session:
        try:
            await session.post("http://localhost:8080/api/metrics", json=metrics, timeout=2)
        except Exception as e:
             pass # Dashboard closed
```
