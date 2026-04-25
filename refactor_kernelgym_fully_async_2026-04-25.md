# KernelGym 全异步路径复用 KernelAgent 重构总结

> **日期**：2026-04-25  
> **背景**：全异步训练路径的 `kernelgym_rollout` 与 rllm-071 共卡路径的 `KernelAgent` / `KernelGymEnv` 存在大量重复逻辑。本次重构在保持全异步高并发能力的前提下，复用 `KernelAgent` 消除 prompt / extraction / thinking 等重复代码。

---

## 一、问题与分析

### 1.1 用户原始问题

当前全异步路径（`examples/kernelgym/kernelgym_rollout.py`）本质上重写了共卡路径的：
- `KernelAgent`：prompt 管理、kernel extraction、thinking strip
- `KernelGymEnv`：评测协议、reward 计算、feedback 构建

**能否直接复用共卡的 `KernelGymEnv` 和 `KernelAgent`，以减少重复代码？**

### 1.2 分析结论

| 组件 | 结论 | 原因 |
|------|------|------|
| `KernelAgent` | **可以复用** | 纯状态管理层，无同步 IO，接口天然适配 async |
| `KernelGymEnv` 评测核心 | **不建议复用** | `httpx.Client` + `ray.get()` + `time.sleep()` 同步阻塞，与 `RolloutExecutor` 的单 event loop 高并发模型不兼容 |
| `KernelGymEnv` reward/feedback | **逻辑已复用** | `KernelGymRewardOps` + `async_submit_and_poll` 已对齐 071 的三种 reward 函数和评测协议 |

#### `KernelGymEnv` 不可复用的代码级证据

`KernelGymEnv` 的评测核心 `_HybridHttpWorker` 存在四处同步阻塞点，会直接冻结 `RolloutExecutor` 的 asyncio event loop，导致 `max_concurrent_rollout` 个 rollout 全部串行：

1. **`httpx.Client`**（同步 HTTP 客户端）
   ```python
   self._client = httpx.Client(...)
   resp = self._client.post(f"{self.server_url}/evaluate", json=task_data)
   s = self._client.get(f"{self.server_url}/status/{task_id}")
   ```

2. **`ray.get()`**（同步等待 Ray Actor）
   ```python
   return ray.get(self._rate_limit_worker.get_current_count.remote())
   ```

3. **`time.sleep()`**（主动阻塞）
   ```python
   time.sleep(self._backoff(attempt, base=2 if resp.status_code == 429 else 5))
   time.sleep(1.0)  # poll 循环中每秒阻塞一次
   ```

4. **`step()` 本身是同步方法**
   ```python
   def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, dict]:
   ```

`RolloutExecutor.generate_trajectory` 运行在 **同一个 Ray Actor 进程的同一个 event loop** 中。任何同步阻塞调用都会卡住 event loop，其他并发 rollout 全部暂停。

> 当前全异步代码必须使用 `async_submit_and_poll`（`httpx.AsyncClient` + `await asyncio.sleep`）的原因正在于此——只有 `await` 才会把控制权交还 event loop。

---

## 二、开发计划

### Phase 1：对齐 `KernelAgent` 到 rllm-071 逻辑
- 替换 `_REVISION_USER_TEMPLATE` 为 071 版本（要求 ````python` 块）
- 合并 071 的 `extract_kernel_code` helper
- `update_from_model` 增加 import 前缀 + `class ModelNew:` → `class ModelNew(nn.Module):` patch
- 保留当前的 thinking partition 行为（`</think>` 存在时先 partition 再 extraction）

### Phase 2：重构 `kernelgym_rollout` 复用 `KernelAgent`
- 删除 `_SYSTEM_PROMPT`、`_INITIAL_USER_TEMPLATE`、`_REVISION_USER_TEMPLATE`
- 删除 `_strip_thinking`、`_content_for_kernel_extraction`、`_extract_kernel_code`
- 删除手动 `messages` 列表维护
- 内部驱动 `KernelAgent` 管理对话状态：`update_from_env` → `chat_completions` → `update_from_model`
- 评测仍走 async 路径（`async_submit_and_poll` + `KernelGymRewardOps`）
- `protocol.Trajectory` 仍由 `kernelgym_rollout` 自己维护（与 `agents/agent.Trajectory` 类型不同，不强行映射）

### Phase 3：`train_kernelgym_fully_async.py` 最小改动
- 无需改动，`make_rollout_fn` 的 closure 参数与新的 `kernelgym_rollout` 签名完全兼容

### Phase 4：验证
- Mock 模式端到端测试
- Prompt 格式检查
- Kernel extraction 行为检查
- Trajectory 返回检查

---

## 三、实际改动内容

### 3.1 `rllm/rllm/agents/kernelgym_agent.py`

**改动前**：rllm 当前版本的 `KernelAgent`
- `_REVISION_USER_TEMPLATE` 要求 `<kernel>` 标签
- `update_from_model` 仅做 `</think>` partition，返回原始 response
- 无 `extract_kernel_code` 逻辑

**改动后**：对齐 rllm-071 逻辑
- `_REVISION_USER_TEMPLATE` 替换为 071 版本：
  ```
  Now you have received the server feedback ...
  Return an improved Triton implementation named `ModelNew` as a single ```python``` block.
  ```
- 新增 `extract_kernel_code(solution_str)` helper：
  - 按优先级匹配 `# Kernel Implementation`、````python # Kernel`、 `# Your implementation:`、`# Generated kernel:`
  - 回退到提取最后一个 ` ``` ` 代码块
  - 最终回退到整个字符串
- `update_from_model` 重构：
  1. 保留 thinking partition（`</think>` 存在时分离 thought 和 answer）
  2. 对 answer 调用 `extract_kernel_code`
  3. 自动添加 import 前缀：`import triton`、`import triton.language as tl`、`import torch`、`import torch.nn as nn`
  4. Patch `class ModelNew:` → `class ModelNew(nn.Module):`
  5. 返回 `Action(action=kernel_code.strip())`

### 3.2 `rllm/examples/kernelgym/kernelgym_rollout.py`

**删除的重复逻辑**（约 70 行）：
- `_SYSTEM_PROMPT`、`_INITIAL_USER_TEMPLATE`、`_REVISION_USER_TEMPLATE`
- `_strip_thinking()`、`_content_for_kernel_extraction()`、`_extract_kernel_code()`
- 手动 `messages` 列表的初始化、追加、thinking strip 调用

**保留的逻辑**：
- `_build_feedback()`：构建人类可读的 feedback 字符串传给 agent
- `_evaluate_and_score_turn()`：async 评测核心（`async_submit_and_poll` + `KernelGymRewardOps`）
- `protocol.Trajectory` 维护：满足 `RolloutExecutor` 的返回类型要求

**新的 `kernelgym_rollout` loop 结构**：
```python
agent = KernelAgent(system_prompt=system_prompt)
agent.reset()
agent.update_from_env(observation=task, reward=0.0, done=False, info={})

for turn in range(max_turns):
    messages = agent.chat_completions
    response_msg, output = await client.chat_completion(messages, ...)
    action = agent.update_from_model(response_msg.get("content", ""))
    
    # 评测仍走 async 路径
    last_result = await _evaluate_and_score_turn(task, action.action, ...)
    
    feedback = _build_feedback(...)
    agent.update_from_env(observation=feedback, reward=last_reward, done=False, info=last_result)
```

**关键接口映射**：

| fully-async 侧 | `KernelAgent` 接口 | 说明 |
|---------------|-------------------|------|
| 首回合 task dict | `update_from_env(observation=task, ...)` | agent 初始化 system + initial user prompt |
| 获取当前 messages | `agent.chat_completions` | 自动 strip thinking（除最后一条） |
| LLM 生成后 | `agent.update_from_model(response)` | 提取 kernel_code（遵从 071 逻辑） |
| feedback 后 | `agent.update_from_env(observation=feedback, ...)` | 追加 revision user message |

### 3.3 未改动的文件

- `rllm/environments/kernelgym/kernelgym_reward_ops.py`：reward 逻辑已与 071 对齐，无需改动
- `rllm/environments/kernelgym/kernelgym_eval_hybrid_async.py`：async 评测核心保持现状
- `rllm/environments/kernelgym/kernelgym_env.py`：同步 env 保持现状
- `rllm/experimental/fully_async/rollout_executor.py`：executor 接口不变
- `examples/kernelgym/train_kernelgym_fully_async.py`：入口逻辑无需改动

---

## 四、跨版本差异清单（复用前已对齐）

| 差异点 | `rllm` 当前（重构前） | `rllm-071` | 重构后决策 |
|--------|---------------------|-----------|----------|
| `_REVISION_USER_TEMPLATE` | 要求 `<kernel>` 标签 | 要求 ````python` 块 | **统一为 071 版本** |
| `update_from_model` 的 extraction | 简单 `max(pool, key=len)` | `extract_kernel_code()` + 自动加 import / patch class | **统一为 071 版本** |
| `update_from_env` 的 observation | 当前未使用 | 直接 `format(feedback=observation)` | 行为不变，fully-async 传入人类可读 feedback 字符串 |
| `Trajectory` 类型 | `protocol.Trajectory`（`sequences`） | `agents/agent.Trajectory`（`steps`） | **不映射**，`kernelgym_rollout` 自己维护 `protocol.Trajectory`，agent 只负责对话状态 |

---

## 五、验证结果

### 5.1 静态检查
- `python -m py_compile` 通过：`kernelgym_agent.py`、`kernelgym_rollout.py`
- `kernelgym_rollout` 模块 import 通过

### 5.2 `KernelAgent` 单元级验证
- ✅ 首回合 `update_from_env(task_dict)` 正确初始化 system + user prompt（2 条 messages）
- ✅ `update_from_model(response_with_thinking)` 正确分离 thinking、提取 kernel code、添加 import 前缀、patch `class ModelNew(nn.Module):`
- ✅ revision turn `update_from_env(feedback_string)` 正确追加 user message，包含 "Server feedback (status/metrics/errors):"
- ✅ `chat_completions` property 正确 strip 非最后一条 assistant message 的 thinking 块
- ✅ `_REVISION_USER_TEMPLATE` 已切换为 071 的 ````python` 要求

### 5.3 端到端验证（待执行）
```bash
RLLM_KERNELGYM_MOCK_EVAL=1 bash examples/kernelgym/train_kernelgym_fully_async.sh
```
- 观察 rollout 是否正常完成多轮对话
- 检查日志中的 prompt 格式是否与 071 一致
- 确认 `update_from_model` 返回的 `action.action` 已自动添加 import 前缀和 class patch
- 确认 `RolloutExecutor` 能正常将 `protocol.Trajectory` 放入 `TrajectoryGroup`

---

## 六、回滚风险

- `kernelgym_rollout.py` 只被 `train_kernelgym_fully_async.py` 引用，改动影响范围有限。
- `kernelgym_agent.py` 被共卡路径和全异步路径共用，但改动完全对齐 071 逻辑，风险可控。
- 若出现问题，可直接回滚上述两个文件。
