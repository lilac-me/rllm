# Stage Integration Work Log

> 整合分支 `verl-main` 的开发日志。
> 目的：从 stage1/stage2 工作日志中**只保留真正必要**的 commit，做一份干净的"Qwen3.6 × OpenHands × NPU AgentSDK 路径专项适配"成果分支。
> 起点：`b2538771` (fix time) — 这是 obs 分支 cherry-pick 的第 2 条，再往前是 `35bd9e05`（修复 stepwise + completion_ids）。

---

## 范围声明

**只保留**对以下路径有效的改动：
- rllm 入口 `AgentSdkTrainer` (`rllm/trainer/verl/agent_sdk_trainer.py`)
- rllm 引擎 `AgentSdkEngine` (`rllm/engine/agent_sdk_engine.py`)
- rllm rollout `VerlEngine` (`rllm/engine/rollout/verl_engine.py`，**注意非** `rllm/experimental/rollout/`)
- examples/openhands_sdk/* (整套 OpenHands 接入)
- rllm.parser (Qwen3.5/3.6 chat template)
- rllm.sdk.* (LiteLLM proxy / trace store / metadata slug)

**排除**：改动只触及以下路径的 commit
- `rllm/experimental/verl/verl_backend.py` (走 `VerlBackend` / `UnifiedTrainer` 路径，不是 stage1 走的 `AgentSdkTrainer`)
- `rllm/experimental/rollout/verl_engine.py` (跟 `rllm/engine/rollout/verl_engine.py` 是**两个不同文件**，stage1 不导入)
- `rllm/experimental/verl/transform.py` (`VerlBackend` 路径的 trajectory 转换)
- `rllm/experimental/verl/fully_async/*` (fully_async trainer，stage1 不用)
- `rllm/trainer/verl/agent_ppo_trainer.py` (跟 `AgentSdkTrainer` 同根但不同入口；stage1 走 SDK)

---

## W1 Upstream cherry-pick Review（重新审视）

之前的 review 把"改了 VerlBackend 路径"的 cherry-pick 标 ⏸ 但建议保留（理由：未来 path 转换地基）。**用户校准要求：专项适配工作不需要这些"未来地基"，纯粹按 AgentSDK 路径必要性筛**。

按这个标准重新判断每条：

| Commit | 改动文件 | 是否必要 | 理由 |
|---|---|---|---|
| `35bd9e05` 修复 stepwise + completion_ids 长度 | **6 个文件**，关键 2 个：`rllm/engine/rollout/verl_engine.py:87` 加 truncate 调用 + `rllm/engine/rollout/rollout_engine.py` 新加 `_truncate_completion_after_eos` base helper | ✅ **必要** | stage1 用的 `verl_engine.py` 就是这个；修的 bug 是"vllm 返回的 completion_ids 在 EOS 之后可能还有多余 token，没截掉会被当成 LLM 输出训进 PPO"；SDK 路径同样存在 |
| `b2538771` fix time | `scripts/eval_pass_at_k.py` | ❌ **不必要** | eval 脚本，stage1/stage2 不用。但已经是新分支起点 commit，自然在分支里 |
| `029ee36f` pad multi-turn batch to lcm | `rllm/experimental/verl/verl_backend.py` only | ❌ **不必要** | 只改 VerlBackend；AgentSdkTrainer 有自己的 `_pad_dataproto_to_world_size` |
| `58b15e4a` zero response_mask on padded rows | `rllm/experimental/verl/verl_backend.py` only | ❌ **不必要** | 只改 VerlBackend；AgentSdkEngine 用 `is_pad_step` non_tensor flag 区分 padding |
| `697afd79` rollout_probs_diff masking | **`rllm/experimental/verl/metrics.py` (new file +66) + 3 others** | ✅ **必要** | `metrics.py` 是这个 PR 新建，定义 `calculate_debug_metrics_compat`，**`agent_sdk_trainer.py:506` 直接 import 使用**。不是"间接受益"——这个函数没 PR 就不存在 |
| `41727530` collapse multi-turn rows | `rllm/experimental/verl/transform.py` + example configs | ❌ **不必要** | transform.py 走 VerlBackend；AgentSdkEngine 走 `transform_results_for_verl` per-step row schema，完全不走 |
| `d17b73e6` raise on aborted rollouts | `rllm/experimental/rollout/verl_engine.py` | ❌ **不必要** | 改的是 `experimental/rollout/verl_engine.py`，stage1 import 的是 `rllm.engine.rollout.verl_engine`——**两个不同文件**。详见下文 |
| `e11a92d0` don't truncate merged | `rllm/experimental/verl/verl_backend.py` | ❌ **不必要** | 跟 `41727530` 配套；VerlBackend 路径 |
| `74721f13` Qwen3.5 chat template | `rllm/parser/chat_template_parser.py` | ✅ **必要** | Qwen3.6 复用 Qwen3.5 chat template，stage1 LiteLLM proxy 回程重建 token 时用此 parser |
| `3c8ee08d` R2/R3 MoE router replay | `rllm/experimental/*` + config schema | ❌ **不必要** | plan §14.5 已锁 `router_replay=disabled`，AgentSdkTrainer 入口不触发 |
| `80d4e7b8` release GPU memory on shutdown | `VerlBackend.shutdown()` | ❌ **不必要** | 改的是 VerlBackend.shutdown；AgentSdkTrainer 有自己的 `shutdown()` (`agent_sdk_trainer.py:928-936`) |
| `ca560dc7` lr_schedule routing | `rllm/experimental/verl/utils.py` (`_SHARED_KEYS`) | ❌ **不必要** | `_SHARED_KEYS` 是 VerlBackend sync_config 用的；stage1 通过 hydra override 直接传 megatron 特定 key |

**结论**：12 条 cherry-pick 里 **真正对 AgentSDK 路径必要的是 3 条**（35bd9e05 + 697afd79 + 74721f13），剔除 9 条。

剔除占比 75%。**整合策略**：之前推荐"全拿，为未来留地基"，校准后是"只拿当前 AgentSDK 路径必要的"。

### Verify 方法的反思

第一轮 review 把 `35bd9e05` 错判成"AgentPPOTrainer 路径无关"，根因是看 `git show --stat | head -8` 只读到前 2 个文件就下结论，没翻完整 stat（这个 PR 实际改 6 个文件，关键的 `verl_engine.py:87` 在第 4 个）。

**正确 verify 方法**（后续 W2 / Stage2 review 用）：
1. `git show --stat <sha>` 看**完整**文件列表，不要 head
2. 每个文件 path 跟 stage1 实际 import 链（`agent_sdk_trainer.py` / `agent_sdk_engine.py` / `verl_engine.py` / parser / sdk.proxy）对比
3. **读 patch hunks**：被 import 的 module 内具体改的函数，是不是 stage1 调用栈触发的
4. 同名易混淆文件特别注意：`rllm/engine/rollout/verl_engine.py` ≠ `rllm/experimental/rollout/verl_engine.py`

### Bonus 发现（W1 不解决，stage2/3 follow-up）

stage1 用的 `rllm/engine/rollout/verl_engine.py:89` 直接 `finish_reason = token_output.stop_reason`，**没有 raise on aborted**。d17b73e6 修的 silent abort bug 在 stage1 路径**确实存在**，但 d17b73e6 修的是 `rllm/experimental/rollout/verl_engine.py`——修在错位置，stage1 路径没受益。

如果 stage2 真训练撞到 partial trajectory（weight sync 中断 + abort 流到训练），需要把 d17b73e6 的逻辑应用到 `rllm/engine/rollout/verl_engine.py`。但 d17b73e6 本身**仍然剔除**——它修错位置了。

---

## 关于 d17b73e6 的详细解释（回答 review 问题 2）

PR 标题：`fix(verl): raise on aborted rollouts and normalize stop_reason at source`

**改的是什么**：`rllm/experimental/rollout/verl_engine.py` 的 `VerlEngine.get_token_output_from_token_input` 和 `_assemble_model_output`。

**修复的问题**：

`VerlEngine` 调 `server_manager.generate(...)` 拿 `token_output` 后，token_output 的 `stop_reason` 字段可能是这些值：
- `"completed"` (正常完成)
- `"length"` (打到 max_tokens)
- `"stop"` (碰到 stop_token)
- `"aborted"` (rollout 被中断 — 典型场景：weight sync 跑到一半，server_manager 主动 abort 这一轮 generation)

修复前：
```python
# 在 _assemble_model_output 里反查表 mapping
reason_mapping = {"aborted": "abort", "completed": "stop"}
finish_reason = reason_mapping.get(token_output.stop_reason, token_output.stop_reason)
```
→ "aborted" 被 silent 映射成 "abort"，跟正常 "stop" 一起流到下游，**下游训练把 partial token sequence 当成完整 rollout 训**。

修复后：
```python
# 在 get_token_output_from_token_input 里
if token_output.stop_reason in ("aborted", "abort"):
    raise RuntimeError("Rollout aborted")  # exceptional path
token_output.stop_reason = "length" if len(token_output.token_ids) >= max_tokens else "stop"
```
→ aborted 立即 raise，workflow 层 catch 后能决定 retry / drop；剩下的 stop_reason 统一 normalize 成 `"length"` / `"stop"`，反查表删掉。

**为什么对 stage1 不直接生效**：

stage1 `AgentSdkTrainer` / `AgentSdkEngine` import 的是 `rllm.engine.rollout.verl_engine`：
```bash
$ grep -rn "from rllm.engine.rollout.verl_engine\|from rllm.experimental.rollout.verl_engine" rllm/ examples/
rllm/engine/agent_execution_engine.py:115:    from rllm.engine.rollout.verl_engine import VerlEngine
rllm/engine/agent_sdk_engine.py:22:           from rllm.engine.rollout.verl_engine import VerlEngine
rllm/trainer/verl/agent_workflow_trainer.py:35: from rllm.engine.rollout.verl_engine import VerlEngine
rllm/trainer/verl/agent_sdk_trainer.py:39:     from rllm.engine.rollout.verl_engine import VerlEngine
```

而 d17b73e6 改的是 `rllm/experimental/rollout/verl_engine.py`——**两个不同文件**。stage1 路径完全不导入 `experimental.rollout`，所以这个 fix 不生效。

**潜在风险（不影响整合决策，但要 track）**：

stage1 用的那个 `verl_engine.py`（`rllm/engine/rollout/verl_engine.py`）可能有同样的 silent abort 流转问题。但 stage1 L3 PASS + 真训练已经能跑，说明：
- 要么那个文件已经处理 aborted 正确
- 要么 stage1 weight sync 时序下 abort 不发生
- 要么 abort 发生但 partial trajectory 被 transform_results_for_verl 后续路径过滤掉

不阻塞整合决策——`d17b73e6` 仍然剔除。如果 stage2 真训练撞到 partial trajectory 问题，再回头查 `rllm/engine/rollout/verl_engine.py` 是否需要同步类似 fix。

---

## 后续工作

- [x] 核实 `35bd9e05` → 必要（改 `rllm/engine/rollout/verl_engine.py:87`）
- [ ] W2 阶段 review：决定 stage1 路径搭建 + 第一轮 debug 的 16 条 commits 哪些必要
- [ ] W2.20-W2.30 deep debug 阶段 review：14 条都是核心 bug fix，预期大多数必要
- [ ] Stage2 阶段 review：6 条 commits，跟 stage1 强耦合
- [ ] 出整合 cherry-pick 计划：从 b2538771 依次 cherry-pick 必要 commits
- [ ] 跑回归 sanity（L3）确认整合后链路完整

---

## 遗留 follow-up（不在整合 W1 cherry-pick 范围，需 stage2/3 期间单独处理）

### F1: stage1 路径 silent abort partial trajectory bug

**症状**：weight sync mid-generation 时 rollout 被 abort，`token_output.stop_reason="aborted"` 直接流到训练，partial token sequence 被当成完整 trajectory 训进 PPO。

**位置**：`rllm/engine/rollout/verl_engine.py:89`

```python
# 当前状态（stage1 用的版本）：
token_output: TokenOutput = await self.server_manager.generate(...)
completion_ids: list[int] = token_output.token_ids
logprobs: list[float] = token_output.log_probs
completion_ids, logprobs = self._truncate_completion_after_eos(completion_ids, logprobs)

finish_reason = token_output.stop_reason   # ← 直接用，aborted 也会被当 finish_reason
```

**为什么 W1 cherry-pick 没解决**：upstream PR `d17b73e6` 修了同款 bug，但修在 `rllm/experimental/rollout/verl_engine.py`（VerlBackend 路径用的另一个 verl_engine）——和 stage1 路径的 `rllm/engine/rollout/verl_engine.py` 是**两个不同文件**。`d17b73e6` 应用到错位置，本专项 cherry-pick 时已剔除。

**触发条件**（什么时候真撞到）：
- 训练步频高 + weight sync 时序紧（每 step 末 update_weights）
- generation 长（OpenHands rollout 多 turn 时尤其）
- rollout 跨过 weight sync 边界的概率 > 0

**修复方法**（参照 d17b73e6 思路应用到 stage1 路径）：

```python
# rllm/engine/rollout/verl_engine.py 在 token_output 取到之后立刻 check
token_output: TokenOutput = await self.server_manager.generate(...)
if token_output.stop_reason in ("aborted", "abort"):
    raise RuntimeError("Rollout aborted by weight sync or server")
# normalize 剩下的 stop_reason
if len(token_output.token_ids) >= sampling_params["max_tokens"]:
    token_output.stop_reason = "length"
elif token_output.stop_reason not in ("length", "stop"):
    token_output.stop_reason = "stop"
```

workflow 层 catch 这个 RuntimeError，决定是 drop 还是 retry。

**是否做的判断标准**：
- 看 stage2 N 步 sanity 时有没有诊断 metric 异常（特别是 `rollout/log_probs_fill_rate < 1.0` 或 reward 偶发跳变）
- 如果稳定 = 暂时不需要修
- 如果偶发 trajectory 异常 = 必须修

**优先级**：🟡 中（stage2 sanity 阶段先观察 metric，撞了再修）

---

## verl-main cherry-pick 计划（W1 阶段）

| commit | 已在分支祖先 | 需 cherry-pick | 顺序 |
|---|---|---|---|
| `35bd9e05` | ✅ (parent of b2538771) | — | 已 in |
| `b2538771` | ✅ (HEAD) | — | 已 in |
| `74721f13` Qwen3.5 parser | — | ✅ | 1 |
| `697afd79` rollout_probs_diff + new metrics.py | — | ✅ | 2 |

其余 9 条 W1 cherry-pick 全部不在 verl-main。

---

## Decision log

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-05-28 | verl-main 起点选 `b2538771` 而非 `origin/openhands_ascend` | 保留 obs 分支已有的 2 条小 fix (`35bd9e05` + `b2538771`)，后续 cherry-pick 从此点开始；不带 W1 的 10 条 upstream PR |
| 2026-05-28 | W1 cherry-pick 中 10 条标 ⏸ 的全部剔除 | 这些 PR 改的是 `VerlBackend` / `experimental/*` 路径，不被 `AgentSdkTrainer` 触发；保留它们只是为 path 转换留地基，本专项不要 |
| 2026-05-28 | `697afd79` 保留 | `metrics.py` 是这个 PR 新建，`calculate_debug_metrics_compat` 是 stage1 `agent_sdk_trainer.py:506` 的直接依赖 |
| 2026-05-28 | `74721f13` 保留 | Qwen3.5 chat template parser 是 Qwen3.6 实际使用的 parser |
| 2026-05-28 | `35bd9e05` 升级为 ✅ 必要 | 第一轮 verify 只读 `git show --stat` 前 2 行，漏看了关键的 `rllm/engine/rollout/verl_engine.py:87` truncate 调用——stage1 用的就是这个文件，修的 EOS-后多余 token 截断 bug 在 SDK 路径同样存在；review 方法论同步加严（见 §W1 末段） |
