# Stage Integration Work Log

> 整合分支 `verl-main` 的开发日志。
> 目的：从 stage1/stage2 工作日志中**只保留真正必要**的 commit，做一份干净的"Qwen3.6 × OpenHands × NPU AgentSDK 路径专项适配"成果分支。
> 起点：`b2538771` (fix time) — 这是 obs 分支 cherry-pick 的第 2 条，再往前是 `35bd9e05`（修复 stepwise + completion_ids）。

---

# 🚀 Pre-context for new conversation（30 秒回到上下文）

**新对话从这里开始读**。整合工作已完成 cherry-pick + squash 阶段，处于待环境验证 + cleanup 收尾状态。

## TL;DR

| Quick fact | Value |
|---|---|
| 整合分支 | `verl-main` (本地分支，未 push) |
| 当前 HEAD | `f8845ff3` "docs: Stage2 review + squash decision (5 commits)" |
| 起点 | `b2538771` "fix time" (obs 分支 cherry-pick 之一) |
| ahead | **18 commits ahead of b2538771** |
| 已完成 | W1 cherry-pick / W2 squash / W3 squash (拆 3) / Stage2 squash (拆 5) / commit message rewrite / 整合分支用户接手指南 / Layer A 静态验证 |
| 待做 | (1) stage3+ cleanup commit (不需要环境) / (2) Layer B 启动验证 (需 NPU) / (3) Layer C 训练验证 (需 NPU + 真实数据) |
| 关键源仓 | rllm 本地 `/Users/yeji/Documents/Code/Python/Qwen36/rllm` |
| 关键 log 文档 | 本文件 `STAGE_INTEGRATION_LOG.md` + `STAGE2_FIT_LOOP_AUDIT.md` + `QWEN36_OPENHANDS_AGENT_RL_PLAN.md` |

## verl-main 当前 commit 列表（按时间倒序，新 → 旧）

```
f8845ff3 docs: Stage2 review + squash decision (5 commits)
fa750e61 fix(train-script): DooD precheck FATAL
50bc9f96 feat(stage1): Dr.GRPO + std0 metric (drop random fallback)
2e1886ad fix(openhands): container/workspace cleanup + docker wait timeout
f258836b fix: stage2 audit findings + observability metrics
e35c4c65 docs: stage2 fit-loop audit + plan config locks
96c98f05 docs: update stale SHAs after commit message rewrite
de6038aa docs: W3 squash 拆分决策记录（初版 1 commit → 拆 3 commit）
35f12439 fix(trainer): AgentSdkTrainer DataProto→TensorDict bridge  ← 核心架构 fix
ff7dc3f2 fix(stage1): L3 unblock — context/iter/padding configs
34cfe866 fix(openhands): DooD workspace alignment + diag tool
0e4dd7e9 docs: W3 review + 整合分支用户接手指南
0d2888d1 docs: W2 squash decision + 注释下放 + dbg cleanup deferred
82499320 feat(stage1): training script for Qwen3.6-35B-A3B × OpenHands × NPU  ← 训练脚本主体
6ac64baf docs(plan): snapshot stage1 plan v2.2 (pre-implementation)
50ab196d Fix rollout_probs_diff masking to respect response_mask in Verl trainers (#503)
a52de00a fix(parser): Qwen3.5 chat template support + simple_math example (#504)
34c3ee68 docs: stage integration review log — W1 done (verl-main baseline)
----- b2538771 起点 -----
```

## 关键决策（已锁定，不要再讨论）

1. **整合分支专项**：只覆盖 AgentSDK 路径 (`AgentSdkTrainer` / `AgentSdkEngine` / `rllm.engine.rollout.verl_engine`)，**不**带 `VerlBackend` / `experimental/*` / `AgentPPOTrainer` 路径
2. **mock/test 工具不进 baseline**：所有 mock LLM / mock rollout / preflight / `prepare_npu_operator_data` 工具 cherry-pick 时剔除
3. **commit messages 无 stage1 日志绑定**：W2.X / plan §13.x / audit lesson #X 全部已 rewrite 移除（用 `git filter-branch --msg-filter` 实施）
4. **dbg() helper 推迟到 stage3+ cleanup commit**：W2/W3 squash 时都把 dbg() 函数 + 调用站点保留，等整合完成后一次性清理
5. **3 个 stage2 临时 default 一起保留 + 一起解锁**：`OPENHANDS_MAX_ITERATIONS=1` + `rejection_sample.enable=False` + `norm_adv_by_std_in_grpo=False` (Dr.GRPO) — 接 condenser + reward 稳定后 stage3 解锁
6. **整合分支 Squash 粒度**：W2 squash 1 个 (训练脚本搭建是同一 feature)；W3 squash 拆 3 个 (跨 OpenHands runtime / config / Trainer 架构 3 个子系统)；Stage2 squash 拆 5 个 (跨 audit / runtime / 算法 / precheck 等子系统)

## 当前 working tree 状态

clean（最新 commit `f8845ff3` 之后无未提交改动）。

## 下一步快速决策

新对话开始时直接读 §"接下来的工作（按优先级）" 章节（见本文件末尾）。**最高优先**是 stage3+ cleanup commit（不需要环境，~30 分钟内完成）。

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
- [x] W2 阶段 review + squash 整合（见下方 §W2）
- [x] W3 阶段 review + squash 拆分 (3 commits)（见下方 §W3）
- [x] Stage2 阶段 review + squash 拆分 (5 commits)（见下方 §Stage2）
- [ ] 跑回归 sanity（L3）确认整合后链路完整（需要环境）
- [ ] **stage3+ cleanup commit**：替换 `dbg()` helper 为标准 `logger.debug(...)`（agent_sdk_trainer.py + openhands_agent.py）；reconcile plan markdown version reference in script header

---

## W2 Setup + Debug Round 1 Review (16 commits → 1 squash commit)

**最终决策**：全部 11 个必要 commit squash 成单 commit `feat(stage1): training script for Qwen3.6-35B-A3B × OpenHands × NPU` (`82499320`)，每个 fix 的技术决策保留在脚本 inline 注释里。5 个剔除 (`c1ff82e3` + 4 个 mock/test infra)。

### 各 commit 一句话回顾

| Commit | 决策 | 一句话 |
|---|---|---|
| `c1ff82e3` | ❌ 剔除 | 修 `3c8ee08d` (R2/R3) 副作用，3c8ee08d 已剔除 |
| `5082c854` W2.2 | ✅ 合并 | 新建 train script baseline（v2.5 hydra config + 8-NPU 并行度 TP=2 EP=4） |
| `b07a55d7` W2.0 | ❌ 剔除 | layered test infra（mock LLM / mock rollout / preflight / orchestrator）开发期工具 |
| `d1fda898` W2.9 | ✅ 合并 | `vanilla_mbridge=False`（NPU 必须 NVIDIA Megatron-Bridge，pypi mbridge 0.15.1 不识别 qwen3_5_moe） |
| `25054b71` W2.10 | ✅ 合并 | 18 处对齐 verl NPU 分支（13 个 outright omission + 5 个 coupled-config） |
| `845a0585` W2.11 | ✅ 合并 | batch sanity（DP_SIZE / PPO_MINI_BATCH_SIZE 约束 fail-fast 检查） |
| `bd475579` | ❌ 剔除 | prepare_npu_operator_data.py mock 数据准备工具 |
| `d4482bf1` W2.12 | ❌ 剔除 | --source mock 选项 |
| `ecda05d7` W2.13 | ❌ 剔除 | mock 数据准备脚本的 prompt schema fix |
| `4073fb1d` W2.14 | ✅ 合并 | Megatron-Bridge PYTHONPATH 3-路 dispatch（source clone / pip / WARN） |
| `fff72535` W2.15 | ✅ 合并 | drop hardcoded HCCL_IF_IP；改为 env opt-in（audit lesson #7） |
| `e480c657` W2.16 | ✅ 合并 | enforce_eager=False 默认（vllm setdefault cudagraph_mode + enforce_eager=True 冲突） |
| `86065972` W2.17 | ✅ 合并 | 删 obs-legacy 3 个 vllm engine_kwargs（swap_space / cpu_offload_gb / enable_prefix_caching，vllm-ascend 不识别） |
| `07d98394` | ✅ 合并 | 路径标准化 1：MODEL_PATH / ARTIFACT_DIR / TRACE_DB_PATH / logs / SAVE_PATH 从 `/home/t00893162/` → `/home/docker/` + `/workspace/results/` |
| `4dfb7dc9` | ✅ 合并 | 路径标准化 2：撤回 07d98394 留的 `/workspace/rllm-071/...` 用户特定 hardcode |
| `6b1f522e` | ✅ 合并 | 路径标准化 3：`dbg()` log path 改 `/workspace/results/`；注释 `ASCEND_LAUNCH_BLOCKING=1`（改 `# debug only` 替代原 `# TODO`） |

### Squash 实施细节

执行步骤：
1. `git reset --hard 50ab196d`（回 W1 末）
2. `git cherry-pick 8d8b1fe6`（plan baseline 独立 commit）
3. `git cherry-pick --no-commit -X theirs` 11 个 W2 必要 commit（plan markdown 冲突自动接 incoming）
4. Edit train script line 49 `# TODO` → `# debug only`（唯一需要补的注释；其余 inline 注释 cherry-pick 时已经齐了）
5. `git commit` 一次 squash 全部 staged changes
6. update STAGE_INTEGRATION_LOG.md（本节）

### 注释下放清单（已在脚本里完成）

每条 fix 在脚本中的 inline 注释位置（读者直接看脚本就能 follow 决策）：

| Fix | 脚本注释位置 |
|---|---|
| vanilla_mbridge=False (W2.9) | `.sh` line 310-313, 373（两路 actor/ref + 详细 root cause） |
| 18 omissions (W2.10) | `.sh` 各参数旁标 `verl NPU case :191` 等来源 |
| batch sanity (W2.11) | `.sh` line 167-194（约束公式 + fail-fast assertion） |
| Megatron-Bridge PYTHONPATH (W2.14) | `.sh` line 114-126（3-路 dispatch + WARN） |
| HCCL env opt-in (W2.15) | `.sh` line 52-67（详细 obs hardcode root cause + opt-in 用法） |
| enforce_eager=False (W2.16) | `.sh` line 385-390（cudagraph setdefault 冲突说明） |
| drop obs vllm kwargs (W2.17) | `.sh` line 411-415（删的 3 个 kwargs 列出 + vllm-ascend CLI lag 原因） |
| 路径标准化 | 不需要注释（路径已是标准值） |
| ASCEND_LAUNCH_BLOCKING 注释 | `.sh` line 49（`# debug only` 替代原 `# TODO`） |

### dbg() helper 处理

**保留到 stage3+ cleanup commit 一次性处理**：

- 当前状态：`agent_sdk_trainer.py:dbg()` 用 `/workspace/results/` 路径（W2 squash 合并后的版本，从 6b1f522e 来）
- 同样 dbg() helper 在 `examples/openhands_sdk/openhands_agent.py` 也有
- 多处 `dbg(...)` 调用散布在 W2 / W3 / stage2 各阶段
- 如果 W2 时就删 dbg → W3 / stage2 cherry-pick 时撞冲突
- 决策：W2/W3/stage2 全部整合完后，加单独 cleanup commit 替换 `dbg(...)` 为 `logger.debug(...)` 或直接删

### 修订后统计

**16 → 11 必要 → squash 成 1 commit**。剔除占比 31%，整合 commit 数压缩 91%。

### Verify 方法论补充

3 条 commit message 写得像 dev iteration 临时改动（`"customize"` / `"change"`），实际是 path 标准化——**必须读 patch 才能区分**，不能只看 commit message。

新增 verify check：commit message 含 `"customize"` / `"change"` / `"hack"` / `"tmp"` / `"fix me"` 等"非正式"词时，强制 `git show` 完整 patch。

---

## W3 (W2.18-W2.30) Deep Debug Review (14 commits → 1 squash commit)

`903c6165` → `c4efcc07`：L3 闭环硬骨头修复 — DooD workspace 对齐 + L3 unblock (max_model_len / max_prompt_length / MAX_ITER) + Trainer DataProto→TensorDict bridge (W2.27/28/29/30)。

### 各 commit 决策

| Commit | 时间 | 决策 | 一句话 |
|---|---|---|---|
| `903c6165` W2.18 | May 22 | ❌ **剔除** | mock_rollout returns float — 改 mock_rollout.py，跟 W2 阶段 mock 工具整体剔除一致 |
| `feb85471` change abs path | May 26 06:41 | ✅ **合并** | openhands_agent.py 的 dbg() log path 标准化 `/home/t00893162/` → `/workspace/results/`（跟 W2 squash 里 6b1f522e 的 agent_sdk_trainer.py dbg path 同性质，两处 dbg helper 一起在 stage3+ cleanup commit 里删）|
| `fdb8e236` bugfix dood -v mount | May 26 15:35 | ❌ **剔除** | 中间态把 workspace_temp 改成 `/tmp/openhands_workspace`，10 分钟后被 `bedbb1b0` (W2.20) 改成 `/home/docker/openhands_workspace` 取代 |
| `bedbb1b0` W2.20 | May 26 15:45 | ✅ **合并** | workspace_temp hardcode `/home/docker/openhands_workspace`（plan §14.5 锁定）+ train script 顶部 fail-fast precheck（目录存在 + 可写 + bind mount fstype 检查）|
| `15686e0e` precheck relax | May 26 16:12 | ✅ **合并** | precheck 改用 `findmnt -T`（walk up mount tree，让 `-v /home/docker:/home/docker` 这种祖先 mount 也通过） |
| `e5e6c9ce` W2.21 | May 26 | ✅ **合并** | `actor_rollout_ref.rollout.max_model_len` 32768 → 49152（plan §14.5 锁定，OpenHands 首轮 prompt 30k+ 需要更大 cap） |
| `2a2fdbb7` W2.22 | May 26 | ⚠️ **partial cherry-pick** | 保留 `OPENHANDS_MAX_ITERATIONS=1` 改动（stage2 lock）；跳过 `reward = random.random()` 改动（stage2 A 已删，避免 squash 里带"将撤销代码"） |
| `e2ff129f` add debug db py | May 26 17:24 | ✅ **合并** | 新建 `diagnose_trace_store.py`（276 行独立诊断工具，不依赖训练栈，只读 SQLite + docker inspect 检查 trace store 跟 OpenHands session_uid 对齐情况；W2.23 配套，可复用 ops tool） |
| `4f941da2` W2.23 | May 26 | ✅ **合并** | `data.max_prompt_length` 8192 → 32768（plan §14.5 锁定，OpenHands 首轮 prompt 30k+，过滤阈值低则全 step 被 skip）|
| `90690d55` W2.24 | May 26 | ❌ **剔除** | random fallback 在 reward==0 path 的修补 — stage2 A 已完全删除 fallback，整个 commit 无需 |
| `dd229d51` W2.27 | May 26 | ✅ **合并** | **核心 trainer fix**：AgentSdkTrainer 加 3 步 DataProto→TensorDict bridge（`_compute_old_log_prob` / `_compute_ref_log_prob` / `_update_actor` 调 worker 前的 `to_tensordict` + `left_right_2_no_padding` + `assign_non_tensor` metadata 全套），verl PR #2733 漂移导致 rllm trainer 漏 port 的根本修复 |
| `5eb4224f` W2.28 | May 26 | ✅ **合并** | `batch.meta_info["temperature"]` per-PPO-iter 一次性 set（mirror verl `fit():1395`；Megatron forward_step 读 `batch["temperature"]` 无 default，缺会 KeyError） |
| `81cb31f0` W2.29 | May 26 | ✅ **合并** | `actor_rollout_ref.model.use_remove_padding` + `actor.megatron.use_remove_padding` True → False（plan §14.5 锁定，MindSpeed GDN 不支持 packed seq） |
| `c4efcc07` W2.30 | May 26 | ✅ **合并** | **W2.27 bridge β-deferred output 后处理**：worker 返回的 nested tensor 经 `no_padding_2_padding` + key rename（worker "log_probs"/"entropy" → rllm "old_log_probs"/"entropys"）+ `from_tensordict` 重包，避免下游 `select_idxs` 撞 `NotImplementedError(aten.index.Tensor)` |

### W3 修订后统计

**14 → 9 完整 + 1 partial 必要 = 10 处合并，4 剔除**：

| 剔除 | 一句话理由 |
|---|---|
| `903c6165` W2.18 | mock 工具 |
| `fdb8e236` DooD 中间态 | 10 分钟后被 W2.20 取代 |
| `90690d55` W2.24 | stage2 A 已完全删 fallback |

W2.22 是 partial cherry-pick：保留 MAX_ITER=1，跳过 random fallback hunk。

### Partial cherry-pick (W2.22) 实施细节

W2.22 patch 改了 3 个文件：
- `plan markdown`：保留（plan 自然演进）
- `train_openhands_qwen36_npu.sh`：`OPENHANDS_MAX_ITERATIONS="${OPENHANDS_MAX_ITERATIONS:-1000}"` → `:-1` —— **保留**（plan §14.5 锁定）
- `openhands_agent.py`：`reward = 0.0` → `reward = random.random()` + STAGE1 MOCK 注释 —— **跳过**（squash 后保持 stage2 A 的 reward=0.0 状态）

实施方式：`git cherry-pick --no-commit -X theirs 2a2fdbb7` 后 manual edit `openhands_agent.py` 撤回 random 部分。

### MAX_ITER=1 保留的一致性理由

MAX_ITER=1 看起来像调试值，但本质上跟 plan §14.5 其他两个 stage2 临时 default 同性质：

| 临时 default | stage2 lock 理由 | stage3+ 解锁条件 |
|---|---|---|
| `OPENHANDS_MAX_ITERATIONS=1` | 多 turn prompt 撞 max_model_len 49k | 接 condenser 后改 30+ |
| `rejection_sample.enable=False` | 单 turn 真 reward `is_correct=False` 概率高，开了 batch 全 drop | reward 信号稳定后切 True |
| `norm_adv_by_std_in_grpo=False` (Dr.GRPO) | std=0 风险 + reward ≤1 | stage3+ 收敛对比时切回 True |

三者要么一起留，要么一起剔。后两者是 stage2 工作核心成果不能剔，所以 MAX_ITER=1 同保留。stage3+ cleanup 时三个一起解锁。

### W3 squash 计划（初版 → 拆分版）

**初版尝试**：11 个必要 commit 全部 squash 成 1 个 commit。第一次按 W2 squash 同模式做出来后回头评估，发现：

- W2 11 个 commits 是"同一 feature 演进"（训练脚本逐步对齐），squash 成 1 commit 合理
- **W3 11 个 commits 跨 3 个完全不同子系统**：OpenHands runtime / 训练脚本 config / Trainer 核心架构 bridge —— squash 成 1 commit 让 trainer bridge 这种**核心架构修复**埋在 git log 里看不见

**拆分版**（最终采用）：拆 3 个 commit，每个 logical scope 单一：

| # | Commit message | 包含原 commits | 性质 |
|---|---|---|---|
| 1 | `fix(openhands): DooD workspace alignment + diag tool (W2.20-W2.23 runtime path)` | feb85471 + bedbb1b0 + 15686e0e + e2ff129f | OpenHands runtime / DooD |
| 2 | `fix(stage1): L3 unblock — context/iter/padding configs (W2.21-W2.29)` | e5e6c9ce + 2a2fdbb7 (partial) + 4f941da2 + 81cb31f0 | 训练脚本 / algorithm config |
| 3 | `fix(trainer): AgentSdkTrainer DataProto→TensorDict bridge (W2.27/W2.28/W2.30)` | dd229d51 + 5eb4224f + c4efcc07 | **核心架构 fix** |

Trade-off：W3 阶段整合分支多 2 个 commit（从 1 个变 3 个），换来 trainer bridge 这种核心修复在 git log 直接 visible + 单子系统 revert 容易。

**实施步骤**：

1. `git reset --hard 0e4dd7e9`（回 W3 log 之后，丢弃初版 squash）
2. Group 1: `git cherry-pick --no-commit -X theirs feb85471 bedbb1b0 15686e0e e2ff129f` → commit `34cfe866`
3. Group 2: 同上 `e5e6c9ce 2a2fdbb7 4f941da2 81cb31f0` → manual edit `openhands_agent.py` 撤回 W2.22 random fallback + 删 `import random` → commit `ff7dc3f2`
4. Group 3: 同上 `dd229d51 5eb4224f c4efcc07` → commit `35f12439`

最终 W3 阶段贡献 4 commits：1 log + 3 squash commits。

---

## Stage2 Review (8 commits → 5 squash commits)

Stage2 是这次对话从一开始做的工作——audit + 实施工作 + 配套修复（本对话内 task #1-#10 对应的 8 个 commits）。整合到 verl-main 时按 logical sub-theme 拆 5 个 squash commit。

### 8 个 input commits 分类决策

| 原 commit | 决策 | 一句话 |
|---|---|---|
| `82d01ab3` audit doc + plan locks | ✅ → Group 1 | STAGE2_FIT_LOOP_AUDIT.md (新建 155 行) + plan §14.5 加 stage2 lock 表行 (use_critic / sum_pi_squared / rejection_sample)；性质：docs only |
| `38f18b10` trainer audit fix | ✅ → Group 2 | fit_agent 全 drop 兜底 continue + driver 侧 rollout_log_probs 填充率 metric + MFU observability 提取 |
| `a22295a5` engine audit fix | ✅ → Group 2 | 删 hardcoded `max_prompt_length = 16384` + DEBUG print 残留 + engine 侧 rollout_log_probs 填充率统计 (透出 meta_info) |
| `0dce238c` plan soft-constraint backfill | ✅ → Group 1 | plan §0.1 / §0.2 "可调 / 起步" → "已实测 = X" + footer note (verl 权威基线 vs stage1 锁定值区别) |
| `b3a82c4f` openhands runtime cleanup | ✅ → Group 3 | 默认 docker rm -f + KEEP_OPENHANDS_CONTAINER=1 escape；默认 shutil.rmtree workspace + KEEP_OPENHANDS_WORKSPACE=1 escape；恢复 _CONTAINER_TIMEOUT 在 docker wait 生效 |
| `c8412e94` Dr.GRPO 决策 | ✅ → Group 4 | 删 random fallback (reward=0.0) + algorithm.norm_adv_by_std_in_grpo=False (Dr.GRPO) + rollout/std0_groups + rollout/std0_rate metric |
| `3362e70a` 诊断 metric | ✅ → Group 2 | reward/{mean, std, max, zero_rate} + adv/{has_nan, has_inf, abs_max} defense + prompt_length/{raw_p50, raw_p95, raw_max, raw_mean, filtered_rate} |
| `6324d5f6` DooD precheck FATAL | ✅ → Group 5 | train script `docker ps` FATAL 检查 + workspace fstype overlay/tmpfs WARN → FATAL 升级 |

全部 8 个 commits 都必要——是本对话内为整合分支专门做的工作。**0 个剔除**。

### 5 个 squash commit 输出

| # | Commit subject | 包含 input commits | 性质 |
|---|---|---|---|
| 1 | `docs: stage2 fit-loop audit + plan config locks` | 82d01ab3 + 0dce238c | docs / plan |
| 2 | `fix: stage2 audit findings + observability metrics` | 38f18b10 + a22295a5 + 3362e70a | **核心 audit fix + 配套 metric** |
| 3 | `fix(openhands): container/workspace cleanup + docker wait timeout` | b3a82c4f | 独立运维 |
| 4 | `feat(stage1): Dr.GRPO + std0 metric (drop random fallback)` | c8412e94 | 算法决策 |
| 5 | `fix(train-script): DooD precheck FATAL` | 6324d5f6 | 独立 precheck |

### 拆分逻辑（继承 W3 拆分经验）

跟 W3 拆 3 个 commit 同精神——按 logical sub-theme 拆，避免把不相关子系统混在 1 commit 里：

- **Group 1（docs）独立**：跟 W1/W2/W3 docs 同 pattern
- **Group 2（audit + observability 合并）**：38f18b10 / a22295a5 / 3362e70a 三者性质强相关（都是 audit 主线产物 — 漏 port 修复 + 配套观测）
- **Group 3（openhands runtime）独立**：性质完全不同（运维/稳定性），独立 visible
- **Group 4（算法决策）独立**：Dr.GRPO 是关键算法选择决策（删 fallback + Dr.GRPO + std0 metric 三者强相关），独立 visible
- **Group 5（precheck）独立**：debug session 沉淀的独立 startup fix

### 特殊 cherry-pick 处理

`c8412e94` (Group 4) 的 `openhands_agent.py` 改动跟 W3 squash 已经手工 apply 的 `reward = 0.0` 一致；cherry-pick 时 -X theirs 接受 incoming，结果跟 working tree 一致没新增 diff。其他 3 文件（train.sh / agent_sdk_trainer.py / plan）正常 apply。

### 实施步骤

```bash
# Group 1
git cherry-pick --no-commit -X theirs 82d01ab3 0dce238c
git commit -m "docs: stage2 fit-loop audit + plan config locks"

# Group 2
git cherry-pick --no-commit -X theirs 38f18b10 a22295a5 3362e70a
git commit -m "fix: stage2 audit findings + observability metrics"

# Group 3
git cherry-pick --no-commit -X theirs b3a82c4f
git commit -m "fix(openhands): container/workspace cleanup + docker wait timeout"

# Group 4
git cherry-pick --no-commit -X theirs c8412e94
git commit -m "feat(stage1): Dr.GRPO + std0 metric (drop random fallback)"

# Group 5
git cherry-pick --no-commit -X theirs 6324d5f6
git commit -m "fix(train-script): DooD precheck FATAL"
```

5 个新 commit messages 完全不带 stage1 日志的 W2.X / plan §13.x / audit lesson #X 引用（吸取上一次教训；commit subject + body 都用 logical 描述）。

---

## 整合分支用户接手指南

> **重要前提**：整合分支是 stage1+2 工作成果 baseline，**不是**"立刻能 L3 跑通的 sanity baseline"。下方说明用户怎么快速验证整合后链路 + 怎么调到正式跑状态。

### Step 0：环境前置（一次性）

确保 NPU 节点 main container 启动时挂好 5 个关键 bind mount（参考 W2 squash 后 train script 顶部 precheck 提示）：

```bash
docker run \
  -v /var/run/docker.sock:/var/run/docker.sock      # DooD 必需 \
  -v /home/docker/openhands_workspace:/home/docker/openhands_workspace  # OpenHands 子容器 workspace_temp \
  -v /home/docker/Qwen3.6-35B-A3B:/home/docker/Qwen3.6-35B-A3B          # 模型 60GB+ \
  -v /workspace/results:/workspace/results                                # artifact + trace DB + log \
  -v /dev:/dev --privileged                                               # NPU 设备访问 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro                # CANN driver \
  # ... 其他 NPU mount 见 openhands_agent.py:539-546 \
  <image>
```

train script `bash examples/openhands_sdk/train_openhands_qwen36_npu.sh` 启动时 fail-fast precheck 会**直接 exit 1** 告诉你缺哪个 mount。

### Step 1：快速验证整合链路（10 分钟，单步训练 sanity）

**整合分支没有 mock 数据准备工具**，需要先有一份真实算子数据集 parquet（schema 跟 `_MockNPUOperatorParquetDataset` 期望对齐：含 `prompt`、`instruction`、`op_name`、`arch`、`extra_info` 字段）。

最小可跑配置：

```bash
# 替换 _MOCK_NPU_PARQUET 路径指向你的真实数据 parquet
# (在 examples/openhands_sdk/train_openhands_qwen36_npu.py:60 改)
_MOCK_NPU_PARQUET = "/your/real/operator_dataset.parquet"

# 跑训练 1-2 step 看链路通不通
bash examples/openhands_sdk/train_openhands_qwen36_npu.sh
```

**观察关键 metric**（都是 stage2 D 加的）：
| Metric | 期望 | 含义 |
|---|---|---|
| `adv/has_nan` | = 0 | trainer 数值健康，advantage 无 NaN |
| `adv/has_inf` | = 0 | 同上 |
| `batch/skipped_all_drop` | 接近 0 | 整批 drop 兜底没频繁触发 |
| `rollout/log_probs_fill_rate` | 接近 1.0 | rollout logprobs 跟 response shape 对齐 |
| `prompt_length/raw_p95` | < 32768 | max_prompt_length 没被截 |
| `rollout/std0_rate` | < 1.0 | 组内 reward 有 variance（如果 1.0 说明全组同值，跟 reward 函数有关）|

看到 `Saved checkpoint` log = ckpt 路径通。

### Step 2：开始真正训练前需要解锁的 stage2 临时 default

| 临时 default | 当前值 | 改在哪里 | 解锁条件 |
|---|---|---|---|
| `OPENHANDS_MAX_ITERATIONS` | 1 | `train_openhands_qwen36_npu.sh:158` `export ... :-1}` → `:-30}` 或显式 env | 接 OpenHands condenser 后（让多 turn prompt 不撞 max_model_len） |
| `rllm.rejection_sample.enable` | False | hydra override `+rllm.rejection_sample.enable=True` 或脚本 `ARGS+=(...)` | 真 reward 稳定（is_correct=True 比例 > 你的 drop 阈值） |
| `algorithm.norm_adv_by_std_in_grpo` | False (Dr.GRPO) | hydra override `algorithm.norm_adv_by_std_in_grpo=True` | 想跑经典 GRPO 做收敛对比时 |
| OpenHands rollout 真 reward 路径 | 已启用 | 无需改（stage2 A 已删 random fallback） | — |

### Step 3：正式训练前的 stage3 工作（不在整合分支范围）

- 接 OpenHands `Condenser`（LLM summarize 或 Recent）让多 turn prompt 不线性涨
- 真实算子数据集 + reward 评测协议（这部分 plan §0 划给"相邻团队"）
- 多节点扩展：HCCL_IF_IP / HCCL_NIC_NAME / Ray 端口分配（plan §14.3 已 track）
- 单 turn 撞 std0 时改用 trainer 层注入（方案 0+ 升级为 A2 风格的精准兜底，stage3 决定）

### Quick reference：整合分支文件清单

| 文件 | 性质 | 主要内容 |
|---|---|---|
| `examples/openhands_sdk/train_openhands_qwen36_npu.{py,sh}` | 训练入口 | hydra config + Ray cluster + 各 stage1/2 fix 的 inline 注释 |
| `examples/openhands_sdk/openhands_agent.py` | rollout | OpenHands docker DooD + reward 函数 + workspace 管理 |
| `examples/openhands_sdk/diagnose_trace_store.py` | ops 工具 | trace store ↔ session_uid 对齐诊断（独立脚本） |
| `examples/openhands_sdk/workspace/` | OpenHands 容器内 | Dockerfile + entrypoint + AGENTS.md + skills |
| `rllm/trainer/verl/agent_sdk_trainer.py` | AgentSdkTrainer | DataProto→TensorDict bridge + stage2 audit metrics |
| `rllm/engine/agent_sdk_engine.py` | AgentSdkEngine | transform_results_for_verl + rollout log_probs fill 监控 |
| `STAGE_INTEGRATION_LOG.md` | 本日志 | 整合决策 + 用户接手指南 |
| `QWEN36_OPENHANDS_AGENT_RL_PLAN.md` | 项目 plan | §14.5 配置锁定 + §0 范围声明 + §13.x debug 历史 |

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
| 2026-05-29 | W2 mock/test infra 全部剔除 | b07a55d7 / bd475579 / d4482bf1 / ecda05d7 都是开发期辅助工具，整合分支只保留生产代码 baseline |
| 2026-05-29 | W3/Stage2 review 工作流改为"先 log 等用户确认 + 再 cherry-pick" | 避免直接 cherry-pick 后再回退 |
| 2026-05-29 | W2 最终方案：11 个必要 commit squash 成 1 个 + 决策下放到脚本 inline 注释 | 整合分支是生产代码 baseline 不是 debug archive；git log 简洁、技术决策就近、读脚本直接 follow 决策；dbg() helper 推迟到 stage3+ cleanup commit 一次性删 |
| 2026-05-29 | W3 14 commits → 1 squash commit；剔除 mock W2.18 + DooD 中间态 fdb8e236 + fallback 修补 W2.24；W2.22 partial cherry-pick（保留 MAX_ITER=1，跳过 random fallback hunk）| 跟 W2 同方针——squash 生产代码 baseline + 决策保留脚本注释；W2.22 partial 避免 squash 里带"stage2 A 将撤销代码" |
| 2026-05-29 | `OPENHANDS_MAX_ITERATIONS=1` 跟 `rejection_sample.enable=False` 和 `Dr.GRPO` 一致性地保留（都是 stage2 临时 default） | 三者性质相同（接 condenser + reward 稳定后解锁），不能分开处理；剔除任一会破坏 stage2 工作核心成果 |
| 2026-05-29 | log 加 "整合分支用户接手指南" 章节 | 整合分支不带 mock 工具，L3 不能"直接跑"；用户接手时需明确知道 Step 0 mount / Step 1 验链路 / Step 2 解锁 stage2 default / Step 3 stage3+ 工作 |
| 2026-05-29 | W3 squash 拆分（初版 1 commit → 拆 3 commit） | W2 squash 合理是因为 11 个 commits 是同一 feature 演进；W3 11 个 commits 跨 OpenHands runtime / 训练脚本 config / Trainer 核心架构 bridge 三个完全不同子系统，1 commit 让 trainer bridge 这种核心架构修复在 git log 不可见；拆 3 个 commit 每个 logical scope 单一 |
| 2026-05-29 | Commit message rewrite — 6 个 commits 移除 stage1 日志绑定（W2.X / plan §13.x） | 用户校准要求整合分支独立于 stage1 工作日志；用 git filter-branch --msg-filter 一次性重写，commit content 不变；只更新过期 SHA 在 log 文档里的引用 |
| 2026-05-29 | Stage2 squash 拆 5 commit | 继承 W3 拆分精神——8 个 input commits 跨 audit/observability / runtime / 算法决策 / precheck 4-5 个子系统；按 logical sub-theme 拆 5 个 commit 让每个 scope 单一；audit + observability 合并 1 commit 因为三者性质强相关（都是 audit 主线产物）|

---

## 接下来的工作（按优先级）

整合工作的 cherry-pick + squash + commit message rewrite 已完成。剩下 3 项 follow-up，**新对话应该按下面优先级顺序做**。

### 🔴 优先级 1：Stage3+ cleanup commit（**不需要环境**，~30 分钟）

把整合工作彻底收尾，做一个 cleanup commit 处理之前 W2 / W3 / Stage2 squash 时显式 defer 的项。

**包含 2 项**：

#### 1.1 删除 `dbg()` helper 函数 + 所有调用站点

**当前状态**：`dbg(msg: str)` 函数定义在两处文件，调用散布在多个位置（debug 期手撸的 logger，path standardized 但 helper 本身留着）。

**位置**：
- `rllm/trainer/verl/agent_sdk_trainer.py` 第 64-71 行附近（函数定义）+ 文件内多处调用
- `examples/openhands_sdk/openhands_agent.py` 第 64-75 行附近（函数定义）+ 文件内多处调用

**改法**（选项 A 推荐）：完全删除 `dbg(...)` 调用 + 删 dbg 函数定义；生产代码靠现有 `logger.info(...)` 已经够。

**实施**：

```bash
# 1. grep 所有 dbg() 调用站点
grep -rn "^[^#\"]*dbg(" rllm/trainer/verl/agent_sdk_trainer.py examples/openhands_sdk/openhands_agent.py

# 2. 用 Edit 删除每个 dbg(...) 调用行（注意是行级删除，整行 dbg(...) 都删）
# 不要 sed 因为 Python 缩进容易破坏

# 3. 删 dbg() 函数定义本身（2 处，每处 ~7-10 行）

# 4. python3 -c "import ast; ast.parse(...)" 验证语法 OK

# 5. grep -rn "dbg(" 确认 0 残留（应该只剩字符串内 / 注释内）
```

#### 1.2 reconcile plan markdown 版本引用

**位置**：`examples/openhands_sdk/train_openhands_qwen36_npu.sh:5` 写：
```
# Per QWEN36_OPENHANDS_AGENT_RL_PLAN.md (v2.4):
```

但脚本里实际配置已经超出 v2.4 baseline（W3 / Stage2 改动都进来了）。

**改法**：去掉版本号，只引用 plan 章节名。例如：
```
# Per QWEN36_OPENHANDS_AGENT_RL_PLAN.md key sections:
#   §0.1 / §0.2  Required NPU env + verl-script MoE flags
#   §5.2         max_model_len baseline
#   §14.5        stage1-locked configs (must not flip without re-audit)
```

#### Cleanup commit message 建议

```
chore: stage3+ cleanup — remove dbg() helper + reconcile plan reference

W2/W3/Stage2 squash 时为了避免每阶段 cherry-pick 撞冲突，dbg() helper
函数 + 调用站点都保留，约定整合结束后单独清理。本 commit 实施。

Changes:
  - examples/openhands_sdk/openhands_agent.py
    - remove dbg() helper definition
    - remove all dbg(...) call sites (~N occurrences)
  - rllm/trainer/verl/agent_sdk_trainer.py
    - same

  - examples/openhands_sdk/train_openhands_qwen36_npu.sh
    - header reference "(v2.4)" replaced with chapter names only,
      so the comment doesn't go stale as plan evolves
```

---

### 🟡 优先级 2：Layer A 静态验证（**不需要环境**，~5 分钟）

✅ **已完成**（2026-05-29 本对话内）。结果如下：

| 子项 | 结果 |
|---|---|
| Python syntax (558 文件) | ✅ 0 fail |
| Bash syntax (82 文件) | ✅ 0 fail |
| Import smoke (rllm 顶层 OK；深层 `AgentSdkTrainer` 等缺 torch/verl/rllm[train] — Mac 上 expected) | ✅ |
| Train script precheck（在 Mac 上正确 fail-fast on `/home/docker/openhands_workspace` 不存在）| ✅ |

A.4 关键验证：4 处整合 fix 都在 train script 启动序列里**实际生效**：Megatron-Bridge 3-路 dispatch / OPENHANDS_MAX_ITERATIONS=1 / 路径标准化 / DooD precheck FATAL。

**做 cleanup commit (优先级 1) 之后应该再跑一次** Layer A.1 / A.2 确认没引入语法错误。

---

### ⏸ 优先级 3：Layer B 启动验证（**需要 NPU 环境**）

在 NPU container 内跑完整 train script，看启动序列是否完整。**不需要真实算子数据集**——预期挂在 dataloader（找不到 parquet）。

**前置**：参考 STAGE_INTEGRATION_LOG.md "整合分支用户接手指南" 的 Step 0 mount 列表。NPU container 启动时挂好 5 个关键 bind mount。

**操作**：

```bash
# 在 NPU container 内（main container）
cd <path-to-verl-main-checkout>
bash examples/openhands_sdk/train_openhands_qwen36_npu.sh
```

**期望观察**（按顺序）：

| 阶段 | 期望 |
|---|---|
| Top precheck | docker.sock check + workspace bind mount check + Megatron-Bridge dispatch — 全过 |
| Ray cluster | `ray start --head` 起来 |
| vllm worker (slowest) | 加载 Qwen3.6-35B-A3B 完成（~5-10 min 模型加载）|
| LiteLLM proxy | subprocess 起来 + 监听 PROXY_PORT |
| dataloader | **挂在"找不到 parquet"** ← 这是 expected PASS 标志 |

挂在 dataloader = Layer B PASS（说明 train script 启动序列完整）。

**如果中途挂在别的地方**：
- precheck 挂 → 看具体哪个 mount 缺
- Ray 挂 → 看 NPU 占用 + 端口
- vllm 挂 → 看 vllm-ascend log（compilation/cudagraph/enforce_eager 问题）
- LiteLLM 挂 → 看 PROXY_PORT 是否被占

---

### ⏸ 优先级 4：Layer C 训练验证（**需要 NPU + 真实算子数据集**）

按 STAGE_INTEGRATION_LOG.md "整合分支用户接手指南" Step 1 走：

1. 准备真实算子任务 parquet（schema 跟 `_MockNPUOperatorParquetDataset` 期望对齐，含 `prompt`/`instruction`/`op_name`/`arch`/`extra_info`）
2. 在 `train_openhands_qwen36_npu.py:60` 改 `_MOCK_NPU_PARQUET` 路径指向你的 parquet
3. 跑 1-2 step 训练（默认 `total_training_steps=1` 或显式设 `STAGE1_DRY_STEPS=1`... wait, STAGE1_DRY_STEPS 已经剔除了，所以要在 train_openhands_qwen36_npu.sh 或 hydra override 里改 `trainer.total_training_steps=1`）
4. 观察 stage2 D 加的诊断 metric

**期望 metric 值（健康状态）**：

| Metric | 期望 |
|---|---|
| `adv/has_nan` | = 0（Dr.GRPO 下必 0；sustained > 0 = NaN 来自 KL/ratio/log_prob 别处）|
| `adv/has_inf` | = 0 |
| `batch/skipped_all_drop` | 接近 0（如果高 = real reward 启动初期 is_correct 频繁 False）|
| `rollout/log_probs_fill_rate` | 接近 1.0（< 1.0 = rollout logprobs 跟 response 长度对不齐）|
| `prompt_length/raw_p95` | < 32768（OpenHands real prompt < max_prompt_length cap）|
| `rollout/std0_rate` | 看实际值；如果接近 1.0 = 组内 reward 全同（reward 函数有 variance 问题）|

看到 `Saved checkpoint` log = Layer C PASS。

---

## Reference：整合分支用户接手指南 → 见上文 §"整合分支用户接手指南"

新对话不需要重写这个章节，直接 reference。

---

## 如何 push verl-main

整合分支目前**未 push**。本地完成 stage3+ cleanup + Layer A 复检后，可以 push 给协作者：

```bash
git push -u origin verl-main
```

如果协作者也用 git filter-branch 或 force rewrite 过 verl-main 远端版本，需要 `--force-with-lease`。一般情况第一次 push 直接 `-u origin verl-main` 即可。

