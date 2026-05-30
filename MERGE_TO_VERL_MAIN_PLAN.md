# 合并计划：openhands-refactoring ↔ verl-main

> 状态：**仅计划阶段。** 目标是 reconcile 两套并行的 OpenHands 集成，
> 不是单向合并 —— 两边各有独有价值。本文档先给计划，B 类逐行 reconcile
> 在 `verl-main-merge` 分支上做。

## 0. 分支现状（2026-05-29）

| 分支 | 距 5daa24c0 的 commit 数 | 角色 |
|---|---|---|
| `origin/verl-main` | 72 | Qwen3.6 + AscendC，stage1 主线（活跃） |
| `origin/openhands-refactoring` | 9 | Qwen3-Coder + triton，pqy 血统（本次工作） |
| merge-base | `5daa24c0` | 最近公共祖先 |
| `verl-main-merge`（本分支） | 从 origin/verl-main 拉出 | 做合并的工作分支，不碰你的 working verl-main |

verl-main 独立做了一套完整的 OpenHands 集成（DooD fix、padding fix、cleanup、
DataProto bridge、Dr.GRPO、NPUKernelBench AscendC 数据）。所以这**不是**
"把 openhands-refactoring 加到 verl-main 上"，而是 reconcile 两套用不同方式
解决重叠问题的实现。

## 1. 架构决策：保留 HTTP worker（已修正先前判断）

先前评估说 HTTP worker "冗余"，因为 verl-main 的 bind-mount DooD fix 更简单。
**那是只看 single-host 的偏差，错了。** 决策：**HTTP worker 作为目标 rollout 架构。**

| 维度 | verl-main bind-mount（进程内 docker run） | openhands-refactoring HTTP worker |
|---|---|---|
| 多机 | ❌ 物理不可能 —— `-v /home/docker/openhands_workspace:...` 是 host 本地路径，远端 eval host 的 dockerd 看不到 trainer host 的 fs | ✅ 天然支持 —— worker 在哪台 host，rollout 就在那台 |
| 代码统一性 | 只有 single-host 一条路径；支持多机要再写一套（技术债） | single = worker@localhost，multi = worker@remote，同一条 `_run_remote_eval_worker` |
| workspace 传递 | 依赖 host fs 路径一致（脆弱，DooD 踩过坑） | HTTP tar 传输，路径无关 |
| 成本 | 无额外进程 | worker 进程 + HTTP 序列化（实测 ~132KB/rollout，可忽略） |

bind-mount 是 HTTP worker 在 `worker=localhost` 时的退化特例，且砍掉了多机能力。
保留 HTTP worker；bind-mount 可作为 same-host 快路径保留，或移除。

## 2. 细粒度内容审计

图例：**A** = verl-main 已有等价且不差于我们，丢弃我们的 · **B** = 两边都改，
逐行 reconcile · **C** = openhands-refactoring 独有且有价值，必须 port。

### 2.1 Rollout / engine / trainer（代码）

| 文件 / 函数 | 类别 | 相对 verl-main diff | 说明 |
|---|---|---|---|
| `openhands_agent.py` `_run_remote_eval_worker` + `_tar_directory_b64` + `_manifest_directory` + `_extract_tar_b64_into` | **C** | 独有 | HTTP worker 入口 + helper，verl-main 完全没有 |
| `openhands_agent.py` `_impl_newer_than_metrics` + `_snapshot_success_as_best` | **C** | 独有 | reward 防 stale + 最佳实现快照。**粗暴 merge 极易漏掉。** |
| `openhands_agent.py` `_npu_operator_reward` / `_setup_npu_operator_workspace` / `rollout` / `_run_openhands_container` | **B** | 大 | 两边从不同 pqy 基线 diverge；reward + workspace 逻辑要仔细 reconcile |
| `remote_eval_worker.py` | **C** | 独有（verl-main 无此文件） | HTTP worker server + GC + /health + /admin/reset-locks |
| `agent_sdk_engine.py` `_ENGINE_DEBUG`/`_dprint` gate | **C** | 131 的一部分 | env-gated debug print（RLLM_ENGINE_DEBUG） |
| `agent_sdk_engine.py` rollout_logprobs padding | **B/C** | 131 的一部分 | logprob 支持（对应 openhands-observability "增加rollout_logprobs"）；需 verify verl-main 是否已有 |
| `agent_sdk_engine.py` 其他 | **A** | 131 的一部分 | verl-main +78 行含 DataProto 路径，更新 |
| `agent_sdk_trainer.py` LCM mini_batch padding | **A** | — | **两边字节级相同**（verl-main:998）。丢我们的。 |
| `agent_sdk_trainer.py` DataProto→TensorDict bridge / Dr.GRPO / stage2 audit | **A** | 329 的一部分 | verl-main 独有且更新，保留 verl-main |
| `agent_sdk_trainer.py` config 读取（我们的） | **B** | 329 的一部分 | 仅当保留 config/ 层时需要 |
| `distributed_npu_lock.py` | **B** | 37 | 两边 diverge，需 reconcile |
| `runner.py` | **B** | 33 | 小，reconcile |
| `create_mock_npu_operator_data.py` | **B** | 7 | 琐碎 |

### 2.2 Config / 基础设施（openhands-refactoring 独有）

| 文件 | 类别 | 说明 |
|---|---|---|
| `config/.env.example`、`config/qwen30b.env` + `RLLM_TOPOLOGY` 开关 | **C** | env 驱动配置，解耦机器/路径/拓扑。verl-main 硬编码在 train_*.sh |
| `debug_oom.sh`（worker health check + reset-locks + cleanup） | **C** | verl-main 有自己的 train_openhands_qwen36_npu.sh；reconcile 或保留两者 |
| `OPENHANDS_REFACTORING_LOG.md`（KU1-8） | **C** | 知识沉淀：DooD、NPU-lock、batch-align、LiteLLM flaky 等。作为参考 port |

### 2.3 verl-main 独有（保留，不动）

- DataProto→TensorDict bridge（`35f12439`）
- Dr.GRPO + std0 metric（`50bc9f96`）
- stage2 audit + observability metrics（`f258836b`）
- NPUKernelBench AscendC 数据集（level2/3/4，约 500 文件）—— **AscendC，见 §4**
- bind-mount DooD fix（`34cfe866`）—— 被 HTTP worker 取代但无害

## 3. 当前阶段执行计划（代码合并，不含 skills）

目标：把 openhands-refactoring 的 **C 类代码/基础设施** 落到 verl-main，
**reconcile B 类**，丢弃 A 类。skills 推迟到 §4。

执行顺序（低风险优先）：

1. **C 类干净新增**（无 verl-main 冲突）：
   - `remote_eval_worker.py`（整文件）
   - `config/.env.example` + `config/qwen30b.env`
   - `OPENHANDS_REFACTORING_LOG.md`
2. **C 类并入 openhands_agent.py**（port 独有函数）：
   - HTTP worker：`_run_remote_eval_worker` + tar helpers + `rollout` 里的路由
   - reward：`_impl_newer_than_metrics` + `_snapshot_success_as_best`
   - **决策**：HTTP worker 作主路径，bind-mount 作 fallback 或移除
3. **B 类 reconcile**（逐行，要小心）：
   - `openhands_agent.py` reward / workspace 函数
   - `agent_sdk_engine.py` rollout_logprobs + _dprint（保留 verl-main 更新的部分）
   - `distributed_npu_lock.py`
   - `runner.py`、`create_mock_npu_operator_data.py`
4. **A 类**：直接用 verl-main（padding、DataProto bridge、Dr.GRPO）。
5. **验证**（不跑训练）：py_compile + bash -n + worker@localhost 单 rollout 冒烟。

估算：**3-5 人天**，主要成本在 openhands_agent.py + agent_sdk_engine.py 的 B 类 reconcile。

## 4. 下一阶段（推迟）：triton skills 移植

**当前阶段不做。** 记在这里以免遗漏。

- skills 来源：**用户专门维护的 triton skills 仓（Claude Code 格式，`.claude/skills/`）**
  —— 不是 openhands-refactoring 里的 pqy 4-skill，也不是 verl-main 的 AscendC 5-skill。
  **仓路径待定（需用户提供）。**
- verl-main 当前 skills（AscendC）：ascendc-translator、tilelang-designer、
  case-simplifier、performance-analyzer、trace-recorder
- 移植任务（下阶段）：
  1. 格式适配：`.claude/skills/` → OpenHands `agent_workdir/.agents/skills/`
  2. `AGENTS.md` / `INSTRUCTIONS.md`：AscendC Phase 流程 → triton 流程
  3. `operator_pipeline.sh`（155 diff 行 AscendC↔triton）：验证后端
     evaluate_ascendc.sh / evaluate_tilelang.sh → triton 验证
  4. reward：`metrics.json` schema ↔ `_npu_operator_reward` triton 对齐
  5. 数据集：verl-main NPUKernelBench（AscendC）→ triton KernelBench 数据
     （openhands-refactoring `prepare_kernelbench_openhands_data.py` 是 triton）
  6. port `validate_triton_impl.py`、`ascend_op_gen_agent_triton.md`

## 5. 风险 / 待确认

1. **B 类 reconcile 是真成本** —— openhands_agent.py 和 agent_sdk_engine.py
   从不同 pqy 基线 diverge，不能盲选一边。每个共同函数要做 3-way 对比
   （5daa24c0 base + verl-main + openhands-refactoring）。
2. **rollout_logprobs**：openhands-refactoring 有 logprob padding；port 前需
   verify verl-main 是否已有等价实现（避免双实现）。
3. **HTTP worker vs bind-mount 共存**：决定 bind-mount 是否作 same-host 快路径
   保留还是移除。保留两者 = 两条代码路径都要测。
4. **verl-main 有未提交工作 + in-progress merge**（你的 qwen36）—— 在从
   origin/verl-main 拉的新分支上做，绝不碰你 local 的 verl-main。
5. **distributed_npu_lock.py 已 diverge（37 行）** —— 两边都改，需真实 diff
   review，不能假设相同。

## 6. 不要做的事（别重蹈先前的错）

- ❌ 别把 HTTP worker 当"冗余"丢掉 —— 它是多机架构
- ❌ 别整体 `git merge` openhands-refactoring 进 verl-main —— 15 个文件冲突，
  大部分 engine/trainer 改动已被取代，会 reconcile 329+131 行噪音
- ❌ 别假设 verl-main newer = 每个函数都更好 —— reward 改进
  （_snapshot_success_as_best）和 HTTP worker 是我们的且有价值
- ❌ 当前阶段别动 skills —— 独立议题，要用户的 triton skills 仓
