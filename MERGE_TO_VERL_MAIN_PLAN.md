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
| `openhands_agent.py` HTTP worker（`_run_remote_eval_worker` + `_tar_directory_b64` + `_manifest_directory` + `_extract_tar_b64_into` + `_metadata_for_proxy_url`）| **C 当前并** | 独有 | 路线无关的 rollout 执行机制（docker run OpenHands 容器）。当前阶段并入 + rollout 路由改 1 行 |
| `openhands_agent.py` reward（`_npu_operator_reward` + `_impl_newer_than_metrics` + `_snapshot_success_as_best`）| **推迟 skills 阶段** | 独有/diverge | ⚠️ **路线绑定**：我们读 triton `metrics.json`（success/correctness_ok + 防stale + best快照），verl-main 读 AscendC `trace.md`（Phase 3/4），两套完全不同。跟 triton skills + operator_pipeline.sh 死绑，必须一起换。当前并入会破坏 verl-main AscendC pipeline → reward 全 0 |
| `openhands_agent.py` `_setup_npu_operator_workspace` / `_run_openhands_container` / `_archive_npu_artifacts` / `rollout` 其余 | **A 保留 verl-main** | — | 配套 verl-main AscendC PKG；`_run_openhands_container`（bind-mount）作 HTTP worker 的 localhost fallback |
| `remote_eval_worker.py` | **C** | 独有（verl-main 无此文件） | HTTP worker server + GC + /health + /admin/reset-locks |
| `agent_sdk_engine.py` `_ENGINE_DEBUG`/`_dprint` gate | **C 可选 TODO** | base:63 的一部分 | env-gated debug print。verl-main 无。纯调试辅助，本阶段先不并（插入点要匹配 verl-main 结构有风险），记可选增强 |
| `agent_sdk_engine.py` rollout_logprobs padding | **A 用 verl-main**（已确认） | base:80 的一部分 | 已实测：verl-main 更完善（量化 mismatch + `rollout_log_probs_fill_rate` metric，部分 mismatch 也处理），我们是 all-or-nothing |
| `agent_sdk_trainer.py` LCM padding + DataProto bridge / Dr.GRPO / stage2 audit | **A 用 verl-main**（已确认） | base:our18/vm313 | 我们 18 行=padding，verl-main:998 字节级覆盖；verl-main 另有 bridge/Dr.GRPO 更全 |
| `distributed_npu_lock.py` | **C 已 port** ✅ | base:our37/vm0 | verl-main 没改；已 port 我们的 device_ids（physical id 8-15 支持）|
| `create_mock_npu_operator_data.py` | **A 用 verl-main**（已确认） | base:our4/vm9 | verl-main 覆盖我们的 ×4 + 多修 prompt list[dict] bug |
| `.gitignore` | **已并入** ✅ | base:our4/vm9 | res*.log + .codemate/ 并入 verl-main |
| `runner.py` | **B 待 reconcile** | base:our405/vm386 | 两边各 ~400 行大改，需 3-way |

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
2. **C 类并入 openhands_agent.py（仅 HTTP worker，路线无关）**：
   - HTTP worker：`_run_remote_eval_worker` + tar helpers + 常量 + `rollout` 路由改 1 行
   - bind-mount `_run_openhands_container` 保留作 localhost fallback
   - **reward 不在此阶段**（路线绑定 triton metrics.json，见 §4）
3. **B 类 reconcile**（逐行，要小心）：
   - `runner.py`（两边各 ~400 行，唯一剩下的大 B）
   - 其余 B 已实测落地：distributed_npu_lock(C port✅) / .gitignore(并入✅) /
     create_mock + agent_sdk_trainer + agent_sdk_engine(均判 A 用 verl-main)
4. **A 类**：直接用 verl-main（padding、DataProto bridge、Dr.GRPO、rollout_logprobs、
   create_mock prompt 修复）。
5. **验证**（不跑训练）：py_compile + bash -n + worker@localhost 单 rollout 冒烟。

估算（修正后）：**1.5-3 人天**，主要成本在 openhands_agent.py HTTP worker 并入 +
runner.py 的 3-way reconcile。reward 推迟到 skills 阶段后，当前阶段范围明显缩小。

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
  4. **reward（从当前阶段移来）**：整块 port openhands-refactoring 的
     `_npu_operator_reward` + `_impl_newer_than_metrics` + `_snapshot_success_as_best`
     （读 triton `metrics.json`：success/correctness_ok + 防 stale + best 快照），
     替换 verl-main 的 AscendC `trace.md`/Phase reward。必须跟 operator_pipeline.sh
     的 triton 输出 schema（metrics.json）同步换，否则 reward 读不到。
  5. 数据集：verl-main NPUKernelBench（AscendC）→ triton KernelBench 数据
     （openhands-refactoring `prepare_kernelbench_openhands_data.py` 是 triton）
  6. port `validate_triton_impl.py`、`ascend_op_gen_agent_triton.md`
  7. `_setup_npu_operator_workspace`：换 triton PKG 后可能要调（impl 路径
     `src/{op}_triton_ascend_impl.py` vs AscendC `output/{op}/model_new_ascendc.py`）

## 5. 风险 / 待确认

1. **openhands_agent.py reconcile 是真成本** —— 从不同 pqy 基线 diverge，
   reward 已确认推迟到 skills 阶段，当前只并 HTTP worker（路线无关）。
2. ~~rollout_logprobs~~ **已 verify**：verl-main 更完善（量化 mismatch +
   `rollout_log_probs_fill_rate` metric），用 verl-main，我们的不并。
3. **HTTP worker vs bind-mount 共存**：决定 bind-mount 是否作 same-host 快路径
   保留还是移除。当前决定：保留 bind-mount 作 localhost fallback（HTTP worker
   `_run_remote_eval_worker` 内部 `if not REMOTE_EVAL_URL: 走 bind-mount`）。
4. **verl-main 有未提交工作 + in-progress merge**（你的 qwen36）—— 在从
   origin/verl-main 拉的新分支 `verl-main-merge` 上做，绝不碰你 local 的 verl-main。
   PR 已撤（删 remote 分支），纯本地操作，你 review 后再 push。
5. ~~distributed_npu_lock.py~~ **已确认 + port**：verl-main 没改（base diff=0），
   不是两边都改；已 port 我们的 device_ids 支持。

## 6. 不要做的事（别重蹈先前的错）

- ❌ 别把 HTTP worker 当"冗余"丢掉 —— 它是多机架构
- ❌ 别整体 `git merge` openhands-refactoring 进 verl-main —— 15 个文件冲突，
  大部分 engine/trainer 改动已被取代，会 reconcile 329+131 行噪音
- ❌ 别假设 verl-main newer = 每个函数都更好 —— reward 改进
  （_snapshot_success_as_best）和 HTTP worker 是我们的且有价值
- ❌ 当前阶段别动 skills —— 独立议题，要用户的 triton skills 仓
- ❌ 别把 reward 函数当"通用防 stale"并入 —— 它绑定 triton metrics.json schema，
  跟 AscendC trace.md 不兼容，属 skills 阶段

## 7. reconcile 实测进展（2026-05-29，在 verl-main-merge 分支）

逐文件做了 3-way 对比（5daa24c0 base / verl-main / openhands-refactoring），
精确分类（按相对 base 的改动量，非两分支当前 diff）。已落地的 commit 在
local `verl-main-merge`，未 push（你 review 后再 push）。

| 文件 | 实测结论 | 状态 |
|---|---|---|
| `distributed_npu_lock.py` | C：verl-main base diff=0，port 我们的 device_ids | ✅ done (d81be77f) |
| `.gitignore` | 两边改不同段，并入 res*.log + .codemate/ | ✅ done (d81be77f) |
| `create_mock_npu_operator_data.py` | A：verl-main 覆盖 ×4 + 多修 prompt list[dict] bug | ✅ 用 verl-main |
| `agent_sdk_trainer.py` | A：我们 18 行=padding，verl-main:998 已覆盖；verl-main 另有 bridge/Dr.GRPO | ✅ 用 verl-main |
| `agent_sdk_engine.py` | A：rollout_logprobs verl-main 更优；_dprint gate 记可选 TODO | ✅ 用 verl-main |
| `openhands_agent.py` | C 当前并 HTTP worker；reward 推迟 skills 阶段；其余保留 verl-main | ⏳ 待实施 |
| `runner.py` | B：两边各 ~400 行，待 3-way reconcile | ⏳ 待做 |

**关键发现（修正了 §2.1 先前分类）**：
- reward（`_npu_operator_reward` + 防 stale helpers）**不是通用改进，是 triton
  路线绑定**：我们读 `metrics.json`，verl-main 读 `trace.md`。从"当前 C 类必须并"
  改为"skills 阶段一起换"。当前并会破坏 verl-main AscendC pipeline。
- 先前用"两分支当前 diff 行数"分类不准（把 verl-main 没改的 distributed_npu_lock
  误判 B）。改用"相对 base 的双侧改动量"才能区分 A/B/C。

**当前阶段剩余工作**：openhands_agent.py HTTP worker 并入 + runner.py reconcile
+ C 类干净加（remote_eval_worker.py / config/ / LOG.md）+ 静态验证。
