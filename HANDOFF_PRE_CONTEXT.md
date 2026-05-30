# Handoff pre-context — OpenHands RL 整合项目（接手用）

> 给新对话窗口的 Claude：**先读完整份**，再读 `MERGE_TO_VERL_MAIN_PLAN.md`
> （当前阶段的逐文件 reconcile 决策）和 `OPENHANDS_REFACTORING_LOG.md`
> （KU1-8 知识沉淀）。user 数据驱动很严格：要**源码引用 + line number**，
> 撤回错判断不硬撑，反对拍脑袋。

---

## 1. 项目全景（30 秒）

两条并行线，共享 rllm 框架但 diverge：

| 线 | 模型 × Agent × 算子 | 主分支 | origin |
|---|---|---|---|
| **stage1**（user 主力） | Qwen3.6-35B-A3B × OpenHands × **AscendC** | `verl-main` | lilac-me/rllm |
| **pqy** | Qwen3-Coder-30B-A3B × OpenHands × **Triton** | `debug_oom` / `openhands` | qiangyupei/rllm-openhands |

两者 diverge 点：`5daa24c0`。

**本会话做了两阶段**：

1. **pqy 解耦嫁接** → 把 pqy 的 OpenHands RL pipeline 从 `5daa24c0` 嫁接到
   lilac-me/rllm 的新分支 `openhands-refactoring`，解耦全部硬编码（路径/IP/端口/
   image），引入 env-driven config，加 HTTP remote eval worker，沿途调试
   OOM/batch-padding/NPU-lock/dockerd-stuck/LiteLLM-flaky（KU1-8）。**已 push**。

2. **merge 到 verl-main**（进行中）→ 把 `openhands-refactoring` 的优质内容
   reconcile 进 stage1 的 `verl-main`。在 `verl-main-merge` 分支。**未 push**，
   等 user review + 冒烟 + push。

### 文档地图（阅读顺序）

仓根目录 md 分两批 —— stage1 既有 + 本会话新增：

| 文档 | 来源 | 何时读 |
|---|---|---|
| **本文件** `HANDOFF_PRE_CONTEXT.md` | 本会话 | **最先** — 全局索引 |
| `MERGE_TO_VERL_MAIN_PLAN.md` | 本会话 | 接 OpenHands merge — 逐文件 reconcile 决策 + §7 进度 |
| `OPENHANDS_REFACTORING_LOG.md` | 本会话 | KU1-8 知识（DooD/padding/lock/LiteLLM 等） |
| `QWEN36_OPENHANDS_AGENT_RL_PLAN.md`（160KB） | stage1 既有 | 接 stage1 训练细节 — Qwen3.6 主计划（并行/MoE/megatron flags/§ 编号被各 commit 引用） |
| `STAGE_INTEGRATION_LOG.md` | stage1 既有 | stage1 整合演进日志（F1-F3 findings 等，F3=本会话提到的 all-drop guard） |
| `STAGE2_FIT_LOOP_AUDIT.md` / `UPSTREAM_DELTA_*.md` | stage1 既有 | fit-loop 审计 / 上游 delta 分析 |

接 OpenHands merge 线 → 读前 3 个。接 stage1 训练线 → 加读 QWEN36 plan + STAGE_INTEGRATION_LOG。

---

## 2. 仓库 / 分支地图

| 本地路径 | 角色 | 当前分支 | 注意 |
|---|---|---|---|
| `/Users/yeji/Documents/Code/Python/Qwen36/rllm` | stage1 原仓（lilac-me/rllm） | `verl-main` | **user 的 working 仓，有 in-progress merge + qwen36 未提交工作 — 别碰** |
| `/Users/yeji/Documents/Code/Python/Qwen36/rllm-merge-tmp` | **merge 工作副本**（cp 自上面） | `verl-main-merge` | 本会话第 2 阶段在这里做；独立 .git |
| `/Users/yeji/Documents/Code/Python/Qwen36/rllm-pqy/rllm-openhands` | pqy 仓 | `debug_oom` | pqy remote，本地 fetch 用 |

**远端分支（origin = git@github.com:lilac-me/rllm.git）**：
- `origin/openhands-refactoring` — 阶段 1 成果，**已 push**（pqy 嫁接 + 解耦 + HTTP worker + KU 文档）
- `origin/verl-main` — stage1 主线（活跃，BryanChen408 等在推进）
- `origin/openhands-observability` — pqy 旧分支，含 ZhihaoSun 的 AgentPPOTrainer padding fix（`f3763717`）
- `verl-main-merge` — **本会话第 2 阶段，仅 local，未 push**

---

## 3. 当前现状

### 阶段 1（openhands-refactoring）✅ 完成 + push
- 9 commit on `origin/openhands-refactoring`，base `5daa24c0`
- 内容：pqy OpenHands pipeline 嫁接 + env config（RLLM_TOPOLOGY）+ HTTP worker
  + NPU-lock 清理 + batch padding fix + KU1-8 文档
- 细节见 `OPENHANDS_REFACTORING_LOG.md`

### 阶段 2（verl-main-merge）✅ 代码合并完成，未 push
- base `1464f0c0`（rebase 跟进过 verl-main 最新）
- 8 commit（4 代码 + 4 docs），代码改动 +669/-19，8 文件
- **静态验证通过**：py_compile / bash -n / HTTP worker payload schema
- 逐文件 reconcile 决策见 `MERGE_TO_VERL_MAIN_PLAN.md` §2 + §7
- **待**：user review → 单 rollout 冒烟（需 NPU+docker+ray）→ push

### 阶段 3（skills 移植）⏳ 未启动
- 等 user 提供 **triton skills 仓路径**（Claude Code 格式 `.claude/skills/`）
- 见 `MERGE_TO_VERL_MAIN_PLAN.md` §4

---

## 4. 场景（部署 + 训练）

**机器拓扑**（两种，config RLLM_TOPOLOGY 切换）：
- single-host：1 机 16 NPU，trainer 用 0-7，OpenHands 容器用 8-15
- multi-host：trainer 机 + 独立 eval 机（OpenHands 容器跑 eval 机 NPU）

**DooD（Docker-out-of-Docker）**：trainer 在 dev container 内，挂 host docker.sock，
`docker run` sibling 容器跑 OpenHands。两套方案：
- **HTTP worker**（目标架构，多机）：trainer HTTP POST workspace tar 给
  `remote_eval_worker.py`，worker 在自己 host 解压 + docker run。路径无关。
- **bind-mount**（verl-main 现有，same-host fallback）：`-v <host_path>:/opt/workspace`
  靠 host bind-mount 路径一致。空 `OPENHANDS_REMOTE_EVAL_URL` 时自动 fallback。

**训练**：OpenHands agent 在容器内生成 NPU 算子（triton 或 AscendC），跑 KernelBench
任务，reward 由算子验证产出（triton: metrics.json / AscendC: trace.md）。
**reward 稀疏**（KernelBench 难，初期大量 reward=0）。

---

## 5. 未来路线与目标

| 优先级 | 任务 | 状态 |
|---|---|---|
| P0 | verl-main-merge：user review + 单 rollout 冒烟 + push | 待 user |
| P1 | **skills 移植阶段**（triton 路线整体换） | 等 triton skills 仓路径 |
| P2 | 长跑训练稳定性 + 算法层（entropy collapse、batch solve_none=1.0） | 后续 |

**skills 移植阶段（P1）必须一起换的绑定项**（都是 triton 路线，单独换会 reward 读不到对的文件）：
1. triton skills（user 的仓）→ `agent_workdir/.agents/skills/`（格式适配 `.claude/skills/`→`.agents/skills/`）
2. `AGENTS.md` / `INSTRUCTIONS.md`：AscendC Phase 流程 → triton 流程
3. `runner.py` 的 `_HARDCODED_SYSTEM_PROMPT`：AscendC → triton（本会话保留了 verl-main 的 AscendC 版）
4. `_npu_operator_reward` + `_impl_newer_than_metrics` + `_snapshot_success_as_best`：
   从 openhands-refactoring port（读 triton metrics.json，含防 stale + best 快照）
5. `operator_pipeline.sh`：AscendC/TileLang 验证 → triton 验证
6. 数据集：NPUKernelBench(AscendC) → triton KernelBench
7. **F3 all-drop guard 重评估**（见 §6 KU 关键项）

---

## 6. 关键 KU / 遗留（索引）

完整在 `OPENHANDS_REFACTORING_LOG.md`，这里列接手必知的：

| KU | 一句话 | 处置 |
|---|---|---|
| **F3 all-drop guard**（plan §7.1） | verl-main `1464f0c0` 删了"整 batch uid 全 drop 时 skip"的保护；其触发场景"agent 没写 impl→is_correct 全 False→整 batch drop"正是 triton 稀疏 reward 初期常态 | skills 阶段换 triton reward 后**必须重评估是否恢复 guard**，否则 empty DataProto crash |
| **batch padding**（KU7） | stepwise_advantage 下 batch_size 动态，verl `protocol.py:815` 要求 `% mini_batch_size==0` | 已修：`agent_sdk_trainer.py` LCM(world_size, ppo_mini×rollout_n) padding（verl-main + 我们都有，字节级一致） |
| **LiteLLM flaky**（KU8） | proxy 启动同步 fetch github model-cost-map，flaky 网络致 30s timeout 间歇失败 | `export LITELLM_LOCAL_MODEL_COST_MAP=True` 跳过 fetch |
| **dockerd Created stuck**（KU2） | 高并发 privileged docker run 致容器卡 Created，`docker rm -f`/`inspect` 都 hang | 重启 dockerd；debug_oom.sh 启动前扫尾。根因未定（独立问题） |
| **DooD path 错配**（KU 本质） | dev container 内 `-v <内部路径>` 提交给 host dockerd 找不到 → mount 空目录 → entrypoint 缺失卡 Created | HTTP worker（路径无关）或 bind-mount 同路径 |

---

## 7. 撤回过的错误判断（**不要重犯**）

本会话纠正过的错，新 agent 别再踩：

- ❌ "HTTP worker 冗余，用 verl-main bind-mount 就行" → **错**。HTTP worker 是
  strictly-more-general 的多机架构，bind-mount 是砍掉多机能力的退化特例。
- ❌ "reward 防 stale 是通用改进，当前阶段并入" → **错**。reward 绑定算子路线
  （triton metrics.json vs AscendC trace.md），当前并会破坏 verl-main pipeline。推迟 skills 阶段。
- ❌ "openhands-refactoring 90% 代码冗余" → **草率**。要逐文件 3-way（base/verl-main/ours）
  分 A/B/C，独有价值（HTTP worker、reward 防 stale、device_ids、config）不能漏。
- ❌ 用"两分支当前 diff 行数"分类 A/B/C → **不准**（把 verl-main 没改的
  distributed_npu_lock 误判 B）。要用"相对 base 的双侧改动量"。
- ❌ "BS=3 那次跑的是 max_split=128"（从日志反推）→ user runtime 改过 2048，ground truth 优先。

---

## 8. 关键源码 reference

| 文件 | 关键点 |
|---|---|
| `examples/openhands_sdk/openhands_agent.py` | `_run_remote_eval_worker`（HTTP client，空 URL fallback bind-mount）；`rollout` 路由；`_npu_operator_reward`（verl-main 当前是 AscendC trace.md 版） |
| `examples/openhands_sdk/remote_eval_worker.py` | HTTP worker server：`/run` + `/health` + `/admin/reset-locks` + 启动 GC |
| `examples/openhands_sdk/config/qwen36.env` | verl-main 部署 config，RLLM_TOPOLOGY single/multi |
| `examples/openhands_sdk/train_openhands_qwen36_npu.sh` | stage1 训练入口，`source config` + 超参 + HCCL override + precheck |
| `rllm/trainer/verl/agent_sdk_trainer.py` | `_pad_dataproto_to_world_size`（LCM padding，~L998）；fit_agent（`1464f0c0` 删了 F3 guard） |
| `rllm/engine/agent_sdk_engine.py` | `transform_results_for_verl`（batch 组装，stepwise_advantage） |
| verl `protocol.py:815` | `batch_size % mini_batch_size == 0` assert（megatron DP 对齐） |

---

## 9. user 风格（务必遵守）

- **数据驱动**，反对拍脑袋
- 要**源码引用**（文件 + line number + 机制）
- 反复让你严谨化（"你确认 X 吗？给证据"）
- 撤回错的不掉价，**继续瞎猜会被骂**
- 破坏性操作（git push/reset/force）+ 改 user working 仓前先确认
- "本地操作完我 review 再 push" — 默认不主动 push，commit 留 local
