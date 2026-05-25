# Qwen3.6-35B-A3B × OpenHands Agentic RL 开发计划

> 生成时间：2026-05-23
> 修订时间：2026-05-25 (v2.2 — 基线切到 `openhands_ascend`；软化 GDN 约束；高风险 cherry-pick 标注；明确算子 reward / 编译工具链不在本 plan 范围)
> 配套分析文档：[UPSTREAM_DELTA_QWEN36_AGENT_TRAINING_ANALYSIS.md](./UPSTREAM_DELTA_QWEN36_AGENT_TRAINING_ANALYSIS.md)

## 范围声明（v2.2）

本 plan **只覆盖 rllm 顶端的 agent 适配 + OpenHands 链路 + 训练验证**。

**明确不在本 plan 范围**（由相邻团队/相邻阶段负责）：

- 算子生成 reward 函数 / 评测协议 → 完全在 OpenHands 执行容器内的脚本里实现；rllm 只通过 trajectory 拿到一个 reward 标量，不感知 reward 计算细节。
- ascendC 编译工具链 / CANN 单算子运行环境 → 是算子编译环境适配工作，与 Qwen3.6 训练正交，由容器/环境侧负责。
- 真实算子数据集来源、规模、切分、许可 → 由数据侧负责，rllm 侧只消费 parquet。
- 依赖仓库（verl / vllm-ascend / Megatron-LM / Megatron-Bridge / MindSpeed）的版本管理 → **当前各仓库 HEAD 即为锁定状态**，本 plan 不引入 detach / pin / bump 任务。
- AgentFlow 迁移、dashboard 在 Qwen3.6 episode 上的呈现、端口预算总表、reasoning 字段端到端 schema、checkpoint 后端拍板、bypass rollout / token-id 调试代码归位 → **统一下放到下一阶段 TODO**（见 §9）。

---

## 0. 实际生产环境（已就位）

| 组件 | 实际版本 / 路径 | 状态 |
|---|---|---|
| 硬件平台 | **Ascend NPU 集群**（NPU `n_devices_per_node=16`） | 已就位 |
| verl | `BryanChen408/verl @ hostoom-debug`（基于 volcengine/verl main） | 已就位，含 PR #6318/#6399/#6323 + 自有 OOM debug |
| **verl 启动脚本（参数权威）** | `verl/examples/grpo_trainer/run_qwen3_5_35b_megatron.sh` | **Qwen3.6 已用此脚本验证** |
| Megatron-Bridge | `NVIDIA-NeMo/Megatron-Bridge @ de93536e`（含 PR #2654 Qwen 3.5 recipes） | 已就位 |
| Megatron-LM | `core_r0.16.0`（NPU 路径用 `0.16.1`） | 已就位 |
| MindSpeed | `0.16.0` | 已就位 |
| vLLM | `vllm-ascend` fork | 已就位 |
| Qwen3.6-35B-A3B 加载 | 走 Megatron-Bridge 路径，**已验证** | ✅ |

**这意味着外部依赖栈和训练参数都已经稳定，本计划不再涉及"栈对齐"和"训练参数从零设计"，重心彻底转移到 rllm 顶端 agent 适配 + OpenHands 链路 + 训练验证。**

### 0.1 关键架构约束

| 约束 | 原因 | 设置 | 强度 |
|---|---|---|---|
| Qwen3.5/3.6 用 Gated Delta Net (GDN) linear attention，Megatron-LM 对 GDN 的 packed sequences (THD) 支持不完整 | 上游 verl 脚本 NPU 分支默认关闭 remove padding | `model.use_remove_padding=False`<br>`actor.megatron.use_remove_padding=False`<br>`actor.use_dynamic_bsz=False` | **可调**：以 W2 实测为准。当前 obs/openhands_ascend 分支训练脚本仍是 `True`，若实测能跑通且性能可接受则保留，否则切回 `False`。**不作为硬性强制项。** |
| CUDA_DEVICE_MAX_CONNECTIONS=1 | Megatron 通信顺序保证 | `export CUDA_DEVICE_MAX_CONNECTIONS=1` | **必须** |
| vLLM V1 engine + 禁用 symm mem allreduce | vllm-ascend 兼容性 | `export VLLM_USE_V1=1`<br>`export VLLM_ALLREDUCE_USE_SYMM_MEM=0` | **必须** |
| DEVICE 自动检测 | `python3 -c 'import torch_npu'` 检测 NPU | 脚本默认行为，无需手工 | 默认 |

> ⚠️ swe 配方 `MODEL_USE_REMOVE_PADDING=true` 与 verl 脚本 NPU 分支默认值相反。两条路径都可能在 NPU 上跑通，本 plan 不预先选边——W2 单节点训练时实测决定。

### 0.2 verl 脚本的并行/训练参数（NPU 路径，参数权威）

直接来自 `run_qwen3_5_35b_megatron.sh` NPU 分支：

```bash
# 并行（NPU 1 节点 16 卡）
TP=2 PP=2 CP=1 EP=8 ETP=1 GEN_TP=8
n_devices_per_node=16

# 数据
data.train_batch_size=32
data.max_prompt_length=1024
data.max_response_length=2048
data.truncation='error'
data.filter_overlong_prompts=True

# Actor
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
actor_rollout_ref.actor.use_dynamic_bsz=False              # GDN 默认值（W2 可调）
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl
actor_rollout_ref.actor.entropy_coeff=0

# Megatron-Bridge 集成（核心：use_mbridge=True + vanilla_mbridge=True）
actor_rollout_ref.actor.megatron.use_mbridge=True
actor_rollout_ref.actor.megatron.vanilla_mbridge=True
actor_rollout_ref.actor.megatron.use_remove_padding=False  # GDN 默认值（W2 可调）
actor_rollout_ref.actor.megatron.dtype=bfloat16
actor_rollout_ref.actor.megatron.param_offload=True
actor_rollout_ref.actor.megatron.optimizer_offload=True
actor_rollout_ref.actor.megatron.grad_offload=True

# MoE 关键配置
++actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True

# 重算（显存压力下必需）
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1

# 优化器 offload（NPU 显存关键）
+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
+actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
+actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True

# Rollout
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}    # 8
actor_rollout_ref.rollout.gpu_memory_utilization=0.6              # 注意：0.6 不是 swe 的 0.4
actor_rollout_ref.rollout.n=5                                     # 注意：5 不是 swe 的 8
actor_rollout_ref.rollout.dtype=bfloat16
```

> **本计划下所有训练脚本以上述参数为基线，OpenHands 接入只改 dataset/agent_run_func 入口，不动并行/MoE/优化器参数。**

---

## 1. 目标与边界

### 1.1 核心目标

**用 Qwen3.6-35B-A3B 模型，在 Ascend NPU 集群上跑 agentic RL 训练算子生成 agent，以 OpenHands 作为 agent 运行时框架。**

### 1.2 优先级

| 优先级 | 范式 | 说明 |
|---|---|---|
| P0 | OpenHands × 算子生成 × Ascend NPU | 团队长期方向，本计划主线 |
| P1 | KernelGym × 算子生成 | 已在 Qwen3-8B/30B 验证有效果，本计划仅做"环境同步" |
| 不做 | AgentFlow 迁移 | 延后到本计划交付之后另立专项 |
| 不做 | NVIDIA GPU 路径 | 平台已确定为 Ascend，不维护 GPU 路径 |
| 不做 | 14 个 legacy example 入口 | math_tool / search / deepscaler 等 |

### 1.3 已完成的工作（不需重做）

- ✅ **verl + Megatron-Bridge + Qwen3.6 加载已验证**（你刚完成）
- ✅ OpenHands 容器已支持 Qwen3-Coder 系列（不是 obs 分支 Dockerfile 锁的 0.28）
- ✅ OpenHands 训练循环在 Qwen3-8B/30B 跑通过（未做效果验证）
- ✅ KernelGym 在其他 Qwen 模型上验证有效果
- ✅ verl fork 含 PR #6318：Ascend 上 Qwen3.5-35B + Megatron-Bridge launch script
- ✅ verl fork 含 #6399：Ascend Docker build guidance
- ✅ verl fork 含 OOM debug（host/NPU memory probes）

---

## 2. 基线决策

### 2.1 选择以 `origin/openhands_ascend` 为基线

**不选 `upstream/main` 的硬理由：**

- 官方 PR #526 **物理删除了整个 `rllm/sdk/` 子系统**（50 个文件）+ `AgentExecutionEngine` + `AgentSdkEngine`
- OpenHands 集成深度依赖 `rllm/sdk/proxy`、`session`、`store`、`tracers`
- 迁移到 upstream/main = 把 OpenHands SDK 路径重写为 AgentFlow + model-gateway，**专项 25–45 天**（难度 4/5）

**为何选 `openhands_ascend` 而不是 `openhands-observability`：**

`openhands_ascend` = `openhands-observability` 基础上叠加 5 个提交（包括 ascendc 算子生成的数据准备、单算子 parquet 读取、SDK runner 更新），与本 plan P0 目标（算子生成 agent）直接对齐。

实际差异（运行 `git rev-list --left-right --count origin/openhands_ascend...origin/openhands-observability` = `5\t4`）：

| ascend 独有（基线已带） | obs 独有（需评估 cherry-pick） |
|---|---|
| `4274204b 适配ascendc算子生成` | `095dd639 add gitignore`（可忽略） |
| `dcc7f204 修改数据读取，直接读取本地提前构造好的rl_single_ops.parquet` | `4268ffdd tmd`（人工评估提交内容） |
| `bb5b5c6a 增加数据梳理脚本，json→parquet` | `17430f3a fix time`（轻量修复） |
| `da38a51f openhands: train scripts, runner, and agent SDK updates` | **`c6e36b0a 修复stepwise下打印不全的问题。修复使用completionids的长度问题`**（值得 cherry-pick） |
| `90a150fc add response length debug` | — |

**W1 必做：** 把 obs 独有的 4 个提交逐个判断是否 cherry-pick；`c6e36b0a` 强烈建议拿，其余按内容定。

**选 ascend 为基线的代价（已知技术债务）：**

- `rllm/sdk/*` 50 文件成为 fork 子系统，上游不再修复
- AgentFlow 迁移笔账以后要还（下放下一阶段，见 §9）

### 2.2 三个知识源的分工

```
origin/openhands_ascend         ── 基线，已带 OpenHands SDK + KernelGym + 观测能力 + ascendc 算子数据准备
origin/openhands-observability  ── 仅作为 4 个独有补丁的来源（其中 c6e36b0a 强烈建议拿）
upstream/main                   ── cherry-pick ~10 个 verl 集成层修复（其中 2 个标 P0★ 高风险）
upstream/swe                    ── 仅作 GPU 路径参考，不照搬（swe 配方是 B200/CUDA 栈）
verl@BryanChen408/hostoom-debug ── Qwen3.5/3.6 × Megatron-Bridge × Ascend 已合入，当前 commit 即锁定版本
Megatron-Bridge main            ── Qwen 3.5 recipe 已合入，当前 commit 即锁定版本
```

### 2.3 工作分支

```bash
git -C rllm checkout -b qwen36-openhands-stage1 origin/openhands_ascend
```

---

## 3. 适配工作清单

### 3.1 从 upstream/main cherry-pick 的关键修复

> 这些修复改的是 **rllm 仓库本身**（`rllm/trainer/verl/*`、`rllm-model-gateway/*`、`rllm/parser/*`），**和 verl 仓库内部状态解耦**，所以理论上仍然适用于你的 verl fork 路径。**但必须逐个 cherry-pick + 验证 import 不挂**，因为 verl fork 已经在 verl 上游 main 之后，可能有 API 漂移。

#### P0 — 多轮训练正确性

| Commit | 说明 | NPU 适配注意 |
|---|---|---|
| `68e9107e` | fix(verl): pad multi-turn batch to lcm of dp_size and ppo mini-batch | 通用，与硬件无关 |
| `45c99f1c` | fix(verl): zero response_mask on padded rows | 通用 |
| `d09f6155` | Fix rollout_probs_diff masking to respect response_mask | 通用 |
| `326445bd` | [feat] Collapse multi-turn trajectory steps into a single row | 通用 |
| `44fb9a3c` | fix(verl): raise on aborted rollouts and normalize stop_reason | 通用 |
| `e8539db5` | don't truncate merged multi-turn responses | 通用 |

#### P0 — Qwen / MoE 支持

| Commit | 说明 | NPU 适配注意 |
|---|---|---|
| `e5ba77ea` | fix(parser): Qwen3.5 chat template support | 通用，**Qwen3.6 复用 Qwen3.5 路径，必拿** |
| **`19983fe4` 🔴 P0★** | feat(verl): support R2 and R3 MoE router replay | **高风险 cherry-pick**：当前 rllm 仓库**没有 `rllm/algorithm/` namespace**，本 commit 实质是新增一整套子系统而非小补丁。冲突面大，单独排期 0.5–1 天，cherry-pick 前先在 verl fork 端 grep 是否已有 router replay hook，避免重复实现。Qwen3.6-A3B 训练在 W2/W3 上是否真用 R3 由实测决定，但 cherry-pick 动作本身仍在 W1 内完成。 |

#### P1 — 稳定性

| Commit | 说明 | NPU 适配注意 |
|---|---|---|
| `d314745e` | fix(verl): release GPU memory on Verl backend interrupts | **要看代码是否 NPU 兼容**（可能 hardcode `torch.cuda.empty_cache`） |
| **`e66fc2e4` 🔴 P0★** | fix(verl): route rllm.algorithm.lr_schedule to active backend | **高风险 cherry-pick**：同样依赖 `rllm/algorithm/` namespace。若 `19983fe4` 已经把 namespace 引入，本 commit 是小改；若先于 `19983fe4` 单独拿则会断链。**必须排在 `19983fe4` 之后**。 |

#### 不要拿

| Commit | 原因 |
|---|---|
| `88f409c4` | "Fix training collapse for AgentFlow with verl backend" — 绑死新 AgentFlow |
| `6d5c2c59` | PR #526 删除 SDK 路径，与基线冲突 |

### 3.2 训练脚本参数权威：**verl 脚本 > obs NPU 脚本 > swe 配方**

权威顺序：

1. **verl/examples/grpo_trainer/run_qwen3_5_35b_megatron.sh**（你已验证）—— 并行、MoE、优化器、Megatron-Bridge 参数全部以此为准（见 §0.2）
2. **obs 分支 examples/openhands_sdk/train_open_megatron.sh + examples/swe/train_deepswe_30b_npu.sh** —— OpenHands 入口结构、NPU 集群启动方式
3. **upstream/swe 配方** —— 仅作 GPU 路径对照，**不照搬**（B200/CUDA 栈，与生产环境不同）

**整合方式：**

```
新训练脚本 = verl 脚本的 actor/megatron/rollout 参数
           + obs train_open_megatron.sh 的 OpenHands agent_run_func 接入
           + 算子生成 dataset 替换 geo3k
```

swe 配方里和 verl 脚本不一致的参数（如 `use_remove_padding=true`、`gpu_memory_utilization=0.4`、`rollout.n=8`），**默认以 verl 脚本为准，但 use_remove_padding 在 W2 实测后可调**（见 §0.1）。

### 3.3 swe 的 4 个 verl/rllm patch 取舍

| Patch | 是否需要 | 备注 |
|---|---|---|
| `rllm/experimental/verl/patch.py` RLLM_VLLM_PORT_BASE | **要**（rllm 这一层，与硬件无关） | 多 vllm-ascend 实例同时启动会撞端口 |
| `rllm/experimental/verl/utils.py` HDFS checkpoint | **看情况**（NPU 集群存储是 HDFS 还是 OBS/本地） | Ascend 集群常用 OBS |
| `rllm/trainer/verl/ray_runtime_env.py` runtime env 传播 | **要**（rllm 这一层） | 需追加 HCCL/CANN 相关 env |
| `rllm-model-gateway/.../server.py` token 透出 | **要**（rllm 这一层） | gateway 改动 |

⚠️ verl fork PR #6318 可能已经在 verl 这一层做了类似 patch（特别是端口、runtime env）。**cherry-pick 前先在 verl fork 里 `git log` 搜一下相关关键词**，避免双重 patch 互相覆盖。

### 3.4 Qwen3.6 适配清单（rllm 侧）

#### ✅ 已完成（不需重做）

- ✅ Megatron-Bridge HF→Megatron Qwen3.6 转换（PR #6318 + #2654 + 你的验证）
- ✅ Qwen3.6 模型加载到 NPU
- ✅ verl + Megatron-Bridge 集成路径
- ✅ 训练并行/MoE/优化器参数已固化（见 §0.2，verl 脚本权威）
- ✅ Qwen3.6 architecture 约束已确定（GDN 不支持 THD，见 §0.1）

#### 第一版（关 thinking，最小可跑路径）

- [ ] `rllm/parser/chat_template_parser.py` 验证 Qwen3.5 parser 兼容 Qwen3.6 chat template（cherry-pick `e5ba77ea` 后）
- [ ] vllm-ascend 启动参数加 `--tool-call-parser qwen3_coder`（确认 vllm-ascend 是否支持该 parser）
- [ ] vllm-ascend 启动加 `--reasoning-parser`（即使关 thinking 也保留接口）
- [ ] MoE router replay R3 行为验证（依赖 P0★ cherry-pick 完成且不破坏 import）
- [ ] `use_remove_padding` W2 实测：先按现有训练脚本（`True`）跑，遇 OOM 或挂再切 `False`，结果落到 RUNBOOK
- [ ] 训练 metrics 上报：reasoning_token_ratio = 0（验证关 thinking 生效）

#### 第二版（开 thinking，增量）

- [ ] reasoning content 在 rollout 输出里的处理策略实现
- [ ] response_mask 在 reasoning content 上的策略（**算法决策，见 §5**）
- [ ] multi-turn merge 时 reasoning 保留策略（建议初版只留最近 1 轮）
- [ ] 训练 metrics 加 reasoning_token_ratio 监控

### 3.5 Qwen3.6 适配清单（OpenHands 侧）

> 假设你已有的 OpenHands × Qwen3-Coder 环境是基线，**只列增量**。

#### 第一版（关 thinking）

- [ ] OpenHands LLM config 改为 Qwen3.6 model name + base URL（指向 vllm-ascend）
- [ ] 调用时显式设 `enable_thinking=False`
- [ ] 多模态显式禁用（如果不用 vision）
- [ ] A3B tool format post-processing 复用现有实现（Qwen3-Coder-30B-A3B 同款问题）
- [ ] system prompt 验证（Qwen3.6 SWE 能力强，可能可简化原 Coder prompt）
- [ ] 单 turn / 多 turn tool call smoke

#### 第二版（开 thinking）

- [ ] conversation history 中 reasoning_content 字段处理
- [ ] reasoning 计入 context budget 的核算
- [ ] 长 episode（≥20 turn）reasoning 累积导致 context 爆掉的截断策略
- [ ] thinking 是否让 OpenHands "看到" reasoning（影响 agent 决策路径）

### 3.6 两仓库对接（链路一致性）

```
OpenHands 容器内 messages
    ↓ messages 是否带 reasoning_content
litellm proxy
    ↓ 是否透传 reasoning
vllm-ascend + reasoning-parser
    ↓ 返回 reasoning_content / content / tool_calls 分开
回程
    ↓ OpenHands 收到哪些字段，怎么进 conversation
    ↓ rllm 训练侧从 trace 重建 prompt_ids / response_ids
```

- [ ] 端到端 reasoning 字段在 5 个节点的一致性矩阵
- [ ] prompt_ids / response_ids 重建正确性（vs litellm proxy 录的 token）
- [ ] context budget 实测：典型 OpenHands episode 长度分布
- [ ] reward / loss 在 reasoning 处理策略变化下的曲线对比

### 3.7 Ascend NPU 特定的适配项

#### 网络与通信

- [ ] HCCL 环境变量调优（HCCL_BUFFSIZE、HCCL_EXEC_TIMEOUT 等）
- [ ] GLOO_SOCKET_IFNAME / TP_SOCKET_IFNAME 显式设置
- [ ] NPU 拓扑与 TP/CP/EP 映射检查（同一 NPU 服务器内的 8 卡 affinity）

#### 内存

- [ ] 复用 verl fork OOM debug（host/NPU memory probes）
- [ ] vllm-ascend 的 npu_memory_utilization 调优（vs swe 配方 GPU 的 0.4）
- [ ] Megatron offload 参数适配 NPU 内存特性
- [ ] OpenHands 多并发容器在 NPU 服务器上的 host 内存压力

#### 存储与 checkpoint

- [ ] 选定 checkpoint 后端（HDFS / OBS / 本地共享 fs）
- [ ] use_dist_checkpointing=true 在 NPU 集群的兼容性
- [ ] 模型 cache（HuggingFace hub）路径在 NPU 节点的共享方式

#### 工具链

- [ ] CANN 版本与 Megatron-Bridge / verl 的兼容矩阵记录
- [ ] MindSpeed 是否使用（如果用，集成在哪一层）

---

## 4. 阶段化执行计划

### 4.1 总览

> 由于栈已就位 + Megatron-Bridge × Qwen3.6 已验证，原 W1 大幅压缩。

| 阶段 | 内容 | 工时 | 交付物 |
|---|---|---|---|
| W1 | rllm 分支建立 + cherry-pick + Ascend smoke | **2–3 天** | 分支带齐修复，Qwen3.6 在 vllm-ascend 上推理 smoke 通过 |
| W2 | OpenHands × Qwen3.6（关 thinking）单节点训练 | 5–7 天 | 1 节点 NPU 一步训练 + checkpoint |
| W3 | 多节点 + 长跑 + 评估 | 5–8 天 | 多节点 NPU 训练 + pass@k + checkpoint resume |
| W4（可选） | 第二版：开 thinking | 5–10 天 | thinking 模式训练 + reasoning 处理策略验证 |
| W5（可选） | KernelGym 同步 + AgentFlow 迁移评估 | 3–5 天 | KernelGym 在 Ascend 上 smoke + AgentFlow 立项 |

**P0 交付时间：~2.5 周（W1–W3）** — 比原计划压缩 0.5–1 周，因为栈已就位。
**P0 + 第二版：~3.5 周**

### 4.2 Week 1：分支建立 + cherry-pick + smoke（2–3 天）

**目标：** rllm 分支带齐 cherry-pick + swe patch 残留项 + Ascend 推理 smoke。

**任务：**

```
1.1  建工作分支
     git checkout -b qwen36-openhands-stage1 origin/openhands_ascend

1.2  评估并 cherry-pick obs 独有 4 个 commit (§2.1)
     - c6e36b0a  修复stepwise下打印不全 + completionids 长度问题  → 强烈建议拿
     - 17430f3a  fix time                                          → 看内容
     - 4268ffdd  tmd                                               → 看内容
     - 095dd639  add gitignore                                     → 可忽略

1.3  Cherry-pick upstream/main 的 commit (§3.1)
     ★ 顺序约束：先拿 P0 通用 6 个，再拿 e5ba77ea (parser)，
       最后拿 P0★ 两个高风险（19983fe4 → e66fc2e4），
       因为 e66fc2e4 依赖 19983fe4 引入的 rllm/algorithm namespace
     ★ 逐个 cherry-pick，每次完成后 python -c "import rllm" 验证
     ★ 冲突常见点：rllm/trainer/verl/* 因 verl API 漂移；
       19983fe4 几乎必然冲突（新增子系统）— 单独排 0.5–1 天

1.4  swe 的 4 个 rllm 层 patch (§3.3)
     ★ 先在 verl fork 里 grep 是否已有相同补丁，避免双重
     git diff upstream/main upstream/swe -- rllm/experimental/verl/patch.py
     git diff upstream/main upstream/swe -- rllm/experimental/verl/utils.py
     git diff upstream/main upstream/swe -- rllm/trainer/verl/ray_runtime_env.py
     git diff upstream/main upstream/swe -- rllm-model-gateway/

1.5  Qwen3.6 在 vllm-ascend 上推理 smoke（不带 rllm、不带 OpenHands）
     - 单请求 chat completion（关 thinking）
     - 单请求 tool call（qwen3_coder parser）
     - 多轮 tool call
     - 32k context 长请求

1.6  rllm 分支 import smoke
     - python -c "import rllm; import rllm.sdk"
     - 按 ls rllm/engine/ 实际结果补充对应 import
     - 关键模块加载正常
```

**验收标准：**

- [ ] obs 4 commit 评估结论落到 commit message 或 RUNBOOK
- [ ] upstream cherry-pick 全部应用，每个之后 import 不挂；P0★ 两项顺序正确
- [ ] vllm-ascend 服务 Qwen3.6 单请求/tool call/多轮全通
- [ ] rllm 分支无 import 错误

### 4.3 Week 2：OpenHands × Qwen3.6 单节点训练（5–7 天）

**目标：** 1 节点 NPU 上跑通"OpenHands rollout + 一步训练 + checkpoint"。

**任务：**

```
2.1  OpenHands 配 Qwen3.6
     - LLM base URL 指 vllm-ascend
     - enable_thinking=False
     - 复用现有 Qwen3-Coder 环境 80%

2.2  rllm 训练脚本基于 verl 脚本 + obs OpenHands 入口合成
     ★ 参数权威：verl/examples/grpo_trainer/run_qwen3_5_35b_megatron.sh（NPU 分支）
     ★ Agent 入口：examples/openhands_sdk/train_open_megatron.sh
     - 把 verl 脚本里 actor/megatron/rollout/optim 整段照搬（见 §0.2）
     - 数据 source 从 geo3k 换算子生成数据集
     - 模型路径换 Qwen3.6-35B-A3B
     - 加 OpenHands agent_run_func 接入（AgentTrainer 入口）
     - 加环境变量 CUDA_DEVICE_MAX_CONNECTIONS / VLLM_USE_V1 / VLLM_ALLREDUCE_USE_SYMM_MEM
     - HCCL env 透传到 ray runtime env

2.3  litellm proxy 链路验证
     - proxy 配 vllm-ascend backend
     - metadata slug 路由单容器单 session
     - SQLiteTraceStore 落盘正确

2.4  observer_gateway 边界画死（§5.4）
     - 只记 OpenHands runtime 事件
     - 不碰 token_ids / logprobs / 模型调用

2.5  1 节点单 episode rollout
     - 1 个 OpenHands 容器
     - 1 个算子生成任务
     - 端到端跑出 Trajectory + reward

2.6  1 节点一步训练
     - Megatron-Bridge 加载 Qwen3.6
     - 跑 1 个 ppo iter
     - checkpoint save 成功

2.7  checkpoint resume 验证
     - 从 step 1 加载继续训
     - 数值正确性 spot check
```

**验收标准：**

- [ ] 1 节点单 episode 完整跑完，reward 非 NaN
- [ ] Megatron-Bridge 加载 Qwen3.6 无 dtype/shape mismatch
- [ ] 1 步训练 loss 数值合理（量级与 Qwen3-30B 训练参考值同档）
- [ ] checkpoint save/load 数值一致
- [ ] observer 日志能追溯 OpenHands runtime 事件
- [ ] model gateway / litellm proxy 日志能追溯 token

### 4.4 Week 3：多节点 + 长跑 + 评估（5–8 天）

**目标：** 多节点 NPU 多 episode 并发训练，跑通完整训练循环。

**任务：**

```
3.1  多节点 Ray + vllm-ascend
     - 应用 RLLM_VLLM_PORT_BASE / RLLM_RAY_MASTER_PORT_RANGE patch
     - 多节点 Ray cluster 起来
     - vllm-ascend async server 端口不撞
     - HCCL 通信连通

3.2  多并发 OpenHands 容器
     - 多 episode 并发
     - litellm proxy + metadata slug 路由在并发下正确性
     - Docker daemon + host 内存够用

3.3  N 步训练（建议 N=100）
     - reward 曲线
     - rollout_probs_diff
     - response_mask 正确性
     - NPU 显存稳定
     - 复用 verl fork OOM debug 监控

3.4  pass@k 评估
     - 复用 obs 上 scripts/eval_pass_at_k.py
     - 在 holdout 数据集上跑

3.5  长跑恢复
     - 训练中途模拟节点失败
     - checkpoint resume 后继续
     - 训练曲线连续
```

**验收标准：**

- [ ] 多节点 NPU 稳定跑 ≥ 100 步，无 OOM / 端口冲突 / HCCL 错乱
- [ ] reward 曲线呈上升趋势
- [ ] rollout_probs_diff < 0.1 量级
- [ ] pass@k pipeline 跑通
- [ ] 中断恢复曲线衔接

### 4.5 Week 4（可选）：开 thinking 第二版

**核心算法决策：** 在小规模实验上对比三种策略：

| 策略 | reasoning 在 response_mask 里 | 多轮保留 |
|---|---|---|
| A | 计入（算 loss） | 保留 1 轮 |
| B | mask 掉（不算 loss） | 保留 1 轮 |
| C | 计入 | 保留全部 |

**任务：**

```
4.1  vllm-ascend 启用 reasoning-parser
4.2  rllm 侧 reasoning 处理策略实现（A/B/C 三种）
4.3  OpenHands 侧 reasoning 在 conversation 中的策略
4.4  小规模（1 节点）对比实验：A vs B vs C
4.5  选定策略后扩到多节点
4.6  max_model_len 从 32k 扩到 64k
4.7  NPU 显存利用率重新调优
```

**验收标准：**

- [ ] 三种策略至少 50 步训练曲线对比
- [ ] 选定策略在多节点稳定
- [ ] pass@k 相对关 thinking 第一版有提升（否则回退）

### 4.6 Week 5（可选）：KernelGym 同步 + AgentFlow 立项

```
5.1  KernelGym 在 Ascend × Qwen3.6 上 smoke（不调优）
5.2  AgentFlow 迁移立项 SOW，准备 Q2 启动
```

---

## 5. 决策点（开发前必须对齐）

> 已根据生产环境信息删除部分原决策（栈选择、Megatron 版本、vLLM 版本等已定）。

### 5.1 第一版是否关 thinking

**推荐：关。**

| | 关 thinking | 开 thinking |
|---|---|---|
| 工时 | W1–W3 ≈ 2.5 周 | W1–W4 ≈ 4 周 |
| 算法风险 | 低 | 高（reasoning loss 决策） |
| context 风险 | 低（32k 够） | 高（可能要 64k+） |

### 5.2 max_model_len 第一版定多少

**推荐：32768。**

W3 长跑监控 episode 长度分布，P95 接近 32k 再切 64k。

### 5.3 observer_gateway 责任边界

**必须在 W2 之前画死：**

| 责任 | 归属 |
|---|---|
| 模型调用 token_ids / logprobs / messages | `rllm/sdk/proxy/litellm_server.py` |
| OpenHands runtime 事件（shell、文件、tool、env state） | `examples/openhands_sdk/observer_gateway.py` |
| episode 级 trace 聚合 | observer 引用 model 侧 trace_id，不复制内容 |

**违反这条边界 = AgentFlow 未来迁移成本翻倍。**

### 5.4 checkpoint 存储后端

**需要团队拍板：HDFS / OBS / 本地共享 fs？**

Ascend 集群常用 OBS（华为云对象存储）。swe 的 HDFS patch 是否照搬取决于此。

### 5.5 14 个 legacy example 怎么办

**推荐：本计划期内不动，README 标 deprecated。**

---

## 6. 风险清单（按可能性排序）

> 已根据生产环境删除原 Megatron 转换、栈对齐等已解除风险。

| 优先级 | 风险 | 触发条件 | 缓解 |
|---|---|---|---|
| ★★★★★ | reasoning content 训练侧语义决策错误 | W4 开 thinking | W4 必做 A/B/C 对比实验 |
| ★★★★★ | **OpenHands 长 episode 与 GDN 不支持 THD 冲突**（bshd 模式下显存暴涨） | W3 长 episode 训练 | 严格 use_dynamic_bsz=False + 重算策略 + offload 全开 |
| ★★★★ | 长 episode + thinking 导致 context 爆 | W3/W4 | max_model_len 第一版 32k |
| ★★★★ | A3B 小 MoE tool format 问题 | W2 单 episode | 复用 Qwen3-Coder-30B-A3B 现有 post-processing |
| ★★★★ | **rllm cherry-pick 在 NPU 路径 import 不通** | W1 cherry-pick | 逐个 cherry-pick + verify import，遇阻立即评估 |
| ★★★★ | **HCCL 通信在多 vllm-ascend + Ray 多节点撞配置** | W3 多节点 | 严格应用端口 patch + HCCL env 显式设置 |
| ★★★ | NPU 显存在 Megatron + vllm-ascend colocated 模式下不够 | W2/W3 | 复用 verl fork OOM debug，offload 参数调优 |
| ★★★ | vllm-ascend qwen3_coder parser 支持不完整 | W1 smoke | W1 必须独立验证 |
| ★★★ | litellm proxy 在多并发下成为瓶颈 | W3 多节点 | 压测后决定是否扩 proxy 实例 |
| ★★★ | swe 的 4 个 rllm 层 patch 与 verl fork 内现有补丁重复或冲突 | W1 cherry-pick | grep verl fork 后再合 |
| ★★★ | **CUDA_DEVICE_MAX_CONNECTIONS 等环境变量在 ray runtime env 里没传播** | W3 多节点 | ray_runtime_env.py 显式注入 |
| ★★ | observer_gateway 边界画错连带 dashboard 重做 | W2 边界决策 | §5.3 必须落地 |
| ★★ | OpenHands 容器在 NPU 服务器 Docker daemon 资源不足 | W3 多并发 | 多节点每节点限并发实例数 |

---

## 7. 文件改动清单

### 7.1 需要新增的文件

```
examples/openhands_sdk/train_openhands_qwen36_npu.py     # 基于 train_open_megatron.py + train_deepswe_30b_npu.sh
examples/openhands_sdk/train_openhands_qwen36_npu.sh     # 启动脚本
examples/openhands_sdk/configs/qwen36_actor_npu.yaml     # actor + rollout megatron-bridge 配置
examples/openhands_sdk/configs/qwen36_no_thinking.yaml   # 第一版
examples/openhands_sdk/configs/qwen36_thinking.yaml      # 第二版
QWEN36_RUNBOOK.md                                         # W3 完成后写
```

### 7.2 需要修改的文件

```
rllm/parser/chat_template_parser.py             # cherry-pick Qwen3.5 兼容
rllm/experimental/verl/patch.py                 # 应用 swe 端口分配 patch（如 verl fork 未有）
rllm/experimental/verl/utils.py                 # 应用 swe runtime/checkpoint patch（按需）
rllm/trainer/verl/ray_runtime_env.py            # 应用 swe runtime env 传播 + HCCL/CANN env
rllm-model-gateway/src/.../server.py            # 应用 swe token 透出
examples/openhands_sdk/observer_gateway.py      # §5.3 边界画清
pyproject.toml                                  # 锁定（或不锁定）verl 来源、megatron-bridge 等
```

### 7.3 不要改的文件

```
rllm/sdk/**                                     # 整个 SDK 保留，承认 fork 债务
rllm/engine/agent_sdk_engine.py                 # 同上
rllm/trainer/agent_trainer.py                   # 训练入口不动
examples/{swe,search,math_tool,deepcoder,deepscaler,frozenlake}/*   # legacy 不维护
```

### 7.4 直接参考但不修改的文件

```
verl/examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
                                                # ★ 训练参数权威，NPU 分支照搬到新脚本
examples/swe/train_deepswe_30b_npu.sh           # NPU 集群启动方式参考
examples/openhands_sdk/train_open_megatron.sh   # OpenHands agent_run_func 接入方式
upstream/swe:cookbooks/swe/.../run_swe_training_35b_a3b_megatron.sh
                                                # GPU 路径，仅作对照（参数冲突时不采纳）
```

---

## 8. 验收基线

### 8.1 P0 交付（W1–W3 结束）

- [ ] 多节点 NPU 稳定跑 OpenHands × Qwen3.6-35B-A3B 训练 ≥ 100 步
- [ ] reward 曲线呈上升趋势
- [ ] checkpoint save/load 数值一致
- [ ] pass@k 评估 pipeline 跑通
- [ ] 训练日志可按 train_id / eval_tag 聚合
- [ ] 失败 episode 可追溯到 OpenHands event + model trace

### 8.2 第二版交付（W4 结束，可选）

- [ ] 开 thinking 模式训练稳定
- [ ] reasoning 处理策略经 A/B/C 对比选定最优
- [ ] pass@k 相对第一版有正向提升

---

## 9. 长期演化路径（非本计划范围）

### 下一阶段 TODO（从 stage1 下放）

stage1 不做、留到 stage2+ 处理的功能项：

- **真实算子数据集**：来源 / 规模 / 切分 / 许可 / 与 mock parquet schema 的兼容
- **OpenHands × Qwen3.6 reasoning/tool 字段端到端 schema**（5 节点字段一致性矩阵）
- **checkpoint 后端拍板**：HDFS / OBS / 本地共享 fs；swe HDFS patch 是否吸收
- **dashboard 在 Qwen3.6 episode 上的呈现**：trace_id / train_id 接线、episode 重放
- **节点端口预算总表**：HCCL_HOST_SOCKET_PORT_RANGE / HCCL_NPU_SOCKET_PORT_RANGE / RLLM_VLLM_PORT_BASE / RLLM_RAY_MASTER_PORT_RANGE / Ray master / NPU runtime 内置端口的全局对账
- **bypass rollout / token-id 调试代码归位**：HEAD 私有提交与上游 `d09f6155` 等 patch 的去重
- **legacy example 标 deprecated**：14 个 example 入口具体到 PR/commit 动作
- **GDN bshd 模式下 multi-turn merge / response_mask 的可视化验收**（当 §0.1 实测决定走 `use_remove_padding=False` 时）

### Q2（本计划交付后 1–2 个月）

- 观察 upstream AgentFlow 对 external runtime / slug routing 的演进
- 评估 community "OpenHands × AgentFlow" 参考实现
- 准备 AgentFlow 迁移专项 SOW
- 评估是否把 NPU 适配 PR 给 verl 上游

### Q3

- 启动 AgentFlow 迁移专项（独立分支，25–45 天）
- `rllm/sdk/*` 子系统逐步弱化
- NPU 适配考虑 PR 回 upstream verl/rllm

### Q4

- 基线回归 upstream/main + OpenHands × AgentFlow component
- 长期债务清零

---

## 10. 不做什么（明确边界）

| | 不做的事 | 理由 |
|---|---|---|
| 1 | 不以 upstream/main 为基线 | OpenHands 依赖已删除的 rllm/sdk |
| 2 | 不迁 AgentFlow | 难度 4/5，独立专项 |
| 3 | 不维护 NVIDIA GPU 路径 | 平台定为 Ascend NPU |
| 4 | 不照搬 swe 配方 | swe 是 B200/CUDA 路径，与生产栈不同 |
| 5 | 不升级 OpenHands 容器到 0.54+ | 当前版本已支持 Qwen3-Coder |
| 6 | 不维护 14 个 legacy example | math_tool 等与目标无关 |
| 7 | 不做多模态 vision | Qwen3.6 有 vision encoder 但算子生成纯文本 |
| 8 | 第一版不开 thinking | 控制变量 |
| 9 | 不动 rllm/sdk/* 内部实现 | fork 债务后续清理 |
| 10 | 不升级 Megatron-LM 到 0.17 | 0.16 已验证可用 |
| 11 | 不切换到 mbridge | Megatron-Bridge 已验证 |

---

## 11. 关键命令速查

### 11.1 建分支与 cherry-pick

```bash
cd /Users/yeji/Documents/Code/Python/Qwen36/rllm

# 建工作分支（基线：openhands_ascend，已带 ascendc 数据准备）
git checkout -b qwen36-openhands-stage1 origin/openhands_ascend

# 先评估 obs 独有的 4 个 commit 是否 cherry-pick（c6e36b0a 强烈建议拿）
git log --oneline origin/openhands_ascend..origin/openhands-observability
git show c6e36b0a   # 修复 stepwise 打印 + completionids 长度
# 决策后单独 cherry-pick

# 顺序：先 6 个通用 P0 + parser，再 P0★ 两个高风险（19983fe4 必须先于 e66fc2e4）
SAFE_PICKS="68e9107e 45c99f1c d09f6155 326445bd 44fb9a3c e8539db5 e5ba77ea"
HIGH_RISK="19983fe4 d314745e e66fc2e4"

for c in $SAFE_PICKS; do
    echo "=== Cherry-pick $c ==="
    git cherry-pick $c || { echo "!!! Conflict on $c"; break; }
    python -c "import rllm" && echo "import OK" || { echo "!!! import broken after $c"; break; }
done

# P0★ 高风险逐个手动 cherry-pick + 测试，预计 0.5–1 天
# 19983fe4 引入 rllm/algorithm namespace（新子系统），冲突面大
# e66fc2e4 依赖上一个
for c in $HIGH_RISK; do
    echo "=== Cherry-pick HIGH-RISK $c ==="
    git cherry-pick $c
    # 不自动 break，预期需要人工解决
    python -c "import rllm" || echo "!!! review needed for $c"
done

# 检查 swe rllm 层 patch 是否需要（先看 verl fork 是否已有）
git -C ../verl-BryanChen408 log --oneline --all | grep -iE "port|hccl|runtime_env" | head -20
git diff upstream/main upstream/swe -- rllm/experimental/verl/patch.py
git diff upstream/main upstream/swe -- rllm/trainer/verl/ray_runtime_env.py
```

### 11.2 vllm-ascend Qwen3.6 smoke

```bash
# 服务端（命令以 vllm-ascend 实际 API 为准，下面是模板）
python -m vllm_ascend.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-35B-A3B \
    --tensor-parallel-size 2 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --max-model-len 32768

# 客户端 smoke：关 thinking + tool call
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3.6-35B-A3B",
        "messages": [{"role": "user", "content": "List files in /tmp"}],
        "tools": [{"type": "function", "function": {"name": "list_files", "parameters": {"type":"object","properties":{"path":{"type":"string"}}}}}],
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

### 11.3 NPU 资源 + verl 脚本对齐的环境变量

```bash
# NPU 状态
npu-smi info

# 必须的（来自 verl 脚本）
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# HCCL 环境（建议在训练脚本里显式设）
export HCCL_BUFFSIZE=200
export HCCL_EXEC_TIMEOUT=1800
export HCCL_CONNECT_TIMEOUT=1200

# rllm 端口
export RLLM_VLLM_PORT_BASE=46000
export RLLM_VLLM_PORT_STRIDE=100
export RLLM_RAY_MASTER_PORT_RANGE=25000:25100

# verl 脚本默认 DEVICE 自动检测
# DEVICE=$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)
```

### 11.4 复用 verl 脚本时的参数提取

```bash
# 提取 verl 脚本里的 ACTOR / ROLLOUT / DATA 数组（NPU 分支）
sed -n '/^ACTOR=(/,/^)/p' /Users/yeji/Documents/Code/Python/Qwen36/verl/examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
sed -n '/^ROLLOUT=(/,/^)/p' /Users/yeji/Documents/Code/Python/Qwen36/verl/examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
sed -n '/^DATA=(/,/^)/p' /Users/yeji/Documents/Code/Python/Qwen36/verl/examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
```

### 11.5 验证当前位置

```bash
# 看分叉（基线 = openhands_ascend）
git rev-list --left-right --count origin/openhands_ascend...upstream/main
git rev-list --left-right --count origin/openhands_ascend...origin/openhands-observability
git rev-list --left-right --count upstream/main...upstream/swe

# 看 cherry-pick 后状态
git log --oneline qwen36-openhands-stage1 ^origin/openhands_ascend

# 看依赖仓库当前 commit（这些就是锁定版本，不主动 bump）
git -C ../verl-BryanChen408 log --oneline -3
git -C ../verl log --oneline -3
git -C ../Megatron-Bridge log --oneline -3
git -C ../Megatron-LM log --oneline -3
git -C ../MindSpeed log --oneline -3
git -C ../vllm-ascend log --oneline -3
```

---

## 12. 修订记录

| 日期 | 版本 | 说明 |
|---|---|---|
| 2026-05-23 | v1.0 | 初版，基于多轮分析与仓库实证形成 |
| 2026-05-23 | v2.0 | **重大修订**：基于生产环境实证（Ascend NPU + verl BryanChen408 fork + Megatron-Bridge + vllm-ascend 已就位且 Qwen3.6 已验证）重写。删除栈对齐、Megatron 转换验证等已完成项；W1 从 5–7 天压缩到 2–3 天；P0 交付从 3 周压缩到 ~2.5 周；明确 swe 配方仅作参考、不照搬；增加 Ascend 特定适配章节 §3.7。 |
| 2026-05-23 | v2.1 | 把 `verl/examples/grpo_trainer/run_qwen3_5_35b_megatron.sh` 提升为**训练参数权威**。新增 §0.1 关键架构约束（GDN 不支持 THD、`use_remove_padding=False`、CUDA_DEVICE_MAX_CONNECTIONS=1 等）+ §0.2 verl 脚本固化参数清单。§3.2 重写权威顺序（verl 脚本 > obs NPU 脚本 > swe）。§3.4 标注训练参数已固化。§6 风险加 GDN/THD 与显存交互、ray runtime env 环境变量传播。§11.3 加 verl 脚本对齐的 env，§11.4 加参数提取命令。明确"rllm 顶端 agent 适配"是真正的工作量，verl 后端已稳定。 |
| 2026-05-25 | v2.2 | **基线切到 `origin/openhands_ascend`**（已带 ascendc 数据准备 + SDK runner 更新），新分支名 `qwen36-openhands-stage1`；§2.1 补 obs 4 个独有 commit 的 cherry-pick 评估表，`c6e36b0a` 强烈建议拿。新增**范围声明**：算子 reward / 评测协议、ascendC 工具链、数据集来源、依赖版本管理均**不在本 plan 范围**（reward 在 OpenHands 容器内自洽，rllm 只拿标量；依赖以"当前 commit 即锁定版本"为准）。§0.1 把 GDN/`use_remove_padding=False` 从硬约束**软化为可调**，W2 实测决定。§3.1 把 `19983fe4`（router replay，新增 `rllm/algorithm` 子系统）与 `e66fc2e4`（lr_schedule，依赖前者）标 **P0★ 高风险 cherry-pick**，明确顺序约束。§4.2 W1 任务相应重排。§11.1 cherry-pick 脚本拆分 SAFE / HIGH_RISK 两段。§9 新增"下一阶段 TODO"清单，把 P1/P2 候选项（真实数据集、reasoning schema、checkpoint 后端、dashboard、端口预算、bypass rollout 归位、legacy example deprecate、GDN bshd 可视化）统一下放。|
