# Qwen3.6-35B-A3B × OpenHands Agentic RL 开发计划

> 生成时间：2026-05-23
> 修订时间：2026-05-25 (v2.10 — batch sanity 联立约束：train_batch_size/ppo_mini_batch_size/rollout.n/DP 联动检查 + fail-fast；新增 §13.14)
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

# Bridge 集成
#
# verl 提供两条 bridge 路径（verl/workers/engine/megatron/transformer_impl.py:173+）：
#   vanilla_mbridge=True  → pypi `mbridge` 库 (`from mbridge import AutoBridge`)
#   vanilla_mbridge=False → NVIDIA Megatron-Bridge (`from megatron.bridge import AutoBridge`)
#
# **NPU 必须 vanilla_mbridge=False**（与 verl 脚本 NPU case override 一致）。
# 原因：pypi mbridge 0.15.1 不注册 qwen3_5_moe model_type，会 raise
#       "Unregistered model type: qwen3_5_moe"；NVIDIA Megatron-Bridge 已有 PR #2654
#       Qwen3.5 recipe。
#
# 注意：plan v2.4 以前的版本（本节早期）错抄了 verl 脚本"主行"的 True，遗漏 NPU
#       case override。W2.9 (v2.8) 修正。
actor_rollout_ref.actor.megatron.use_mbridge=True
actor_rollout_ref.actor.megatron.vanilla_mbridge=False    # NPU 必须 False
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

| Commit | 说明 | 实施后注释 |
|---|---|---|
| `e5ba77ea` | fix(parser): Qwen3.5 chat template support | 已 cherry-pick；Qwen3.6 复用 Qwen3.5 chat template，必拿 |
| **`19983fe4` 🔴 P0★** | feat(verl): support R2 and R3 MoE router replay | 已 cherry-pick，但**实际只在 `VerlBackend` / `UnifiedTrainer` 路径上生效**。stage1 走 `AgentPPOTrainer`（`examples/openhands_sdk/train_open_megatron.py` → `AgentTrainer` → `train_agent_ppo.TaskRunner` → `AgentPPOTrainer`），19983fe4 改动**完全没碰** `rllm/trainer/verl/agent_ppo_trainer.py`。详见 §13.3 "R2/R3 在 stage1 路径上的真实生效情况"。R2 actor 端 50 行也因依赖未引入的 abstraction (`tu`/`batch_td`/`no_padding_2_padding`) 被跳过。 |

#### P1 — 稳定性

| Commit | 说明 | 实施后注释 |
|---|---|---|
| `d314745e` | fix(verl): release GPU memory on Verl backend interrupts | 已 cherry-pick；改动在 launcher 的 try/finally + Ray actor 命名，NPU 兼容性 OK（不依赖 torch.cuda） |
| **`e66fc2e4` 🔴 P0★** | fix(verl): route rllm.algorithm.lr_schedule to active backend | 已 cherry-pick；实际改动是 `_SHARED_KEYS` 表加两行 lr_warmup_steps 映射，规模远小于预想 |

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
| 2026-05-25 | v2.3 | **W1 cherry-pick 全部落地** (12 commits, 见 §13.1)；本地全仓 syntax 通过；NPU 真实 import + vllm-ascend smoke 移交。**修正 R2/R3 在 stage1 路径上的真实生效情况**（§13.3）：stage1 走 `AgentPPOTrainer`（不是 `VerlBackend`），而 `19983fe4` 改动文件清单**完全不含** `rllm/trainer/verl/agent_ppo_trainer.py`——意味着 stage1 路径上 R2/R3 入口都不生效，无论是走 rllm 入口还是 verl native CLI（rllm 的 batch 组装路径绕开了 verl 原生 dataloader，rollout 端 routing 数据没法流到训练端）。stage1 建议**关 router_replay**，靠 KL coef / PPO clip 控制 off-policy；W2/W3 reward 不稳时按需 backport。§3.1 P0★ 表加"实施后注释"列。§3.3 swe 4 个 rllm 层 patch 全部 stage1 跳过（§13.2）。|
| 2026-05-25 | v2.4 | **NPU smoke 第一轮跑出两类失败：(A) verl-BryanChen408 fork 已把 rollout 抽象重写**（`AsyncLLMServerManager` 重命名为 `LLMServerClient` 搬到 `verl/workers/rollout/llm_server.py`；`AgentLoopManager.{server_addresses,server_handles,global_load_balancer}` 三属性搬到 `LLMServerManager`；构造签名都变了）—— **不是我们 cherry-pick 引入的，upstream/main 也 broken，只是 W1 NPU smoke 第一次实际验证暴露**。**决策**：在 verl-BryanChen408 fork 加薄兼容层（路线 B），rllm 一行不改。**(B) 19983fe4 合并漏带 `RolloutCorrectionConfig` 类定义**，已从 upstream/main 补回（commit `c1ff82e3`）。verl shim 实施：分支 `qwen36-rllm-compat`，commit `881a98d7`，2 文件 +89 行：(1) `agent_loop.py` 给 `AgentLoopManager` 加可选 `_server_manager` kwarg + 3 shim property + 末尾 `AsyncLLMServerManager(LLMServerClient)` 兼容类；(2) `ray_trainer.py:902` 把 `_server_manager=self.llm_server_manager` wire 进 `AgentLoopManager.create(...)`。新增 §13.6（verl shim）、§13.7（stage1 safe config: 关 router_replay + 关 KL loss + 监控 `rollout_probs_diff` / `pg_clipfrac` / `approx_kl`）、§13.8（NPU smoke 第二轮重写）。|
| 2026-05-25 | v2.5 | **W1 全部完成进入 W2**。NPU smoke 第二轮 20/20 import OK + vllm-ascend Qwen3.6 4 项推理 smoke 全过（含 qwen3_coder tool call 单轮/多轮、关 thinking、32k context）。**W2.2 stage1 训练脚本交付**：新增 `examples/openhands_sdk/train_openhands_qwen36_npu.{py,sh}`（py 100 行/sh 290 行），融合 verl 脚本 NPU 分支参数 + OpenHands docker 链路 + §13.7 安全配置。Surgical changes vs `train_open_megatron.sh`：env 补 `CUDA_DEVICE_MAX_CONNECTIONS=1` / `VLLM_ALLREDUCE_USE_SYMM_MEM=0`；ARGS 补 §0.2 MoE 4 项 + `vanilla_mbridge=True`；改 `calculate_log_probs=True` / `kl_loss_coef=0.0`；新增 `+rllm.algorithm.router_replay=disabled`；并行改成单节点 TP=2 EP=4 ETP=1（用户决策）；`max_model_len` 改 32k；rollout TP=8。Legacy `train_open_megatron.{py,sh}` 保留作参考。新增 §13.9（脚本交付清单 + W2.3 起 NPU 验证入口）。|
| 2026-05-25 | v2.6 | **W2.0 stage1 分段测试基础设施落地（B 档）**。设计 6 层逻辑模型 + L1..L4 fail-fast orchestrator，把端到端跑挂时的定位时间从小时级压到秒级。新文件：(1) `preflight_qwen36_npu.py` Layer 1 隔离（imports / dataset / rollout signature / hydra compose / AgentTrainer ctor 5 个子检查）；(2) `mock_rollout.py` Layer 4 drop-in 假 rollout（确定性 trajectory，无 docker/LLM）；(3) `mock_llm_server.py` Layer 2 FastAPI mock OpenAI-compatible server（已本地 curl 验通，schema 与真 vllm-ascend 一致）；(4) `stage1_test_layered.sh` orchestrator（L1/L2a/L2b/L3/L4 任选子集、bash 3.2 兼容、独立 log、summary 表）。train 脚本新增三个互相正交的 env hatch：`PREFLIGHT_ONLY=1` / `STAGE1_DRY_STEPS=N` / `STAGE1_MOCK_ROLLOUT=1`。新增 §13.10。|
| 2026-05-25 | v2.7 | **L2b 跑挂暴露 verl fork 第二轮 API 漂移**：`verl/workers/megatron_workers.py` / `fsdp_workers.py` 已合并成 backend-agnostic `engine_workers.py`（只有 `ActorRolloutRefWorker` + `TrainingWorker`）。rllm 三处 import 仍是旧路径。verl-BryanChen408 分支 `qwen36-rllm-compat` commit `85159408` 加 2 个 shim 文件（+139 行）re-export `engine_workers`，并 alias `AsyncActorRolloutRefWorker=ActorRolloutRefWorker`（async 改 config 驱动）、`CriticWorker=TrainingWorker`（stage1 GRPO 不实例化 critic）。累计 verl fork compat 文件清单详见 §13.11。NPU 节点重 pull verl fork 后重跑 L2b。|
| 2026-05-25 | v2.8 | **L2b 第二次重跑（带 W2.8 worker shim）走到 `engine.initialize()` 挂在 `mbridge.AutoBridge.from_config` "Unregistered model type: qwen3_5_moe"**。**根因 audit 错误**：plan §0.2 v2.1~v2.7 抄 verl 脚本 ACTOR 主行 `vanilla_mbridge=True`，**漏读 NPU case override 块**（verl 脚本 :196 明确 NPU 用 `vanilla_mbridge=False`）。两个独立 bridge 库澄清：`mbridge`（pypi 0.15.1，不支持 qwen3_5_moe）vs `Megatron-Bridge`（NVIDIA-NeMo，含 Qwen3.5 recipe，你环境已装）；`vanilla_mbridge` 是两者的开关，NPU 必须 False。修正：plan §0.2 + §13.9 改 False + 长注释解释；`train_openhands_qwen36_npu.sh` actor + ref 两处改 False + 详细注释；新增 §13.12 documenting 根因 + audit 教训（抄上游脚本要扫 case/if/elif override）。不需要升级 mbridge pypi 包。|
| 2026-05-25 | v2.9 | **训练脚本完整校对（W2.10）**：W2.9 修 vanilla_mbridge 后做完整 verl NPU 分支 vs stage1 脚本 diff，发现 17 项遗漏 + 4 项 dynamic_bsz 耦合不一致。用户决策：use_dynamic_bsz 走 verl 套（False + micro_batch=1 + max_token cap）。补 18 项：`CPU_AFFINITY_CONF=1` / `trust_remote_code=True` / `algorithm.use_kl_in_reward=False` / `data.truncation='error'` / `data.filter_overlong_prompts=True` / `megatron.dtype=bfloat16` / `actor.megatron.use_remove_padding=True` / `actor.checkpoint.strict=False` / `attention_backend=auto` / `moe_token_dispatcher_type=alltoall` / `use_naive_l2norm=True` / `overlap_cpu_optimizer_d2h_h2d=True` / `rollout.dtype=bfloat16` + 4 项耦合切换（actor/ref/rollout 三处 `use_dynamic_bsz=False` 同步，`max_token=16384`——按数据规模等比放大 verl 4096，因 OpenHands prompt+response=12288）。不补 `model_engine=megatron`（hydra defaults 链已带入）。34 项 stage1 必需 key 全部就位。新增 §13.13 + audit 教训 4 条（case/if 扫描、耦合识别、cap 等比放大、hydra defaults 追到底）。|
| 2026-05-25 | v2.10 | **batch sanity 联立约束（用户指出）**：`BATCH_SIZE=1` < `ppo_mini_batch_size=4` 违反 verl actor.py:224。深挖发现还有 DP 维度约束：`total_trajectories (BATCH×ROLLOUT) >= DP_size` 不然 DP rank 分不到 sample。stage1 默认改为 `BATCH_SIZE=1 ROLLOUT_N=4 PPO_MINI_BATCH_SIZE=${BATCH_SIZE}`（耦合）；脚本顶部加 fail-fast sanity check（两个约束秒级 reject，不进 Ray）；plan §13.14 写联立约束矩阵 + W3 scaling 例子 + 为什么 ROLLOUT_N=4（GRPO group baseline + 覆盖 DP=4）。|

---

## 13. W1 实施记录（2026-05-25）

### 13.1 cherry-pick 实际清单（12 commits）

执行顺序与 §4.2 / §11.1 一致。所有 commit 都在 `qwen36-openhands-stage1` 分支上，全部 `python -c "import rllm"` OK，全仓 `py_compile` 通过（rllm/ 308 文件 + rllm-model-gateway/ 33 文件 = 0 错误）。

**obs 独有（2 / 4 拿）：**

| Commit | 决策 | 实际冲突 |
|---|---|---|
| `c6e36b0a` 修复stepwise下打印不全 + completionids 长度 | ✅ 拿 | 1 处冲突，与 HEAD `90a150fc add response length debug` 并列保留 |
| `17430f3a` fix time (eval_pass_at_k.py) | ✅ 拿 | 无 |
| `4268ffdd` "tmd" (KernelGym 脚本重命名 + memory_compact.txt) | ❌ 跳 | 重命名易破坏下游 + commit message 含糊 + KernelGym stage1 不重点 |
| `095dd639` add gitignore | ❌ 跳 | 仅 +3 行 gitignore，意义不大 |

**upstream SAFE 批次（7 / 7 拿）：**

| Commit | 冲突情况 |
|---|---|
| `68e9107e` pad multi-turn batch to lcm(dp_size, ppo mini-batch) | DU agentcore_math/train_verl.sh (git rm，legacy)；verl_backend.py 2 段冲突手术合并（取 upstream 重构 `_get_dp_size`/`_get_aggregate_dp_size`） |
| `45c99f1c` zero response_mask on padded rows | auto-merge |
| `d09f6155` rollout_probs_diff masking respect response_mask | verl_backend.py + agent_sdk_trainer.py 2 段冲突，全取 upstream `calculate_debug_metrics_compat` |
| `326445bd` Collapse multi-turn steps + unified merge metrics | DU 2 个 legacy 文件 (git rm)；transform.py 4 段 + tinker/transform.py 2 段全取 upstream（`--theirs`）；pyproject.toml 加 harbor extras + tool.uv.sources rllm-model-gateway editable |
| `44fb9a3c` raise on aborted rollouts + normalize stop_reason | auto-merge |
| `e8539db5` don't truncate merged multi-turn responses | auto-merge |
| `e5ba77ea` Qwen3.5 chat template support + simple_math example | 新增 example 文件，无冲突 |

**upstream HIGH_RISK 批次（3 / 3 拿，但 19983fe4 部分跳过）：**

| Commit | 冲突情况 |
|---|---|
| `19983fe4` R2/R3 router replay | **6 个文件冲突**（modify/delete utils.py + verl_backend.py + verl_launcher.py + common/config.py + base.yaml + rollout/verl_engine.py + dataclass.py）。决策：sync_config 框架完整保留；R3 入口完整保留（verl_engine.py + transform.py + dataclass.py）；**R2 actor 端 50 行跳过**——upstream 那段引用 `tu`/`batch_td`/`no_padding_2_padding`/`bypass_mode`/`rc`，是上游另一波重构（未在 cherry-pick 范围内）引入的符号，硬合会让 import 直接挂；common/config.py from_config 保留 HEAD 调用约定 + 补 router_replay 新字段；base.yaml 取 upstream 并集但 use_rllm 保留 HEAD `false`；verl_launcher.py 自动补 `hydra_overrides = kwargs.get(...)` 和 `logger = logging.getLogger(...)` |
| `d314745e` release GPU memory on Verl backend interrupts | verl_launcher.py 1 段冲突，HEAD 旧 `WorkflowTaskRunner.remote()` 直调 vs upstream 加 HydraConfig 捕获 + try/finally + Ray actor 重命名。手动合并：保留 HEAD class 名 `WorkflowTaskRunner`，吸收 upstream 的 try/finally + hydra_overrides 转发 |
| `e66fc2e4` route lr_schedule to active backend's optim key | 1 段小冲突，接受 upstream（`_SHARED_KEYS` 表加 2 行 lr_warmup_steps 映射）；实际改动比 plan §3.1 预想小很多 |

### 13.2 swe 的 4 个 rllm 层 patch — 全部 stage1 跳过

| Patch | swe vs main 规模 | stage1 跳过原因 |
|---|---|---|
| `rllm/experimental/verl/patch.py` | +156 / -0 | 主要是 `RLLM_VLLM_PORT_BASE` 多节点端口分配；plan §9 "节点端口预算总表"已下放；HEAD 上 patch.py 已被 19983fe4 大改 (HEAD vs swe = +449/-59)，盲合风险高 |
| `rllm/experimental/verl/utils.py` | +162 / -4 | 主要是 HDFS checkpoint 逻辑；plan §5.4 "checkpoint 后端拍板"已下放 |
| `rllm/trainer/verl/ray_runtime_env.py` | +37 / -1 | HEAD 已有 +82 行 NPU env 透传（obs 自加），swe 37 行重合度高，W2 实测前不动 |
| `rllm-model-gateway/*` | +102 / -29 | 主要是 token 透出；stage1 第一版用 vllm-ascend 原生 OpenAI API 不强依赖；plan §9 "reasoning schema" 已下放 |

verl-BryanChen408 fork 已有完整 NPU 端口/HCCL 处理（`HCCL_HOST_SOCKET_PORT_RANGE` 等），与 rllm 层 swe patch 不重叠也不冲突。

### 13.3 R2 / R3 在 stage1 路径上的真实生效情况（v2.3 修正）

**stage1 训练入口实际调用链：**

```
examples/openhands_sdk/train_open_megatron.py
  → from rllm.trainer.agent_trainer import AgentTrainer
  → AgentTrainer.train()
  → from rllm.trainer.verl.train_agent_ppo import TaskRunner
  → AgentPPOTrainer  (rllm/trainer/verl/agent_ppo_trainer.py)
```

**stage1 不走的路径：**

- `rllm/experimental/verl/verl_backend.py` (`VerlBackend`)
- `rllm/experimental/verl/verl_launcher.py` (`WorkflowTaskRunner`)
- `rllm/experimental/unified_trainer.py` (`UnifiedTrainer`)

**19983fe4 改动的全部文件清单（`git diff-tree --name-only`）：**

```
rllm/experimental/common/config.py
rllm/experimental/config/rllm/base.yaml
rllm/experimental/rollout/verl_engine.py     ← R3 rollout-side 编码
rllm/experimental/verl/dataclass.py           ← R3 字段
rllm/experimental/verl/transform.py           ← R3 解码 + 填充
rllm/experimental/verl/utils.py               ← sync_config (R2/R3 共用)
rllm/experimental/verl/verl_backend.py        ← R2 actor-side propagate（被跳过那 50 行）
rllm/experimental/verl/verl_launcher.py
rllm/trainer/tinker/tinker_backend.py
rllm/trainer/tinker/transform.py
```

**关键事实**：19983fe4 **完全没碰** `rllm/trainer/verl/agent_ppo_trainer.py`（也没碰 `agent_sdk_trainer.py`、`agent_workflow_trainer.py`）。也就是说，19983fe4 加的 router_replay 入口**只在 `VerlBackend` / `UnifiedTrainer` 路径上生效**。

**关于"走 verl native CLI 应该能用 R2/R3" 的修正**：

之前 §13.0 / 对外解释里说过 R2 备选可以走 verl native CLI。这个结论**只在 stage1 不走 `AgentPPOTrainer` 的情况下成立**。实际情况是：

- verl-BryanChen408 fork 端确实有完整 R2/R3 **后端能力**（PR #5219、#5185 等 megatron worker 层面），这部分独立于 rllm
- **R2** (training-side record)：verl megatron worker 在 proximal forward 时记 routing → 返回 output tensordict → **rllm AgentPPOTrainer 不会主动把 routed_experts 从 output 抽出来塞进 batch** → actor update 拿不到 → R2 失效
- **R3** (rollout-side record)：verl rollout worker 记 routing → 但 rllm 的 `AgentExecutionEngine` / `AgentPPOTrainer` 自己组装 batch（不走 verl 原生 dataloader），**rollout 端 routing 数据没法流到训练 batch** → verl training forward 收到的 batch 里没 `routed_experts` → R3 失效

**所以 stage1 路径上**：

| | rllm CLI 入口 (`rllm.algorithm.router_replay=R3`) | verl native CLI (`actor.router_replay.mode=R3 rollout.enable_rollout_routing_replay=true`) |
|---|---|---|
| `VerlBackend` 路径 | ✅ R3 完整可用，R2 缺 actor-side propagation | ✅ 与左相同 |
| **`AgentPPOTrainer` 路径（stage1 实际走）** | ❌ rllm 19983fe4 入口在这个 trainer 上完全不触发 | ❌ rllm 自己组装 batch 时不传 routing；verl worker 即使能记录也用不上 |

**stage1 的建议处理**：

1. **第一选择**：不开 router_replay（`rllm.algorithm.router_replay=disabled` + 不传 verl native router_replay）。靠 `actor.use_kl_loss=True` + `kl_loss_coef=0.01` + PPO clip 控制 off-policy 漂移
2. **W2/W3 监控指标**：`rollout_probs_diff_max/mean/std`（来自 `calculate_debug_metrics_compat`，d09f6155 已 cherry-pick 生效）—— 如果 diff < 0.1 量级，router_replay 不是必须项
3. **如果 W3 reward 不稳且 rollout_probs_diff 大**：才考虑把 R3 数据流 backport 到 `AgentPPOTrainer`（把 `transform.py` R3 解码逻辑接到 agent_ppo_trainer.py 的 batch 组装路径），这是独立的小工作（预计 1–2 天），不在 stage1 范围

**plan §0.2 verl 脚本里的 `R3 default` 是 `VerlBackend` 路径的默认；stage1 走 `AgentPPOTrainer`，这个默认对我们不适用。**

### 13.4 本地 import smoke 结果

20 个关键模块：5 OK / 15 缺重依赖 (`torch_npu` / `verl` / `vllm` / `omegaconf` / `openai`)，**0 个 SyntaxError/NameError/AttributeError**。本地 Mac 不装 NPU 训练栈是预期；NPU 节点上跑 `git checkout qwen36-openhands-stage1` 后应该 20/20 OK。

全仓 `py_compile`：rllm/ 308 文件 + rllm-model-gateway/ 33 文件 = 0 错误。

### 13.5 W2 进入条件（NPU 节点验证清单）

NPU 节点执行清单（命令见交付物 `RUNBOOK_W2_PREFLIGHT.md` —— 如未创建，可参考会话记录里的"W2 进入条件验证清单"）：

- [ ] **NPU 节点拉到** `qwen36-openhands-stage1` 分支（需先 `git push -u origin qwen36-openhands-stage1`）
- [ ] **真实 import smoke**：20/20 模块在装齐 `torch_npu` + verl@BryanChen408 + Megatron-Bridge + vllm-ascend 后全部 OK
- [ ] **`npu-smi info`** 与 plan §0.2 `n_devices_per_node=16` 一致
- [ ] **vllm-ascend Qwen3.6 4 项 smoke**（W1.7）：关 thinking 单请求 / qwen3_coder tool call / 多轮 tool call / 32k context

**5.2 tool call 是关键失败点**——若 vllm-ascend 不支持 `qwen3_coder` parser，要么 fallback 到 generic tool parser，要么在 OpenHands 侧加 A3B tool 格式 post-processor（plan §3.5 第一版 task 之一）。

### 13.6 verl-BryanChen408 fork rllm-compat shim（W1.8 新增）

**问题**：NPU 第一轮 smoke 暴露 11/20 模块 import 失败，根因是 verl-BryanChen408 fork 已经把 rollout 抽象重写（基于 verl main 的最新形态），与 rllm 当前仍引用的 verl 0.7.x API 不兼容：

| | rllm（含 upstream/main）当前期待 | verl-BryanChen408 fork 实际 |
|---|---|---|
| 符号 `AsyncLLMServerManager` | `verl.experimental.agent_loop.agent_loop` | **重命名为 `LLMServerClient`，搬到 `verl/workers/rollout/llm_server.py`** |
| 构造签名 | `(config, servers=..., load_balancer_handle=...)` | `LLMServerClient(config, load_balancer_handle=...)` |
| `AgentLoopManager` 属性 | `.server_addresses` / `.server_handles` / `.global_load_balancer` | **三属性全部搬到 `LLMServerManager`**（`llm_server.py:326-337`） |

**注意：rllm 上游 main 也有这个问题** —— 不是 W1 cherry-pick 引入的，只是 W1 NPU smoke 第一次实际验证暴露。

**决策（用户已同意）**：路线 B —— 在 verl-BryanChen408 fork 加一层薄兼容层，rllm 一行不改。

**实施**：分支 `qwen36-rllm-compat` (verl-BryanChen408)，commit `881a98d7`，改动 2 个文件、+89 行：

1. **`verl/experimental/agent_loop/agent_loop.py`**：
   - 给 `AgentLoopManager.__init__` 加可选 kwarg `_server_manager=None`
   - 加 3 个 shim property（`server_addresses` / `server_handles` / `global_load_balancer`）forward 到 `self._server_manager`
   - 末尾新增 `class AsyncLLMServerManager(LLMServerClient)` 继承类，接受 legacy `servers=` / `load_balancer_handle=` kwargs，转发到新 `LLMServerClient` 构造

2. **`verl/trainer/ppo/ray_trainer.py:902-904`**：
   - 在 `AgentLoopManager.create(...)` 调用处加 `_server_manager=self.llm_server_manager`
   - rllm 子类继承 `RayPPOTrainer` 后，`self.async_rollout_manager.server_addresses` 等自动可用

**没 wire 的路径**（stage1 不走，留作未来）：
- `verl/trainer/main_ppo_sync.py`（用的是 `AgentLoopManagerTQ` 子类，rllm 不依赖）
- `verl/experimental/fully_async_policy/fully_async_rollouter.py`（fully-async 路径）
- `verl/experimental/one_step_off_policy/ray_trainer.py`（one-step-off-policy 路径）

**NPU 节点上拉取 shim**：

```bash
# 在 NPU 节点上的 verl 工作目录（实际路径替换）
cd /workspace/verl
git fetch <bryan-remote>     # 或 git remote add bryan git@github.com:BryanChen408/verl.git && git fetch bryan
git checkout qwen36-rllm-compat     # 注意：可能需要先 push 到 origin
# 如果分支只在本地 Mac 上：
#   (本地)  git -C /Users/yeji/Documents/Code/Python/Qwen36/verl-BryanChen408 push -u origin qwen36-rllm-compat
#   (NPU)   git fetch origin && git checkout -b qwen36-rllm-compat origin/qwen36-rllm-compat
```

**Stage2 退出 shim 的条件**：等上游 rllm 跟进 `LLMServerClient`-based API 后，删除这两段改动即可。

### 13.7 Stage1 安全训练配置（关 router_replay + 关 KL loss + 监控指标）

基于 §13.3 结论（router_replay 在 `AgentPPOTrainer` 路径上不生效），stage1 训练脚本采用以下配置：

```bash
# === router_replay 关闭（在 AgentPPOTrainer 上即便开了也无效，明确关掉避免误解）===
# 不传 rllm.algorithm.router_replay 即默认 disabled
# 不传 actor.router_replay.mode 也不传 rollout.enable_rollout_routing_replay

# === KL loss 关闭（业界惯例，包括 verl 脚本 NPU 分支自身 kl_loss_coef 较小）===
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.kl_loss_coef=0.0

# === PPO clip 保持开启（verl 默认即可）===
actor_rollout_ref.actor.clip_ratio=0.2
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.28
actor_rollout_ref.actor.entropy_coeff=0      # NPU 分支默认

# === learning rate 保守起步（与 verl 脚本 NPU 分支一致）===
actor_rollout_ref.actor.optim.lr=1e-6
```

**W2/W3 必看的 4 个 off-policy 健康指标**（前 2 个由 cherry-pick `d09f6155` 自动上报，后 2 个 verl 原生自带）：

| 指标 | 含义 | 健康 | 报警 |
|---|---|---|---|
| `rollout_probs_diff_mean` | rollout-engine vs train-engine logprob 一致性 | < 0.05 | > 0.1 → 考虑 router_replay backport |
| `rollout_probs_diff_max` | 最坏单 token 偏差 | < 0.5 | > 1.0 |
| `actor/pg_clipfrac` | 每步 PPO clip 触发比例 | < 0.1 | > 0.2 → 降 lr 或减小 ppo_mini_batch_size |
| `actor/approx_kl` | actor 当前 vs 上一 ppo iter 的 KL | < 0.02 | > 0.05 → 临时开 kl_loss_coef=0.001 救场 |

**判定规则（W3 长跑 100 步后）**：

1. 4 个指标全 healthy → 配置 OK，继续
2. reward 在涨但 `rollout_probs_diff_mean > 0.1` → 不阻塞，记录
3. `actor/pg_clipfrac > 0.3` → **降 lr 或 ppo_mini_batch_size**，不是开 KL loss 的事
4. `actor/approx_kl` 在 5–10 步内从 0.01 跳到 > 0.1 → **临时开 `kl_loss_coef=0.001`** 救场，并 dump batch 看是不是 reward 异常
5. reward 长期不涨且 `rollout_probs_diff` 大 → 才考虑 R3 backport 到 `AgentPPOTrainer`（plan §13.3 列的 1–2 天工作）

### 13.8 NPU smoke 第二轮（W1.7 重做）

第一轮 smoke 暴露的两个问题都已修复（§13.6 verl shim + §13.5 RolloutCorrectionConfig）。第二轮 smoke 步骤：

```bash
# === 0. 同步 fix ===
# 在 NPU 节点上
cd /workspace/rllm      # 或你的 rllm 路径
git fetch origin
git checkout qwen36-openhands-stage1
git pull                # 应拉到 commit c1ff82e3 (RolloutCorrectionConfig fix)

cd /workspace/verl      # 或你的 verl 路径
git fetch <bryan-remote>
git checkout qwen36-rllm-compat
git pull                # 应拉到 commit 881a98d7 (rllm-compat shim)

# === 1. 重跑结构性 import smoke（应 20/20 OK）===
python3 - <<'EOF'
import importlib, traceback
mods = [
    "rllm", "rllm.sdk", "rllm.engine",
    "rllm.engine.agent_execution_engine",
    "rllm.engine.rollout.openai_engine",
    "rllm.engine.rollout.verl_engine",
    "rllm.engine.rollout.rollout_engine",
    "rllm.trainer.verl.agent_ppo_trainer",
    "rllm.trainer.verl.agent_sdk_trainer",
    "rllm.trainer.verl.agent_workflow_trainer",
    "rllm.trainer.verl.ray_runtime_env",
    "rllm.experimental.verl.verl_backend",
    "rllm.experimental.verl.verl_launcher",
    "rllm.experimental.verl.transform",
    "rllm.experimental.verl.utils",
    "rllm.experimental.verl.metrics",
    "rllm.experimental.verl.dataclass",
    "rllm.experimental.common.config",
    "rllm.experimental.rollout.verl_engine",
    "rllm.parser.chat_template_parser",
]
fail = 0
for m in mods:
    try:
        importlib.import_module(m); print(f"  OK   {m}")
    except Exception as e:
        fail += 1; print(f"  FAIL {m}: {type(e).__name__}: {e}")
print(f"--- {len(mods)-fail}/{len(mods)} OK ---")
import sys; sys.exit(1 if fail else 0)
EOF

# === 2. 再跑 vllm-ascend Qwen3.6 4 项推理 smoke（W1.7 原文）===
#    关 thinking 单请求 / qwen3_coder tool call / 多轮 tool call / 32k context
```

**W2 准入硬性 checklist**（重写）：

- [ ] §13.8 步骤 1：20/20 OK，无 `AsyncLLMServerManager` ImportError、无 `RolloutCorrectionConfig` NameError
- [ ] §13.8 步骤 2：4 项 vllm-ascend Qwen3.6 smoke 全通

通过即可进 W2。

### 13.9 Stage1 训练脚本已交付（W2.2 完成）

**新增文件**（plan §7.1 列出，本次 landed）：

| 文件 | 行数 | 作用 |
|---|---|---|
| `examples/openhands_sdk/train_openhands_qwen36_npu.py` | 100 | hydra entry point（与 `train_open_megatron.py` 等价，独立文件防 stage1 改动污染 legacy） |
| `examples/openhands_sdk/train_openhands_qwen36_npu.sh` | 290 | shell wrapper，融合 verl 脚本 NPU 分支参数 + OpenHands docker 链路 + §13.7 安全配置 |

**关键配置（diff 自 `train_open_megatron.sh` 的 surgical changes）**：

```bash
# env (plan §0.1 必须，原脚本缺)
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# §13.7 stage1 safe config
algorithm.kl_ctrl.kl_coef=0.0                                # 原 0.001
actor_rollout_ref.actor.kl_loss_coef=0.0                     # 原 0.001
actor_rollout_ref.rollout.calculate_log_probs=True           # 原 False
+rllm.algorithm.router_replay=disabled                       # 新增防御性显式

# §0.2 verl 脚本权威 MoE 配置（原全缺）
actor_rollout_ref.actor.megatron.vanilla_mbridge=False   # NPU 必须 False，见 §0.2 / §13.12
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True

# 并行（用户决策 2026-05-25：TP=2 EP=4 单节点 8 NPU）
actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2          # 原 4
actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
actor_rollout_ref.actor.megatron.context_parallel_size=1               # 原 2
actor_rollout_ref.actor.megatron.expert_model_parallel_size=4          # 原注释
actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1         # 新增
# ref.megatron 同步上述并行配置

# rollout（plan §5.2 stage1 起步 32k）
actor_rollout_ref.rollout.tensor_model_parallel_size=8                 # 原 4，单节点全 TP
actor_rollout_ref.rollout.max_model_len=32768                          # 原 131072
```

**保持不变的部分**：HCCL/GLOO 网络 env、OpenHands docker config、LiteLLM proxy (PROXY_PORT=5000)、Ray cluster 重启逻辑、profiler 注释段、AgentTrainer + rollout 入口。

**legacy 脚本归位**：`train_open_megatron.sh` / `train_open_megatron.py` 保留不动，作为 stage1 之前的参考实现 + 任何要回退测试的对照。

**W2.3 起的 NPU 节点验证**：

```bash
# NPU 节点上
cd /workspace/rllm-071
git fetch origin
git checkout qwen36-openhands-stage1
git pull   # 拉最新

# 跑 stage1
bash examples/openhands_sdk/train_openhands_qwen36_npu.sh
```

期待第一次跑：

1. Ray cluster 起来
2. LiteLLM proxy 启动监听 :5000
3. Megatron-Bridge 加载 Qwen3.6（已验证可行）
4. 单 episode rollout 启动 OpenHands docker，container 内调 proxy → vllm-ascend → 返回 trajectory
5. 一步 ppo iter 跑完，metrics 显示 `rollout_probs_diff_mean` / `pg_clipfrac` / `approx_kl`（前者来自 §13.7 监控，后两者 verl 原生）
6. checkpoint save 到 `default_local_dir`

任何步骤失败的 traceback 告诉我。

### 13.10 Stage1 分段测试基础设施（W2.0，B 档）

**动机**：W2 端到端跑挂时定位成本太高（Ray + LiteLLM proxy + vllm-ascend + Megatron + OpenHands docker + rllm trainer 6 个子系统都可能挂在某一层）。增加分层 hatch + mock + orchestrator，做到"挂在哪一层"在秒级隔离。

**6 层逻辑模型**：

```
L1  preflight              hydra config + import + dataset + AgentTrainer ctor (no Ray/NPU/docker/LLM)
L2a mock_llm_server smoke  curl mock OpenAI server, 验 tool_calls schema
L2b mock_rollout dry-step  STAGE1_MOCK_ROLLOUT=1 + STAGE1_DRY_STEPS=1 → 全 rllm trainer 链路 + Megatron + 1 ppo iter，但 rollout 是 deterministic fake
L3  real-rollout dry-step  STAGE1_DRY_STEPS=1 → 真 OpenHands docker + LiteLLM proxy + vllm-ascend + 1 ppo iter
L4  full training          无 hatch
```

**新增文件**：

| 文件 | 行数 | 作用 |
|---|---|---|
| `examples/openhands_sdk/preflight_qwen36_npu.py` | 230 | Layer 1 隔离：5 子检查（imports / dataset / rollout signature / hydra compose / AgentTrainer construct），fail-fast，pass-through CLI overrides |
| `examples/openhands_sdk/mock_rollout.py` | 110 | Layer 4 隔离：drop-in replacement for `openhands_agent.rollout`，返回 deterministic fake trajectory，无 docker/LLM |
| `examples/openhands_sdk/mock_llm_server.py` | 160 | Layer 2 隔离：FastAPI mock OpenAI-compatible server，返回 schema 与真 vllm-ascend 完全一致的 tool_calls / logprobs / token_ids |
| `examples/openhands_sdk/stage1_test_layered.sh` | 165 | Orchestrator：跨 L1..L4 fail-fast 跑，每层独立 log + summary 表 |

**train 脚本新增 hatch**：

```bash
PREFLIGHT_ONLY=1     # skip Ray + ray job submit，只跑 preflight 后退出
STAGE1_DRY_STEPS=N   # 注入 +trainer.total_training_steps=N，跑 N 步后退出
STAGE1_MOCK_ROLLOUT=1 # train_openhands_qwen36_npu.py 自动 import mock_rollout.rollout
```

三个 hatch 互相正交可组合。orchestrator 内部就是几条 hatch 组合的 wrapper：

| 层 | hatch 组合 |
|---|---|
| L1 | `PREFLIGHT_ONLY=1 bash train_openhands_qwen36_npu.sh` |
| L2a | `python3 -m examples.openhands_sdk.mock_llm_server --port 18000` + 2 个 curl 验 schema |
| L2b | `STAGE1_MOCK_ROLLOUT=1 STAGE1_DRY_STEPS=1 bash train_openhands_qwen36_npu.sh` |
| L3 | `STAGE1_DRY_STEPS=1 bash train_openhands_qwen36_npu.sh` |
| L4 | `bash train_openhands_qwen36_npu.sh` |

**NPU 节点使用**：

```bash
# 全跑
bash examples/openhands_sdk/stage1_test_layered.sh

# 只跑 L1+L2a（不需要装 verl/Megatron 的轻量验证）
bash examples/openhands_sdk/stage1_test_layered.sh L1 L2a

# 跑到 L2b 即停（验证 rllm trainer + Megatron 不依赖 OpenHands docker）
bash examples/openhands_sdk/stage1_test_layered.sh L1 L2b

# 跳过 L1（已知通过）直接 L3 dry-step
bash examples/openhands_sdk/stage1_test_layered.sh L3
```

**fail-fast 行为**（用户决策）：每层 fail 立即停，summary 表里其余层标 `(skipped)`。覆盖：`STAGE1_HALT_ON_FAIL=0 bash stage1_test_layered.sh`。

**logs**：`/tmp/stage1-layered/L{1,2a,2b,3,4}.log` + mock server log。orchestrator 退出前打 summary 表。

**第一次 W2 推荐路径**：

1. `bash stage1_test_layered.sh L1 L2a` —— 4 秒内验：rllm 链路 import 健康 + mock LLM 链路 OK
2. `bash stage1_test_layered.sh L2b` —— ~10–30 分钟（要起 Megatron + 1 ppo iter），验：rllm trainer 全链路 + Megatron-Bridge 加载 Qwen3.6 + 一步训练 + checkpoint save（mock rollout 跳过 docker/LLM）
3. `bash stage1_test_layered.sh L3` —— 验：OpenHands docker + LiteLLM proxy + vllm-ascend 真实联调
4. `bash stage1_test_layered.sh L4` —— 进入正式训练

L1/L2a 通过 L2b 挂 → 100% 是 rllm trainer / Megatron / NPU 问题（与 OpenHands 无关）。  
L2b 通过 L3 挂 → 100% 是 OpenHands / LiteLLM proxy / vllm 链路问题（与 trainer / Megatron 无关）。

**stage2 退出 hatch**：mock 文件保留，只是 orchestrator 默认配置改成 L4 only。三个 env var 永久保留作 debug tool。

### 13.11 verl fork shim 第二轮：megatron_workers / fsdp_workers（W2.8）

**触发**：L2b 跑 `mock_rollout + dry-step` 时挂在
`ModuleNotFoundError: No module named 'verl.workers.megatron_workers'`，定位到 verl fork 的第二轮 API 漂移。

**漂移内容**：fork 把 backend-specific worker 模块（`megatron_workers.py` / `fsdp_workers.py`）**合并到一个 backend-agnostic `engine_workers.py`**，只暴露两个类：

| 旧符号 | 新位置 / 行为 |
|---|---|
| `verl.workers.megatron_workers.ActorRolloutRefWorker` | `verl.workers.engine_workers.ActorRolloutRefWorker`（async/sync 改 config 驱动：`actor_rollout_ref.rollout.mode=async`） |
| `verl.workers.megatron_workers.AsyncActorRolloutRefWorker` | 同上（统一类） |
| `verl.workers.megatron_workers.CriticWorker` | `verl.workers.engine_workers.TrainingWorker`（generic 训练 worker；critic 也用它） |
| `verl.workers.fsdp_workers.*` | 同上（fork 也搬走了） |

**rllm 侧的滞后 import**（与之前一样不是我们 cherry-pick 引入）：

```
rllm/trainer/verl/train_agent_ppo.py:99    AsyncActorRolloutRefWorker (fsdp 分支)
rllm/trainer/verl/train_agent_ppo.py:105   AsyncActorRolloutRefWorker (megatron) ← stage1 触发
rllm/trainer/verl/train_agent_ppo.py:134   CriticWorker (megatron, GRPO 不用但 import 必须通过)
rllm/trainer/verl/train_workflow_pipeline.py:120-123 同（stage1 不走但保险）
```

**实施**（verl-BryanChen408 分支 `qwen36-rllm-compat`，commit `85159408`，新增 2 个文件 +139 行）：

| 文件 | 角色 |
|---|---|
| `verl/workers/megatron_workers.py` | shim：from engine_workers 重导 + alias `AsyncActorRolloutRefWorker = ActorRolloutRefWorker` + `CriticWorker = TrainingWorker` |
| `verl/workers/fsdp_workers.py` | 同样的 shim（stage1 不走 fsdp，但 import safety net） |

shim 在 `engine_workers` 模块缺失时 raise 明确错误（不是模糊的 ImportError），方便未来诊断。

**stage1 影响**：
- GRPO actor-only，不实例化 critic → CriticWorker=TrainingWorker 的运行时签名差异（TrainingWorker 要 `TrainingWorkerConfig`）不被触发
- 真正实例化的是 `AsyncActorRolloutRefWorker → ActorRolloutRefWorker`，async 行为由 `rollout.mode=async` 触发（stage1 训练脚本里已经是 `actor_rollout_ref.rollout.mode=async`）

**累计 verl fork compat 文件清单**（`qwen36-rllm-compat` 分支）：

| commit | 文件 | 解决问题 |
|---|---|---|
| `881a98d7` | `verl/experimental/agent_loop/agent_loop.py` | `AgentLoopManager` 三 property + `AsyncLLMServerManager` shim class |
| `881a98d7` | `verl/trainer/ppo/ray_trainer.py` | wire `_server_manager` 到 `AgentLoopManager.create()` |
| `85159408` | `verl/workers/megatron_workers.py` | shim：re-export engine_workers |
| `85159408` | `verl/workers/fsdp_workers.py` | shim：同上 |

**stage2 退出条件**：等上游 rllm 跟进 verl 8.x worker API 后整体删除。

**NPU 节点拉新 commit**：

```bash
cd /workspace/verl
git fetch origin
git pull origin qwen36-rllm-compat   # 应拉到 85159408
```

然后重跑：

```bash
cd /workspace/rllm-071
bash examples/openhands_sdk/stage1_test_layered.sh L2b
```

### 13.12 `vanilla_mbridge` NPU 必须 False（W2.9）

**触发**：L2b 重跑（带 W2.8 worker shim 后）走到 `engine.initialize()` 时挂在：

```
File "verl/workers/engine/megatron/transformer_impl.py", line 178, in _build_tf_config
    bridge = AutoBridge.from_config(self.model_config.hf_config, dtype=self.param_dtype)
File "mbridge/core/auto_bridge.py", line 50, in from_config
    raise ValueError("Unregistered model type: qwen3_5_moe, now only support
                      dict_keys(['deepseek_v3', 'llama', 'qwen2', 'mimo',
                      'mixtral', 'qwen2_5_vl', 'qwen2_moe', 'qwen3', 'qwen3_moe',
                      'glm4_moe', 'glm4v', 'glm4v_moe', 'gemma3', 'internvl_chat']))
```

**重要澄清：两个独立库**

| | `Megatron-Bridge` (NVIDIA-NeMo) | `mbridge` (pypi 包) |
|---|---|---|
| 来源 | github.com/NVIDIA-NeMo/Megatron-Bridge | github.com/ISEEKYAN/mbridge → pypi |
| 用途 | offline HF↔Megatron 权重转换 + recipes | runtime build TransformerConfig + model |
| plan §0 装的版本 | `de93536e`（含 PR #2654 Qwen3.5 recipe） | `0.15.1`（pypi）— **不支持 qwen3_5_moe** |

verl 提供两条 bridge 路径（`verl/workers/engine/megatron/transformer_impl.py:173+`）：

| `vanilla_mbridge` | 走的库 | import 路径 |
|---|---|---|
| `True` | pypi `mbridge` | `verl/models/mcore/mbridge.py` → `from mbridge import AutoBridge` |
| `False` | NVIDIA `Megatron-Bridge` | `verl/models/mcore/bridge.py` → `from megatron.bridge import AutoBridge` |

**verl 自己的 Qwen3.5 NPU 脚本明确 NPU case override 为 False**（`run_qwen3_5_35b_megatron.sh:196`）：

```bash
case "${DEVICE}" in
    npu)
        ACTOR+=(
            actor_rollout_ref.actor.megatron.vanilla_mbridge=False    # ← NPU override
            ...
        )
```

**plan §0.2 早期版本的 audit 错误**（v2.1 ~ v2.7）：抄了 verl 脚本 ACTOR 主行参数 `vanilla_mbridge=True`，**没读 case override 块**。stage1 训练脚本（v2.5）继承了这个错误，导致 L2b 重跑挂。

**修正**：
- plan §0.2 改正 + 长注释说明两个 bridge 路径区分 + NPU 必须 False 的根因
- plan §13.9 W2.2 记录里的 `vanilla_mbridge=True` 改 False
- `train_openhands_qwen36_npu.sh` 改 actor + ref 两处 `vanilla_mbridge=False`，加详细注释解释

**stage1 影响**：不需要升级 mbridge pypi 包，也不需要装新 NVIDIA Megatron-Bridge 版本（你环境 `de93536e` 已经够）。

**通用教训（写给 stage2 / 类似项目）**：

抄上游脚本主行参数时要扫一下 case/if/elif 块的 override —— 主行表的"权威值"可能被 device/strategy/mode 分支 case override，audit 时容易漏。Plan §0.2 接下来对照 `run_qwen3_5_35b_megatron.sh` 应该重做一次 case-aware 抄录。

**下一步**：NPU 节点上 `git pull` rllm 拉到这个修正后重跑 L2b。

### 13.13 训练脚本完整校对（W2.10）

**触发**：W2.9 vanilla_mbridge 修复时意识到 plan §0.2 只抄了 verl 脚本 ACTOR 主行，遗漏 NPU case override。
做完整 verl NPU 分支 vs stage1 脚本 diff，发现还有 **17 项遗漏**（分高/中风险，外加 4 项配置耦合不一致）。

**用户决策（2026-05-25）**：`use_dynamic_bsz` 整组耦合 config 走 verl 套（False + micro_batch=1 + max_token cap）。

**所有补全（18 项）**：

| 类别 | key | 值 | 来源 |
|---|---|---|---|
| env | `CPU_AFFINITY_CONF=1` | yes | verl 脚本 :191 NPU case |
| model | `trust_remote_code=True` | yes | verl MODEL 段 |
| algorithm | `use_kl_in_reward=False` | yes | verl ALGORITHM 段 |
| data | `truncation='error'` | yes | verl DATA 段 |
| data | `filter_overlong_prompts=True` | yes | verl DATA 段 |
| actor | `megatron.dtype=bfloat16` | yes | verl ACTOR 段 |
| actor | `megatron.use_remove_padding=True` | 与 model 层同步 | plan §0.1 软约束 |
| actor | `checkpoint.strict=False` | yes | verl NPU case :194 |
| actor.megatron.tf_cfg | `attention_backend=auto` | yes | verl ACTOR 段 |
| actor.megatron.tf_cfg | `moe_token_dispatcher_type=alltoall` | yes | verl NPU case :199 |
| actor.megatron.tf_cfg | `use_naive_l2norm=True` | yes | verl NPU case :200 |
| actor.optim.override | `overlap_cpu_optimizer_d2h_h2d=True` | yes | verl 段（之前误注释） |
| rollout | `dtype=bfloat16` | yes | verl ROLLOUT 段 |
| 耦合切换 | `actor.use_dynamic_bsz` | True→**False** | 用户决策（verl 套） |
| 耦合切换 | `actor.ppo_micro_batch_size_per_gpu` | (缺)→**1** | verl 套必需 |
| 耦合切换 | `actor.ppo_max_token_len_per_gpu` | 32768→**16384** | 按数据规模放大（max_prompt+max_response=12288 + 33% headroom；verl 主行 4096 在我们数据下放不下） |
| 耦合切换 | `ref.log_prob_use_dynamic_bsz` | True→False；`ref.log_prob_max_token_len_per_gpu` 32768→16384 | 与 actor 套一致 |
| 耦合切换 | `rollout.log_prob_use_dynamic_bsz` | (缺)→False；`rollout.log_prob_max_token_len_per_gpu` (缺)→16384 | 与 actor 套一致 |

**没补的**（确认不需要）：

| key | 不补理由 |
|---|---|
| `EXTRA=(model_engine=megatron)` | rllm `agent_ppo_trainer_megatron.yaml` → `ppo_megatron_trainer.yaml` → `override model_engine: megatron`，hydra defaults 链已自动带入 |
| `pip install -U git+https://github.com/ISEEKYAN/mbridge.git` | NPU 走 `vanilla_mbridge=False`（NVIDIA Megatron-Bridge），不依赖 pypi mbridge |

**保留的差异**（plan 已记录原因）：

- 并行 TP=2 PP=1 EP=4（用户决策单节点 8 卡）vs verl TP=2 PP=2 EP=8（16 卡）
- `data.max_prompt_length=8192` / `max_response_length=4096`（OpenHands 长 episode）vs verl 1024/2048（geo3k 短题）
- `kl_loss_coef=0.0` / `use_kl_loss=False` / `+rllm.algorithm.router_replay=disabled` / `clip_ratio_low/high=0.2/0.28`（§13.7 安全配置）
- `max_model_len=32768`（plan §5.2）
- `rollout.n=${ROLLOUT_N}`（默认 1，bring-up 阶段）
- `trainer.test_freq=20 / total_epochs=100`（长 episode 调小 eval 频率 + 长跑）

**对照执行**：

```bash
# 34 项必需 key 自动校验
for key in CUDA_DEVICE_MAX_CONNECTIONS=1 VLLM_USE_V1=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0 CPU_AFFINITY_CONF=1 \
           trust_remote_code=True kl_loss_coef=0.0 router_replay=disabled vanilla_mbridge=False \
           megatron.dtype=bfloat16 rollout.dtype=bfloat16 \
           moe_aux_loss_coeff moe_z_loss_coeff moe_permute_fusion moe_grouped_gemm \
           moe_token_dispatcher_type=alltoall use_naive_l2norm=True \
           use_flash_attn=True attention_backend=auto use_kl_in_reward=False \
           checkpoint.strict=False calculate_log_probs=True \
           max_model_len=32768 use_dynamic_bsz=False \
           ppo_micro_batch_size_per_gpu=1 ppo_max_token_len_per_gpu=16384 \
           log_prob_use_dynamic_bsz=False log_prob_max_token_len_per_gpu=16384 \
           expert_model_parallel_size=4 tensor_model_parallel_size=2 expert_tensor_parallel_size=1 \
           overlap_cpu_optimizer_d2h_h2d=True data.truncation data.filter_overlong_prompts; do
    grep -q "$key" examples/openhands_sdk/train_openhands_qwen36_npu.sh && echo "  ✓ $key" || echo "  ✗ MISSING $key"
done
# 已全 ✓
```

**audit 教训整理**（plan §13.12 续写）：

1. **抄上游 array 时务必扫 case/if/elif override**（W2.9）
2. **耦合 config（use_dynamic_bsz / use_remove_padding 等）要识别为"套装"**，不能单改一项（W2.10）
3. **数据规模差异 → cap 类参数等比放大**（W2.10：verl 4096 → 我们 16384）
4. **hydra defaults 链要追到底**（W2.10：rllm 引用的 ppo_megatron_trainer.yaml 已经是 verl 8.x stub）

### 13.14 batch sanity：`train_batch_size`/`ppo_mini_batch_size`/`rollout.n`/`DP` 联立约束（W2.10 续）

**触发**：用户指出 `BATCH_SIZE=1` 默认值 < `actor.ppo_mini_batch_size=4`，verl actor.py:224 会直接 reject。

进一步发现这不只是整除问题，而是**联立约束**——还有 DP 维度。

**verl 实际约束**（from `verl/workers/config/actor.py:222` + `verl/trainer/ppo/ray_trainer.py:1267`）：

```
DP_size = N_GPUS / (TP × PP × CP)                         = 8 / 2 = 4  (stage1 single-node)
total_trajectories = train_batch_size × rollout.n
effective_mini_batch = ppo_mini_batch_size × rollout.n

约束 1 (硬 check, actor.py:224):  train_batch_size >= ppo_mini_batch_size
约束 2 (DP 切分, implicit):       total_trajectories >= DP_size
约束 3 (PPO 完整 iter):           total_trajectories % effective_mini_batch == 0
```

**stage1 默认 (修正后)**：

```bash
BATCH_SIZE=1           # 每 step 1 prompt → 1 个 OpenHands docker 容器（bring-up 稳）
ROLLOUT_N=4            # 每 prompt 4 rollouts（GRPO group baseline；同时 1×4=4=DP，所有 rank 都有活）
PPO_MINI_BATCH_SIZE=1  # = BATCH_SIZE，保证 train_batch_size >= ppo_mini_batch_size
```

验证：

| 约束 | 计算 | 结果 |
|---|---|---|
| 1 | 1 >= 1 | ✓ |
| 2 | 1×4=4 >= 4 | ✓ |
| 3 | 4 % (1×4=4) == 0 | ✓ |

**W3 scaling 例子**（用户改 env 即可）：

| Env | total | mini_eff | DP | 合法? |
|---|---|---|---|---|
| `BATCH_SIZE=4 ROLLOUT_N=4` | 16 | 16 | 4 | ✓ 1 PPO iter |
| `BATCH_SIZE=4 ROLLOUT_N=8 PPO_MINI_BATCH_SIZE=2` | 32 | 16 | 4 | ✓ 2 PPO iter |
| `BATCH_SIZE=2 ROLLOUT_N=2` | 4 | 4 | 4 | ✓ 最小 |
| `BATCH_SIZE=1 ROLLOUT_N=1` | 1 | 1 | 4 | ❌ DP 切不下去 |
| `BATCH_SIZE=2 PPO_MINI_BATCH_SIZE=4` | (n/a) | (n/a) | (n/a) | ❌ batch < mini |

**fail-fast sanity check 已加在脚本顶部**：

```bash
DP_SIZE=$(( N_GPUS / 2 ))   # TP=2, PP=1, CP=1
_total_samples=$(( BATCH_SIZE * ROLLOUT_N ))
if [[ ${_total_samples} -lt ${DP_SIZE} ]]; then
    echo "[stage1] FATAL: BATCH × ROLLOUT < DP ($DP_SIZE)" >&2; exit 2
fi
if [[ ${BATCH_SIZE} -lt ${PPO_MINI_BATCH_SIZE} ]]; then
    echo "[stage1] FATAL: BATCH < MINI" >&2; exit 2
fi
```

挂在脚本顶部秒级 reject 配置错误，不浪费 30s 启动 Ray 才挂在 verl validate()。

**为什么默认 `ROLLOUT_N=4` 而不是 1**：
- DP=4 必须 >=4 个 samples
- GRPO 需要 group baseline（n>=2 才有意义）
- 4 是平衡 OpenHands docker 串行执行时间（每 prompt 4 rollouts ≈ 8-15 分钟单 step）与 statistical signal 的最小可用值

**为什么 `PPO_MINI_BATCH_SIZE` 默认耦合 `BATCH_SIZE` 而不是固定值**：
- BATCH_SIZE=1 时 mini=1（1 个 mini-batch）
- BATCH_SIZE=4 时 mini=4（仍 1 个 mini-batch）
- 用户 W3 scale up 时改 BATCH_SIZE，mini 自动跟随，**避免再踩 1 vs 4 这种坑**
- 想分多个 mini-batch (e.g. 4 prompts × 2 mini-batches) 时显式传 `PPO_MINI_BATCH_SIZE=2`
