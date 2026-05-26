# Qwen3.6-35B-A3B × OpenHands Agentic RL 开发计划

> 生成时间：2026-05-23
> 修订时间：2026-05-26 (v2.30 — W2.30 W2.27 bridge β-deferred 输出后处理：worker 输出 nested tensor 经 `from_tensordict` 一刀切流入 batch，下游 `select_idxs(tensor[idxs])` 撞 nested tensor `NotImplementedError`。修：3 个 bridge 补 verl 自家 `no_padding_2_padding` + key rename + from_single_dict 输出后处理。新增 §13.33 + audit 教训第 23 条。stage2 TODO follow-up #1 ✅。前置 v2.29-v2.20 全链。)
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
- ✅ ~~**rllm `AgentSdkTrainer` 同步 verl `_compute_old_log_prob` 的 3 步桥接**~~ — **已 W2.27 完成**（rllm `dd229d51` + verl-fork revert `9998be02`，见 §13.30）。
  - **W2.27 (β) 未做的 stage2 follow-up**：
    - ✅ ~~**逐字段 `no_padding_2_padding` 输出转换**~~ — **W2.30 完成**（commit `c4efcc07`，见 §13.33）
    - ✅ ~~**`update_actor` 输出 `rename_dict + from_single_dict` 重包**~~ — **W2.30 完成**（同上）
    - **routed_experts / teacher_logprobs 处理**：verl 自家 `_compute_old_log_prob:1230` 拿 `routed_experts`，stage1 不开 distillation 用不到，跳过（如 W3 开 MoE expert load balancing 监控，再补）
- **🟡 修 `agent_sdk_engine.py:624` 硬编码 `max_prompt_length = 16384`**（plan §13.26 / W2.23 follow-up，用户决策推迟）：
  - **问题**：W2.23 `data.max_prompt_length=32768` 让 step 过滤通过，但 [agent_sdk_engine.py:624](rllm/engine/agent_sdk_engine.py:624) line 622 读 config 后 line 624 立刻硬编码覆盖 `max_prompt_length = 16384`，导致实际 PPO 只拿 16K tokens，30K - 16K = 14K 浪费。`[DEBUG format]` 打印 line 623 暴露是别人调试残留
  - **stage1 不修的理由**（用户决策 2026-05-26）：stage1 走 random reward fallback，训练信号本来就 mock，扔 14K tokens 不影响"验证 trainer 链路"目标；且当前 line 624 截到 16k 反而保护 NPU KV cache 不被 30k 拖爆。W3 真训练前必须解决
  - **修法**：删 [:624](rllm/engine/agent_sdk_engine.py:624) `max_prompt_length = 16384` 那一行，让 line 622 读到的 config 值直接生效；同步评估 NPU KV cache 余量是否撑得住 32K
- **🟡 stage2 follow-up：把 verl-fork `85159408` worker re-export shim 移到 rllm 侧**（plan §13.11 W2.8 follow-up，用户决策推迟；原本预留 W2.28 编号已被 temperature fix 占用，此项无固定 W-编号，等真做时分配）：
  - **问题**：W2.8 在 verl-fork 加了 `verl/workers/megatron_workers.py` + `verl/workers/fsdp_workers.py` 两个 shim re-export `engine_workers` 内容，让 rllm `train_agent_ppo.py:99/105/122/134` 4 个老 import 工作。属"次优 side"——rllm 这 4 个 import 完全在 `rllm/trainer/verl/` 可改区
  - **stage1 不修的理由**（用户决策 2026-05-26）：W2.8 shim 功能上完全正常没引起 bug；做 W2.28 现在 = 引入未必要变动，可能踩 `AsyncActorRolloutRefWorker` vs `ActorRolloutRefWorker` 行为微妙差异
  - **修法**：rllm 4 处 import 改为 `from verl.workers.engine_workers import ActorRolloutRefWorker as AsyncActorRolloutRefWorker` 和 `from verl.workers.engine_workers import TrainingWorker as CriticWorker`（或更彻底改用新名）+ revert verl-fork `85159408`
- **🔴 删除 stage1 random reward fallback**（plan §13.25 + §13.27，代码块在 [openhands_agent.py:717-740](rllm/examples/openhands_sdk/openhands_agent.py:717)）：
  - **触发条件（4 个必须同时满足）**：
    1. OpenHands `Condenser` 接入完成（多 turn prompt 不再线性增长）
    2. `OPENHANDS_MAX_ITERATIONS` 从 1 拉回正常值（30+）
    3. 真训练 log 里 `grep "STAGE1_MOCK keeping random"` 极少出现或为空
    4. 4 个 rollout 之间的 reward variance 来源于真信号，不是随机噪声
  - **修法（纯代码改动，无 env / config）**：恢复 `reward = 0.0` 初值 + 删 `if real_reward > 0.0` 条件覆盖 + 真值无条件覆盖 + 删 `import random`（如其他代码不用）
  - **after 状态**（[openhands_agent.py 注释里有完整版本](rllm/examples/openhands_sdk/openhands_agent.py:717)）：
    ```python
    reward = 0.0
    try:
        output = _run_openhands_container(...)
        reward = _npu_operator_reward(task, metrics_dir, output)
    except Exception:
        logger.exception(...)
    finally:
        _archive_npu_artifacts(...)
    return reward
    ```
  - **W3 验收时必查**：跑前先 grep production log 看 STAGE1_MOCK 还出不出现；不出现 = 可以放心删

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
| 2026-05-25 | v2.11 | **prepare 脚本加 `--source mock` 模式（W2.12）**：用户希望 32 条 mock 跑 layered smoke 不依赖外部数据。复用既有 4 个 ascendc 算子模板（vector_add / matmul / softmax / layer_norm），轮转生成任意 N 条（默认 32），前 4 条用原名，从第 5 条起加 `_NNNN` 后缀保 op_name 唯一。Mock 模式 yield rl_single_ops-style shape 走 `_row_to_record` 统一管线，与 real-data 同路径，reward_model 多 `{kind:'mock', template}` 标识便于调试时区分。本地验证 32 条全转 + split 模式 (val_frac=0.1 → train=29 val=3)。Plan §13.15 续写对比表（legacy `create_mock_npu_operator_data.py` 16 行固定 vs 新 mock 模式任意数量）。|
| 2026-05-25 | v2.12 | **mock parquet 在 dataset filter 挂在 jinja `No user query found in messages`（W2.13）**。根因：3 个 prepare/mock 脚本（`prepare_npu_operator_data.py`、legacy `create_mock_npu_operator_data.py`、`create_mock_npu_ascend_operator_data.py`）都用 `json.dumps([{...}])` 存 prompt 字段，但 verl `_build_messages` 和 `doc2len` 都直接拿 `doc[prompt_key]` 当 list iterate，无 `json.loads`。字符串被按字符 iterate → 找不到 user message。修复：三处都改成直接 `list[dict]`（pyarrow 原生支持 list[struct]）。为什么之前 mock 跑过没挂：之前没开 `filter_overlong_prompts=True`（W2.10 新加），doc2len 路径不触发。W2.10 + W2.13 是一起的。Audit 教训第 5 条：肉眼读 parquet 看不出 string vs list 时要 `type()` 验证。|
| 2026-05-25 | v2.13 | **W2.13 prompt 修复后 L2b 推进到 verl `_build_tf_config` 挂在 `ModuleNotFoundError: No module named 'megatron.bridge'`**（W2.14）。根因：NPU 环境的 NVIDIA Megatron-Bridge 是源码 clone（`/workspace/Megatron-Bridge`）未 pip install，`from megatron.bridge import AutoBridge` 找不到。修：训练脚本顶部加三路 dispatch（源码路径 → PYTHONPATH；pip 装了 → 不动；都没 → WARN）。可用 `MEGATRON_BRIDGE_DIR` env 覆盖默认路径。Audit 教训第 6 条：依赖装载方式（pip / 源码 clone / 系统包）跨环境不一致，启动脚本应该把源码路径作为可配置 env + 路径不存在显式 WARN。|
| 2026-05-25 | v2.14 | **L2b 在 Megatron-Bridge load_weights_hf_to_megatron → torch.distributed.broadcast 挂 HCCL error code 6（W2.15）**。**用户关键提示**：verl 自己跑 Qwen3.6 OK，rllm 基于 verl，所以问题在 rllm 这层引入。根因：obs 历史脚本硬编码 `HCCL_IF_IP=80.48.5.88` + `nic_name=ens1f3`，当前 NPU 节点 IP 是 80.48.5.65，HCCL_IF_IP 不存在 → HcclGetRootInfo 挂。verl 自己脚本 env 段只有 3 行（`CUDA_DEVICE_MAX_CONNECTIONS` / `VLLM_USE_V1` / `VLLM_ALLREDUCE_USE_SYMM_MEM`），没设这些 HCCL/socket env，靠容器默认。修：改成 opt-in 模式，`HCCL_IF_IP_OVERRIDE` / `HCCL_NIC_NAME` / `HCCL_FORCE_PORT_RANGE` env 才生效。Audit 教训第 7 条：硬编码 IP/MAC/NIC 跨机必爆，machine-specific identifier 不允许进库代码。**Audit 策略调整**：W2.10 只对齐了 ARGS 段，env 段从 obs 历史继承了 10+ 行硬编码 NPU/HCCL/vLLM env，全是潜在地雷；W2.15 先解决最毒的 HCCL_IF_IP，其他等 L2b 跑过再清理。|
| 2026-05-25 | v2.15 | **W2.15 修 HCCL 后 vllm worker 启动 exit 2（W2.16）**。继续应用 verl-self-OK 诊断原则：审 verl vllm_async_server.py:237 写死 compilation_config.setdefault cudagraph_mode=FULL_AND_PIECEWISE，无论 user 怎么配 cudagraph 默认开。verl 自己脚本不设 enforce_eager（yaml 默认 false），cudagraph 真生效跑通。我们 obs 历史脚本 enforce_eager=True # TODO 与 cudagraph 矛盾，vllm-ascend worker init 时挂。改成 env 可控默认 False，ROLLOUT_ENFORCE_EAGER=True 可 opt-out。Audit 教训第 8 条：下游覆盖上游默认时要看上游 setdefault 隐式 fill。同时让用户去 /tmp/ray/session_latest/logs/worker-*.err 拿 vllm 真 stderr 验证。|
| 2026-05-25 | v2.16 | **W2.16 改 enforce_eager 后 vllm worker 仍 exit 2（W2.17）**。拿 ray log 看到真实 vllm error: `unrecognized arguments: --swap-space 0`。obs 历史脚本通过 `engine_kwargs.vllm.<key>=...` 给 vllm CLI 加了 3 个参数（swap_space / cpu_offload_gb / enable_prefix_caching），vllm-ascend 不识别（vllm fork lag）。verl 自己脚本不设这三个，靠 vllm 默认。注释 3 行，保留 tool_call 必需的 2 行。Audit 教训第 9 条：vllm-ascend 是 vllm fork，CLI 参数有 lag；删到只剩业务必需的。**连续 W2.15/W2.16/W2.17 都是同模式**：obs 历史 vs verl-self-OK 差异 → 删 obs 多余 → 跑过；剩余可疑 obs env/config 暂不动，待 L2b 跑过再清理。|
| 2026-05-25 | v2.17 | **L2b 通过所有 setup 阶段（vllm + LiteLLM + Megatron load + actor.reset 全部 OK），挂在 step 1 start 后 AgentSdkEngine assertion Must be a list of Trajectory（W2.18）**。根因：mock_rollout 返回 list[dict]，但 AgentSdkEngine 接受三种类型（float/list[BaseTrajectory]/tuple），dict 不是 BaseTrajectory 触发 assertion。**真 openhands_agent.rollout 末尾 return reward (float)** 走 (a) float 分支，但它 docstring 写 List with one trajectory dict 误导了我。改 mock_rollout 返回 float 0.5 对齐真 rollout。Audit 教训第 10 条：不要相信 docstring，看 return 语句。L2b 进展：vllm-ascend Qwen3.6 + cudagraph capture + LiteLLM proxy + Megatron-Bridge load Qwen3.6 + HCCL broadcast + actor.reset 全部跑过，整个 setup 链路通；W2.18 之后剩 trainer 内部 mock 数据流 + PPO step。|
| 2026-05-26 | v2.18 | **L2b mock 路径走完它能走的最远（W2.19）**：W2.18 修后 4 episode 全 rollout success reward 0.5，但 mock_rollout 不调真 LLM，trace store 空，transform_results_for_verl pad_sequence 拿到 empty list 挂。这是 mock 天然边界（PPO step 需要真 token data，标量 reward 不够）。用户决策跳 L2b 进 L3。**L2b setup 验证使命已完成**（W2.8-W2.18 累积验过：rllm import / verl-fork shim / Megatron-Bridge load Qwen3.6 / HCCL / vllm-ascend cudagraph / LiteLLM proxy / actor.reset / step 1 进入 / rollout 调用 / AgentSdkEngine process_task）。新增 §13.22 documenting L2b 边界 + L3 准备清单 + audit 教训第 11 条（分层 mock 设计时先画清覆盖范围 vs 真链路依赖边界）。|
| 2026-05-26 | v2.19 | **新增 §14 新对话接手指南（Onboarding）**：用户问当前 plan 是否够新对话续工作。补 9 个 subsection：14.1 TL;DR（30 秒回上下文）、14.2 git refs cheatsheet（两仓库当前分支 + 关键 commit）、14.3 NPU 节点环境（本地 vs NPU 路径对照表）、14.4 关键代码文件清单（stage1 入口 / 辅助 / 业务 / shim 分类）、14.5 当前进度 + 下一步（已完成 / 当前位置 / 用户该做的 / L3 挂点预期）、14.6 已知 limitations + 历史踩坑（mock 边界 / verl×vllm-ascend 不兼容 / verl setdefault 暗坑 / 两个 bridge 库 / 数据 schema 陷阱 / verl API 漂移模式 / 机器特定信息不进代码 / stage1 safe config / batch sanity）、14.7 常见用户 prompt → 处理模式表、14.8 哪节看哪个 reference 表、14.9 plan 维护规则。让新对话 Claude 5 分钟内回到上下文。|
| 2026-05-26 | v2.30 | **W2.30 W2.27 bridge β-deferred 输出后处理**。W2.29 修 GDN 后 L3 推进到 advantage 阶段，挂在 `_remove_padding → batch.select_idxs → tensor[idxs_torch]` 对 nested tensor `NotImplementedError(aten.index.Tensor)`。源码验证完整 6 层因果链。这是 plan §13.30 W2.27 末段明确预测过的 β-deferred 挂点。修：3 个 bridge 都补 mirror verl 自家 `_compute_old_log_prob:1227-1246` / `_compute_ref_log_prob:1203-1208` / `_update_actor:1283-1287` 的输出后处理（extract 字段 + no_padding_2_padding + rename keys / from_single_dict）。Key rename 关键：worker 出 `log_probs`/`entropy` (singular)，rllm 读 `entropys` (plural)。新增 §13.33 + audit 教训第 23 条（包装 worker bridge 时 input prep 和 output post-processing 不可分割）。stage2 TODO 的 β-deferred follow-up #1 标 ✅。|
| 2026-05-26 | v2.29 | **W2.29 use_remove_padding True→False：GDN 不支持 packed seq**。源码验证完整因果链（5 层：train script → engine config → data_format → preprocess_packed_seqs → Qwen3-VL → MindSpeed GDN raise）。**完全对齐 plan §0.1 早期警告**。修：train script 2 处 `True → False`，对齐 verl NPU 脚本默认。W2.27 bridge `left_right_2_no_padding` 不动（源码验证 verl 自家 `_compute_old_log_prob` 无条件调它，与 use_remove_padding=False 配合工作）。新增 §13.32 + audit 教训第 22 条（"plan §0.1 软约束应在 L3 第一次失败时强制全部重审"，stage2 TODO 加"§0.1/§0.2 软约束状态回填"工作）。|
| 2026-05-26 | v2.28 | **W2.28 W2.27 extension：rllm batch.meta_info 漏 `temperature` 字段**。L3 推进到 megatron `transformer_impl.py:841 forward_step` 报 `KeyError: temperature`。源码验证：verl 在 `fit:1394-1395` 每 ppo iter batch 构造后立刻 set，rllm `transform_results_for_verl` 路径不走 fit batch 构造 → 没 set。W2.27 update_actor bridge 内 set 了但太晚（line 534 vs compute_log_prob line 418）。修：在 rllm `agent_sdk_trainer.py` 紧贴 `batch.meta_info["global_token_num"]` 现有行后全局只设一次 `temperature` + `multi_turn`，删 W2.27 update_actor bridge 里的重复 set。新增 §13.31 + audit 教训第 21 条（port 多阶段 trainer 桥接时要 port 上游 fit-loop 层 metadata 注入，不仅是 bridge 方法内的）+ stage2 TODO 加"完整 verl fit ↔ rllm fit_agent diff 审计"避免类似漏洞。|
| 2026-05-26 | v2.27 | **W2.27 supersede W2.26：rllm 侧补 trainer 桥接 + revert W2.26 worker-entry shim**。用户两轮追问推动重审：第一次"为什么这么多问题"逼出真根因（rllm trainer 漏 verl 自家 mid-#2733 加的 3 步桥接）；第二次"逐 commit 评估"催出全面 audit。结论 W2.26 是唯一"side 选错"的 commit。修：(1) rllm `agent_sdk_trainer.py` 3 处（compute_log_prob/compute_ref_log_prob/update_actor）补 5 步桥接（to_tensordict + left_right_2_no_padding + tu.assign_non_tensor + 调 worker + DataProto.from_tensordict），mirror verl `RayPPOTrainer._compute_old_log_prob/_compute_ref_log_prob/_update_actor`；(2) revert verl-fork `7e30674e`。Stage2 TODO 加 follow-up：逐字段 no_padding_2_padding 输出转换（W2.27 用 from_tensordict 整体转，未验证下游 `.batch["entropys"]` shape 期望）。新增 §13.30 + audit 教训第 19 + 20 条。W2.28 候选：W2.8 worker re-export shim 也可移到 rllm 侧（4 个 import 更新），不紧急。|
| 2026-05-26 | v2.26 | **verl-fork shim #4：engine_workers 3 个 batch 方法入口 DataProto→TensorDict 转换（W2.26）**：W2.25 后 `infer_batch` 又挂在下面 12 行的 `data.keys()` 上（DataProto 无 .keys()）。根因 W2.25 同源扩大版：3 个 batch 方法（train_mini_batch/train_batch/infer_batch）整段都按 TensorDict 写但实际收 DataProto，`tu.assign_non_tensor` / `tu.make_iterator` / `data.shape` 全踩。修：3 方法入口加 3 行 `if not isinstance(data, TensorDict): data = data.to_tensordict()`，幂等，下游零改动。`to_tensordict()` 是 verl 上游 native API（protocol.py:1102）。verl-fork compat 累计 4 个 commit: `881a98d7` + `85159408` + `370e1148` + `7e30674e`。新增 §13.29 + audit 教训第 18 条（PR #2733 是渐进 migration，同方法连续 type error 优先怀疑同源；修在入口比逐行修 helper 更稳）。|
| 2026-05-26 | v2.25 | **verl-fork shim #3：tu.pop 缺失 key 守卫（W2.25）**：W2.24 后 L3 重跑前 5 层全闭环（pad_sequence/step/episode/is_valid/进 PPO step），挂在 `actor_rollout_compute_log_prob` 内部：`verl/utils/tensordict_utils.py:759 tu.pop` 调 `tensordict.pop("no_lora_adapter", _sentinel)` 时，实际 `data` 是 DataProto（非 TensorDict，dispatch 默认传 DataProto），DataProto.pop 签名是 `(batch_keys, ...)` 取 list，把字符串当字符迭代，每个 char 撞 `assert key in self.batch.keys()`。tu.get 有同样守卫但 tu.pop 漏了。修：verl-fork 在 tu.pop 加 `if key not in tensordict: return default` 守卫（4 行），对齐 tu.get 行为。修在 verl-fork qwen36-rllm-compat `370e1148`，**rllm 一行不改**（沿用 W2.8/W2.14 同样模式）。新增 §13.28 + audit 教训第 17 条（verl API drift 持续暴露；helper 函数都要审 sentinel/default 处理是否对齐）。verl-fork 累计 compat 3 个：`881a98d7` agent_loop + `85159408` workers + `370e1148` tu.pop。|
| 2026-05-26 | v2.24 | **W2.22 random fallback 漏洞修正（W2.24）**：用户预判 W2.23 后 4 rollout 全 reward=0.0 → GRPO `std=0` → NaN advantage → PPO 仍挂。根因：W2.22 我设的 `reward = random.random()` 初值会被 try 内 `reward = _npu_operator_reward(...)` **无条件覆盖**，return 0.0 是正常返回路径（不是 exception）会把 random 刷掉。fallback 只在 except 触发，reward=0 不触发 = W2.22 描述里没说清的逻辑漏洞。修：try 内加 `if real_reward > 0.0: reward = real_reward` 条件覆盖（3 行）。W3 transition 零摩擦：真 reward > 0 自然接管。新增 §13.27 + audit 教训第 16 条（fallback 触发条件要精确写在 docstring + 单测覆盖；test_reward_pipeline.py 未来要加 `reward != 0.0` 回归 case）。|
| 2026-05-26 | v2.23 | **L3 `data.max_prompt_length` 过滤掉所有 step（W2.23）**：W2.22 后 4 rollout 全部 200 OK + reward=0.0 完成，但依然 `pad_sequence empty`。用户深度诊断 path：(1) 写 `diagnose_trace_store.py` 验 DB 干净 → 排除 DB；(2) `json_extract` 比对 `data.session_name` vs `metadata.session_name` 全对齐 → 排除字段错位；(3) 沿代码下查到 `agent_sdk_engine.py:563` 用 `data.max_prompt_length=8192` 过滤 step，OpenHands 真 prompt 30721 全 skip → trajectory 无 valid step → episode drop → batch 空。修：1 行，`data.max_prompt_length` 8192 → 32768 给 OpenHands 实际 prompt 留出空间。顺手发现 `agent_sdk_engine.py:624` 硬编码 `max_prompt_length=16384` 是 debug 残留（"[DEBUG format]" 打印泄露），对 stage1 反而是 KV cache 保护（padding 阶段 left-truncate 到 16K）→ 列入 stage2 cleanup 不动。新增 §13.26 + audit 教训第 15 条：分层 mock 漏覆盖"OpenHands 实际 prompt 量级 vs dataloader 阈值"。教训：调试数据流问题需 grep warning 级别日志（"Skipping step..." 被 info 淹没）。新增诊断工具 `diagnose_trace_store.py`（可复用）。|
| 2026-05-26 | v2.22 | **L3 改换策略：MAX_ITERATIONS=1 + random fallback reward（W2.22）**：W2.21 升 max_model_len 到 49152 后第二轮 prompt 47105 又越界 1。观察：每 turn 增长 ~16k，盲升 context 不是路。用户决策 stage1 只验 trainer 链路、不在乎 reward 真假。修：(1) `OPENHANDS_MAX_ITERATIONS` 默认 1000 → 1，prompt 锁首轮 ~30k，永不超 49k；(2) rollout fallback reward `0.0` → `random.random()`，避免 GRPO `std=0` → NaN 让 PPO 跑得动；(3) 不加新 env gate（"env 已太多"）。Trade-off：纯噪声 PPO 信号，stage1 不在乎，W3 + condenser 后 real reward 接管自然 dormant。新增 §13.25 + audit 教训第 14 条（mock_rollout 该返随机 [0,1) 暴露 GRPO std=0 corner case）。|
| 2026-05-26 | v2.21 | **L3 第一轮 LLM call context off-by-one（W2.21）**：W2.20 DooD fix 通过后容器跑 120 秒到 vllm tokenize 阶段，`30721 prompt + 2048 output = 32769 > max_model_len 32768`，越界 1 token。根因：OpenHands 默认 system prompt + tools + AGENTS.md 第一轮就 30k tokens（Qwen3-coder agent 重 prompt 通病，结构性解法在 OpenHands condenser，stage1 不碰）。修：[train_openhands_qwen36_npu.sh:451](rllm/examples/openhands_sdk/train_openhands_qwen36_npu.sh:451) `max_model_len` 32768 → 49152，给 16k buffer。不动 OpenHands max_tokens（影响生成质量）也不动 data.max_prompt_length（那是 training dataloader 过滤阈值，不限制 runtime LLM call）。Qwen3.5/3.6 native 支持 256K，49k 完全在模型能力内；NPU KV cache 多吃 50% 但 batch=1 dry-step 扛得住。W4 plan §4.5 预排的 64k 升级保留给后续"P95 接近 40k"触发。新增 §13.24 + audit 教训第 13 条：分层 mock 设计漏了"真业务 context budget"约束，下次 L2a mock_llm_server 加边界 case。|
| 2026-05-26 | v2.20 | **L3 首跑 DooD workspace path alignment（W2.20）**：4 个 OpenHands 容器在 ~3s 内全 exit，reward=0.0，trace store 空 → pad_sequence empty list（与 L2b 同症状但根因不同）。根因：rllm 主进程跑在一个 main container 内，`workspace_temp` 写在 main container overlay 上；`openhands_agent` 用 `-v {workspace}:/opt/workspace` 起子容器，**这条 `-v` 由宿主 dockerd 解析**，看不到 main container 内的路径 → 宿主新建空目录挂到子容器 → entrypoint.py 不存在 exit 127。修复（用户决策：不引入新 env，stage1 env 已过多 + 不污染 /tmp）：(1) `openhands_agent.py:225` workspace_temp **hardcode** `/home/docker/openhands_workspace`（NPU 节点 docker user home，持久路径稳定）+ 长注释说明 DooD 约束；(2) 训练脚本顶部加 **fail-fast precheck**：`workspace_temp` + `artifact_dir` dir 不存在/不可写直接 `exit 1` 并提示 bind mount 修复指令；soft warn 不是 bind mount（用 findmnt 检测，不阻塞，因为 findmnt 不一定每个 image 都装）；(3) 用户运维侧 main container 启动加 `-v /home/docker/openhands_workspace:/home/docker/openhands_workspace`。新增 §13.23 + audit 教训第 12 条：DooD 下任何 `-v <src>:<dst>` 的 `<src>` 必须宿主可见（在宿主 shell `ls <src>` 能看到内容才合法）。**Stage 2 演化路线**：用户决策不烤 skills 进 image（迭代成本高），多机时走 OBS 拉取（首选，Ascend 集群默认有）/ NFS / 自有 HTTP 服务（最重，非必要不引入）。stage 2 进 §9 下一阶段 TODO 跟踪。|

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

### 13.15 真实数据集 prepare 脚本（W2.11）

**触发**：用户想直接用真实算子数据集（kernelgym 风格 JSONL）训 stage1，问能否复用
`examples/kernelgym/prepare_kernelbench_data.py`。

**结论**：不能直接复用。两份脚本输出 schema 完全不一致：

| | `prepare_kernelbench_data.py` | stage1 训练脚本期待 |
|---|---|---|
| 格式 | JSONL | parquet |
| 顶层 schema | `{task: {...}, backend: "..."}` | `{prompt: str, extra_info: {...}}` |
| 字段命名 | `task.problem_id` / `task.reference_code` / `task.prompt` | `extra_info.op_name` / `extra_info.task_code` / `extra_info.instruction` |
| 注入路径 | `DatasetRegistry.register_dataset()` | verl 直接读 parquet (`data.train_files=<parquet>`) |

**新增 `examples/openhands_sdk/prepare_npu_operator_data.py`**（460 行，含 W2.12 mock 模式）支持 4 种输入源：

```bash
# 1. kernelgym 风格 JSONL（用户实际格式）
python3 -m examples.openhands_sdk.prepare_npu_operator_data \
    --source local-jsonl \
    --input ./data/drkernel_rl_data.jsonl \
    --output ./examples/openhands_sdk/real_ops_train.parquet \
    --scenario npu_ascend_operator --arch ascend910b

# 2. rl_single_ops 风格 JSON（既有 mock 的源）
python3 -m examples.openhands_sdk.prepare_npu_operator_data \
    --source local-json --input ./rl_single_ops.json \
    --output ./examples/openhands_sdk/rl_single_ops.parquet \
    --scenario npu_ascend_operator --arch ascend910b

# 3. HuggingFace dataset（本地 hf_dataset/ 目录或远端 Hub）
python3 -m examples.openhands_sdk.prepare_npu_operator_data \
    --source hf --input ./hf_dataset/drkernel-rl-data --hf-split train \
    --output ./examples/openhands_sdk/real_ops_train.parquet

# Train/val 切分
python3 -m examples.openhands_sdk.prepare_npu_operator_data \
    --source local-jsonl --input ./data/drkernel_rl_data.jsonl \
    --output-train ./examples/openhands_sdk/real_ops_train.parquet \
    --output-val ./examples/openhands_sdk/real_ops_val.parquet \
    --val-frac 0.05 --seed 42
```

**统一输出 schema**（与 mock 完全兼容，superset）：

```python
{
    "prompt": str,        # json.dumps([{"role":"user", "content": instruction}])
    "extra_info": {
        "instruction":  str,     # 给 LLM 的提示
        "scenario":     str,     # 'npu_operator' / 'npu_ascend_operator'
        "op_name":      str,     # 短标识，metrics 聚合用
        "arch":         str,     # 'ascend910b1' / 'ascend910b'
        "task_code":    str,     # PyTorch 参考实现，写进 OpenHands workspace
        "reward_model": dict,    # 新增字段：{ground_truth, entry_point, backend, problem_id}
                                 # 给容器内 reward 函数读，不破坏现有 rollout
    }
}
```

**核心设计**：

- 单一 `_row_to_record(row, scenario, arch)` 函数自动判别 kernelgym-style (有 `task` 字段) vs rl_single_ops-style (有 `prompt`+`py_code/code`)，源端解析与目标 schema 解耦
- 不认识 schema 的行 `--skip` 计数（前 5 条打 stderr），不阻塞整体转换
- `--limit N` 支持 smoke 模式（先转 N 条试管线）
- `--val-frac` deterministic split（seed 固定）

**本地验证**（4 条 mini kernelgym JSONL → parquet）：

```
✓ top-level columns identical: ['prompt', 'extra_info']
mock extra_info keys: ['arch', 'instruction', 'op_name', 'scenario', 'task_code']
real extra_info keys: ['arch', 'instruction', 'op_name', 'reward_model', 'scenario', 'task_code']
✓ real parquet is a superset of mock keys — rollout will work
✓ split mode: train=3 val=1
```

**接到 stage1 训练**：

train_openhands_qwen36_npu.py 当前指向 `rl_single_ops.parquet`：

```python
_MOCK_NPU_PARQUET = os.path.join(_EX_DIR, "rl_single_ops.parquet")
```

如果你的 prepare 输出叫 `real_ops_train.parquet`，两种接法：

A. **改 .py 里的常量**指向新 parquet（最干净）
B. **生成时 `--output=examples/openhands_sdk/rl_single_ops.parquet`** 覆盖现有文件（最少代码改动）

stage1 起步推荐 B（不动 .py，重跑 prepare 时直接覆盖 parquet）。

**W2.12 内置 mock 模式（`--source mock`）**：

不依赖任何外部数据源，4 个 ascendc 算子模板（vector_add / matmul / softmax / layer_norm）轮转生成任意 N 条；前 4 条用原名，从第 5 条起加 `_NNNN` 后缀保唯一。

```bash
# 32 条 mock，single file
python3 -m examples.openhands_sdk.prepare_npu_operator_data \
    --source mock --mock-count 32 \
    --output examples/openhands_sdk/rl_single_ops.parquet \
    --scenario npu_ascend_operator --arch ascend910b
# 输出：32 行 parquet，extra_info.reward_model 含 {ground_truth, kind="mock", template}
# 用途：layered smoke L2b/L3 不需要真实数据时；快速 bring-up
```

**与既有 `create_mock_npu_operator_data.py` 的关系**：

| | `create_mock_npu_operator_data.py` (legacy, 4×4=16 rows) | `prepare_npu_operator_data.py --source mock` (新) |
|---|---|---|
| 行数 | 固定 16（4 模板 × 4 重复） | 任意 N (`--mock-count N`，默认 32) |
| op_name 唯一 | 否（vector_add 出现 4 次） | 是（加 `_NNNN` 后缀） |
| reward_model | 无 | `{ground_truth, kind="mock", template}` |
| 输出位置 | 固定 `mock_npu_operator.parquet` | 任意 `--output <path>` |
| 走 _row_to_record | 否（直接写 parquet） | 是（与 real-data 统一管线） |

legacy 脚本 stage1 用过、保留不动；新脚本 mock 模式作为统一入口。

### 13.16 prompt 字段类型修正：list[dict] 不是 JSON string（W2.13）

**触发**：用户用 `--source mock --mock-count 32` 生成的 parquet 在 L2b dataset filter 阶段挂在：

```
jinja2.exceptions.TemplateError: No user query found in messages.
  File "verl/utils/dataset/rl_dataset.py", line 256, in doc2len
    tokenized_prompt = tokenizer.apply_chat_template(doc[prompt_key], ...)
```

**根因**：verl 在两条路径上都假设 `doc[prompt_key]` **已经是 `list[dict]`**：

```python
# rl_dataset.py:_build_messages
messages: list = example[key]              # 不 json.loads
for message in messages:                   # 字符串会被按字符 iterate
    ...

# rl_dataset.py:doc2len (plain tokenizer path)
tokenizer.apply_chat_template(doc[prompt_key], ...)   # jinja iterate, 字符串失败
```

我们三个 prepare/mock 脚本都用了 `json.dumps([{"role":"user",...}])`，所以 prompt 字段是 string，jinja iterate 字符找不到 `role=='user'` → "No user query found"。

**修复**：prompt 字段直接存 list[dict]，pyarrow 原生支持 `list[struct]`。

涉及 3 个文件：

| 文件 | 修改 |
|---|---|
| `examples/openhands_sdk/prepare_npu_operator_data.py` | `_row_to_record` 末尾 `json.dumps(...)` → 直接 list |
| `examples/openhands_sdk/create_mock_npu_operator_data.py` | `prompt = json.dumps(...)` → `prompt = [...]` |
| `examples/openhands_sdk/create_mock_npu_ascend_operator_data.py` | 同上 |

每处加注释说明 verl 要求 list、字符串会挂 jinja，避免未来回归。

**本地验证**：

```
prompt type: ndarray   (pyarrow list[struct] → numpy.ndarray of dict)
prompt[0]: role=user content_head=Implement an AscendC kernel ...
✓ 1 user message(s) found in row 0 (previously 0 → jinja crashed)
```

**为什么之前 stage1 跑过 mock 没挂？**

最可能：之前没开 `filter_overlong_prompts=True`（这是 W2.10 新加的，§0.2 verl 权威要求）。`filter_overlong_prompts` off 时，doc2len 路径不触发；prompt 字段后续怎么解 / 有没有别的兜底 json.loads 没追到底。**v2.10 把 `filter_overlong_prompts=True` 加上后，这条 latent bug 立刻显形**。

也就是说：W2.10 完整对齐 verl NPU 分支 → 触发 W2.13 修复 prompt 类型 → 这两个修是一起来的，缺一不可。

**audit 教训补充（§13.13 第 5 条）**：

5. 跨数据/训练边界的 schema 不能假设"看起来对就行" —— pyarrow 能存 list[struct] 也能存 string，但 verl 只接受 list，肉眼读 parquet 看不出 string 还是 list 时要 `type()` 一下。

### 13.17 Megatron-Bridge PYTHONPATH 自动注入（W2.14）

**触发**：L2b 走过 dataset filter 后挂在：

```
File "verl/workers/engine/megatron/transformer_impl.py", line 192, in _build_tf_config
    from verl.models.mcore.bridge import AutoBridge
File "verl/models/mcore/bridge.py", line 17, in <module>
    from megatron.bridge import AutoBridge
ModuleNotFoundError: No module named 'megatron.bridge'
```

**根因**：NPU 环境的 Megatron-Bridge 是源码 clone（`/workspace/Megatron-Bridge`），**未 pip install**，所以 `megatron.bridge` 不在 Python path 上。

**修复**：训练脚本顶部加路径自动注入：

```bash
MEGATRON_BRIDGE_DIR="${MEGATRON_BRIDGE_DIR:-/workspace/Megatron-Bridge}"
if [[ -d "${MEGATRON_BRIDGE_DIR}/src/megatron/bridge" ]]; then
    export PYTHONPATH="${MEGATRON_BRIDGE_DIR}/src:${PYTHONPATH}"
elif python3 -c "import megatron.bridge" 2>/dev/null; then
    echo "[stage1] Megatron-Bridge: already importable (pip-installed)"
else
    echo "[stage1] WARN: not found AND not pip-installed; training will crash"
fi
```

三路 dispatch：
- 源码 clone 存在 → 加入 PYTHONPATH
- 已 pip install → 不做事
- 都没有 → 显式 WARN（不立即 exit，让 preflight smoke 也能跑）

`MEGATRON_BRIDGE_DIR` 默认 `/workspace/Megatron-Bridge`，可用环境变量 override。

**通用教训补充（§13.13 第 6 条）**：

6. **依赖装载方式（pip vs 源码 clone vs 系统包）跨环境不一致** —— stage1 启动脚本应该把"源码 clone 路径"作为可配置 env，并加路径不存在的显式 WARN，避免又跑半天才发现是 PYTHONPATH 没接上。

### 13.18 删 HCCL_IF_IP 硬编码 — 对齐 verl 自己跑的默认（W2.15）

**触发**：L2b 走过 `_build_tf_config` + Megatron-Bridge load HF weights，挂在 `torch.distributed.broadcast`：

```
RuntimeError: createHCCLCommOrigin:torch_npu/csrc/distributed/ProcessGroupHCCL.cpp:2314
HCCL function error: HcclGetRootInfo(&hcclID), error code is 6
[ERROR] (PID:1131801, Device:0, RankID:6) ERR02200 DIST call hccl api failed.
```

Traceback 头部显示 worker `pid=1131801, ip=80.48.5.65` — **当前节点 IP 是 80.48.5.65**。

**用户关键提示**：verl 自己跑 Qwen3.6 是 OK 的。rllm 基于 verl，所以排查方向必须是"rllm 引入了什么 verl 自己跑时没有的差异"。

**根因**：obs 分支历史脚本硬编码：

```bash
nic_name="ens1f3"
export HCCL_IF_IP=80.48.5.88        # ← 当前节点没有这个 IP
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
```

当前 NPU 节点不存在 IP 80.48.5.88，HCCL `HcclGetRootInfo` 找不到对应 NIC → error code 6。

verl 自己的脚本（`run_qwen3_5_35b_megatron.sh`）**根本不设这些**，靠 NPU 容器默认 HCCL 配置就 OK。

**修复**：改成 opt-in 模式：

```bash
# 默认（与 verl 自己跑一致）：完全不设 HCCL_IF_IP / SOCKET_IFNAME / port range

# 需要 override（多 NIC 机器、端口冲突等）时启动前 export：
HCCL_IF_IP_OVERRIDE=10.1.2.3 \
HCCL_NIC_NAME=enp0s8 \
HCCL_FORCE_PORT_RANGE=1 \
bash train_openhands_qwen36_npu.sh
```

三个独立 opt-in env：
- `HCCL_IF_IP_OVERRIDE` — 多 IP 机器选哪个 NIC 通信
- `HCCL_NIC_NAME` — GLOO/TP/HCCL SOCKET_IFNAME 一致设
- `HCCL_FORCE_PORT_RANGE` — 同节点跑多 verl 实例时强制端口段（含 HCCL_INTRA_ROCE/PCIE）

**Audit 教训补充（§13.13 第 7 条）**：

7. **硬编码 IP/MAC/NIC 名称跨机器必爆**。obs 历史脚本对某个 dev 节点是 OK 的，但 stage1 跑在新节点上立刻挂。原则：machine-specific identifier 不允许出现在 commit 进库的代码里，要么用 env override 要么 runtime detect（如 `hostname -I`）。

**Audit 整体策略调整**：

之前 W2.10 做"完整对齐 verl NPU 分支"是对的方向，但只对齐了 `ARGS` 段参数，**没对齐 env 段**。verl 自己脚本 env 段只有 3 行：

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
```

我们脚本 env 段从 obs 历史继承了 10+ 行硬编码 NPU/HCCL/vLLM env，**这些都是 stage1 的潜在地雷**。W2.15 解决 HCCL_IF_IP 这个最炸的，其他 env（如 `ASCEND_LAUNCH_BLOCKING=1`、`VLLM_ATTENTION_BACKEND=TORCH_SDPA` 等）等到 L2b 跑过再视情况清理。

### 13.19 enforce_eager 对齐 verl 默认（W2.16）

**触发**：L2b 走过 Megatron load + HCCL broadcast 后，vllm worker 启动时 `exits with an exit code 2`，ray actor 死亡。debug 输出能看到 vllm CLI args 但没有 vllm 内部 stderr。

**verl 自己 vllm worker 关键代码**（`vllm_async_server.py:234-249`）：

```python
compilation_config = engine_kwargs.pop("compilation_config", None) or {}
compilation_config.setdefault("cudagraph_mode", "FULL_AND_PIECEWISE")   # ← 写死
# ...
args = {
    "enforce_eager": self.config.enforce_eager,
    "compilation_config": compilation_config,
    ...
}
```

verl **永远把 `cudagraph_mode='FULL_AND_PIECEWISE'` 传给 vllm**。verl 自己脚本 yaml 默认 `enforce_eager: false`（不显式设），cudagraph 真生效，vllm-ascend 跑 Qwen3.6 OK。

我们脚本从 obs 历史继承了 `actor_rollout_ref.rollout.enforce_eager=True # TODO`，导致 `enforce_eager=True` + `cudagraph_mode=FULL_AND_PIECEWISE` 矛盾，vllm-ascend worker init 时 exit 2。

**修复**：改成 env 可控，默认 False（对齐 verl）：

```bash
# 默认（与 verl 一致）
actor_rollout_ref.rollout.enforce_eager=False

# 如 NPU cudagraph warmup 太久或挂，可临时 opt-out
ROLLOUT_ENFORCE_EAGER=True bash train_openhands_qwen36_npu.sh
```

**待用户确认**（同时跑 W2.16 + 拿 stderr 验证）：

```bash
# 在 NPU 节点
ls -la /tmp/ray/session_latest/logs/ | grep "worker-" | tail -5
grep -l "vllm" /tmp/ray/session_latest/logs/worker-*.err 2>/dev/null | head -3
cat /tmp/ray/session_latest/logs/worker-<vllm-worker-pid>.err | tail -50
```

如果 W2.16 改完跑过：cudagraph 确实是根因。
如果还挂：stderr 会显示真实错误，可能是别的（如 `--logprobs_mode processed_logprobs` 或 `--enable_sleep_mode` vllm-ascend 不支持）。

**Audit 教训补充（§13.13 第 8 条）**：

8. **下游覆盖上游默认值时要看上游 setdefault / 隐式 fill 的行为**。verl 的 vllm CLI 不只是 user-config 透传，还有 setdefault 自动填充（compilation_config 就是例子），下游显式设的某些值会与 setdefault 自动填的形成矛盾。审 verl 类似自动 fill 行为：`grep -rn "setdefault\|_apply_quantization" verl/workers/`。

### 13.20 删 obs 历史 vllm engine_kwargs（W2.17）

**触发**：W2.16 改 enforce_eager 后 vllm worker 仍 exit 2。Ray log 显示真实 vllm CLI parse error：

```
:job_id:02000000
:actor_name:InstrumentedvLLMHttpServer
usage: default_worker.py [-h] {serve} ...
default_worker.py: error: unrecognized arguments: --swap-space 0
```

**根因**：obs 历史脚本通过 verl 的 `engine_kwargs.vllm.<key>=...` 透传机制给 vllm CLI 加了 3 个参数：

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=0
+actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb=0
+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=False
```

verl 把它们转成 `--swap-space 0` 等 vllm CLI args。**vllm-ascend 不识别这些参数**（vllm-ascend fork 早，vllm 主线后加的 args 没跟）。

verl 自己脚本根本不设这三个，靠 vllm 默认 → 能跑通。

**修复**：注释三行，对齐 verl 默认。tool call 必需的两个保留：

```bash
# 保留
+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True   # tool call 必需
+actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=qwen3_coder    # 同

# 注释（vllm-ascend 不识别）
# +actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space=0
# +actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb=0
# +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=False
```

**vllm-ascend 不识别的 vllm 参数**：累计这次 + 之前 (`--logprobs_mode processed_logprobs` 等)，后续如需 disable swap / prefix_caching 要查 vllm-ascend 对应参数名（可能 vllm-ascend 用 `--ascend-swap-space` 之类的）。

**Audit 教训补充（§13.13 第 9 条）**：

9. **vllm-ascend 是 vllm 的 fork，CLI 参数与 vllm 主线有 lag**。obs 历史脚本可能是为 GPU+vllm 写的，迁到 vllm-ascend 上时凡是 `engine_kwargs.vllm.*` 透传的参数都要"上下游 fork 是否同步"逐个验证。最简方式：删到只剩业务必需的（如 tool_call_parser），其他靠默认。

**W2.17 体现的 audit 模式**（连续 W2.15/W2.16/W2.17 三次都是同一类）：

```
obs 历史 vs verl-self-OK 差异 → 删 obs 多余 → 跑过
```

W2.15 删 HCCL_IF_IP 硬编码 / W2.16 改 enforce_eager / W2.17 删 vllm engine_kwargs，全是同样模式。剩余可疑 obs 历史 env / config：

- `actor_rollout_ref.rollout.max_num_seqs=4`（vllm-ascend 应该支持但 verl 默认不设）
- `actor_rollout_ref.rollout.max_num_batched_tokens=8192`（同）
- `actor_rollout_ref.rollout.enable_chunked_prefill=True`（同）
- `actor_rollout_ref.rollout.free_cache_engine=True`（同）
- `++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096`（同）
- 顶部 env：`ASCEND_LAUNCH_BLOCKING=1` / `VLLM_ATTENTION_BACKEND=TORCH_SDPA` / `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128` / `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` / `VLLM_ENGINE_ITERATION_TIMEOUT_S=...` / `RAY_DEBUG_POST_MORTEM=0` / `RAY_DEDUP_LOGS=0` / `VLLM_ASCEND_ENABLE_NZ=0` / `OOM_SNAPSHOT_*`

这些先不动（"非必要不改"原则），等 L2b 跑过看哪些真的是必需的，再视情况清理。

### 13.21 mock_rollout 返回类型对齐真 rollout（W2.18）

**触发**：L2b 走过 vllm + LiteLLM proxy 起来 + Megatron-Bridge load + actor.reset() + Qwen35VLMoEBridge 模型转换完成，到 "epoch 0, step 1 started"，然后立刻挂：

```
Error in execute_tasks: Must be a list of Trajectory, retrying...
AssertionError: Must be a list of Trajectory
  File "rllm/engine/agent_sdk_engine.py", line 220, in process_task_with_retry
    assert all(isinstance(t, BaseTrajectory) for t in output)
```

**根因**：`AgentSdkEngine.process_task_with_retry` 接受 rollout 函数 3 种返回形式（agent_sdk_engine.py:213-223）：

```python
(a) float | int | bool        → 单纯 reward 标量
(b) list[BaseTrajectory]      → trajectories (有 assertion 检查类型)
(c) tuple[(payload, metrics)] → 带 metrics 的 payload
```

**真 `openhands_agent.rollout` 末尾返回 `reward` (float)** 走 (a)。但它的 docstring 写：

```python
"""
Returns:
    List with one trajectory dict: {name, steps, reward}.
"""
```

**docstring 撒谎**——实际看代码 line 730 `return reward`。

我之前写 `mock_rollout.py` 时被 docstring 误导，返回了 `list[dict]` → 走 (b) 分支但里面是 dict 不是 BaseTrajectory → assertion 挂。

**修复**：mock_rollout 返回 `float` (0.5)，与真 rollout 对齐：

```python
def rollout(*args, **kwargs) -> float:
    return 0.5
```

L2b 设计目的（exercise rllm trainer step / Megatron / PPO 框架，不验真 rollout）依然满足：scalar reward 进 trainer，trajectory 数据从 trace store 取（mock 模式下 trace store 是空的，但 trainer 框架能跑过 1 step）。

**Audit 教训补充（§13.13 第 10 条）**：

10. **不要相信 docstring，看 return 语句**。`openhands_agent.rollout` docstring 写 "List with one trajectory dict"，实际 `return reward` 返回 float。Mock / shim / adapter 类代码尤其要 grep `return` 实证，不能照 docstring 抄签名。

**L2b 进展确认**：在挂到 W2.18 之前，L2b 已经走过：
- vllm-ascend Qwen3.6 server 启动 + cudagraph FULL_AND_PIECEWISE capture
- LiteLLM proxy 启动 + admin/reload 200
- Megatron-Bridge 加载 Qwen3.6 (Qwen35VLMoEBridge converting 100%)
- HCCL broadcast + actor.reset() 完成
- Step 1 start

也就是 W2.8/W2.9/W2.13/W2.14/W2.15/W2.16/W2.17 一连串修复**全部生效**，rllm × verl × vllm-ascend × Megatron-Bridge × NPU 链路 setup 阶段已通。剩下是 trainer 内部 mock 数据流（W2.18）+ PPO step 框架。

### 13.22 L2b mock 路径天然边界 + 跳 L3 决策（W2.19）

**W2.18 修 mock_rollout 返回 float 后，L2b 走得更远**：

```
Generating trajectories: 100% 4/4
[xxx:0:1] Rollout completed with reward: 0.5
[xxx:1:1] Rollout completed with reward: 0.5
[xxx:2:1] Rollout completed with reward: 0.5
[xxx:3:1] Rollout completed with reward: 0.5
INFO: POST /admin/flush-tracer 200 OK
Episode xxx:0:1 has no valid trajectories, dropping it from the batch
Episode xxx:1:1 has no valid trajectories, dropping it from the batch
Episode xxx:2:1 has no valid trajectories, dropping it from the batch
Episode xxx:3:1 has no valid trajectories, dropping it from the batch
RuntimeError: received an empty list of sequences
  at agent_sdk_engine.py:614 pad_sequence(prompts)
```

**根因 — L2b mock 路径的天然边界**：

`AgentSdkEngine.transform_results_for_verl` 从 trace store（LiteLLM proxy 录的 SQLite）拿真 LLM 调用记录，build `prompt_ids / completion_ids / logprobs` 喂给 PPO。**mock_rollout 只返回 reward 标量，不调 LLM**，trace store 是空的 → 4 个 episode 全 drop → batch 空 → `pad_sequence(empty_list)` 挂。

PPO step 需要真 token data，mock 标量 reward 不够。要让 L2b 真跑通 PPO step，必须 mock_rollout 真调一次 LLM（哪怕通过 mock_llm_server）—— 那就和 L3 差不多了。

**用户决策（2026-05-26）**：跳过 L2b 直接进 L3。L2b setup 验证使命已完成。

**L2b 已验证的（W2.8-W2.18 累积成果）**：

| 子系统 | 状态 |
|---|---|
| rllm import 链 + verl-fork shim | ✓ |
| Megatron-Bridge PYTHONPATH | ✓ |
| HCCL broadcast | ✓ |
| Megatron-Bridge load Qwen3.6 | ✓ (Qwen35VLMoEBridge 转换 5916/5916) |
| vllm-ascend Qwen3.6 server + cudagraph capture | ✓ |
| LiteLLM proxy 启动 + admin/reload | ✓ |
| AgentSdkTrainer.init_workers | ✓ |
| epoch 0 step 1 进入 | ✓ |
| rollout 函数调用（mock × 4） | ✓ (reward 0.5 × 4) |
| AgentSdkEngine.process_task_with_retry | ✓ (W2.18 后) |

**L2b 未验的（mock 天然不能验）**：

| 子系统 | 状态 |
|---|---|
| 真 LLM 调用 + trace store filling | ❌ mock 跳过 |
| PPO batch 构造 | ❌ 上一项依赖 |
| actor.compute_log_prob | ❌ 同 |
| PPO loss / backward / optimizer.step | ❌ 同 |
| checkpoint save / load | ❌ 同 |

剩余子系统**只能 L3 真 rollout 验**。

### L3 准备清单

L3 = `STAGE1_DRY_STEPS=1 bash train_openhands_qwen36_npu.sh`（不设 `STAGE1_MOCK_ROLLOUT`，走真 OpenHands docker 路径）。

**前置检查**：

| 项 | 命令 | 期待 |
|---|---|---|
| docker daemon 在跑 | `docker info \| head -5` | 不报错 |
| OpenHands image 已 build | `docker images \| grep openhands-triton-env` | 有 v1 |
| 容器能调 host | `docker run --rm --network host openhands-triton-env:v1 curl -fsS http://localhost:5000/health` | 200 OK |
| OPENHANDS_ARTIFACT_DIR 可写 | `mkdir -p /workspace/results/openhands_results && touch $_/test && rm $_/test` | OK |
| mock_npu_operator reward 函数能在容器内跑 | 看 openhands_agent.py `_npu_operator_reward` 是否容器内可执行 | 函数体内只用 fs，OK |

**可能挂点排序（按风险递减）**：

1. **docker image 不存在 / build 失败** —— FORCE_BUILD=1 重 build 看
2. **容器内调 LiteLLM proxy 不通** —— `host.docker.internal:5000`（默认）vs `--network host` + `localhost:5000` 看 OpenHands 容器 entrypoint 怎么传 LLM_BASE_URL
3. **OpenHands 容器内 entrypoint 异常 exit** —— 看 `OPENHANDS_ARTIFACT_DIR` 下容器 log
4. **第一次 LLM 调用太慢导致 timeout** —— 已设 `OPENHANDS_CONTAINER_TIMEOUT=1800`（30 分钟）
5. **PPO step 真挂**（最值得期待——意味着前面全通了）

**Audit 教训补充（§13.13 第 11 条）**：

11. **分层 mock 的"覆盖范围 vs 真链路依赖"边界要早画**。L2b 设计初衷"绕 docker/LLM 验 trainer"是对的，但低估了 PPO step 对真 token data 的硬依赖（标量 reward 不够）。下次设计 mock 层，先在 plan 里明确该层能验/不能验的子系统清单（这次 plan §13.10 没画清楚 PPO step 是否在 L2b 范围内）。

### 13.23 DooD workspace path alignment：workspace_temp 必须落在宿主可见路径（W2.20）

**L3 第一次跑出意外快速失败**：4 个 OpenHands 容器 docker run 成功（拿到 container id），但**总耗时 2.96 秒就 4 个全部 reward=0.0 退出**，trace store 空 → 4 episode 全 drop → 同 L2b 的 `pad_sequence empty list`。容器内 entrypoint stderr 报 `/opt/workspace/entrypoint.py: No such file or directory`。

**根因 — Docker-out-of-Docker volume 路径未对齐**：

rllm 主进程跑在一个 "main container" 里（NPU 节点上的训练容器）。`openhands_agent.rollout` 用 docker SDK 在**宿主 dockerd 上**起子容器，emit 的 `-v {workspace}:/opt/workspace`：

```
main container 内：
    /workspace/rllm-071/examples/openhands_sdk/workspace_temp/trajectory-XXX/   ← shutil.copytree 写在这里
              ↓ openhands_agent emit "-v {workspace}:/opt/workspace"
宿主 dockerd 在宿主 FS 找 /workspace/rllm-071/.../workspace_temp/trajectory-XXX
              ↓ 找不到 → 默认行为：宿主上新建空目录
子 OpenHands 容器看到 /opt/workspace/ 是空的 → entrypoint.py 不存在 → exit 127
```

**关键点**：`docker run -v <src>:<dst>` 的 `<src>` 始终由**执行 docker daemon 的内核**解析，永远不是调用方容器的视角。这是 DooD 拓扑的基础约束，文档里很少提，第一次踩必坑。

**修复（W2.20，code + 文档）**：

| 文件 | 改动 | 行号 |
|---|---|---|
| `examples/openhands_sdk/openhands_agent.py` | `workspace_temp` hardcode 为 `/home/docker/openhands_workspace` + DooD 注释 | [:220-228](rllm/examples/openhands_sdk/openhands_agent.py:220) |
| `examples/openhands_sdk/train_openhands_qwen36_npu.sh` | 启动前 fail-fast 检 `workspace_temp` + `artifact_dir` 存在 + 可写；soft warn 不是 bind mount | 紧接 `OPENHANDS_ARTIFACT_DIR` |

**用户决策（2026-05-26）**：
- stage1 env var 已积累过多，workspace_temp 路径**不引入新 env**，直接 hardcode 在 `openhands_agent.py`。代价：换路径要改 code；收益：脚本 env 表清爽，新人接手不用追溯一条 env 链。
- 不用 `/tmp/openhands_workspace`（容易污染 tmp / 重启丢），落在 `/home/docker/openhands_workspace`（NPU 节点 docker user 的 home，持久 + 路径稳定）。
- 训练脚本顶部加 fail-fast precheck：dir 不存在或不可写直接 `exit 1`，提示用户加 bind mount；soft warn 不是 bind mount（用 `findmnt` 检测，不阻塞）。

**用户操作（main container 启动侧）**：main container 启动命令需加同路径 bind mount（rllm 仓不管 main container 怎么起，这步在用户的运维脚本里）：

```bash
# 宿主上准备
mkdir -p /home/docker/openhands_workspace

# main container 启动加 bind mount（源/目的同路径）
docker run ... \
    -v /home/docker/openhands_workspace:/home/docker/openhands_workspace \
    ...
```

**验证方式**（在宿主 shell，不是 main container 内）：rollout 跑起来后宿主上能看到 `/home/docker/openhands_workspace/trajectory-*/agent_workdir/INSTRUCTIONS.md`，子容器才看得到。

或者直接靠训练脚本顶部的 precheck：跑 `bash stage1_test_layered.sh L3` 时，如果 main container 没加 bind mount，脚本会在进入 Ray 之前就 fail 并打印明确的修复指引。

**为什么这个解法是"同路径 bind mount"而不是"路径翻译"**：DooD 拓扑里，**main container 内的路径 = 宿主路径 = 子容器 -v 的 source 路径**，三者必须一致。如果搞路径映射（"main container 内 X 在宿主上是 Y"），代码就要维护一张映射表，每次起容器都查表，下次接手的人很难调试。同路径 bind mount 是工业标准做法（k8s 的 hostPath、Jenkins agent 跑 docker build 都是这个套路）。

**为什么 `_WORKSPACE_PKG`（模板源）不用宿主可见**：模板只在 main container 内被 `shutil.copytree` 读取，destination 才挂到子容器。所以**只有 workspace_temp 这个 destination 需要同路径 bind mount**，源模板 + rllm 源码这些都不用动。这点很重要 —— 用户可以继续在 main container 内迭代 `examples/openhands_sdk/workspace/` 模板，**改完立刻生效**（每次 rollout 重新 copytree），不需要重 build image。

**Audit 教训第 12 条**：**DooD 下任何 `docker run -v <src>:<dst>` 的 `<src>` 必须是宿主可见路径**。判断方法：在宿主 shell（不是 main container 内）`ls <src>`，能看到内容才合法。代码里 emit 这种 mount 字符串的位置（[openhands_agent.py:546](rllm/examples/openhands_sdk/openhands_agent.py:546)）值得在 docstring / 注释里显式标"这里的 path 是宿主 dockerd 视角"。

**Stage 2 演化路线（用户决策 2026-05-26）**：

skills 快速迭代场景下，stage 2 不走"workspace 烤进 image"（迭代成本高），三个备选：

| 路线 | 适用 | 评估 |
|---|---|---|
| OBS / S3 拉取（Ascend 集群默认有） | 模板存对象存储，节点本地缓存 | **首选**：不写自有 service，对象版本天然支持 |
| 自有 HTTP 服务 | 需要复杂 skill 拼装逻辑或细粒度访问控制 | 多一个 critical service，部署/监控/HA 成本，非必要不引入 |
| 多机共享 NFS | 节点已有共享 fs | 路径直挂，跨节点透明 |

stage 1（W2.20）= 同路径 bind mount，工时 < 30 分钟。stage 2 真正切多机时再选 OBS 或 NFS，进 §9 下一阶段 TODO 跟踪。

### 13.24 L3 第一轮 LLM call context off-by-one：32k → 49152（W2.21）

**症状**：W2.20 DooD fix 通过后，L3 第一个 rollout 容器跑 120 秒（vs 之前 3 秒），但**vllm 端 tokenize 阶段抛 `VLLMValidationError`**：

```
This model's maximum context length is 32768 tokens. However, you requested
2048 output tokens and your prompt contains at least 30721 input tokens,
for a total of at least 32769 tokens.
```

`30721 + 2048 = 32769 > 32768`，**差 1 个 token 越界**。

**根因**：OpenHands 默认 system prompt + 全部 tools 定义（terminal/file_editor/bash 等）+ `agent_workdir/AGENTS.md` + `INSTRUCTIONS.md`，第一轮 LLM call 就 30k tokens。Qwen3-coder agent 框架历史性的"重 system prompt"问题，不是 stage1 该结构性解决的（结构性解法在 OpenHands condenser / prompt compaction，下放到 W3/W4）。

**结构性观察**：30k 只是**第一轮**就吃掉的，多 turn 会随对话历史线性增长。L3 dry-step 只跑 1 iter 所以单次 unblock 够用；W3 长跑必须有 compaction 机制或更大 context。

**修复（W2.21，一行）**：[train_openhands_qwen36_npu.sh:451](rllm/examples/openhands_sdk/train_openhands_qwen36_npu.sh:451) `max_model_len` 32768 → **49152**。

| 数 | 含义 |
|---|---|
| 49152 | new max_model_len（48k） |
| 30721 | first-turn observed prompt |
| 2048 | OpenHands LLM client 默认 output budget |
| 49152 − 30721 − 2048 = **16383** | 给 prompt 后续增长 + output 留的 buffer |

**为什么不一步升到 65536**：
- 49152 在 NPU KV cache 上比 32k 多 50% 占用，65536 多 100%，dry-step 都扛得住但 49k 更保守；
- W4 plan §4.5 已预排 64k 升级，留给后续以"实测 P95 接近 max_model_len"为触发条件升；
- 49152 在 W2.21 是为了精准解决"30k+2k 越界 1"的 issue，不是为了 W3+ 长跑做容量规划。

**为什么不动 OpenHands `max_tokens=2048`**：那是 OpenHands LLM client 默认，改了影响生成质量（容器内 output 被截断），权衡不如直接拉 context。

**为什么不动 `data.max_prompt_length=8192`**：这是**训练数据 dataloader 阶段的过滤阈值**，不限制 OpenHands runtime LLM call。对 L3 不相关。

**Audit 教训第 13 条**：分层 mock 设计时（plan §13.10）覆盖了"trainer 链路"和"OpenHands docker 链路"两条主轴，但**漏了"真业务 context budget"这条隐式约束**。下一次设计 mock 时，给 L2a mock_llm_server 加一条"返回 prompt token count > model max_model_len 的边界 case"，提前在 L2a 暴露这类越界，不让它溜到 L3。

stage1 此后默认 `max_model_len=49152`。W3 监控 episode 长度 P95，超 40k 再考虑 65536。

### 13.25 改换策略：MAX_ITERATIONS=1 + random fallback reward（W2.22）

**症状**：W2.21 升 max_model_len 到 49152 后，L3 重跑 — **第二轮 LLM call 47105 tokens + 2048 output = 49153 又越界 1**：

```
This model's maximum context length is 49152 tokens. However, you requested
2048 output tokens and your prompt contains at least 47105 input tokens,
for a total of at least 49153 tokens.
```

**观察**：每个 turn prompt 增长 ~16k（=LLM 响应 + 上一轮 tool 执行结果），盲目升 context 走不通 —— 65k 也只多撑 1 轮。

**用户决策（2026-05-26）**：stage1 不是为了让 agent 真解算子题，**只为了让 trainer 链路（rollout → trace → batch → PPO step → checkpoint）跑起来**。reward 用随机数 mock 都可以接受。

**修复（W2.22，两改动）**：

| 文件 | 改动 | 行号 |
|---|---|---|
| `train_openhands_qwen36_npu.sh` | `OPENHANDS_MAX_ITERATIONS` 默认 1000 → **1** | [:159](rllm/examples/openhands_sdk/train_openhands_qwen36_npu.sh:159) |
| `openhands_agent.py` rollout | `reward = 0.0` 初始值 → **`reward = random.random()`**（fallback uniform [0,1)） + 加 `import random` | [:717](rllm/examples/openhands_sdk/openhands_agent.py:717) |

**为什么 MAX_ITERATIONS=1**：
- 1 个 turn = 1 次 LLM call，prompt 锁定在首轮 baseline ~30k tokens，永远不会增长到 49k
- 仍然产生真实 LLM 调用 → trace store 有数据 → PPO batch 能构造
- 副作用：agent 没机会执行 tool / 写代码 / 跑评测，reward 函数大概率拿不到有效 output → 0 分

**为什么 fallback reward = `random.random()` 而不是 `0.0`**：
- GRPO advantage = `(r - mean_r) / std_r`
- 如果 4 个 rollout reward 全 0（W2.22 后大概率发生，因为 MAX_ITERATIONS=1），std=0，advantage 算出 NaN → PPO step 挂
- random uniform [0,1) 保证 4 个 rollout 之间有 variance，PPO step 跑得动
- 真 reward 成功时（`_npu_operator_reward` 返回正常值）会覆盖随机初值，不影响真训练信号

**信号保真度权衡**：random fallback 是**纯噪声 PPO 信号**，训练不出来真模型 —— 但 stage1 不在乎，目的是验证 trainer 链路。W3 + condenser 后，agent 能真 multi-turn 工作，real reward 接管，random fallback 自然 dormant（不发生异常就不用）。

**为什么不加新 env gate `STAGE1_MOCK_REWARD=1`**：用户决策"env 已经太多"。当前实现：fallback 只在异常分支生效，正常 reward 计算成功就用真的，不需要 env 控制。

**Audit 教训第 14 条**：分层测试理论上能验"trainer 跑得动"，但真业务 reward 全 0 这种 corner case 在 mock 里没暴露过（mock_rollout 写死 0.5 有 variance；真业务可能 4 个 rollout 同 0）。下次 mock 设计时 mock_rollout 应该返回**随机 [0,1)** 而非常数 0.5，提前在 L2b 暴露 GRPO 在 std=0 时的行为（虽然 mock_rollout 现在用不上了，但作为 mock 设计规范保留）。

stage1 此后默认 `MAX_ITERATIONS=1` + random fallback reward。L3 跑通 → 提 W3 时同步评估"何时把 MAX_ITERATIONS 加回 + condenser 接入"。

### 13.26 `data.max_prompt_length` 卡 step 过滤 → pad_sequence empty（W2.23）

**症状**：W2.22 fix 后 L3 重跑，4 个 rollout 全 `reward=0.0` 完成（不是 random fallback，是真 reward 函数返回 0），但**依旧 `RuntimeError: received an empty list of sequences`** 在 `transform_results_for_verl` line 614 `pad_sequence(prompts, ...)`。

**用户排查路径**（耗时但极有价值）：

1. **怀疑 DB 写入失败** → 写诊断脚本 `diagnose_trace_store.py` 一次性查 traces/trace_sessions/metadata/slug 解码 → **结论：DB 完全干净**（4 个 rollout 全有 trace，session_uid 完全对齐，session_name 在 data 和 metadata 列都存在且一致）
2. **怀疑 `trace.data.session_name` vs `metadata.session_name` 字段错位** → SQL `json_extract(data, '$.session_name')` vs `json_extract(metadata, '$.session_name')` 比对 → **结论：两边都对，完全一致**
3. **沿着代码继续往下查** → 找到 `agent_sdk_engine.py:563`：

```python
max_prompt_length = self.config.data.max_prompt_length  # = 8192
for step_idx, step in enumerate(trajectory.steps):
    prompt_ids = ...
    if len(prompt_ids) > max_prompt_length:    # ← 30721 > 8192，全 skip
        logger.warning(f"Skipping step {step_idx} ... prompt length {len(prompt_ids)} > max_prompt_length {max_prompt_length}")
        continue
    prompts.append(prompt_ids)
# ...
if n_steps == 0:                              # ← skip 完所有 step
    logger.warning(f"Trajectory ... has no valid steps after filtering overlong prompts, skipping")
    continue
# 最后 prompts 列表空 → line 614 pad_sequence([]) 炸
```

**根因**：OpenHands 首轮 prompt 30721 tokens > stage1 默认 `data.max_prompt_length=8192`，所有 step 被过滤 → 所有 trajectory 无 valid step → episode 全 drop → batch 空。

**和之前几条都不相关**：不是 W2.20 DooD、不是 W2.21 max_model_len、不是 W2.22 MAX_ITERATIONS，是**纯粹的 dataloader 过滤阈值与 OpenHands 实际 prompt 大小不匹配**。

**修复（W2.23，1 行）**：[train_openhands_qwen36_npu.sh:328](rllm/examples/openhands_sdk/train_openhands_qwen36_npu.sh:328) `data.max_prompt_length` 8192 → **32768**（盖住 30721 + buffer）。

**顺手发现的独立 issue**（不在本节修）：[agent_sdk_engine.py:622-626](rllm/engine/agent_sdk_engine.py:622) padding/truncation 阶段**硬编码** `max_prompt_length = 16384`：

```python
max_prompt_length = self.config.data.max_prompt_length  # line 622
print("[DEBUG format] max prompt length:", max_prompt_length)
max_prompt_length = 16384                                # ← line 624 硬编码覆盖
prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, ...)
prompts_batch = prompts_batch[:, -max_prompt_length:]    # left-truncate 到 16K
```

是别人留的 debug 代码（"[DEBUG format]" 打印是 dead giveaway）。**对 stage1 反而是好事**：过滤层放行 32k prompt 后，padding 阶段会自动 left-truncate 到 16K（保留尾部 16K tokens），PPO 不会因为单 sequence 30k 而爆显存。该硬编码长期应该改成读 config，列入 stage2 cleanup。

**Audit 教训第 15 条**：分层 mock 设计时（plan §13.10）覆盖了"链路连通"和"功能字段对齐"，但**漏了"OpenHands 真实 prompt 量级 vs stage1 dataloader 过滤阈值"这条约束**。下次 mock 设计时 mock_rollout 应该构造一条 prompt > config 限制的 trajectory，提前在 L2b 暴露这条过滤。

**为什么这条这么难找**：日志线索都是 `warning` 级别（"Skipping step..." / "Trajectory has no valid steps after filtering overlong prompts"），用户看的是 traceback / `Generating trajectories: 100%` / `Rollout completed with reward: 0.0` 这些 info 级别消息，warning 被 info 淹没。**调试 stage1 数据流问题时建议 grep warning**（`grep WARNING ray_log_*.err`）。

stage1 此后默认 `data.max_prompt_length=32768`。W3 视 OpenHands prompt 实际增长，可能再升。

### 13.27 W2.22 random fallback 漏洞修正：reward==0 也走 fallback（W2.24）

**症状预判（用户提问）**：W2.23 fix 后，pad_sequence 应该不再 empty，4 个 rollout 都能进 GRPO。但用户敏锐发现：**4 个 rollout 都返回 `reward=0.0`（因为 MAX_ITERATIONS=1 → agent 没写 impl 文件 → `_npu_operator_reward` 走 "no impl file" 分支返回 0.0），GRPO `advantage = (0 - 0) / 0 = NaN`，PPO step 仍然会挂**。

**根因（W2.22 漏洞）**：W2.22 把初始 `reward = 0.0` 改成 `reward = random.random()`，但 try 内 `reward = _npu_operator_reward(...)` **无条件覆盖初始值**。`_npu_operator_reward` 返回 0.0 是**正常成功路径**（不是 exception），会把 random 初值刷掉。fallback 只在 except 分支生效，正常返回 0.0 不触发。

**这是 W2.22 描述里我自己埋的逻辑漏洞**（plan §13.25 末段"random fallback 永远 in 但正常时 dormant"暗示了它生效条件，但没指出"reward==0 = 正常返回 != exception"这条分支）。用户问起来才修。

**修复（W2.24，3 行）**：[openhands_agent.py:717-738](rllm/examples/openhands_sdk/openhands_agent.py:717) try 分支加 `if real_reward > 0.0:` 条件覆盖：

```python
reward = random.random()  # 初始 fallback（覆盖两种 case）
try:
    output = _run_openhands_container(...)
    real_reward = _npu_operator_reward(...)
    if real_reward > 0.0:           # ← W2.24 关键：只在真信号 > 0 时覆盖
        reward = real_reward
    # else: 保留 random（GRPO variance）
except Exception:
    pass   # reward 也保留 random
```

**行为对照**：

| 场景 | `_npu_operator_reward` 返回 | W2.22 行为 | W2.24 行为 |
|---|---|---|---|
| 容器异常 | 抛异常 | random ✓ | random ✓ |
| MAX_ITERATIONS=1, no impl file | 0.0 | **0.0 ✗ GRPO NaN** | **random ✓** |
| 有 impl 无 trace | 0.2 | 0.2 ✓ | 0.2 ✓ |
| 真解出题 | 0.5/0.8/0.95 | 真值 ✓ | 真值 ✓ |
| W3 condenser 接入 | > 0 主流 | 真值, fallback dormant | 同左，**完全一样** |

**W3 transition 路径**（plan §9 stage2 TODO 也跟踪）：

| W3 阶段 | 用户希望的状态 | Random fallback 状态 |
|---|---|---|
| 现在（W2.x stage1） | trainer 链路跑通 | **engaged**（agent 没机会写 impl，real_reward=0 走 fallback）|
| 早期 W3（condenser 接入，MAX_ITERATIONS 30+，agent 多 turn 工作） | reward 大部分 > 0，少数 0 | dormant 主导，偶发噪声 |
| 中期 W3（reward variance 充足） | reward 全 > 0 | 完全 dormant（log 里 `grep "STAGE1_MOCK keeping random"` 无结果） |
| **W3 验收点：删除 fallback 代码** | grep production log 见无 STAGE1_MOCK | 代码块物理删除，random `import` 删除 |

**用户决策（2026-05-26）**：random 在真实场景不合理，**功能验证完后必须删**。代码侧（W2.24+）已加显式 markers：

- 代码块外**显著注释** `⚠️ STAGE1 MOCK REWARD FALLBACK — MUST REMOVE BEFORE W3 REAL TRAINING ⚠️`
- 4 条 removal criteria 列在 comment 里
- "after" code 内联在 comment 里，删的人直接 copy/paste 就行
- log message 加 `STAGE1_MOCK` prefix → 一行 grep 就能扫到 production 还有没有用 fallback
- plan §9 stage2 TODO 加 🔴 高优先级条目跟踪

**不引入 env gate / config flag** 的理由：用户决策"env 已太多"，且 fallback 删除时机 ≠ runtime toggle 时机（需要前置 condenser 等条件成熟），env 反而误导。物理删代码是正确语义。

**Audit 教训第 16 条**：**fallback 的触发条件要精确写在 docstring/comment 里 + 单测覆盖**。W2.22 我口头说"正常路径不触发"，没说"reward 在 try 内被无条件覆盖"。下次写 fallback：(a) 列出明确的触发 condition；(b) 加 unit test 覆盖所有 fallback case。

`test_reward_pipeline.py`（未来要写的 fixture 测试，§13.25 提到过）应该加一个 `assert reward != 0.0` 的回归 case，保证未来代码改动不破坏这个保护。

### 13.28 verl-fork shim #3：`tu.pop` 缺失 key 守卫（W2.25）

**症状**：W2.24 后 L3 重跑，**前 5 层全闭环**：

```
[DEBUG format] len(prompts): 4              ✓ pad_sequence 不空（W2.23 fix）
prompt_lengths: tensor([10205, 10205, ...]) ✓ step 过过滤、line 624 truncate 到合理大小
response_lengths: tensor([256, 336, ...])   ✓ 真生成
unique episode_ids: 4                       ✓ 4 个 rollout 都进 batch
is_valid: Counter({True: 4})                ✓ compact_filtering 不挡
                                            ↓
                            进入 actor_rollout_compute_log_prob
                                            ↓
                            verl-fork tu.pop AssertionError
```

```
File "/workspace/pri/verl/verl/workers/engine_workers.py", line 385, in infer_batch
    no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
File "/workspace/pri/verl/verl/utils/tensordict_utils.py", line 759, in pop
    output = tensordict.pop(key, _sentinel)
File "/workspace/pri/verl/verl/protocol.py", line 741, in pop
    assert key in self.batch.keys()
AssertionError
```

**根因（verl-fork 内部 bug）**：
- `engine_workers.infer_batch` 的 `data: TensorDict` annotation 是骗人的；rllm dispatch 实际传的是 **DataProto** 对象（这是 verl 整个 worker 抽象的常态，dispatcher 名字 `make_nd_compute_dataproto_dispatch_fn` 已说明）
- `DataProto.pop` 签名是 `(batch_keys, non_tensor_batch_keys, meta_info_keys)`，**取 LIST 不取单 key**
- 当 `tu.pop` 调 `tensordict.pop("no_lora_adapter", _sentinel)`：
  - 字符串 `"no_lora_adapter"` 当 `batch_keys` 迭代，逐字符 `'n', 'o', '_', 'l', ...`
  - 每个字符进 `assert key in self.batch.keys()` → 一定 False → AssertionError
- `tu.get` 提前用 `if key not in tensordict: return default` 守卫，所以 4 个 `tu.get` 调用都过；`tu.pop` 漏了同样守卫

**为什么本次才暴露**：`no_lora_adapter` 在 stage1 路径（不开 LoRA）从来没被任何上游写入 `data.batch`。PPO step 1 第一次摸到这条路径 = 第一次触发。`engine_workers.py` 共 7 处 `tu.pop` 调用都有同样脆弱性（`disable_auto_offload`/`mini_batch_size`/`num_mini_batch`/`epochs`/`seed`/`dataloader_kwargs`/`no_lora_adapter`），全靠 `default=` fallback，全会踩同样坑。

**修复（W2.25，verl-fork shim #3）**：[verl/utils/tensordict_utils.py:737](verl-BryanChen408/verl/utils/tensordict_utils.py:737) `tu.pop` 加守卫：

```python
def pop(tensordict, key, default=None):
    """..."""
    if key not in tensordict:        # ← 新加 4 行，行为对齐 tu.get
        return default
    _sentinel = object()
    output = tensordict.pop(key, _sentinel)
    ...
```

**为什么改 verl-fork 而非 rllm**（W2.8 / W2.14 同样模式）：
- bug 在 verl-fork 内部代码路径，rllm 这边不能控制 dispatcher 传什么类型
- rllm `agent_sdk_engine` 走的就是 verl 标准 dispatcher 路径，传 DataProto 是合理的
- 在 verl-fork 加 1 个 compat fix 既不动 rllm，也兼容上游真传 TensorDict 的场景（`key in tensordict` 对 TensorDict 一样 work）

**累计 verl-fork compat 文件**：

| Commit | 文件 | 作用 |
|---|---|---|
| `881a98d7` (W1.8) | `agent_loop.py` shim | `AsyncLLMServerManager` 别名 + `AgentLoopManager` 3 个 property |
| `85159408` (W2.8) | `megatron_workers.py` + `fsdp_workers.py` | re-export `engine_workers` |
| `370e1148` (W2.25) | `tensordict_utils.py:pop` | 缺失 key 守卫 |

**Audit 教训第 17 条**：**verl-fork API drift 不是一次性事件，是持续地暴露**。每次 stage1 链路推进一步，可能解锁下一处 verl 内部代码的 corner case。处理模式：(1) traceback 在 verl/ 内部 → 在 verl-fork compat 分支加 shim 不动 rllm；(2) 否 → 修 rllm。本次第一次在 `utils/` 加 shim（W1.8/W2.8 都在 `workers/`），说明 drift 已经渗到工具层。

未来如果还遇到 `data.batch.keys()` 报错相关 + verl 内部代码，第一时间怀疑同款 DataProto vs TensorDict 类型混淆；`tu.get / tu.pop` 类的 helper 函数都要审一遍 sentinel/default 处理是否对齐。

stage1 此后 verl-fork 用 `qwen36-rllm-compat` 分支 HEAD `370e1148`。

### 13.29 verl-fork shim #4：engine_workers DataProto→TensorDict 入口转换（W2.26）

**症状**：W2.25 修 `tu.pop` 后，同一个 `infer_batch` 又在下面 12 行处挂：

```
File "/workspace/pri/verl/verl/workers/engine_workers.py", line 397, in infer_batch
    if key not in data.keys():
AttributeError: 'DataProto' object has no attribute 'keys'
```

**根因（W2.25 同源 bug 的范围扩大版）**：`TrainingWorker` 的三个 batch 方法 `train_mini_batch` / `train_batch` / `infer_batch` 全都 annotation `data: TensorDict`，但 dispatcher 实际传 DataProto。除了 `tu.pop`（W2.25 已护盘），这三个方法还有其他 TensorDict-only 调用：

| 操作 | 行 | DataProto 是否兼容 |
|---|---|---|
| `data.shape[0]` | infer line 244 类似 | ❌ DataProto 无 .shape |
| `data.keys()` | infer 397, train 343 | ❌ DataProto 无 .keys() |
| `tu.assign_non_tensor(data, ...)` | infer 398, train 344, mini 295 | ❌ asserts TensorDict |
| `tu.make_iterator(data, ...)` | mini 264 | ❌ 内部用 TensorDict.split |

修一行 W2.25 `tu.pop` 不够，整条路径都是 TensorDict 假设。

**为什么 verl 自己跑不踩（深查后的真根因）**：

用户追问"verl 自家 example 走老路径吗？"逼出了**真精确的分歧点**。

`dispatch_lazy_compute_data_proto → BatchData(arg).chunk(...)` 是**类型保持的**（[protocol.py:1271-1289](verl-BryanChen408/verl/protocol.py:1271)）：TensorDict 切片仍是 TensorDict，DataProto 切片仍是 DataProto。dispatcher **不做类型转换**。

差异在 **trainer 层**：

| Trainer | 调 `compute_log_prob` 前 |
|---|---|
| **verl 自家 `RayPPOTrainer._compute_old_log_prob`**（[ray_trainer.py:1212-1226](verl-BryanChen408/verl/trainer/ppo/ray_trainer.py:1212)） | `batch_td = batch.to_tensordict()` + `left_right_2_no_padding(batch_td)` + `tu.assign_non_tensor(batch_td, calculate_entropy=..., compute_loss=False, ...)` → 传 TensorDict |
| **rllm `AgentSdkTrainer`**（[agent_sdk_trainer.py:418](rllm/trainer/verl/agent_sdk_trainer.py:418)） | 直接 `self.actor_rollout_wg.compute_log_prob(batch)`，**0 步桥接** → 传 DataProto |

verl 自家 trainer 的 `# TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free` 注释自己承认这是 mid-migration 的临时桥接代码。**rllm trainer 没同步搬这段桥接**。所以 rllm 链路上 worker 收到 DataProto，verl 自家链路上 worker 收到 TensorDict —— 同一个 `engine_workers.compute_log_prob`，两种入参，verl 自家不踩，rllm 路径踩。

**修复（W2.26，verl-fork shim #4）**：3 个方法**入口加 3 行**类型检查 + 转换：

```python
def train_mini_batch(self, data: TensorDict) -> TensorDict:
    # rllm-compat: dispatcher passes DataProto despite annotation
    if not isinstance(data, TensorDict):
        data = data.to_tensordict()    # protocol.py:1102 native verl API
    ...  # 剩余代码不变
```

`DataProto.to_tensordict()` 是 verl 上游已有的合法 API（[protocol.py:1102](verl-BryanChen408/verl/protocol.py:1102)），把 batch + non_tensor_batch + meta_info 合并成单个 TensorDict。**幂等**（TensorDict 入参直接跳过）。

**修法选择反思（真根因被揭示后）**：

3 种可能修法：

| 修法 | 改动面 | 缺点 |
|---|---|---|
| (A) **worker 入口转换**（W2.26 实施这条） | verl-fork engine_workers 3 方法 × 1 行 | **不补元数据**（compute_loss/no_lora_adapter 全靠 tu.get/tu.pop 的 default 兜底）；与 verl 自家 trainer 行为有语义差（verl 自家会显式设 compute_loss=False，rllm 路径默认 True） |
| (B) **rllm trainer 补桥接**（照搬 verl `_compute_old_log_prob` 3 步） | rllm agent_sdk_trainer.py 多处（compute_log_prob + update_actor 都要补） | 涉及 stage1 不在乎的 metadata 语义决策（calculate_entropy 该 True 还是 False?）；rllm 改动面更大 |
| (C) **dispatcher 端转换**（让 dispatcher 自动 DataProto→TensorDict） | verl-fork decorator 共享代码 | 最干净，但 verl 上游 #2733 part-N 才补，我们超前实施风险高（上游决定与我们不一致时回退困难） |

**选 (A) 理由**：
- "fix in verl-fork, rllm 不动" 沿用 plan §2.1 决策
- 一处修，下游全 callers 受益（不止 rllm）
- 上游 part-N 完成 migration 后，dispatcher 自己会做转换，我们 (A) 的 shim 变 no-op，**可以无副作用删掉**

**(A) 留下的潜在语义偏差**：`compute_loss` 默认 True（rllm 路径），但 verl 自家在 _compute_old_log_prob 里显式 False。stage1 不在乎；如果某条路径后续因这个差异挂，再补 metadata 注入（要么在 rllm trainer 端，要么在 worker 入口注入 default metadata）。

**为什么 W2.25 `tu.pop` 守卫保留**：W2.25 守卫硬化的是 helper 本身（"任何 caller 传缺失 key 都不该 crash"），与 W2.26 入口转换正交。W2.26 让 engine_workers 路径不再触发该守卫，但其他可能的 caller 仍受益。

**累计 verl-fork compat 文件**：

| Commit | 文件 | 作用 |
|---|---|---|
| `881a98d7` (W1.8) | `agent_loop.py` shim | `AsyncLLMServerManager` 别名 + `AgentLoopManager` 3 个 property |
| `85159408` (W2.8) | `megatron_workers.py` + `fsdp_workers.py` | re-export `engine_workers` |
| `370e1148` (W2.25) | `tensordict_utils.py:pop` | 缺失 key 守卫（generic 硬化） |
| `7e30674e` (W2.26) | `engine_workers.py` 3 个 batch 方法 | 入口 DataProto→TensorDict 转换 |

**Audit 教训第 18 条**：**verl PR #2733 是个跨多 part 的渐进式 migration**（DataProto → TensorDict），我们 stage1 恰好踩在中间状态。

**精确根因（用户深查后揭示）**：不是"verl 自家走老路径"（我最初的错猜），而是 **verl 自家 trainer (`RayPPOTrainer._compute_old_log_prob`) 在 worker 调用前显式做了 `batch.to_tensordict() + left_right_2_no_padding + tu.assign_non_tensor` 3 步桥接**（verl 自己注释承认 "TODO: remove after migration done"），**rllm `AgentSdkTrainer` 没同步搬这段桥接代码**。两条链路共享同一个 worker (`engine_workers.compute_log_prob`)，但抵达 worker 的入参类型不同（TensorDict vs DataProto）。

**模式识别**：traceback 内若同一个 worker 方法连续报多个 type-related 错（W2.25 tu.pop assertion → W2.26 keys() AttributeError），**优先 diff verl 自家 trainer 与 rllm trainer 调同一个 worker 方法前的 prep code**，找出"verl 做了 rllm 没做"的桥接步骤。然后选 worker 入口补 / trainer 端补 / dispatcher 补三种修法之一。

**Audit 反思**：W2.20-W2.24 一连串的"独立" bug 看起来散乱，但 W2.25/W2.26 暴露真根因后回看，**全部都是同一个 migration gap 的不同表现** —— rllm trainer 没跟上 verl trainer 加的桥接，每深一层 verl worker 内部代码就触发一种 type 假设违规。**Stage2 应在 rllm 端做一次性的 trainer 桥接补全**（plan §9 stage2 TODO 加这条），与 W3 AgentFlow 迁移合并思考。

stage1 此后 verl-fork 用 `qwen36-rllm-compat` 分支 HEAD `7e30674e`。

**⚠️ W2.27 后续 supersede 此节**：W2.26 worker-entry shim 被 revert（verl-fork `9998be02`），改为在 rllm `AgentSdkTrainer` 加 3 步桥接（更架构正确），见 §13.30。

**⚠️ W2.28 又补一层**：rllm batch.meta_info 漏 `temperature` 字段被 megatron forward_step 撞到，见 §13.31。

### 13.30 rllm 侧补 trainer 桥接 supersede W2.26 worker-entry shim（W2.27）

**触发原因**：用户两轮深查指出我策略错。第一次问"为什么这么多问题"逼出真根因（rllm trainer 漏 verl 自家 trainer 在 mid-#2733 加的 3 步桥接）；第二次问"按 性价比 / 架构发展 是否应该改 rllm 侧"，我重新评估后承认 W2.26 worker-entry shim 是 **side 选错**：

- W2.26 入口转换只覆盖类型问题，**未补 metadata**（`compute_loss` 默认 True，verl 自家 trainer 显式设 False）
- W2.26 fork 债务：上游 PR #2733 part-N 完成后 shim 可能仍要 reconcile
- W2.27 改 rllm 侧 trainer = 既补类型也补 metadata，且与 AgentFlow 迁移方向一致

**实施（W2.27）**：

| 改动 | 仓 | Commit |
|---|---|---|
| `rllm/trainer/verl/agent_sdk_trainer.py` 3 处补 3 步桥接（mirror verl `RayPPOTrainer._compute_old_log_prob/_compute_ref_log_prob/_update_actor`） | rllm | `dd229d51` |
| revert `7e30674e` (W2.26 worker-entry shim) | verl-fork | `9998be02` |

**每处的桥接 = 5 步**（源自 [verl ray_trainer.py:1188-1289](verl-BryanChen408/verl/trainer/ppo/ray_trainer.py:1188)）：

1. `batch_td = batch.to_tensordict()` —— DataProto → TensorDict（verl `protocol.py:1102` native API）
2. `batch_td = left_right_2_no_padding(batch_td)` —— padded → no-padding（必须，因为我们 stage1 设 `use_remove_padding=True`）
3. `tu.assign_non_tensor(batch_td, **metadata)` —— 注入方法特定 metadata：
   - compute_log_prob: `calculate_entropy=True, calculate_sum_pi_squared=False, compute_loss=False`
   - compute_ref_log_prob: `calculate_entropy=False, compute_loss=False`（+ `no_lora_adapter=True` 若 `ref_in_actor`）
   - update_actor: `calculate_entropy=actor.calculate_entropy OR entropy_coeff!=0, distillation_use_topk=False, global_batch_size/mini_batch_size=ppo_mini_batch_size*rollout.n, epochs=ppo_epochs, seed=data_loader_seed, dataloader_kwargs={shuffle}, compute_loss=True`
4. 调 worker（传 TensorDict）
5. `DataProto.from_tensordict(output)` —— TensorDict → DataProto，让 rllm 下游 `.batch[...]` / `.meta_info["metrics"]` API 仍可用

**W2.27 (β) 范围未做的事**（标 stage2 TODO）：

- **逐字段 `no_padding_2_padding` 输出转换**：verl 自家 `_compute_old_log_prob` 对 log_probs/entropy/sum_pi_squared 各自做 `no_padding_2_padding(field, batch_td)` 还原 padded shape；W2.27 用 `DataProto.from_tensordict` 整体转，**不做逐字段重塑**。如果 rllm 下游 `old_log_prob.batch["entropys"]` 期望 padded shape 但拿到 no-padding，会再挂一次。**未验证**，等 L3 跑出来看
- **routed_experts / teacher_logprobs 处理**：verl 自家有这两个 corner case，stage1 用不到，跳过
- **update_actor 输出整形（`rename_dict("actor/") + from_single_dict`）**：verl 自家做完整重命名 + 重包，rllm 只读 `actor_output.meta_info["metrics"]`，`DataProto.from_tensordict` 应当能填这个字段（NonTensorData → meta_info），**未验证**

**Audit 反思（重审 W2.20-W2.26 整批 commit 后）**：

User 第二轮提的"逐 commit 评估"逼出 4 类 commit 分布：

| 类别 | 例 | 状态 |
|---|---|---|
| Side 选对 + 实施正确 | W2.20/W2.21/W2.23 | ✅ 保留 |
| Side 选对但实施有质量瑕疵 | W2.22 fallback 漏 reward=0 (→W2.24 修)；W2.23 `agent_sdk_engine.py:624` 硬编码未一并修（stage2 cleanup） | ⚠️ 已补 / 进 stage2 |
| Side 选对且通用硬化 | W2.25 `tu.pop` 守卫 | ✅ 保留（rllm 不调，verl 内部 helper 硬化合理在 verl-fork） |
| **Side 选错** | **W2.26 worker-entry shim** | ❌ **W2.27 revert + 移到 rllm** |
| Side 次优但能工作 | W2.8 `megatron_workers/fsdp_workers` re-export shim（rllm 4 个旧 import 都在 `train_agent_ppo.py` 可改区） | ⚠️ **W2.28 候选**，不紧急 |
| 选 verl-fork 是因为 rllm 改动会触碰 `rllm/sdk/*` 不动区 | W1.8 agent_loop shim | ✅ 保留 |

**Audit 教训第 19 条**：side 选择不是"rllm 不动是绝对原则"，而是 case-by-case 评估：
- 改 rllm 是否触碰 `rllm/sdk/*` 子系统？ → 是 → verl-fork shim
- 改 rllm 是否只在 `rllm/trainer/verl/` / `examples/` 等可改区？ → 是 → 优先 rllm 侧
- 是 verl 内部 generic 助手的 bug？ → verl-fork
- 是 rllm 缺了 verl 自家 trainer 已做的桥接？ → **rllm 侧（不要图省事做 worker-entry shim）**

**Audit 教训第 20 条**：**用户两轮追问（"为什么这么多问题" + "逐 commit 评估"）改变了我对策略的认识**。下次实施 shim 前先问"verl 自家 trainer 在调这个 worker 前做了什么？rllm 是不是没做？"，避免无脑选 worker-entry shim。

### 13.31 `batch.meta_info["temperature"]` 漏设（W2.28，W2.27 extension）

**症状**：W2.27 bridge 落地后 L3 重跑推进到 megatron forward_step，挂在：

```
File "verl/workers/engine/megatron/transformer_impl.py", line 841, in forward_step
    temperature = batch["temperature"]
KeyError: 'key "temperature" not found in TensorDict with keys [\'attention_mask\', ..., \'metrics\', ..., \'use_remove_padding\']'
```

TensorDict 里 43 个 key 列出来了，**没 `temperature`**。

**源码验证根因**（[verl ray_trainer.py:1394-1395](verl-BryanChen408/verl/trainer/ppo/ray_trainer.py:1394) + [transformer_impl.py:841](verl-BryanChen408/verl/workers/engine/megatron/transformer_impl.py:841)）：

| 位置 | 谁 set 谁 read | rllm 是否做 |
|---|---|---|
| verl `fit:1394-1395` | `batch = DataProto.from_single_dict(...); batch.meta_info["temperature"] = rollout_config.temperature` —— **每 ppo iter batch 构造后立刻设** | ❌ rllm `transform_results_for_verl` 不做 |
| verl `_update_actor:1252-1253` | defensively 再设 | ✅ W2.27 做了，但 update_actor 是 line 534 调，**在 compute_log_prob (line 418) 之后** |
| megatron `forward_step:841` | `temperature = batch["temperature"]` **无 default fallback** | — |

W2.27 在 update_actor bridge 内 set 了，但 **compute_log_prob 时还没设**，megatron forward_step 第一次进入就 KeyError。

**修复（W2.28）**：在 `agent_sdk_trainer.py` 全局只设一次，紧贴 `batch.meta_info["global_token_num"]` 现有那一行后：

```python
# mirror verl fit:1394-1395
_rollout_cfg = self.config.actor_rollout_ref.rollout
batch.meta_info["temperature"] = _rollout_cfg.temperature
batch.meta_info["multi_turn"] = (
    _rollout_cfg.multi_turn.enable if hasattr(_rollout_cfg, "multi_turn") else False
)
```

并删 W2.27 update_actor bridge 里那两行重复 set（全局已设，dead write）。

**关于 multi_turn**：grep verl 内部代码无 reader（只在 ray_trainer.py:1251 set），但为对齐 verl pattern 一并设（防御未来上游加 reader）。

**Audit 教训第 21 条**：**porting 多阶段 trainer 桥接代码时，要把上游 fit-loop 层的 metadata 注入也一起 port**，不仅是每个 bridge 方法内的 metadata。W2.27 只 mirror 了 3 个 bridge 方法，漏掉 verl fit 主循环的 temperature 全局 set —— 这种"在外层 set 一次、所有下游受益"的 pattern 容易在按方法 mirror 时被忽略。

**Audit 反思（流程层）**：连续 W2.25/26/27/28 都是"verl 自家做了 rllm 没做"的同源 bug。每次只看 traceback 直接报错那个变量，看不到完整的"verl fit loop 还做了什么、rllm fit_agent 没做"的对比。**下次进 stage1 cleanup 时应做一次完整的 verl `fit` ↔ rllm `fit_agent` 全文 diff**，把 verl 在 fit 里所有 `batch.meta_info[...]` set / `batch.non_tensor_batch[...]` set / `batch.batch[...]` modify 的位置列清单，逐个检查 rllm 等价位置是否做。**stage2 TODO 加这条"做一次性 fit-loop diff 审计"**。

W2.28 后预期挂点：进入 megatron forward 计算本身（如 mp / shape / dtype 问题），就是真算法层了。

### 13.32 `use_remove_padding=True` → GDN packed seq 不支持（W2.29）

**症状**：W2.28 推进后 L3 跑到 megatron forward + Qwen3-VL text_model + decoder + self_attention，挂在：

```
File "/workspace/MindSpeed/mindspeed/core/ssm/gated_delta_net.py", line 292
NotImplementedError: GDN does not support packed sequence for now.
```

**源码验证完整因果链**（**不再有"推断"环节**）：

| 阶段 | 代码 | 行为 |
|---|---|---|
| Stage1 train script | `actor_rollout_ref.model.use_remove_padding=True` | 配置 |
| verl engine config 注入 | `engine_workers.py:113` `self.engine_config.use_remove_padding = self.model_config.get("use_remove_padding", False)` | 拷到 engine config |
| verl 决定 data_format | `transformer_impl.py:908` `data_format = "thd" if use_remove_padding else "bshd"` | True → thd |
| verl 调 preprocess | `model_forward.py:74-80` `if data_format == "thd": packed_seq_params = preprocess_packed_seqs(...)` | 构造 packed_seq_params |
| 传给 Qwen3-VL.forward | `model_forward.py:96` `input_args = dict(packed_seq_params=packed_seq_params, ...)` | model 收到非 None packed_seq_params |
| Qwen3VL → text_model → decoder → transformer_layer → self_attention (GDN) | call chain | 路径深，每层都透传 |
| **MindSpeed GDN raise** | `gated_delta_net.py:292` `if packed_seq_params is not None: raise NotImplementedError("GDN does not support packed sequence for now.")` | 撞墙 |

**根因**：plan §0.1 早预警 —— Qwen3.5/3.6 用 Gated Delta Net (GDN) linear attention，MindSpeed 的 GDN 实现**不支持 THD packed**。stage1 起步注释自己写了 "可调，挂了再切 False"。**L3 跑通就是实测决定的时刻** —— 必须 False。

**修复（W2.29，2 行）**：

| 文件 | 改动 |
|---|---|
| `train_openhands_qwen36_npu.sh:346` | `actor_rollout_ref.model.use_remove_padding=True` → `=False` |
| `train_openhands_qwen36_npu.sh:380` | `actor_rollout_ref.actor.megatron.use_remove_padding=True` → `=False` |

**W2.27 bridge 不动**：源码读 verl `_compute_old_log_prob:1217` 是无条件调 `left_right_2_no_padding`，verl NPU 脚本 use_remove_padding=False 跑通，说明这两个组合 **配合工作**（engine 的 `data_format` 由 use_remove_padding 决定，bridge 的 nested 转换是独立步骤）。

**Audit 教训第 22 条**：**plan §0.1 "软约束"应在 L3 第一次失败时强制重审**，而不是后续每次崩才单点修。W2.20 进入 L3 阶段时就该把 §0.1 所有 "可调/软约束" 一次性 review：
- `use_remove_padding` ← W2.29 才修
- `use_dynamic_bsz` ← W2.10 已对齐 verl 套（False）
- 其他还有没有？需做完整 §0 / §0.1 / §0.2 软约束清单 vs train script 实际值的对账

**Stage2 TODO 加这条 "§0.1 软约束实测后状态回填" 工作**：把所有 "可调 / 起步" 标记的 config，在 L3-L4 跑通后回到 §0.1 表 + train script 注释里换成"已实测 = X"。

W2.29 后预期挂点：真进 megatron 内部数值层（shape/mp/dtype），剩下要不就是 stage1 setup 闭环成功。

### 13.33 W2.27 bridge β-deferred 输出后处理（W2.30）

**症状**：W2.29 修 GDN 后 L3 推进到 advantage 计算阶段，挂在：

```
File "rllm/trainer/verl/agent_sdk_trainer.py:533, in fit_agent
    batch = self._remove_padding(batch)
File "rllm/trainer/verl/agent_sdk_trainer.py:890, in _remove_padding
    batch = batch.select_idxs(non_pad_step_indices)
File "verl/protocol.py:662, in select_idxs
    source={key: tensor[idxs_torch] for ...}
NotImplementedError: aten.index.Tensor
```

**源码验证完整因果链**：

| 阶段 | 代码 | 行为 |
|---|---|---|
| W2.27 bridge step 2 | `left_right_2_no_padding(batch_td)` | `input_ids` / `position_ids` 变 nested tensor |
| W2.27 bridge step 4 | `compute_log_prob(batch_td)` | worker 用 nested 输入算 log_probs, entropy |
| W2.27 bridge step 5 | `DataProto.from_tensordict(old_log_prob_td)` 一刀切 | 保留 nested tensor 在 DataProto.batch |
| rllm line 458 | `batch = batch.union(old_log_prob)` | nested tensor 进入 batch |
| rllm line 533 / 890 | `batch.select_idxs(non_pad_step_indices)` → `tensor[idxs_torch]` | 对 nested tensor index |
| torch nested_tensor.py:359 | `raise NotImplementedError(func)` | `aten.index.Tensor` 没 dispatch |

**这是 plan §13.30 W2.27 末段"β-deferred follow-up #1"明确预测过的挂点**（"逐字段 `no_padding_2_padding` 输出转换没做 ... 若 rllm 下游 `old_log_prob.batch["entropys"]` 期望 padded shape 但拿到 no-padding shape，跑挂时回头补"）—— **现在补**。

**修复（W2.30）**：照搬 verl `_compute_old_log_prob:1227-1246` / `_compute_ref_log_prob:1203-1208` / `_update_actor:1283-1287` 的输出后处理。

```python
# verl _compute_old_log_prob:1227-1246 mirror
_entropy = tu.get(old_log_prob_td, "entropy")
_log_probs = tu.get(old_log_prob_td, "log_probs")
_entropy = no_padding_2_padding(_entropy, batch_td)       # nested → padded
_log_probs = no_padding_2_padding(_log_probs, batch_td)
_result = {"old_log_probs": _log_probs.float(), "entropys": _entropy.float()}  # rename！
old_log_prob = DataProto.from_tensordict(tu.get_tensordict(_result))
```

**Key rename 关键**（grep 验证）：
- worker 输出 `log_probs` / `entropy` (singular)
- rllm `agent_sdk_trainer.py:454` 读 `old_log_prob.batch["entropys"]` (plural)
- 必须 rename，否则下游 KeyError

三个 bridge 都补了：

| bridge | extract | no_padding_2_padding | rename / wrap |
|---|---|---|---|
| compute_log_prob | `entropy`, `log_probs` | 两个 tensor | → `old_log_probs`, `entropys` |
| compute_ref_log_prob | `log_probs` | 一个 tensor | → `ref_log_prob` |
| update_actor | `metrics` (NonTensorData) | n/a | `rename_dict("actor/")` + `perf/mfu/actor` key swap + `from_single_dict(data={}, meta_info={"metrics": ...})` |

**Audit 教训第 23 条**：**包装 verl worker call 的 bridge，必须 mirror verl 自家的输入 prep + 输出 post-processing 两端**。W2.27 只 mirror 了输入 prep（to_tensordict + left_right_2_no_padding + assign_non_tensor），输出端图省事用了 `from_tensordict` 一刀切，留下 nested tensor 污染 batch 的隐 bug。预测有此问题（β-deferred），但没立刻补。下次 mirror 任何 worker bridge：**input prep + output post-processing 是一对，不可分割**，宁可一次性多写 20 行也别留 deferred 项。

stage2 TODO 的 "W2.27 β-deferred follow-up #1" 已 ✅ done（本次 W2.30）。

---

## 14. 新对话接手指南（Onboarding）

**给在新对话窗口接手的 Claude / 工程师**：先读 §14.1（30 秒 TL;DR），然后按需深入 §14.2-§14.9。其余 §0-§13 是 history / reference，按需检索。

### 14.1 TL;DR — 30 秒回到上下文

**目标**：Qwen3.6-35B-A3B × OpenHands × Ascend NPU 跑 agentic RL（算子生成 agent）。
**基础设施**：rllm（agent RL framework）跑在 verl-BryanChen408 fork（Ascend 适配的 verl）+ Megatron-Bridge + vllm-ascend + Megatron-LM。
**项目阶段**：W1 全完成（cherry-pick + 适配），W2 进行中（stage1 训练 bring-up）。
**当前位置**：W2.22 切换 stage1 策略：`MAX_ITERATIONS=1`（锁 prompt 首轮 ~30k 永不超 49k）+ rollout fallback reward `random.random()`（防 GRPO std=0 NaN）。L3 等待此 fix 后重跑验证 trainer 链路。
**已知开放项**：stage1 走"轻链路验证"路线 — agent 单 turn 不解题、reward 噪声 → 训练信号是 mock 的，目的只是让 PPO step 跑起来。**真训练**需要 OpenHands condenser（长 context 压缩）+ 多 turn agent，下放 W3/W4。

### 14.2 Git refs cheatsheet

**rllm 仓库**（`/Users/yeji/Documents/Code/Python/Qwen36/rllm`）：

```
当前分支: qwen36-openhands-stage1
基线:    origin/openhands_ascend
push 到:  origin/qwen36-openhands-stage1
```

| 类别 | 数量 | 关键 commit |
|---|---|---|
| W1 obs cherry-pick | 2 | `c6e36b0a` + `b2538771` (基线之上前两条) |
| W1 upstream SAFE | 7 | `029ee36f` ~ `74721f13` |
| W1 upstream HIGH_RISK | 3 | `3c8ee08d` (R2/R3) `80d4e7b8` (GPU mem) `ca560dc7` (lr_schedule) |
| W1 RolloutCorrectionConfig 修复 | 1 | `c1ff82e3` |
| W1 plan v2.2/v2.3/v2.4 | 3 | `8d8b1fe6` `3818073b` `76c0cff4` |
| W2 stage1 训练脚本 | 1 | `5082c854` |
| W2 分层测试 | 1 | `b07a55d7` |
| W2 vllm/megatron/HCCL/script 修复（W2.9-W2.18） | 9 | 见 `git log` |
| 用户人工调整（路径） | 多 | `feb85471` `fdb8e236` 及之后人工 rebase |
| W2.20 DooD workspace path alignment | 1 | `bedbb1b0` (hardcode `/home/docker/openhands_workspace` + fail-fast precheck) |
| W2.20 precheck 放宽（findmnt -T） | 1 | `15686e0e` |
| W2.21 max_model_len 32k→49152 unblock L3 | 1 | `e5e6c9ce` |
| W2.21 §14 onboarding 同步 | 1 | `87f03325` |
| W2.22 MAX_ITERATIONS=1 + random fallback reward | 1 | `2a2fdbb7` |
| W2.23 `data.max_prompt_length` 8192→32768 + 诊断工具 | 1 | `4f941da2` |
| W2.24 random fallback 条件覆盖修正 | 1 | `90690d55` |
| W2.24 fallback removal markers（doc-only） | 1 | `bf0943cf` |
| W2.25 verl-fork shim #3 tu.pop（在 verl-fork 仓） | 0 | （rllm 仅 plan 更新；verl-fork `370e1148`） |
| W2.26 verl-fork shim #4 入口 DataProto→TensorDict | 0 | （rllm 仅 plan 更新；verl-fork `7e30674e`，**W2.27 revert**） |
| W2.27 rllm AgentSdkTrainer 补 3 处桥接 + revert W2.26 | 1 | rllm `dd229d51` + verl-fork `9998be02` |
| W2.28 rllm 全局补 `batch.meta_info["temperature"]`（megatron forward_step 必读） | 1 | rllm `5eb4224f` |
| W2.29 train script `use_remove_padding=False` (GDN 不支持 packed) | 1 | rllm `81cb31f0` |
| W2.30 W2.27 bridge 补输出后处理（no_padding_2_padding + key rename + from_single_dict） | 1 | rllm `c4efcc07` |
| 最新 | — | rllm `c4efcc07` + verl-fork `9998be02` |

**verl-BryanChen408 仓库**（`/Users/yeji/Documents/Code/Python/Qwen36/verl-BryanChen408`）：

```
当前分支: qwen36-rllm-compat
基线:    hostoom-debug
push 到:  origin/qwen36-rllm-compat
```

| Commit | 内容 |
|---|---|
| `881a98d7` | shim 1: `AsyncLLMServerManager` 别名 + `AgentLoopManager` 3 个 property |
| `85159408` | shim 2: `megatron_workers.py` + `fsdp_workers.py` re-export `engine_workers` |
| `370e1148` | shim 3: `tu.pop` 缺失 key 守卫（W2.25） |
| `7e30674e` | shim 4: engine_workers 3 batch 方法入口 DataProto→TensorDict 转换（W2.26） — ⚠️ **W2.27 revert（commit `9998be02`），改在 rllm 侧 trainer 补桥接** |
| `9998be02` | revert W2.26 |

### 14.3 NPU 节点环境（已知路径）

| | 本地 Mac | NPU 节点 |
|---|---|---|
| rllm 工作目录 | `/Users/yeji/Documents/Code/Python/Qwen36/rllm` | `/workspace/rllm-071` |
| verl-BryanChen408 | `/Users/yeji/Documents/Code/Python/Qwen36/verl-BryanChen408` | `/workspace/pri/verl` |
| Megatron-Bridge | `/Users/yeji/Documents/Code/Python/Qwen36/Megatron-Bridge` | `/workspace/Megatron-Bridge` |
| Megatron-LM | `/Users/yeji/Documents/Code/Python/Qwen36/Megatron-LM` | (需确认实际路径) |
| MindSpeed | `/Users/yeji/Documents/Code/Python/Qwen36/MindSpeed` | (需确认实际路径) |
| vllm-ascend | `/Users/yeji/Documents/Code/Python/Qwen36/vllm-ascend` | `/workspace/vllm-ascend` |
| Qwen3.6 模型 | n/a | `/home/docker/Qwen3.6-35B-A3B` |
| trace SQLite | n/a | `/workspace/results/rllm-openhands-traces.db` |
| OpenHands artifacts | n/a | `/workspace/results/openhands_results` |

**NPU 节点 IP**：节点上 vllm server 实际监听 `80.48.5.88:39557`（多 NIC 机器，88 是可达 IP；65 是 ray actor 启动 IP）。**不要硬编码任一 IP 进脚本**（W2.15 教训）。

### 14.4 关键代码文件清单

**stage1 入口（必读）**：
- `examples/openhands_sdk/train_openhands_qwen36_npu.{py,sh}` — 训练入口
- `examples/openhands_sdk/stage1_test_layered.sh` — 分层 smoke orchestrator (L1/L2a/L2b/L3/L4)

**stage1 辅助**：
- `examples/openhands_sdk/preflight_qwen36_npu.py` — L1 隔离 (config + import + dataset + ctor)
- `examples/openhands_sdk/mock_rollout.py` — L2b mock rollout（返回 float reward，**不调 LLM**，天然不能验 PPO step）
- `examples/openhands_sdk/mock_llm_server.py` — L2a FastAPI mock OpenAI server
- `examples/openhands_sdk/prepare_npu_operator_data.py` — 4-source 数据准备脚本（mock/local-jsonl/local-json/hf）

**stage1 业务代码**（rllm 老分支已有，stage1 不动）：
- `examples/openhands_sdk/openhands_agent.py` — 真 rollout 函数（末尾 `return reward` 是 float，docstring 撒谎说 list[dict]，已被坑过）
- `examples/openhands_sdk/workspace/` — OpenHands docker 容器内 entrypoint

**verl-fork shim**（不用碰，pull 即可）：
- `verl/experimental/agent_loop/agent_loop.py` — 加 `AsyncLLMServerManager` 别名 + `AgentLoopManager` 3 property
- `verl/workers/megatron_workers.py` / `verl/workers/fsdp_workers.py` — re-export `engine_workers`
- `verl/trainer/ppo/ray_trainer.py:902` — `_server_manager` kwarg wire

### 14.5 当前进度 + 下一步

**已完成**：
- ✓ W1 全部 (cherry-pick + verl-fork shim + RolloutCorrectionConfig 修复)
- ✓ W2.0 分层测试基础设施
- ✓ W2.1-W2.2 训练脚本 + OpenHands LLM 配置
- ✓ W2.8-W2.18 持续修复 setup 链路（HCCL/Megatron-Bridge/vllm/mock_rollout）
- ✓ W2.19 跳 L2b 决策（mock 天然边界，setup 已验完）
- ✓ L1/L2a/L2b（部分） — 全部走过 setup chain
- ✓ W2.20 DooD workspace path alignment（hardcode `/home/docker/openhands_workspace` + fail-fast precheck + findmnt -T）
- ✓ W2.21 max_model_len 32k→49152 解决 OpenHands 30k 重 prompt 首轮越界 1 token
- ✓ W2.22 改换策略：MAX_ITERATIONS=1 + rollout fallback reward random.random()（stage1 只验 trainer 链路、不在乎 reward 真假）
- ✓ W2.23 `data.max_prompt_length` 8192→32768 解决 `agent_sdk_engine.py:563` step 过滤把所有 trace step 全 drop 的问题；过程中诊断完 DB 链路完全干净（trace_store / session_uid / data vs metadata 字段对齐都验过）
- ✓ W2.24 W2.22 random fallback 漏洞修正：reward==0 也走 fallback（之前只在 exception 触发，正常返回 0 直接覆盖；加 `if real_reward > 0.0` 条件覆盖）
- ✓ W2.25 verl-fork shim #3 `tu.pop` 缺失 key 守卫：前 5 层全闭环后挂在 verl 内部 `infer_batch → DataProto.pop(str)`，4 行修在 verl-fork 不动 rllm
- ✓ W2.26 verl-fork shim #4 engine_workers 3 batch 方法入口 DataProto→TensorDict 转换：W2.25 后 `data.keys()` 又挂，根因同源（mid-migration），修方法入口 1 行/方法 × 3 方法，下游零改动 ⚠️ **被 W2.27 supersede**
- ✓ W2.27 rllm 侧补 trainer 桥接（3 处 mirror verl `_compute_old_log_prob/_compute_ref_log_prob/_update_actor`）+ revert W2.26 verl-fork shim：side 修正
- ✓ W2.28 rllm 全局补 `batch.meta_info["temperature"]` + `multi_turn`，mirror verl `fit:1395`。megatron `forward_step:841` 无 default 读 temperature 撞 KeyError。删 W2.27 update_actor bridge 里重复 set
- ✓ W2.29 `use_remove_padding` True→False：GDN packed seq 不支持 (MindSpeed `gated_delta_net.py:292`)。完整 5 层源码验证因果链，对齐 plan §0.1 早期警告 + verl NPU 脚本默认
- ✓ W2.30 W2.27 bridge β-deferred 输出后处理：worker 输出 nested tensor 经 from_tensordict 留在 batch，select_idxs 撞 `NotImplementedError(aten.index.Tensor)`。3 bridge 补 no_padding_2_padding + key rename + from_single_dict

**当前位置**：W2.30 完成 —— W2.27 bridge 输出端补 mirror verl 自家逐字段 `no_padding_2_padding` + key rename + from_single_dict 后处理。3 个 bridge（compute_log_prob/compute_ref_log_prob/update_actor）全部对齐 verl 自家行为。**β-deferred 全部完结**。**剩余未验证项**：(a) 还可能有 verl fit loop 漏 port 的 metadata（W2.28 audit #21）；(b) Megatron 内部 mp/shape/dtype；(c) NPU KV cache OOM。verl-fork compat 仍 3 处有效。

**下一步（用户该做的）**：

```bash
# 本地 push
git -C /Users/yeji/Documents/Code/Python/Qwen36/rllm push origin qwen36-openhands-stage1
# verl-fork 上次 push 之后没动，HEAD 仍是 85159408

# NPU 节点拉新（main container 内）
cd /workspace/rllm-071 && git pull   # 到 e5e6c9ce

# 宿主侧（main container 之外）确认 bind mount + 目录都 OK
mkdir -p /home/docker/openhands_workspace
# main container 启动命令需带：-v /home/docker/openhands_workspace:/home/docker/openhands_workspace
# （或挂父目录 -v /home/docker:/home/docker）

# 跑 L3（脚本顶部 precheck 会先验 bind mount 状态）
cd /workspace/rllm-071
bash examples/openhands_sdk/stage1_test_layered.sh L3
```

**W2.22 后预期 L3 挂点**（按风险递减）：
1. NPU KV cache OOM（49k context 比 32k 多 ~50% 显存；`gpu_memory_utilization=0.6` 应该够，挂的话日志在 `/tmp/ray/session_latest/logs/worker-*.err`）
2. tool call schema 在 30k 长 prompt 下解析错乱（Qwen3-coder parser 边界）
3. OpenHands 容器内 entrypoint 业务异常（reward 计算 / eval 脚本挂；现在 rollout fallback 兜底，应该不至于让 episode 飞掉）
4. **PPO step 真挂**（最值得期待 — 意味着 setup chain 全通了）

**已解除的挂点**（W2.20/W2.21/W2.22 修过，不该再出现）：
- ~~docker image 不存在 / build 失败~~（用户已 build 过 + 用过）
- ~~容器内调 LiteLLM proxy 不通~~（W2.20 DooD 修了路径，proxy 端 host.docker.internal + `--add-host host-gateway` 已生效）
- ~~OpenHands 容器内 entrypoint.py 不存在~~（W2.20 修了 workspace_temp）
- ~~vllm context length 越界~~（W2.21 升 49152 + W2.22 MAX_ITERATIONS=1 锁 prompt ~30k）
- ~~GRPO std=0 NaN~~（W2.22 random fallback 兜底 reward variance）
- ~~episode 全 drop（trace store 空）~~（MAX_ITERATIONS=1 仍产生 1 次真 LLM call，trace store 有数据）

### 14.6 已知 limitations / 历史踩坑（避免重复踩）

**Mock vs 真链路**：
- `mock_rollout` 只返 float reward，**不调 LLM → trace store 空 → PPO step 没法跑**（plan §13.22）。要验 PPO step 必须走 L3 真 rollout。
- `mock_llm_server` 与 mock_rollout 互不调用，前者只为 L2a curl smoke 用。

**verl × vllm-ascend 不兼容点**：
- vllm-ascend 不识别 `--swap-space` / `--cpu-offload-gb` / `--enable-prefix-caching` 等 vllm 主线新参数（W2.17）
- 还可能不识别：`--logprobs_mode processed_logprobs` / `--enable_sleep_mode`（vllm 0.10+ 引入，未实测，如挂同样删）

**verl `setdefault` 暗坑**：
- `vllm_async_server.py:237` 写死 `compilation_config.setdefault("cudagraph_mode", "FULL_AND_PIECEWISE")` → 强制开 cudagraph
- `enforce_eager=True` 会与之矛盾，vllm-ascend 直接 exit 2（W2.16）
- 审 verl 类似 setdefault：`grep -rn "setdefault" verl/workers/`

**两个独立的 "bridge" 库**（plan §13.12）：
- `mbridge`（pypi 0.15.1）— 不支持 qwen3_5_moe，**NPU 不能用**
- `Megatron-Bridge`（NVIDIA-NeMo，src clone）— 你环境装的这个，NPU 必须 `vanilla_mbridge=False` 走这条
- 默认 PYTHONPATH 注入：`MEGATRON_BRIDGE_DIR=/workspace/Megatron-Bridge`（可 env override）

**rllm 数据 schema 陷阱**（plan §13.16）：
- parquet 的 `prompt` 字段**必须是 `list[dict]`**，不是 JSON string
- 已修 3 个数据脚本（prepare + 2 个 legacy mock），未来加新脚本注意

**rllm × verl API 漂移**（plan §13.6 §13.11）：
- verl-BryanChen408 fork = verl main 最新；rllm 仍假设 verl 0.7.x
- 已知 2 处 shim：rollout 抽象（`AsyncLLMServerManager` → `LLMServerClient`）+ worker 抽象（`megatron_workers/fsdp_workers` → `engine_workers`）
- 未来 verl-fork 升级可能暴露第 3 处漂移，按 W2.8/W2.14 同模式处理：fork 端 shim，rllm 不动

**机器特定信息不进代码**（plan §13.18，audit 教训第 7 条）：
- `HCCL_IF_IP` / `nic_name` / 节点 IP 一律不允许硬编码
- 现在全是 env opt-in：`HCCL_IF_IP_OVERRIDE` / `HCCL_NIC_NAME` / `HCCL_FORCE_PORT_RANGE`

**DooD volume path alignment**（plan §13.23，audit 教训第 12 条）：
- rllm 主进程跑在 main container 内，`openhands_agent` 用 docker SDK 起子容器；`-v <src>:<dst>` 的 `<src>` **由宿主 dockerd 解析**，不是 main container 视角
- 任何要给子容器看的目录（典型：`workspace_temp/`）必须落在 main container 和宿主**同路径 bind mount** 的位置
- 当前 hardcode `/home/docker/openhands_workspace`（[openhands_agent.py:225](rllm/examples/openhands_sdk/openhands_agent.py:225)），main container 启动加 `-v /home/docker/openhands_workspace:/home/docker/openhands_workspace`（或挂父目录 `-v /home/docker:/home/docker`）
- 训练脚本顶部 fail-fast precheck（[train_openhands_qwen36_npu.sh:163+](rllm/examples/openhands_sdk/train_openhands_qwen36_npu.sh:163)）：dir 不存在/不可写直接 `exit 1`；soft warn 不是 bind mount（用 `findmnt -T` 走 mount tree，能识别父目录 bind mount）
- 排错：在**宿主 shell**（不是 main container 内）`ls /home/docker/openhands_workspace/`，能看到 `trajectory-*` 子目录才合法

**OpenHands 重 system prompt → 长 context 压力**（plan §13.24，audit 教训第 13 条）：
- Qwen3-coder agent setup 第一轮 LLM call 就 30k tokens（system prompt + 全部 tools 定义 + AGENTS.md + INSTRUCTIONS.md），不是 bug，是 OpenHands × Qwen3-coder 的常态
- W2.21 把 `max_model_len` 升到 49152 给 16k buffer 解决 dry-step 越界 1 token，但多 turn 长跑会线性增长（每 turn 加 ~16k tool result），**49k 不是长跑容量**
- W2.22 用 `MAX_ITERATIONS=1` 锁单 turn 绕开此问题（stage1 只验链路），不是结构解
- 结构性解法在 OpenHands `Condenser`（LLMSummarizing / Recent / NoOp 几种），或精简 system prompt / 减 tools，**stage1 不动**
- W3 长跑前必须监控 episode 长度 P95：超 40k 升 65536，超 60k 必须上 condenser
- L2a `mock_llm_server` 当前没模拟"prompt > max_model_len 的边界"，下一轮 mock 设计要补

**stage1 unblock 策略：MAX_ITERATIONS=1 + random fallback reward**（plan §13.25 + §13.27，audit 教训第 14 + 16 条）：
- stage1 不在乎 agent 真解题，只在乎 trainer 链路（rollout → trace → batch → PPO step → checkpoint）跑起来
- `OPENHANDS_MAX_ITERATIONS=1` 锁单 turn，prompt 永远 ~30k，trace store 有真 LLM 数据 → PPO batch 能构造
- rollout fallback reward = `random.random()`：**两种情况都生效**（W2.24 修正）：
  - (a) exception 路径（容器 crash / reward 函数 raise）
  - (b) `_npu_operator_reward` 返回 0.0（"no impl file"）→ `if real_reward > 0.0` 条件不通过 → 保留 random
- 真 reward > 0 时自动接管真值（W3 transition 零摩擦），但 **W3 验收完后必须物理删除 fallback 代码块**（用户决策：random 在真实训练不合理）
- 代码侧已加 4 个清晰的 removal markers：
  - [openhands_agent.py:717](rllm/examples/openhands_sdk/openhands_agent.py:717) 前的 ⚠️ block + after-code 内联
  - log message 全部 `STAGE1_MOCK` prefix（grep 容易）
  - §9 stage2 TODO 🔴 高优先级条目跟踪
  - removal criteria 4 条同时满足才删（condenser + MAX_ITER 拉回 + reward 真稳定 > 0 + variance 来自真信号）
- **训练信号是 mock 的**：stage1 纯噪声 advantage，不要期待 reward 曲线上升 / 模型学到东西
- 不加 `STAGE1_MOCK_REWARD` env gate（已决策"env 太多"，且 fallback 删除时机 ≠ runtime toggle，env 反而误导）

**`data.max_prompt_length` vs OpenHands 真 prompt 量级**（plan §13.26，audit 教训第 15 条）：
- OpenHands 首轮 prompt 30k+，stage1 默认 `data.max_prompt_length=32768`（W2.23 从 8192 升）必须 ≥ 实际 prompt
- `agent_sdk_engine.py:563` 用此值过滤 step → 过低则所有 step 被 skip → trajectory 无 valid step → episode drop → pad_sequence 空
- 诊断 trick：**调试数据流问题先 grep WARNING**，"Skipping step..." / "Trajectory has no valid steps after filtering overlong prompts" 这些被 info 级别淹没
- 旁注：`agent_sdk_engine.py:624` 硬编码 `max_prompt_length = 16384` 做 padding/truncation 是 debug 残留（"[DEBUG format]" 打印泄露），对 stage1 反而是 KV cache 保护 → stage2 cleanup
- 诊断工具：`examples/openhands_sdk/diagnose_trace_store.py` — 一键检查 DB 完整性 + session_uid 对齐 + slug 解码，未来 trace store 问题第一时间用

**Stage1 安全训练 config**（plan §13.7）：
- `router_replay=disabled`（stage1 AgentPPOTrainer 路径上 19983fe4 入口不生效，§13.3）
- `use_kl_loss=False` + `kl_loss_coef=0` — 靠 PPO clip 控制 off-policy
- **必须** `rollout.calculate_log_probs=True`（监控 `rollout_probs_diff` / `pg_clipfrac` / `approx_kl`）

**Batch sanity 联立约束**（plan §13.14）：
- 默认 `BATCH_SIZE=1 ROLLOUT_N=4 PPO_MINI_BATCH_SIZE=${BATCH_SIZE}`
- 联立：`BATCH × ROLLOUT >= DP=4` 且 `BATCH >= MINI`
- 脚本顶部 fail-fast 验过

### 14.7 常见用户 prompt → 该怎么处理

| 用户说 | Claude 应该做 |
|---|---|
| "贴 traceback" | 先看 traceback 末尾 file + line；查 §13.6-§13.22 找类似挂点；如果是 verl × vllm-ascend × Megatron 接口问题，**先怀疑 "verl 自己能跑 vs rllm 跑"差异**（这是 W2.15-W2.17 反复出现的模式） |
| "脚本里又改了 X，对不对" | 看 §0.2 verl 脚本 NPU 分支权威表 + §13.13 audit 完整表 + §13.7 stage1 safe config |
| "新加 Y 数据集" | 参考 §13.15 `prepare_npu_operator_data.py`，加 `--source <name>` 分支；schema 见 §13.16（prompt 必须 list[dict]） |
| "L3 挂在 Z" | 看 §13.22 L3 prep checklist + 可能挂点表；常见诊断 `docker logs` / `/tmp/ray/session_latest/logs/worker-*.err` |
| "PPO 不收敛" | 看 §13.7 监控阈值表，调 lr / mini_batch（先不开 KL loss） |
| "想加 OpenHands 新功能" | 不动 `rllm/sdk/*`（fork 债务，§2.1）；改 `examples/openhands_sdk/` 业务代码 |
| "改并行" | 看 §0.2 verl 脚本 + §13.13 用户决策（TP=2 EP=4 单节点 8 卡）；§13.14 联立约束必须满足 |

### 14.8 快速 reference — 哪节看哪个

| 想知道 | 看哪节 |
|---|---|
| 项目目标 / 不做什么 | §1 / §10 |
| 硬件 / 依赖版本 | §0 |
| 训练参数权威（verl 脚本 NPU 分支完整摘录） | §0.2 |
| 基线分支决策 / cherry-pick 列表 | §2 / §3.1 / §13.1 |
| stage1 安全训练 config | §13.7 |
| Batch sanity 约束 | §13.14 |
| 分层测试架构（L1-L4） | §13.10 |
| 全 W1+W2 实施记录 | §13.1-§13.22 |
| verl-fork shim 详情 | §13.6 + §13.11 |
| 已知 audit 教训（11 条） | §13.13/.14/.16/.17/.18/.19/.20/.21/.22 散布 |
| NPU smoke 命令 | §13.8 |
| L3 prep checklist | §13.22 末段 |

### 14.9 plan 维护规则

- **每个新挂点修复都加 §13.x 子节** + audit 教训累加 + 修订表新 row
- **每次 sh/py 改动同步反映在 §13.13 或对应章节**（避免代码与文档漂移）
- **commit message 引用 W2.x 编号**，task list 维持同步
- **plan 永远基于 verl 自己能跑 Qwen3.6 这个事实**，rllm 这边的 bug 优先怀疑"rllm 引入的差异"（W2.15-W2.17 共同模式）
