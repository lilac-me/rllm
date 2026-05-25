# rLLM Upstream 差异与 Qwen3.6-35B Agent 训练迁移分析

分析时间：2026-05-23

本报告基于本地私仓 `lilac-me/rllm`、官方仓库 `rllm-org/rllm` 抓取到的 `upstream/*` refs，以及当前生产目标：

- 在私仓历史分支 `origin/openhands-observability` 的基础上继续 OpenHands/KernelGym/观测能力开发。
- 面向 Qwen3.6-35B-A3B 级别模型启动 agent RL 训练。
- 已经全面更新或准备更新 `verl`、Megatron、vLLM。
- 需要判断当前代码离官方主线多远、社区是否已有可复用改动、以及怎样迁移风险最低。

## 1. 当前仓库状态

本地当前检出：

```text
branch: main
tracking: origin/main
remote: git@github.com:lilac-me/rllm.git
worktree: clean
```

官方上游已抓取为本地远端 refs：

```text
upstream/main = https://github.com/rllm-org/rllm.git main
```

关键提交：

| Ref | Commit | 时间 | 说明 |
| --- | --- | --- | --- |
| `upstream/main` | `24eca95876809aba296acce96b91490c2e367ebf` | 2026-05-21 21:47:26 -0700 | Add a work OpenSearch-VL using awesome rllm to the project list (#593) |
| `origin/main` | `7a48826ff743872b3463faf50d888a33313f8639` | 2026-04-17 16:46:20 +0800 | add fully_async examples |
| `origin/openhands-observability` | `095dd6399aac8d861c98a0b9c2219f77e17a6874` | 2026-05-13 06:37:35 +0000 | add gitignore |

## 2. 分叉距离结论

### 2.1 私仓 main 与官方 main

```text
merge-base(origin/main, upstream/main) = ea623cc317fd202e55989b164582d44cea3313af
merge-base commit time = 2026-04-16 21:34:50 -0700
merge-base subject = chore(scripts): simplify and pin megatron dependencies (#498)

origin/main...upstream/main = 1 / 64
diff = 777 files changed, 34461 insertions(+), 57457 deletions(-)
```

解释：

- `origin/main` 只比官方 `upstream/main` 多 1 个私有提交：`7a48826f add fully_async examples`。
- 官方 `upstream/main` 已经比私仓 `origin/main` 多 64 个提交。
- 这 64 个提交中包含多个对训练稳定性、gateway、AgentFlow、Qwen3.5 parser、MoE router replay、multi-turn merge、verl backend 重构的关键改动。

### 2.2 openhands-observability 与官方 main

```text
merge-base(origin/openhands-observability, upstream/main) = 6e32e614716af032e8b6c7ae77e7847750e67f41
merge-base commit time = 2026-03-22 22:08:47 -0500
merge-base subject = feat: add cross-episode Store for sharing state across workflow instances (#453)

origin/openhands-observability...upstream/main = 111 / 108
diff = 833 files changed, 38980 insertions(+), 56994 deletions(-)
```

解释：

- `origin/openhands-observability` 不是简单“落后官方 main”，而是已经和官方主线双向分叉。
- 私有侧有 111 个提交，主要集中在 OpenHands SDK、KernelGym、观测 dashboard、Megatron/verl 训练脚本、pass@k、debug、NPU/Ascend 等。
- 官方侧有 108 个提交，包含一次较大的架构演进：从 legacy Agent/Environment/AgentExecutionEngine 逐步转向 AgentFlow/gateway/统一 eval/train。

### 2.3 openhands-observability 与私仓 main

```text
merge-base(origin/openhands-observability, origin/main) = 6e32e614716af032e8b6c7ae77e7847750e67f41
origin/openhands-observability...origin/main = 111 / 45
diff = 154 files changed, 7160 insertions(+), 1317 deletions(-)
```

解释：

- `openhands-observability` 也没有吸收私仓 `origin/main` 后来的 45 个提交。
- 这些 45 个提交里已经有 `verl 0.7.1`、Megatron 依赖、fully async、agentcore 等基础更新。
- 因此当前真实迁移不是一段线性 rebase，而是“三方整理”：`upstream/main`、`origin/main`、`origin/openhands-observability`。

## 3. 官方 main 的关键变化

官方 `upstream/main` 最近 64 个提交中，和当前生产目标最相关的是这些方向。

### 3.1 AgentFlow/gateway 成为主路径

代表提交：

```text
6d5c2c59 [feat] Drop the legacy Agent + Environment + AgentExecutionEngine stack; replace with AgentFlow cookbooks (#526)
8662bc4e [feat] Unified Task / Runner / Harness — AgentFlow eval stack refactor (#514)
36250e61 feat(eval): unify eval and training behind AgentFlowEngine + gateway (#564)
ecf19e73 refactor: move agent_flow_engine.py out of experimental, rename to agentflow_engine.py (#565)
fa66ef14 feat(train, eval): unify train + eval on AgentFlowEngine for harbor/sandbox (#588)
```

影响：

- 旧的 `rllm.sdk` 和 legacy agent/environment 路径被弱化或移除。
- 官方主线训练和评估逐步统一到 `AgentFlowEngine`、gateway、task/runner/harness。
- 你的 `examples/openhands_sdk` 和 `rllm/environments/kernelgym` 仍有大量旧路径依赖，直接 merge 会产生结构性冲突。

迁移判断：

- 不建议继续在旧 `AgentExecutionEngine` 上堆新生产能力。
- OpenHands/KernelGym 应优先迁移成新 AgentFlow/gateway 消费方式。
- 如果短期只需要跑通生产任务，可以保留私有旧路径，但要隔离在独立 cookbook/example 下，不要污染官方主路径。

### 3.2 verl 后端训练稳定性显著增强

代表提交：

```text
68e9107e fix(verl): pad multi-turn batch to lcm of dp_size and ppo mini-batch (#506)
45c99f1c fix(verl): zero response_mask on padded rows to keep loss magnitude stable (#508)
d09f6155 Fix rollout_probs_diff masking to respect response_mask in Verl trainers (#503)
88f409c4 [feat] Fix training collapse for AgentFlow with verl backend (#511)
326445bd [feat] Collapse multi-turn trajectory steps into a single row (verl) + unified merge metrics (#512)
44fb9a3c fix(verl): raise on aborted rollouts and normalize stop_reason at source (#542)
d314745e fix(verl): release GPU memory on Verl backend interrupts (#568)
e66fc2e4 fix(verl): route rllm.algorithm.lr_schedule to the active backend's optim key (#585)
```

影响：

- 这些改动对 agent RL 非常关键，尤其是 multi-turn、长上下文、rollout/logprob 对齐、padding/mask/loss 稳定性。
- 你的私有分支里有早期 `verl 0.7.1` 兼容和 `rollout_logprobs`、completion ids 长度修复，但官方主线已经修复了更多通用问题。

迁移判断：

- 训练主路径应基于官方 `upstream/main` 的 verl backend，而不是把私有 `origin/openhands-observability` 的旧 verl trainer 整体保留。
- 私有分支中的 `train_id`、`eval_tag`、pass@k、observer 透传、NPU/HCCL/GLOO env 透传、特定 debug 能力，应作为补丁迁到新主路径。

### 3.3 Qwen3.5、Qwen3-VL、MoE router replay 相关支持

代表提交：

```text
e5ba77ea fix(parser): Qwen3.5 chat template support + simple_math example (#504)
1a5c8703 fix(verl): make Qwen3-VL workflow path actually see images (#549)
5d6c222d fix(verl+geo3k): make Qwen3-VL geo3k cookbook actually train (#548)
19983fe4 feat(verl): support R2 and R3 MoE router replay (#543)
```

影响：

- Qwen3.6-35B-A3B 与 Qwen3.5/Qwen3 MoE 路径相近，但官方 `main` 没有直接提供完整 Qwen3.6-35B-A3B 训练配方。
- `R2/R3 MoE router replay` 对 MoE 模型 rollout/training 对齐非常重要。
- Qwen3.5 chat template support 是 Qwen3.6 适配时必须关注的基础能力。

迁移判断：

- 官方 `main` 是 Qwen3.6 适配的必要基础，但不是完整答案。
- Qwen3.6-35B-A3B 的完整运行配方主要在 `upstream/swe`，需要单独评估和迁移。

### 3.4 gateway 稳定性与 trace store

代表提交：

```text
94663217 feat(gateway): clean up sessions and improve trace store robustness (#500)
a6cc59aa fix(gateway): Retry TCP connections if connection failure happens because pooled TCP connections go stale (#505)
b896e1f9 fix(gateway): explicit trace-store selection; memory default; validation (#590)
5f541e4d fix(gateway): disable client keepalive to avoid idle-connection race with uvicorn (#592)
```

影响：

- 你的 OpenHands observability 场景依赖稳定 trace、session、observer/gateway 行为。
- 官方 gateway 的稳定性修复值得优先吸收。
- 私有 observer gateway 与官方 model gateway 功能有重叠，需要明确边界。

迁移判断：

- 官方 `rllm-model-gateway` 应作为 LLM token/logprob/trace 的基础组件。
- 私有 `examples/openhands_sdk/observer_gateway.py` 应定位为 OpenHands 事件观测层，而不是重复实现模型调用链路。

### 3.5 配置与依赖变化

官方 `upstream/main` 的 `pyproject.toml` 关键版本：

```toml
verl==0.7.1
vllm==0.17.0
torch>=2.10.0
torchvision>=0.23.0
transformers>=4.55.0,<5.0.0
flash-attn==2.8.1
```

私有 `origin/openhands-observability` 的 `pyproject.toml` 关键版本：

```toml
verl==0.7.1
vllm>=0.10.2,<=0.12.0
torch>=2.8.0
flash-attn>=2.8.1
litellm[proxy]>=1.80.0,<1.82.7
rllm.experimental.cli.main:cli
```

影响：

- 私有分支依赖已经明显落后于官方 main。
- 但官方 main 的 `vllm==0.17.0` 对 Qwen3.6-35B-A3B 未必是最终生产栈，`upstream/swe` 已经显式使用 vLLM 0.18.0。

迁移判断：

- 通用 rLLM 代码应先对齐官方 main。
- Qwen3.6-35B-A3B 生产环境应参考 `upstream/swe` 的窄版本栈，而不是盲目使用主线 `pyproject.toml` 默认。

## 4. 私仓 openhands-observability 的生产价值

`origin/openhands-observability` 的私有改动主要集中在以下方面。

### 4.1 OpenHands SDK 与观测能力

主要文件：

```text
examples/openhands_sdk/openhands_agent.py
examples/openhands_sdk/observer_gateway.py
examples/openhands_sdk/observer_cli.py
examples/openhands_sdk/persistence.py
examples/openhands_sdk/dashboard/*
examples/openhands_sdk/workspace/*
tests/openhands-sdk/test_openhands_agent.py
```

价值：

- 这是当前私仓最强的生产差异化资产。
- 包含 agent workspace、observer gateway、事件持久化、dashboard、分布式锁、rootfs/nsjail 等。
- 对 OpenHands 类 agent 训练和调试有直接价值。

风险：

- 与官方主线的 AgentFlow/gateway 重构重叠。
- 若直接 merge，可能会同时维护两套 session/trace/event 抽象。

建议：

- 保留 OpenHands 业务层，但重接官方 `AgentFlowEngine` 和 `rllm-model-gateway`。
- 将 observer 定义为“OpenHands runtime events”观测，不要让它承担模型 gateway 的 token/logprob 责任。

### 4.2 KernelGym 与 Ascend/NPU 相关能力

主要文件：

```text
examples/kernelgym/*
rllm/environments/kernelgym/*
rllm/agents/kernelgym_agent.py
datasets/kernelbench/*
hf_dataset/kernelbench/*
examples/swe/train_deepswe_30b_npu.sh
```

价值：

- 这是面向算子生成、KernelBench、Ascend/NPU 的私有生产资产。
- 与 Qwen3.6-35B agent 训练不是完全相同任务，但训练框架、rollout、验证、pass@k、观测能力可复用。

风险：

- 官方主线已删除或重构不少 legacy env/agent 路径。
- KernelGym 需要迁到新 task/runner/harness 或保持 cookbook 隔离。

建议：

- 先不把 KernelGym 混入核心 rLLM engine。
- 将其迁为 `cookbooks/kernelgym` 或保留在 `examples/kernelgym`，通过新 AgentFlow adapter 进入训练。

### 4.3 私有 verl/Megatron/trainer 改动

相关文件：

```text
rllm/trainer/verl/*
rllm/experimental/verl/*
rllm/experimental/fully_async/*
tmp_folder_for_debug/megatron_workers.py
examples/fully_async/train_fully_async_mega.sh
examples/openhands_sdk/train_open_megatron.sh
examples/kernelgym/train_kernelgym_megatron*.sh
```

价值：

- 包含大量你们为了跑通 Megatron、rollout logprobs、bypass rollout、debug checkpoint、OOM、tokenizer mismatch 做的修复。
- 对生产调试仍有价值，尤其是 `train_id`、`eval_tag`、pass@k、token id 对齐、HCCL/GLOO env 透传。

风险：

- 官方 main 后来已经重构 verl backend，很多私有改动可能已被上游覆盖或接口过时。
- 直接保留私有 trainer 可能会绕开官方最新修复。

建议：

- 逐提交审查，不要整体搬运。
- 优先迁移“生产观测字段”和“环境透传/评估指标”，不要迁移旧 trainer 主流程。

## 5. upstream/swe 分支分析

`upstream/swe` 是当前官方远端里最接近你目标场景的非主线分支。

分支状态：

```text
upstream/swe = f341760029113f90ea83a604ee10db57fa1368d5
commit time = 2026-05-23 07:13:47 +0800
subject = docs: add SWE status report
upstream/main...upstream/swe = 4 / 28
```

说明：

- `upstream/swe` 比 `upstream/main` 更新，包含 28 个独有提交。
- 它不是主线，不能假设 API 稳定。
- 但它明确包含 Qwen3.6-35B-A3B + Megatron + vLLM 0.18 的运行配方。

### 5.1 Qwen3.6-35B-A3B Megatron 配方

关键文件：

```text
cookbooks/swe/swe/training_scripts/run_swe_training_35b_a3b_megatron.sh
cookbooks/swe/launchers/arnold_entrypoint_swe_qwen36_35b_a3b_megatron.sh
cookbooks/swe/launchers/launch_swe_qwen36_35b_a3b_megatron_b200.yaml
cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
```

关键参数：

```text
MODEL_REPO=Qwen/Qwen3.6-35B-A3B
MODEL_CACHE_DIR=/tmp/hf_cache/hub/models--Qwen--Qwen3.6-35B-A3B
NNODES=4
NGPUS_PER_NODE=8
ACTOR_TP=2
ACTOR_CP=2
ACTOR_PP=1
ACTOR_EP=1
ROLLOUT_TP=2
ROUTER_REPLAY=R3
ROLLOUT_DISTRIBUTED_EXECUTOR_BACKEND=uni
MODEL_USE_REMOVE_PADDING=true
```

训练侧配置重点：

```text
actor_rollout_ref.actor.strategy=megatron
actor_rollout_ref.actor.megatron.use_mbridge=true
actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
actor_rollout_ref.actor.megatron.sequence_parallel=true
actor_rollout_ref.actor.megatron.use_distributed_optimizer=true
actor_rollout_ref.actor.megatron.use_dist_checkpointing=true
actor_rollout_ref.actor.megatron.param_offload=true
actor_rollout_ref.actor.megatron.grad_offload=true
actor_rollout_ref.actor.megatron.optimizer_offload=true
```

rollout 侧配置重点：

```text
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
actor_rollout_ref.rollout.max_model_len=32768
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.rollout.calculate_log_probs=true
actor_rollout_ref.rollout.n=8
actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=qwen3_coder
actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni
```

MoE/rollout correction 重点：

```text
algorithm.rollout_correction.bypass_mode=true
algorithm.rollout_correction.rollout_is=null
algorithm.router_replay=R3
rllm.algorithm.router_replay=R3
```

判断：

- 这是目前最接近“Qwen3.6-35B-A3B agent 训练”的社区代码。
- 其中 SWE task 业务不一定要迁移，但 Megatron/vLLM/verl 栈、router replay、端口规避、checkpoint、runtime env 很有价值。

### 5.2 vLLM 0.18 / torch 2.10 / transformers 5.3 窄栈

`setup_verl_vllm018_qwen35.sh` 明确说明：

```text
torch 2.10.0 + CUDA 12.9
vLLM 0.18.0
transformers 5.3.0
Megatron-Core 0.17.0
Megatron Bridge 0.4.0
mbridge 0.15.1
flash-linear-attention 0.4.1
flash-attn 2.8.3
TransformerEngine release_v2.12
```

判断：

- 官方 `main` 的 `vllm==0.17.0` 未必覆盖 Qwen3.6-35B-A3B 的完整生产需求。
- 如果你的生产栈已经“全面更新 verl、Megatron、vLLM”，需要对齐 `upstream/swe` 这种窄版本组合，而不是只看 rLLM 主线 pyproject。
- `transformers 5.3` 和 Qwen3.6 tokenizer/template 行为要重点验证，尤其是 tool call、reasoning parser、MTP、MoE routing。

### 5.3 upstream/swe 对 rLLM 核心的增量 patch

相对 `upstream/main`，`upstream/swe` 在核心路径上的增量约：

```text
rllm-model-gateway/src/rllm_model_gateway/server.py       +21
rllm-model-gateway/src/rllm_model_gateway/session_router.py +3
rllm/experimental/common/transform.py                      +5
rllm/experimental/unified_trainer.py                       +29
rllm/experimental/verl/patch.py                            +156
rllm/experimental/verl/utils.py                            +166
rllm/experimental/verl/verl_backend.py                     +24
rllm/parser/chat_template_parser.py                        +13
rllm/trainer/verl/agent_workflow_trainer.py                +22
rllm/trainer/verl/ray_runtime_env.py                       +38
```

核心价值：

- `RLLM_VLLM_PORT_BASE` / `RLLM_VLLM_PORT_STRIDE`：为每个 colocated vLLM async server 分配 deterministic port range，规避多 replica 同时启动时 TCPStore/端口竞争。
- `RLLM_RAY_MASTER_PORT_RANGE`：给 Ray worker group master port 指定范围，规避 ephemeral port race。
- HDFS checkpoint resume/load/save。
- HuggingFace checkpoint upload。
- gateway 暴露 vLLM token fields。
- Qwen/Megatron 相关 hotfix，包括 MTP export 和 checkpoint JSON serialization。

判断：

- 这些 patch 和生产大规模多节点 Qwen3.6 训练高度相关。
- 但它们混在 SWE cookbook 和 Arnold/HDFS/Modal/B200 环境里，不能无脑整分支合并。
- 推荐“按功能最小迁移”，优先迁移端口、router replay、checkpoint、runtime env、Qwen3.6 config。

### 5.4 upstream/swe 与私仓生产场景的关系

相同点：

- 都是 agent 训练，不是单纯 SFT。
- 都需要长上下文、多轮、工具调用、rollout/logprob 对齐。
- 都依赖 Megatron + vLLM + verl。
- 都面向 30B/35B 级别模型。
- 都需要强观测、失败恢复、checkpoint、并发控制。

不同点：

- `upstream/swe` 以 SWE/Arnold/Modal/HDFS/B200 为中心。
- 私仓生产以 OpenHands/KernelGym/Ascend/NPU/观测 dashboard 为中心。
- `upstream/swe` 不直接解决 OpenHands runtime event 观测，也不直接解决 Ascend/NPU。

建议：

- 把 `upstream/swe` 作为 Qwen3.6-35B-A3B Megatron 训练参考实现。
- 不要把 SWE 业务代码作为 OpenHands 生产基础。
- 把其中通用训练栈抽出来，接到私仓 OpenHands/KernelGym AgentFlow。

## 6. 其他值得保持开放关注的官方分支

用户明确希望不要完全排除其他开放分支。以下分支不一定立刻合入，但建议保留关注窗口。

### 6.1 `upstream/feat/verl-fully-async-mode`

状态：

```text
upstream/main...upstream/feat/verl-fully-async-mode = 15 / 1
unique commit = 6362a8cb feat(verl): fully-async separated mode for VerlBackend
```

相关文件：

```text
rllm/experimental/config/rllm/backend/verl.yaml
rllm/experimental/unified_trainer.py
rllm/experimental/verl/async_agent_loop.py
rllm/experimental/verl/utils.py
rllm/experimental/verl/verl_backend.py
rllm/experimental/verl/verl_launcher.py
```

价值：

- 如果 Qwen3.6-35B agent 训练需要更强 rollout/train 分离，fully-async separated mode 值得评估。
- 当前私仓也有 `examples/fully_async/*`，但很可能已经落后于官方最新思路。

风险：

- 不是 main，接口可能不稳定。
- 与 `upstream/swe` 的 Megatron 配方可能存在重复或冲突。

建议：

- 作为第二优先级观察。
- 若主线 `upstream/main` + `upstream/swe` 的 async rollout 不满足吞吐，再单独验证。

### 6.2 `upstream/feat/async-trainer-reliability`

状态：

```text
upstream/main...upstream/feat/async-trainer-reliability = 5 / 1
unique commit = 870d9a7b Improve async trainer reliability
```

相关文件：

```text
rllm/experimental/buffer.py
rllm/experimental/common/transform.py
rllm/experimental/sync_coordinator.py
rllm/experimental/unified_trainer.py
tests/experimental/test_async_training_hardening.py
```

价值：

- 大规模 agent 训练常见失败点是 async buffer、同步协调、异常恢复。
- 这个分支可能会补足 `upstream/main` 的训练可靠性。

建议：

- 不急于合入，但应在跑长作业前看是否已进入 main。
- 若未进入 main，可单独 cherry-pick 测试。

### 6.3 `upstream/verl-zmq-jobid-backport`

状态：

```text
upstream/main...upstream/verl-zmq-jobid-backport = 22 / 2
unique commits:
ea84ce79 feat(verl/patch): backport volcengine/verl#6246 (job-id-keyed colocated IPC)
79e499e7 fix(verl): release GPU memory on Verl backend interrupts
```

价值：

- 多 vLLM/verl colocated IPC 场景可能遇到 ZMQ/job-id 混淆。
- 对多节点、多 replica 的 Qwen3.6 训练有潜在价值。

建议：

- 与 `upstream/swe` 的 port range patch 一起评估。
- 如果生产中出现 IPC 混线、ZMQ、rollout worker 错连，应优先看该分支。

### 6.4 `upstream/fix/verl-collapse`

状态：

```text
upstream/main...upstream/fix/verl-collapse = 50 / 1
unique commit = de641a70 update attention mask in verl to use length instead of checking for non-padding tokens
```

价值：

- 与训练 collapse、attention mask、padding token 判断有关。
- 对 tokenizer/pad id 特殊的 Qwen 系模型有潜在价值。

建议：

- 不直接合入，先确认 main 是否已通过其他方式解决。
- 如果出现 loss collapse 或 padding mask 异常，单独验证。

### 6.5 `upstream/gateway-refactor`

状态：

```text
upstream/main...upstream/gateway-refactor = 66 / 12
```

价值：

- 包含 normalized request/response、parser backend、multimodal gateway 等改动。
- 对未来 OpenHands observer/gateway 边界设计有参考价值。

风险：

- 明显是 WIP/refactor 分支。
- 与官方 main 已合入部分 gateway 改动有重叠。

建议：

- 作为设计参考，不作为当前生产基线。

### 6.6 `upstream/dev-harbor` 与 `upstream/feat-train-harbor-sandbox`

价值：

- remote sandbox、Harbor runtime、train/eval 统一、cloudflared tunnel、opencode/mini-swe-agent/oracle。
- 如果 OpenHands 未来要接远程 sandbox/runtime，这些分支有参考意义。

建议：

- 当前 Qwen3.6-35B 训练主线不依赖 Harbor。
- 但 OpenHands/远程 sandbox 观测长期可能需要借鉴。

## 7. 推荐迁移路线

### 7.1 不推荐的路线

不建议在 `origin/openhands-observability` 上直接执行：

```bash
git merge upstream/main
```

原因：

- 分叉过大：`111 / 108`。
- 官方删除/迁移了大量 legacy agent/environment/sdk 路径。
- 私仓也有大量 trainer/engine/gateway 修改，冲突会混在一起，难以判断保留哪边。
- 即使解决冲突，也容易得到一个“能 import 但训练语义错乱”的混合树。

### 7.2 推荐路线：以官方 main 为新基线，迁移私有能力

建议创建新分支：

```bash
git -C rllm checkout -b qwen36-agent-upstream upstream/main
```

迁移顺序：

```text
1. 基线：upstream/main
2. 迁移私有 OpenHands/KernelGym 独立文件
3. 迁移 upstream/swe 中 Qwen3.6/Megatron/vLLM 最小必要文件和 patch
4. 迁移私有 trainer 观测字段和生产指标
5. 重接 OpenHands/KernelGym 到 AgentFlow/gateway
6. 小规模 smoke test
7. 多节点短跑
8. 长跑和恢复/评估验证
```

### 7.3 私有代码迁移优先级

P0：必须迁移

```text
examples/openhands_sdk/openhands_agent.py
examples/openhands_sdk/observer_gateway.py
examples/openhands_sdk/persistence.py
examples/openhands_sdk/workspace/*
examples/openhands_sdk/train_openhands.py
examples/openhands_sdk/train_open_megatron.py
tests/openhands-sdk/*
```

P1：强烈建议迁移

```text
scripts/analyze_pass_at_k.py
scripts/eval_pass_at_k.py
scripts/run_pass_at_k_eval.sh
train_id / eval_tag / pass@k / observer metadata 相关改动
HCCL / GLOO / NCCL / NPU env 透传
rollout_logprobs 和 token id 对齐相关 debug 逻辑
```

P2：按任务需要迁移

```text
examples/kernelgym/*
rllm/environments/kernelgym/*
rllm/agents/kernelgym_agent.py
datasets/kernelbench/*
hf_dataset/kernelbench/*
examples/math_tool/*megatron*
```

P3：只作参考或归档

```text
tmp_folder_for_debug/megatron_workers.py
memory_compact.txt
results/pass_at_k/summary.json
零散 debug checkpoint/log commit
```

### 7.4 upstream/swe 迁移优先级

P0：Qwen3.6-35B-A3B 训练必须参考

```text
cookbooks/swe/swe/training_scripts/run_swe_training_35b_a3b_megatron.sh
cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
```

P1：大规模多节点稳定性

```text
rllm/experimental/verl/patch.py 中 RLLM_VLLM_PORT_BASE patch
rllm/experimental/verl/utils.py 中 RLLM_RAY_MASTER_PORT_RANGE / checkpoint 逻辑
rllm/trainer/verl/ray_runtime_env.py 中 runtime env 传播
rllm-model-gateway token fields 暴露相关改动
```

P2：如果使用 Arnold/HDFS/B200

```text
cookbooks/swe/launchers/arnold_entrypoint_swe_qwen36_35b_a3b_megatron.sh
cookbooks/swe/launchers/launch_swe_qwen36_35b_a3b_megatron_b200.yaml
cookbooks/swe/launchers/swe_artifact_utils.sh
```

P3：暂不迁移

```text
SWE task/evaluator/Modal/SWE-bench 业务代码
cookbooks/swe/external/*
```

除非 OpenHands 生产目标转向 SWE-Bench，否则这些业务代码不是当前主线。

## 8. Qwen3.6-35B-A3B 适配风险清单

### 8.1 模型与 tokenizer/parser

风险：

- Qwen3.6 与 Qwen3.5/Qwen3 的 chat template、tool call parser、reasoning 输出格式可能存在差异。
- vLLM parser 需要确认 `qwen3_coder` 是否适配你的 agent 工具调用格式。
- 官方 main 有 Qwen3.5 parser 支持，但没有完整 Qwen3.6 主线配方。

验证：

```text
1. 单请求 OpenAI-compatible chat completion。
2. 带 tool call 的多轮请求。
3. gateway strip_vllm_fields=false 时 token_ids/logprobs 是否完整。
4. 训练侧 tokenizer encode/decode 与 rollout completion ids 是否一致。
```

### 8.2 MoE router replay

风险：

- Qwen3.6-35B-A3B 是 MoE/A3B 路径，rollout/training router 不一致可能导致 off-policy 或 logprob mismatch。
- `upstream/swe` 默认 `ROUTER_REPLAY=R3`，这说明社区已经把 router replay 当作生产默认。

验证：

```text
1. R3 默认跑 smoke。
2. 对比 disabled/R2/R3 下 rollout_probs_diff、loss、reward。
3. 确认 router replay metrics 是否上报。
```

### 8.3 Megatron CP/TP/PP/EP

风险：

- `ACTOR_CP=2` 依赖 Megatron Gated DeltaNet THD support 和 remove-padding。
- Megatron-Core、Megatron Bridge、mbridge 版本不匹配会导致 checkpoint/export/transformer_config 问题。
- `upstream/swe` 中存在 Megatron CP2 overlay 和 MTP export hotfix，说明这部分仍不完全稳定。

验证：

```text
1. Megatron import smoke。
2. HF -> Megatron load。
3. 一步训练。
4. checkpoint save/load。
5. Megatron -> HF export 或推理恢复。
```

### 8.4 vLLM async rollout 多节点端口竞争

风险：

- 多 replica 同时启动 vLLM V1 multiprocess executor 时，TCPStore 端口可能 race。
- Ray worker group master port 也可能与 ephemeral port 冲突。

建议默认环境变量：

```bash
export RLLM_VLLM_PORT_BASE=46000
export RLLM_VLLM_PORT_STRIDE=100
export RLLM_RAY_MASTER_PORT_RANGE=25000:25100
```

前提：

- 需要迁移 `upstream/swe` 中相关 patch，或确认你使用的 verl/vLLM 已上游解决。

### 8.5 长上下文与 loss/mask

风险：

- Qwen3.6 agent 训练会产生长 prompt、长 response、多工具调用。
- padding、response_mask、multi-turn merge 错误会直接导致 loss 异常。

必须吸收官方 main 修复：

```text
68e9107e pad multi-turn batch
45c99f1c zero response_mask on padded rows
d09f6155 rollout_probs_diff masking
326445bd collapse multi-turn trajectory steps
e8539db5 don't truncate merged multi-turn responses
```

## 9. 建议的阶段性执行计划

### 阶段 A：建立新基线

```bash
git -C rllm checkout -b qwen36-agent-upstream upstream/main
```

目标：

- 不修改 `origin/openhands-observability`。
- 保持旧生产分支可回退。
- 新分支只承载迁移后的生产路线。

### 阶段 B：最小迁移 OpenHands

目标：

- 先迁移 `examples/openhands_sdk` 独立文件。
- 不急于迁移旧 trainer 修改。
- 让 OpenHands agent 能在新 rLLM import、启动、产生日志。

验收：

```text
pytest examples/openhands_sdk/workspace/tests
pytest tests/openhands-sdk/test_openhands_agent.py
```

### 阶段 C：接入官方 AgentFlow/gateway

目标：

- 明确 OpenHands runtime event observer 与 rLLM model gateway 的边界。
- 将训练入口从旧 trainer/engine 迁到新 AgentFlow/gateway。

验收：

```text
单样本 rollout 能生成 Episode/Trajectory/Step。
gateway 能记录 token_ids/logprobs/messages。
observer 能记录 OpenHands workspace/tool/event。
```

### 阶段 D：迁移 Qwen3.6/Megatron 栈

目标：

- 从 `upstream/swe` 迁移 Qwen3.6-35B-A3B 训练参数和必要 patch。
- 先跑非 OpenHands 的最小 agent flow，再接 OpenHands。

验收：

```text
1 GPU/1 node import smoke。
1 node 8 GPU Megatron load + rollout smoke。
4 node 32 GPU 一步训练。
checkpoint save/load。
validation before train。
```

### 阶段 E：恢复私有生产观测能力

目标：

- 迁移 `train_id`、`eval_tag`、pass@k、bypass debug、token mismatch debug。
- 迁移 HCCL/GLOO/NCCL env 透传。
- 确保 dashboard/observer 与新训练主路径一致。

验收：

```text
训练日志可按 train_id/eval_tag 聚合。
pass@k 可离线分析。
失败 episode 可追溯到 OpenHands event 和 gateway trace。
```

## 10. 最终建议

推荐生产基线：

```text
base = upstream/main
plus = selected private OpenHands/KernelGym assets
plus = selected upstream/swe Qwen3.6/Megatron/vLLM patches
plus = selected private observability/training metadata patches
```

不推荐：

```text
base = origin/openhands-observability
then merge upstream/main
```

核心理由：

- 官方主线已经完成较大架构迁移，训练稳定性修复集中在新路径。
- 私有分支的生产价值很高，但主要是业务层和观测层，不适合作为训练基础设施基线继续老化。
- `upstream/swe` 已经提供 Qwen3.6-35B-A3B + Megatron 的近似生产参考，但它是非主线，应按功能摘取。
- 当前最佳策略是“新基线 + 选择性迁移”，不是“大合并”。

## 11. 复现本分析用到的关键命令

```bash
git -C rllm fetch https://github.com/rllm-org/rllm.git '+refs/heads/*:refs/remotes/upstream/*'

git -C rllm status --short --branch
git -C rllm remote -v
git -C rllm branch -vv --all

git -C rllm merge-base origin/main upstream/main
git -C rllm rev-list --left-right --count origin/main...upstream/main
git -C rllm diff --shortstat origin/main...upstream/main

git -C rllm merge-base origin/openhands-observability upstream/main
git -C rllm rev-list --left-right --count origin/openhands-observability...upstream/main
git -C rllm diff --shortstat origin/openhands-observability...upstream/main

git -C rllm merge-base origin/openhands-observability origin/main
git -C rllm rev-list --left-right --count origin/openhands-observability...origin/main
git -C rllm diff --shortstat origin/openhands-observability...origin/main

git -C rllm log --oneline origin/main..upstream/main
git -C rllm log --oneline upstream/main..origin/openhands-observability

git -C rllm for-each-ref --format='%(refname:short) %(objectname:short) %(committerdate:iso8601) %(subject)' refs/remotes/upstream
git -C rllm diff --name-status upstream/main...upstream/swe
git -C rllm show upstream/swe:cookbooks/swe/swe/training_scripts/run_swe_training_35b_a3b_megatron.sh
```
