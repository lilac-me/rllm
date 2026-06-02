# Triton Skills 移植 × 可信 Reward —— 冲突识别与改动方案

> 状态：**待 review 的方案草案**。对象 = 我移植进 OpenHands 的 **triton-\* skills**（cannbot 最新版，PR #205 那套）。
> 目标 = 在不破坏 agent 生成/优化能力的前提下，让 **reward 的算分不再由 agent 掌握**。
>
> **引用出处**（自行 spot-check 行号）：
> - triton-\* 技能：`rllm/`（verl-main）`examples/openhands_sdk/workspace/agent_workdir/` 下的 `AGENTS.triton.md` 与 `.agents/skills/triton-op-verifier/SKILL.md`。
> - 旧 kernel-\* 参照：`rllm-merge-tmp/`（openhands-refactoring）同路径下的 `AGENTS.md`、`tools/operator_pipeline.sh`、`.agents/skills/kernel-verifier/SKILL.md`。
> - reward / 训练机制：`rllm/examples/openhands_sdk/openhands_agent.py`、`rllm/rllm/trainer/verl/agent_ppo_trainer.py`、`verl-BryanChen408`。
>
> 本文件未跟踪，可随意移动/重命名。

---

## 0. 一句话目标

triton-\* skills 现在把**评测的编排、判定、算分、选最终产物**全放在 agent 手里（`AGENTS.triton.md` Phase 0–5）。对 RL 而言这等于把 reward 的计算权交给了被训练对象。
**改动方向**：把"算 reward 的那次评测"收敛成一个**固定入口 + 固定产物**（照搬旧 kernel-\* 的 `operator_pipeline.sh → metrics.json` 模式），由 **backend 对 agent 提交的实现文件重跑**得出；agent 手里的 Phase 0–5 评测降级为 **advisor**（只给它自己迭代用）。

---

## 1. 两个参照系

| | 移植的 triton-\*（现状） | 旧 kernel-\*（参照，更接近可信） |
|---|---|---|
| 评测编排 | **agent 自己编排** Phase 0–5（`AGENTS.triton.md:186-518`） | 一句 `bash tools/operator_pipeline.sh --op_name X`（`AGENTS.md:62`） |
| 评测脚本 | 3 个分立脚本 `validate_triton_impl.py`/`verify.py`/`benchmark.py`，agent 分阶段调 | 同样 3 步，但**封进一个 pipeline**，agent 不分步 |
| 产物 | 散落：`iter_N/`、`opt_iter_N/`、`verify_result*.json`、`perf_result*.json`、`summary.json`、`report.md`（`AGENTS.triton.md:680-717`） | **单一** `metrics.json`（固定 schema，`operator_pipeline.sh:144-176`） |
| 算分 | **agent 读自家 JSON、自己几何平均、写 summary.json**（`AGENTS.triton.md:555-603`） | pipeline 写 metrics.json，agent 只读不写 |
| 隔离 | agent 全程掌握 | 提示级墙：`禁止 inspect/modify tools/`（`AGENTS.md:27/40`） |
| reward 入口 | 未接（reward 仍读 AscendC 的 `trace.md`/`model_new_ascendc.py`，`openhands_agent.py:311-366`） | `AGENTS.md:166`「reward 来自 metrics.json」 |

> 结论：旧 kernel-\* 的"单入口 + 单产物 + 禁自写/禁inspect"本来就更可信，可直接作为 triton-\* 的改造目标形态。

---

## 2. 冲突清单（移植的 skills ↔ 可信 reward）

> 都是"谁在什么时候调什么脚本、输出什么到哪"层面的冲突。出处均为 `AGENTS.triton.md`（除非注明）。

| # | 冲突点（现状） | 出处 | 与可信 reward 的矛盾 | 处理方向 |
|---|---|---|---|---|
| **C1** | **最终产物由 agent 选 + 写在随机路径**：Phase 5 在 `optimized_code.py` / `generated_code.py` 间选一个，拷成 `{op_name}_generated.py`，工作目录是 `triton_ascend_output/op_..._{timestamp}_{rid}/`（随机） | `:111`、`:530-535`、`:714` | judge 需要一个**确定的、backend 已知的**待评文件；现在路径随机、且"哪个是最终"由 agent 决定 | 固定 submission 路径（backend 定义），技能把最终 impl 写到该处；judge 只认这一个文件 |
| **C2** | **被打分的数值由 agent 计算并落盘**：`summary.json.perf_data`（speedup 几何平均、passed_cases…）全是 agent 读自家 verify/benchmark JSON 后自己算的 | `:555-603` | 若 reward 读 `summary.json` → **可直接伪造**；连几何平均聚合都是 agent 做的 | reward **绝不读** `summary.json`/`*_result.json`；judge 重跑后自己算 |
| **C3** | **pass/fail 判定权在 agent**：Phase 3.3 要求"当前会话打开 verify_result.json 读 passed/total 做相等比较" | `:240-252`、`:278-289` | 喂进 reward 的 correctness 结论是 agent 中转的 | judge 自己跑 verify、自己读 passed/total |
| **C4** | **性能参数可控 + 基线可注入 + 正确性闸门可绕过**（✅已核实源码）：① `--warmup/--repeats` agent 传；② `--skip_framework` 时 framework 延迟**不测量、直接返回 agent 传入的 `--framework_latency_ms`**；③ `--verify_not_required` 可跳过 L1 正确性闸门 | `benchmark.py:485-487`(skip→返回传入值)、`526-529`(speedup=framework/impl)、`1013-1014/1018-1021`(argparse)；GPU 用法 `AGENTS.triton.md:292/470` | speedup 分母由 agent 定 → `--skip_framework --framework_latency_ms 9999` 可令 speedup **任意大（无 clamp）**；`--verify_not_required` 可对错 kernel 也出 perf 数 | judge 固定 warmup/repeats/seed、**自测** framework 基线、**禁用** `--skip_framework/--framework_latency_ms/--verify_not_required`；GPU 模式是否进 RL reward 见决策点 |
| **C5** | **没有单一固定评测入口**：评测被拆进 Phase 3/4 多步、跨 verifier+optimizer 两技能，还有"基线复用直接 cp Phase 3 结果"的捷径 | `:227-309`、`:419-484`、`:427-433`、`:459-466` | backend 要复算就得**复刻 agent 的多步编排**，不可行 | 把 validate→verify→benchmark 封成**一个固定入口**（仿 `operator_pipeline.sh`），输入(impl, task)→输出单一 `metrics.json`；advisor 与 judge 都调它 |
| **C6** | **impl 模块名是评测契约的一部分且会变**：`--triton_impl_name` 取 `triton_ascend_impl`/`triton_baseline`/`triton_optimized`，verify/benchmark 据此推 `verify_result_{name}.json` | `SKILL.md:229-236`、`:435`、`:470` | judge 不知道最终交的是 baseline 还是 optimized，难以统一重跑 | backend 把提交 impl **归一**成固定模块名/文件再重跑 |
| **C7** | **参考 Model 在 agent 工作目录、且 agent 可构造**：Phase 1 由 agent 转写 `{op_name}.py`；GPU 模式甚至**手写 PyTorch 参考**或让 `Model` 直接返回 `gpu_output` | `:153-157`、`:254-256` | 参考实现=ground truth，若用 agent 工作目录那份 → agent 控制了对错标准 | judge 用 **backend/数据集自己的参考 Model**，不用 agent 工作目录副本 |
| **C8** | **benchmark-pass ≠ correctness-pass 的语义陷阱**：文档自己警告"`perf_result.json` 的 passed_cases 只是 benchmark 进程没崩，不等于精度通过" | `:540-543`、`:567` | reward 若误读 perf 的 passed_cases 当正确率 → 把"精度错但能跑"的 kernel 当通过 | reward 的正确性闸门**只取 judge 的 verify 结果**，不取 benchmark pass 数 |

---

## 3. 改动方案

整体 = **把旧 kernel-\* 的单入口模式套到 triton 脚本上 + 外部 judge**。分三层：

### 3.1 新增：单一固定评测入口（最关键）

仿 `operator_pipeline.sh`（`rllm-merge-tmp/.../tools/operator_pipeline.sh`），为 triton 脚本建一个固定入口，例如：

```
triton_eval_pipeline.sh --op_name X --impl <impl_file> --task <reference_model_file>
  → 内部固定顺序: validate_triton_impl.py → verify.py → benchmark.py（参数写死）
  → 输出单一 metrics.json（固定 schema: success/ast_check_ok/correctness_ok/perf_data.speedup_vs_torch/error_type）
```

要点：
- 解决 **C5/C6**：一个入口、固定 impl 名、固定产物，advisor 与 judge 都调它。
- 参数（warmup/repeats/seed/容差）**写死在入口里**，且 `--skip_framework`/`--framework_latency_ms`/`--verify_not_required` 这三个危险开关**不向 agent 暴露**（已核实它们能任意抬 speedup / 绕过正确性闸门，`benchmark.py:485-487/1018-1021`）。framework 基线一律入口内自测。→ 解决 **C4**。
- 复用旧 pipeline 已有的 NPU 设备锁（`operator_pipeline.sh:10-43`，`EVAL_*`/`distributed_npu_lock.py`）。
- **可验证性（决策 #1 硬性要求）**：pipeline 必须能被验证——带一个自检：用一个**已知正确**的 impl（期望 `success=true`+合理 speedup）和一个**已知错误**的 impl（期望 `correctness_ok=false`）各跑一遍，核对 metrics 是否符合预期，作为**正式 RL 训练前的 precondition gate**。否则"评测器自己坏了却照常发 reward"会污染整批训练。建议形态：`triton_eval_pipeline.sh --self-check` 或 `fixtures/{good,bad}_impl.py` + 启动前校验脚本。

### 3.2 技能层：降级为 advisor，收紧三处

triton-\* 的 Phase 0–5 工作流**保留**（它对 agent 的生成/优化能力有用），但：
- **算分剥离**：Phase 5 不再产出"被 reward 读的" `summary.json` 作为权威分——它只是 agent 的自查记录（advisor）。reward 不看它（解决 **C2**）。
- **统一走固定入口**：Phase 3.3/3.5/4.2/4.3 里 agent 的 verify/benchmark 调用，改为调用 3.1 的固定入口（in-loop 自测），不再各自传参/各自落盘散 JSON。
- **固定最终产物**：Phase 5 把选定的最终 impl 写到 backend 约定的**固定路径 + 固定模块名**（如 `submission/{op}_impl.py`），作为唯一对外交付物（解决 **C1/C6**）。保留"禁止自写测试"（`SKILL.md:14`、`AGENTS.triton.md:761`）并补上"禁止 inspect/修改评测入口"（仿 `AGENTS.md:27/40`）。

### 3.3 backend 层：judge（权威重跑）

接 triton reward 时（现在还没接，reward 还指向 AscendC 的 `trace.md`，`openhands_agent.py:326-345`）：
- rollout 结束后，backend 拿 **agent 提交的固定路径 impl**（仅此一个产物，解决 C1），
- 用 **backend 自己的参考 Model**（数据集来源，非 agent 工作目录，解决 **C7**）+ **3.1 的同一个固定入口** + **固定参数**，
- 在 **agent 写不到的干净环境**重跑一次 → `reward = f(metrics)`。
- reward 函数用 `_reward_from_metrics` 那套分级（`openhands_agent.py:296-308`），正确性闸门取 judge 的 **verify** 结果（解决 **C8**），**不读** agent 的任何 `*_result.json`/`summary.json`。
- 可选：judge 比对 agent 自报 metrics vs judge 真值，打 `hack_mismatch_rate` 监控。

**现有 worker 现状（已核实 `remote_eval_worker.py`，352 行）**：它是个 HTTP server，`do_POST` 里 `_run_container`(:303) 跑完 agent 容器后，把**整个 `agent_workdir` tar+b64 回传**(:311) + `exit_code` + 日志 tail(:312-313)；**自身不算任何 reward/metrics**——reward 目前是 trainer 侧读回传的 workdir 算的。

- **判分落点（决策 #5/#6）**：在 `_run_container` 之后、回传之前，worker 多做一步"在一个**干净容器**（同镜像）里对**提交的 impl** 重跑固定入口"，把 `metrics` 塞进现有 payload（复用现成 tar/HTTP transport，无需新通道）。trainer 收到 `metrics` → `reward_fn` 映射成标量。→ **worker 做可信 eval、trainer 做 reward 曲线**：安全性来自"eval 在哪跑"（worker、隔离、有 NPU），灵活性来自"标量在哪算"（trainer、可调、有 batch 上下文）。
- **关于决策 #4**：judge 只对**最终提交的那个 impl** 重跑，**不依赖** agent 跑出的 baseline/optimized 数值是否可信——我们不"相信"agent 真跑过 pipeline，而是自己重跑。所以"只评 optimized（最终提交）"成立；baseline 对 reward 无用（它只是 agent 自己优化时的对照）。

> 这样 advisor（agent in-loop 跑固定入口）与 judge（backend 重跑同一入口）**同源**；agent 篡改 advisor 喂不动 reward（judge 只吃 impl 文件）。这一点的训练侧前提已验证：reward 落在末步末 token、outcome 级（`agent_ppo_trainer.py:1101-1122`）；observation 不进 loss（`data_process.py:208-216`）。

---

## 4. 与旧 kernel-\* 的对应关系（直接可复用）

| 目标能力 | 旧 kernel-\* 现成实现 | 用法 |
|---|---|---|
| 单入口跑全流程 | `tools/operator_pipeline.sh` | 作为 `triton_eval_pipeline.sh` 的骨架 |
| 单一 metrics.json + schema | `operator_pipeline.sh:144-176` | 直接沿用 schema_version 2 |
| NPU 设备锁 | `operator_pipeline.sh:10-43` | judge 重跑也抢锁 |
| 禁自写/禁 inspect | `kernel-verifier SKILL.md:14`、`AGENTS.md:27/40` | 补进 triton 的 AGENTS/SKILL |
| reward 读 metrics | `AGENTS.md:166` + `_reward_from_metrics` | judge 产 metrics → reward |

triton 比 kernel-\* 多出来需额外处理的：**多 case 几何平均**（已定沿用）、**Phase 4 优化产物选择**（已定只评最终）、~~**GPU Kernel 模式**~~（**已定移除**，见 §5）——前两处仍是 C2 高发区，改造时盯。

---

## 5. 决策点（已定 / 待定）

| # | 决策 | 结论 |
|---|---|---|
| 1 | 入口形态 | **已定**：封成一个 shell pipeline（仿 `operator_pipeline.sh`）。**+硬性要求**：pipeline 必须可被验证，正式 RL 前用 fixture 自检确认能被正确使用（见 §3.1）。 |
| 2 | GPU 模式是否进 RL reward | **已定：移除**。RL 不走 GPU Kernel 模式（它本是 GPU→Ascend 移植评估用，不是策略训练用）。落地范围见表下。 |
| 3 | 多 case 哪些 shape 进 reward | **已定**：单算子**全部 shape 几何平均**，沿用 cannbot skills 评分逻辑（`AGENTS.triton.md:297-307`、`benchmark.py:_geomean_speedup` ≈ `:651-656`）。不留 held-out。 |
| 4 | 优化产物评哪个 | **已定**：只评 agent **最终提交**的 impl（Phase 4 成功=optimized，失败=generated）。judge 自己重跑它，不需要 baseline、也不需要信任 agent 的数值（见 §3.3）。 |
| 5 | 提交物回传 | **已查**：现有 worker 已把整个 `agent_workdir` tar+b64 回传（`remote_eval_worker.py:311`）——judge 复用这条 transport（见 §3.3）。 |
| 6 | reward 谁算 | **已定**：worker 做权威 eval 重跑、回传 `metrics`；trainer 端 `reward_fn` 映射成标量（见 §3.3）。 |

> **决策 #2 — 什么是 GPU Kernel 模式**（你问没理解的那条）：triton-\* 有两种输入。常规模式输入一个 PyTorch `Model` 当参考。**GPU Kernel 模式**（`AGENTS.triton.md:90-107/145-170`）输入的是一段 **GPU 上的 Triton kernel（`@triton.jit`）+ 一个 `.pt`**（含 `input_data`、可选 `gpu_output`），目标是把它移植到 Ascend 并和 **GPU 的性能**比。此时：① `Model` 可能直接 `return gpu_output`（预存输出）→ 精度比对天然通过（`:153-156/254-256`）；② benchmark 用 `--skip_framework --framework_latency_ms <从 vllm_gpu_perf.csv 读的 GPU 延迟>`（`:292/470`）当基线。
>
> **为什么对 RL reward 危险**：这俩都让"基线/正确性"绕过真实测量——基线延迟是 agent 从 CSV 读进来的数、`Model` 还可能不算就返回预存输出，等于把 §2 的 **C4+C7 同时打开**。**结论（已采纳）：RL 训练移除 GPU 模式**（它本是给"GPU→Ascend 移植评估"用的，不是给策略训练用的）。

### GPU 模式移除落地范围（决策 #2 → 行动项）

要动的位置（`AGENTS.triton.md`）：Phase 0 检测(`:90-107`)、Phase 1 模式 B(`:145-170`)、Phase 3.3 特殊处理(`:254-256`)、Phase 3.5/4.3 的 `--skip_framework/--framework_latency_ms`(`:292/:468-470`)、Phase 5 GPU 报告 + summary 扩展(`:544-547/:605-638`)、约束表 GPU 行(`:752`)、frontmatter `argument-hint` 的 pt_file/gpu_perf_csv(`:20-23`)。外加：**RL 数据集排除 GPU-kernel 输入任务**；固定入口（§3.1）本就不暴露那两个 flag。

### 5.1 决策（已定）

| # | 决策 | 结论 |
|---|---|---|
| D1 | GPU 移除力度 | **物理删净**：`AGENTS.triton.md` 的 GPU 分支（上列 8 处）+ 数据集排除 GPU-kernel 任务，不留 dead code。 |
| D2 | 提交契约 | **`output/submission/{op}_impl.py`，类名 `ModelNew`**；skill 写它、judge 读它。 |
| D3 | judge 参考来源 | **数据集自带 `{op}.py` + `{op}.json`**（用户数据集即 json+py 格式），不用 agent 工作目录副本。 |
| D4 | reward 曲线 | **沿用 `_reward_from_metrics`**（fail 0.2 / ast 0.3 / correct 0.4 / success 0.5+0.5·speedup/2），后续可调。 |
| D5 | judge eval 镜像 | **复用现有镜像**（openhands 交互镜像带 torch_npu+triton）；不新建轻量镜像。 |

> 原决策 #5（提交物回传）因 #6（judge 跑 worker 上）消解：impl 在 worker 本地，judge 直接读。

### 5.2 决策（已定）

| # | 决策 | 结论 |
|---|---|---|
| E1 | judge 脚本+参考来源 | **方案 (b)**：worker 在**解包 workspace tar 之后、跑 agent 容器之前**，把 canonical `{op}.py/.json` + pinned `verify.py/benchmark.py` cp 到一个 **agent 容器挂不到的 judge-only 目录**；judge 用这份快照重跑。无需 worker 访问数据集、无需新增下发字段。 |
| E2 | reward 边界值 | 无 submission→0.0；agent 容器崩(exit≠0)→0.0；judge 基础设施错误(B 类，如 NPU OOM)→**丢样本(mask)**，不给 0、不污染策略。 |
| E3 | judge 镜像 | openhands 交互镜像，每次起**新容器**对 submission 重跑。 |
| E4 | benchmark.py 清理 | 连带删 `--skip_framework/--framework_latency_ms`（GPU 专用，删 GPU 后无调用者）。 |

> **E1 落点 + 为什么选 (b)**：hook 在 `remote_eval_worker.py` `do_POST` 的 `_extract_tar_b64`(:268) 之后、`_run_container`(:303) 之前 cp 一份到 worker tmp（**不在** agent 挂载的 workspace 内）。canonical 件本就在 trainer 下发的 tar 里（dispatch 时写入、agent 尚未运行），解包即快照——trust 等价于方案 (a)「单独下发数据集」，但省掉「每个 worker 挂数据集 / 每次请求塞参考」的 plumbing。

### 5.3 本轮新增决策

- **D6（task 提取）｜已定：RL 去掉 `triton-task-extractor`（Phase 0/1）**。题目**外部预封装**成数据集条目（`{op}.py`+`{op}.json`），agent 直接拿参考、只写 impl——同旧 kernel-\*（`AGENTS.md:51-54`「user message 已含完整参考，无需单独 task 文件」）。RL 用的 triton skill 5→**4**（designer / coding / verifier / latency-optimizer）。附带坐实 C7/D3（参考全外部，agent 不再构造参考）。
- **D7（轨迹抓取）｜不需要 Phase 6**。Phase 6 是 Claude-Code 专用「从 `/root/.claude/projects/session.jsonl` 导出会话」，OpenHands 无此路径。轨迹框架层已抓：① 训练级 → **SQLite trace store**（`SqliteTraceStore`，`agent_sdk_engine.py:28/90`，路径 `config.rllm.sdk.store.path`；proxy 从 vLLM 收集落盘 `:96` → `trace_to_step` 还原 Episodes/Steps）；② 人读副本 → `conversation.log`/`conversation_result.json`（`openhands_agent.py:598/601`），worker 回传(`:312`)、可经 `OPENHANDS_ARTIFACT_DIR` 归档(`:369/397`)。要抓就指向 SQLite store 或设 `OPENHANDS_ARTIFACT_DIR`。

---

## 6. 这次 review 的对象：移植时具体改了什么

> 源 = cannbot PR #205 的 triton-\* skills（Claude-Code 形态）；目标 = OpenHands 非交互式 RL。
> ⚠️ triton-\* 在 `rllm/` 里是**未跟踪**文件（`git status` 全 `??`），PR #205 原件不在本地——所以下面是"**适配规则 + 当前文件状态核实**"，**不是 git diff**。

### 6.1 已应用的适配（已核实落在文件里）

| 改动 | 为什么 | 核实点 |
|---|---|---|
| 工具名重映射 `bash`→`terminal`、`read/write/edit`→`file_editor` | OpenHands 工具名与 Claude-Code 不同 | 5 个 skill body 均含 `terminal`/`file_editor` |
| 移除 `question` 工具 | 非交互式 RL 无人工、无 question 工具 | `triton-task-extractor/SKILL.md:143/150` 显式声明"没有/禁止 question 工具"；`AGENTS.triton.md:61` 同 |
| Jinja `{{ arch }}` → 具体值 | OpenHands 不渲染 Jinja | triton-\* 中无 `{{...}}`；arch 改为 env/`npu-smi` 推断（`AGENTS.triton.md:88`） |
| Skill 由"可调用工具"改写为"指导文档" | OpenHands 的 skill 不是可调用工具 | `AGENTS.triton.md:30-38`；各 SKILL.md"你没有名为 X 的专用工具…用 terminal/file_editor"（`triton-op-verifier/SKILL.md:20-26`） |
| 删掉 Phase 6 会话导出 | OpenHands/rllm 框架层抓轨迹 | `AGENTS.triton.md:82` |
| 编排器作为 **inert** `AGENTS.triton.md` 交付 | loader 只把 `agents.md`/`agent.md` 当 active，`.triton.md` 休眠 | 文件名即 `AGENTS.triton.md`（未激活）；与 AscendC skills + `npu-arch` 共存（11 个） |

### 6.2 移植残留 / 不完整处（review 重点）

| # | 残留 | 位置 | 影响 |
|---|---|---|---|
| R1 | **已决定清理**：删 `argument-hint`（Claude-Code 字段），对齐 OpenHands 格式 `name`/`description`[/`triggers`]（参照 kernel-\* skills）；I/O 信息并入 `description` | 各 `SKILL.md` frontmatter | 清理项 |
| R2 | `tools:` 块 runner 不读（`runner.py:573-577` 硬编码 `TerminalTool`+`FileEditorTool`），纯冗余 | `AGENTS.triton.md:6-11` | **已定删除**（§7.3） |

> Review 分两层：**6.1/6.2 = 移植本身对不对**；**§2 = 移植的评测编排与可信 reward 的冲突**。

---

## 7. AscendC / Triton 路径开关 + 整洁化（本轮新增）

目标：triton 训练路径里**不出现任何 AscendC 相关内容**；两条路径用一个开关区分。

### 7.1 当前 AscendC 泄漏点（已核实，`openhands_agent.py`）

| 泄漏点 | 位置 | 现状 |
|---|---|---|
| INSTRUCTIONS 模板写死 AscendC | `_NPU_INSTRUCTION_TEMPLATE:214`「生成ascendC算子…」→ 写入 INSTRUCTIONS.md(`:255`) | 恒为 AscendC |
| reward 写死 AscendC | `:772 reward = _npu_operator_reward(...)`（读 `model_new_ascendc.py`+`trace.md`，`:317/326-345`），**无分支** | 无路由 |
| workspace 全量拷贝 | `:234 copytree(_WORKSPACE_PKG)` → 把**两套 skills（11 个）+ `AGENTS.md`(AscendC) + `AGENTS.triton.md`** 全塞进每个 rollout | 两套都在、AscendC orchestrator 默认生效 |
| dataset .json 源写死 | `:264 NPUKernelBench/level*/{op}.json`（NPUKernelBench = AscendC 数据集） | AscendC 专属 |
| 归档产物写死 | `:400-401 model_new_ascendc.py / model_new_tilelang.py` | AscendC 产物名 |
| 已有但未路由的字段 | `:484 operator_backend = task.get("operator_backend","ascendc")` | **读了没用**——正好做开关 key |

### 7.2 开关设计

**开关 key = 复用 `operator_backend`（task 字段，`:484` 已存在）**，取值 `triton`/`ascendc`，由数据集条目带。`_setup_npu_operator_workspace` 与 reward 按它路由：

| 路由对象 | triton 分支 | ascendc 分支 |
|---|---|---|
| skills 装配 | 只拷 **4 triton skill + npu-arch** 进 `.agents/skills/` | 只拷 **AscendC 那套** |
| orchestrator | triton 编排器作 `AGENTS.md` | AscendC `AGENTS.md` |
| INSTRUCTIONS | triton 指令模板 | `_NPU_INSTRUCTION_TEMPLATE` |
| dataset .json/.py | triton json+py 数据集 | NPUKernelBench |
| reward | **judge metrics → `_reward_from_metrics`**（§3.3） | `_npu_operator_reward` |
| 归档产物 | `{op}_triton_ascend_impl.py` 等 | `model_new_ascendc.py` 等 |

→ triton route 装配出的 workspace **零 AscendC 文件**。

**源布局（最终形态）**：源工作目录物理拆成 `workspace/agent_workdir.triton/` + `workspace/agent_workdir.ascendc/` 两个**纯净变体**（各含自己的 `AGENTS.md` / `.agents/skills/` / `tools/`）；`workspace/` 顶层 `entrypoint.py`/`rllm_entrypoint`/`Dockerfile`/`tests` **共享不复制**。dispatch 时 `_setup` 拷共享 infra（`ignore agent_workdir.*`）+ 按 route 拷对应变体 → 运行时 `agent_workdir`。**无运行时裁剪**（已移除 apply_route）。

### 7.3 整洁化（R1/R2 → 执行）

删所有 skill 的 `argument-hint`（R1）；删 `AGENTS.triton.md` frontmatter 的 `tools:`/`temperature:` 等冗余块（R2，runner 不读）。统一最小 frontmatter：`name`/`description`[/`triggers`]。

### 7.4 这块还要你明确的

- **F1｜开关 key**：复用 `operator_backend`（task 字段）当 route key？建议复用（已存在、数据集驱动、天然 per-task）。
- **F2｜源布局**：**已定**——物理拆成两个 `agent_workdir.{triton,ascendc}` 纯净变体（见 §7.2 最终形态），dispatch 按 route 拷其一，无运行时裁剪。
- **F3｜npu-arch 归属**：triton route 带 npu-arch（PR #205 = 5 skill + 共享 npu-arch）；AscendC route 是否也需要 npu-arch？需你确认。

---

## 附：现状产物树（供对照，来自 `AGENTS.triton.md:680-717`）

```
triton_ascend_output/op_{op_name}_{timestamp}_{rid}/   ← 随机路径 (C1)
├── {op_name}.py / .json          ← agent 构造的参考 (C7)
├── sketch.txt
├── output/
│   ├── generated_code.py / optimized_code.py   ← 最终产物由 agent 选 (C1)
│   ├── perf_result.json                         ← agent 落盘 (C2)
│   ├── iter_N/{generated_code.py, verify/verify_result.json, perf_result.json, log.md}
│   └── opt_iter_N/{optimized_code.py, verify/{verify_result_baseline,optimized}.json, *_perf_result.json}
├── {op_name}_generated.py        ← Phase 5 最终代码 (C1)
├── summary.json                  ← agent 算分落盘，reward 绝不能读这个 (C2)
└── report.md
```
