# cannbot Skills → OpenHands RL 迁移指南

> 用途：cannbot 的 triton skills 迭代出新版后，照本指南快速把新版迁移进 OpenHands RL 工作流。
> 迁移 = **两层适配** + 一张 **mismatch 检查清单** + 一套 **可复用流程**（附 grep 命令）。
> 配套设计见 `EXTERNAL_SCORER_DESIGN.md`（可信 reward / judge / 路径开关）。

---

## 0. 两个差异层（先建立心智模型）

cannbot 的 skill 是给"**独立 agent + 离线 batch 评测**"设计的；OpenHands RL 是"**框架给定输入/固定产物 + 在线 judge 算分**"。差异分两层，**必须都过一遍**：

- **形态层**：Claude-Code → OpenHands（工具名、skill 形态、frontmatter）。
- **约定层**：独立 agent → RL pipeline（工作目录、输入来源、提交产物、评测入口、reward 来源）。

> 经验：**形态层**改动机械、好查；**约定层**藏得深——cannbot 往往在某个 Phase 沿用了自建目录 / 自建任务 / 自定产物路径，跑 RL 时和框架约定冲突，且不报错只是 reward 拿不到。务必逐 Phase 审。

---

## A. 形态层适配（Claude-Code → OpenHands）

| 改动 | 为什么 | 怎么改 | 怎么验 |
|---|---|---|---|
| `bash`→`terminal`、`read/write/edit`→`file_editor` | OpenHands 工具名不同 | skill body 内替换 | grep 无 `bash`/`read/write/edit` 工具引用 |
| 移除 `question` 工具 | 非交互 RL 无人工、无此工具 | 删调用，写明"无 question 工具、自动推进" | grep `question` 只剩"禁止"说明 |
| `{{ jinja }}` → 具体值 | OpenHands 不渲染 Jinja | `{{ arch }}` 等换成 env/`npu-smi` 推断 | grep 无 `{{...}}` |
| skill 改写为"指导文档" | OpenHands skill 不是可调用工具 | 写明"没有名为 X 的工具，用 terminal/file_editor 手动做" | — |
| frontmatter 清理 | OpenHands 只认 `name`/`description`[/`triggers`]，工具由 `runner.py` 硬编码 | 删 `argument-hint`/`allowed-tools`/`tools:`/`temperature:` | grep 这些字段为 0 |
| 删 Phase 6（会话导出） | OpenHands/rllm 在框架层抓轨迹（SQLite trace store + conversation.log），agent 无法也无需导出 | 删 Phase 6 段 | grep 无 `session.jsonl`/`/root/.claude` |

---

## B. 约定层适配（独立 agent → RL pipeline）—— 重点

| cannbot 约定（要改） | RL 约定（改成） |
|---|---|
| 自建带时间戳工作目录 `triton_ascend_output/op_..._{ts}_{rid}/` | **固定 `agent_workdir`**，不新建目录（容器 `WORKSPACE_BASE=/opt/workspace/agent_workdir`） |
| Phase 0/1 自建/提取任务（task-extractor、GPU 检测） | `src/{op}.py`(+`.json`) **框架预置**；task-extractor 不用；arch 走 `OPERATOR_ARCH` |
| 产物散落：`iter_N/generated_code.py`、`{op}_generated.py`、`summary.json`、`report.md` | **唯一提交物 `output/submission/{op}_impl.py`**（类 `ModelNew`），不写 report/summary |
| 编排原始 `verify.py`/`benchmark.py` + 手建 `verify_dir`/`iter_N` + baseline cp | **统一固定入口** `tools/triton_eval_pipeline.sh`（内部 AST→verify→benchmark，参数写死），读 `metrics.json` |
| reward 来自 `summary.json` / 离线 batch 脚本 | reward 由**框架侧 judge 重跑**固定入口算分；agent 不产 reward 产物 |
| GPU Kernel 模式（`Model` 返回 `gpu_output` + `--framework_latency_ms`） | **移除**——RL 不用；它是 speedup 注入口（基线被 agent 喂） |
| 评测危险 flag `--skip_framework/--framework_latency_ms/--verify_not_required` 暴露给 agent | 入口写死、不暴露；脚本里这些 flag 删掉或不被任何调用方传入 |

---

## C. mismatch 检查清单（迁移时逐条勾）

- [ ] 工作目录：无自建 timestamp 目录，全部相对 `agent_workdir`？
- [ ] 输入：用框架给定的 `src/{op}.py`(+`.json`)，没自己构造/改写参考？
- [ ] 提交物：最终实现落在 `output/submission/{op}_impl.py`、类名 `ModelNew`？
- [ ] 评测：只经 `triton_eval_pipeline.sh`，没直接编排 `verify.py`/`benchmark.py`/建 `iter_N`？
- [ ] reward：没产出被 reward 依赖的 `summary.json`/`report.md`？
- [ ] GPU 模式：彻底移除？
- [ ] 危险 flag：`--skip_framework/--framework_latency_ms/--verify_not_required` 不暴露给 agent？
- [ ] frontmatter：无 `argument-hint`/`allowed-tools`/`tools:`/`temperature:`？
- [ ] 工具名/jinja/question：已适配？
- [ ] 领域知识（见 F）：保留未动？

---

## D. 可复用迁移流程

1. **拷新版 skills** 到 triton 变体 `workspace/agent_workdir.triton/.agents/skills/`（**不带 task-extractor**），编排器作该变体根下的 `AGENTS.md`。两条路径物理分成 `agent_workdir.triton/` + `agent_workdir.ascendc/` 两个**纯净变体**（各含自己的 AGENTS.md/skills/tools），共享 `workspace/` 顶层 infra；dispatch 按 route 拷对应变体 → 运行时 `agent_workdir`（**无运行时裁剪**）。
2. **形态层**（A 表）：grep + 改。
3. **约定层**（B 表）：
   - **重写 `AGENTS.triton.md` 工作流**为 RL 原生（固定 workdir、`src/{op}.py` 给定、`output/submission/{op}_impl.py`、固定入口评测、去 GPU/iter/summary）。这是最大头。
   - **verifier SKILL.md** 改成"调 `triton_eval_pipeline.sh` + 读 `metrics.json`、禁止自写测试"（参照 kernel-verifier 形态）。
   - **coding / designer / optimizer** 经验上 body 多为 generic（产物落点由编排器给），通常只需 frontmatter 清理——但仍要逐个确认其"输出要求"段没硬编码 cannbot 路径。
4. **残留 grep**（见 E），确认 0 cannbot 约定残留。
5. **验证**：`python3 tests/test_op_route.py`（路由单测，无 NPU）；NPU 机上 `triton_eval_pipeline.sh --self-check`（good/bad fixture）；再跑一次真实 rollout。

---

## E. 审查用 grep 命令（直接复制粘贴）

```bash
cd .../workspace/agent_workdir.triton      # triton 变体（ascendc 同理换 .ascendc）
# 约定层残留（应只剩 scripts 内部 + 正确的"不写 report/summary"约束）
grep -rIn -E "triton_ascend_output|opt_iter|generated_code|optimized_code|_generated\.py|summary\.json|report\.md|NPUKernelBench|gpu_output|GPU Kernel|skip_framework|framework_latency_ms|verify_not_required|/iter_" AGENTS.md .agents/skills/
# 形态层 frontmatter 残留（应为 0）
grep -rIn -E "argument-hint|allowed-tools|^tools:|^temperature:" AGENTS.md .agents/skills/*/SKILL.md
# jinja / question
grep -rIn -E "\{\{.*\}\}|tool.*question" AGENTS.md .agents/skills/
# 确认新约定已就位（应 >0）
grep -rIl -E "output/submission/\{op_name\}_impl\.py|triton_eval_pipeline\.sh" AGENTS.md .agents/skills/*/SKILL.md
```

---

## F. **不要动**的部分（保留 cannbot 的领域知识）

这些是 cannbot 的核心价值，迁移时**原样保留**，只搬不改：

- **PyTorch 退化政策**（forward 禁用 torch.*/F.* 计算、允许的支持性操作、launch 传张量不传 ptr）。
- **Short Kernel Rules**（elementwise/matmul/reduction/transpose/softmax 的实现要点）。
- **错误分类 + Triton 错误类型**（Type1/2/3 退化、`triton_ub_overflow`/`triton_compile_failed`/`correctness_failed` 等 → 修复方向）。
- **精度判定**（MERE/MARE 双门限、dtype 阈值表）——这是评测器的标准，agent 了解即可、不可改。

> 原则：**约定层（路径/工作流/评测/reward）全改成 RL 的；领域层（算子/kernel/精度知识）全保留。**
