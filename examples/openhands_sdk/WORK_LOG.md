# 工作日志 — Triton skills 迁移 + 可信 reward（triton-eval 分支）

> 分支 `triton-eval`（基于 `verl-main-merge`），**全部改动未 commit**，供整体 review。
> 配套：`EXTERNAL_SCORER_DESIGN.md`（设计+决策）、`CANNBOT_SKILLS_MIGRATION_GUIDE.md`（迁移流程）。

## 1. 做了什么（按主题）

1. **可信 reward 架构**：评测拆成 advisor（agent in-loop 自测）与 judge（backend 权威重跑）。reward 由 judge 对**提交产物**重跑固定入口算分，agent 改不到。设计/决策见 `EXTERNAL_SCORER_DESIGN.md`。
2. **AscendC/Triton 路径开关**：`op_route(task)` 按 `task["operator_backend"]` 路由（**默认 triton**）。源工作目录物理拆成两个**纯净变体** `agent_workdir.triton/` + `agent_workdir.ascendc/`（各含自己的 AGENTS.md/skills/tools），共享 `workspace/` 顶层 infra；dispatch 按 route 拷对应变体 → 运行时 `agent_workdir`（triton 路径**零 AscendC**，无运行时裁剪）。
3. **cannbot→OpenHands 迁移**：triton-* skills 从 cannbot 形态适配到 OpenHands RL——重写 `AGENTS.triton.md` 工作流为 RL 原生（固定 `agent_workdir`、`src/{op}.py` 给定、`output/submission/{op}_impl.py` 提交、固定入口评测，去自建目录/任务构造/GPU/iter/summary）；verifier 改固定入口；清理 frontmatter 残留。审查方法沉淀进迁移指南。
4. **固定评测入口** `triton_eval_pipeline.sh`：AST→verify→benchmark→单一 `metrics.json`，参数写死、不传危险 flag，带 `--self-check`。
5. **worker judge**：snapshot（E1，agent 跑前快照 canonical 脚本+参考）→ 干净容器重跑固定入口 → 把 `judge_metrics.json` 写回 `agent_workdir`（随 tar 回传）→ `_triton_judge_reward` 读它算 reward。
6. **E4 安全清理**：benchmark.py 移除 `--skip_framework/--framework_latency_ms/--verify_not_required` 的 CLI（speedup 注入口 / 闸门旁路）。

## 2. 文件清单

**新增（untracked）**
| 文件 | 作用 |
|---|---|
| `EXTERNAL_SCORER_DESIGN.md` | 设计 + 决策记录（D1-D7 / E1-E4 / 冲突 C1-C8） |
| `CANNBOT_SKILLS_MIGRATION_GUIDE.md` | cannbot→OpenHands 迁移流程（复用） |
| `WORK_LOG.md` | 本文件 |
| `op_route.py` | 路由键（dispatch 按它拷对应 agent_workdir 变体；仅 stdlib，可单测） |
| `tests/test_op_route.py` | 路由单测（无 NPU） |
| `workspace/agent_workdir/AGENTS.triton.md` | RL 原生编排器（重写） |
| `workspace/agent_workdir.triton/` | triton 纯净变体（AGENTS.md/skills/tools；含重写的 verifier、E4 改的 benchmark.py、固定入口、fixtures） |
| `workspace/agent_workdir/tools/triton_eval_pipeline.sh` | 固定评测入口（advisor+judge 共用） |
| `workspace/agent_workdir/tools/fixtures/{good,bad}/` | self-check fixtures（good 期望值待 NPU 确认） |

**改动（tracked）**
| 文件 | 改动 |
|---|---|
| `openhands_agent.py` | `operator_backend` 路由（instruction/skills/json/reward 分支）+ 填 `_triton_judge_reward` |
| `remote_eval_worker.py` | worker judge：`_snapshot_canonical`/`_run_judge`/`_judge_and_record` + `do_POST` 接线 |
| `agent_workdir.ascendc/` | AscendC 纯净变体（5 skill + operator_pipeline.sh + scripts + NPUKernelBench；`git mv` 归位） |

## 3. 本地已验证（无 NPU）

- `python3 -m py_compile openhands_agent.py op_route.py remote_eval_worker.py benchmark.py` 全过。
- `python3 tests/test_op_route.py` → PASS（默认 triton 路由正确）。
- `validate_triton_impl.py` 对 good/bad fixture → valid:true / regression_type:1。
- E4：危险 flag 的 CLI/`args.*` 引用清零。

## 4. 遗留 TODO（不阻塞，已在代码标注）

- **E2**：judge 基础设施错误（B 类：`judge_no_metrics`/`judge_container_failed`/NPU OOM）当前给低分，应"丢样本(mask)"——需 trainer 侧 `valid_mask` 配合（`openhands_agent.py:_triton_judge_reward` TODO）。
- **多 case `.json`**：是否被 `verify.py` 的 `get_input_groups` 读取，取决于数据集 `{op}.py` 写法（`triton_eval_pipeline.sh` NOTE）。
- **benchmark.py dead 字段**：`BenchmarkConfig.skip_framework/framework_latency_ms` 仍在（默认值，不可达），无害，可后续删净。
- **good fixture**：`tools/fixtures/good/impl.py` 是手写候选，`success` 期望值未经 NPU 验证（见测试 T2）。

---

## 5. 逐步测试阶梯（从无 NPU 到全栈）

> 路径前缀（triton 变体）：`AW=examples/openhands_sdk/workspace/agent_workdir.triton`；`SD=$AW/.agents/skills/triton-op-verifier/scripts`。

### T0 — 路由单测（开发机，无 NPU）
```bash
cd examples/openhands_sdk && python3 tests/test_op_route.py
# 期望：ALL 3 PASS
```

### T1 — AST 闸门（开发机，无 NPU；validate 是纯 AST）
```bash
python3 "$SD/validate_triton_impl.py" "$AW/tools/fixtures/good/impl.py" --json   # 期望 valid:true
python3 "$SD/validate_triton_impl.py" "$AW/tools/fixtures/bad/impl.py"  --json   # 期望 valid:false, regression_type:1
```

### T2 — 固定入口 self-check（**NPU 机**，开训前的 precondition gate）
```bash
cd "$AW" && bash tools/triton_eval_pipeline.sh --self-check
# 期望：good success/correctness = True True；bad correctness_ok=False；末行 [self-check] PASS
# 若 good 不过：说明手写 add kernel 在本 NPU 上跑不通——换一个你数据集里【已知能跑通】的小算子：
#   cp 你的参考 → tools/fixtures/good/{op}.py ；cp 已知正确实现 → tools/fixtures/good/impl.py
#   （bad 用 tools/fixtures/bad/，impl.py 留作纯 torch 退化即可）
```

### T3 — 单算子手动评测（**NPU 机**，验证 pipeline 本身）
```bash
cd "$AW"
mkdir -p output/submission
cp <你的参考>      src/myop.py
cp <一个正确实现>  output/submission/myop_impl.py     # 类名 ModelNew
bash tools/triton_eval_pipeline.sh --op_name myop \
     --impl output/submission/myop_impl.py --task src/myop.py   # 多 case 再加 --json src/myop.json
cat metrics.json   # 看 success / ast_check_ok / correctness_ok / perf_data.speedup_vs_torch
# 再喂一个【错误实现】，确认 correctness_ok=false（验证闸门真在拦）
```

### T4 — worker judge 隔离（**NPU 机**，验证 snapshot+judge 容器链路）
```python
# 在 worker 机上：先确认 op_route 走 triton，再用一个最小请求打 worker。
# 简化做法：直接跑 T5 的单 rollout（它会驱动 worker judge）。若要单独打 worker，
# 参考 openhands_agent.py:_run_remote_eval_worker 的 payload 字段（workspace_tar_gz_b64/proxied_url/task）。
# 关键观测：rollout 期间 `docker ps -a | grep rllm-triton-judge-` 应出现 judge 容器；
#          回传的 agent_workdir 里应有 judge_metrics.json。
```

### T5 — 单 rollout 全栈冒烟（**NPU + 完整栈**，最终验收）
```bash
# 1) 数据集准备一条 triton 样本：task 带 operator_backend="triton"（不带也默认 triton）、
#    op_name、task_code(=src/{op}.py 内容)、可选 task_json。
# 2) train 配置：rollout.n=1、OPENHANDS_MAX_ITERATIONS 设小（如 3）、先 val_only/单步。
# 3) 跑一步，逐项核对：
grep -i "judge reward"          <训练日志>   # [openhands-triton] judge reward=... success=...  → reward 来自 judge、非 raise
ls  <rollout workspace>/agent_workdir/output/submission/   # agent 是否写了 {op}_impl.py
ls  <rollout workspace>/agent_workdir/judge_metrics.json   # judge metrics 是否回传
# dispatched workspace 应是 triton-only：
find <rollout workspace>/agent_workdir/.agents/skills -maxdepth 1 -type d   # 应只剩 triton-* + npu-arch，无 ascendc-*/tilelang 等
```

### 验收判据
- T0/T1 开发机即可过；**T2 是开训硬门槛**（pipeline 自己得先正确）。
- T5 看到 `judge reward=` 且产物/裁剪/judge_metrics 三项齐 → 链路打通，可放量。
- 失败定位顺序：T2（pipeline）→ T3（评测脚本）→ T4（judge 容器）→ T5（路由/产物/回传）。
