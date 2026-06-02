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
| `npukb_to_task.py` | 数据集 prep：NPUKernelBench `{op}.py`+`.json` → verify 格式 `task_code`（烘焙 get_input_groups/get_init_inputs，自包含；仅 stdlib）。见 §5 准备 |
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
- **多 case `.json`**：已由 `npukb_to_task.py` 把 case 烘焙进 task 文件的 `get_input_groups()`（自包含），不再运行时读 json；数据集条目的 `task_json` 可不带（带了也被 `get_input_groups` 优先、`verify.py` 忽略）。见 §5 准备。
- **benchmark.py dead 字段**：`BenchmarkConfig.skip_framework/framework_latency_ms` 仍在（默认值，不可达），无害，可后续删净。
- **good fixture**：`tools/fixtures/good/impl.py` 是手写候选，`success` 期望值未经 NPU 验证（见测试 T2）。

---

## 5. 逐步测试阶梯（从无 NPU 到全栈）

环境变量（triton 变体；按你机器改 `REPO`/`NPUKB`）。**T0/T1 开发机即可（纯 AST，免 NPU）；T2–T5 都在评测容器 `$IMG` 内跑**——`env.sh` 的 conda+Ascend 工具链只存在于镜像里（宿主直接跑会卡在 `[env.sh] conda python not found`）。以下命令默认在**同一个 shell** 里按序执行（变量/数组沿用）：

```bash
REPO=/workspace/rllm-071                                   # 你的 checkout 根
SDK=$REPO/examples/openhands_sdk
AW=$SDK/workspace/agent_workdir.triton                     # triton 纯净变体（canonical 源）
SD=$AW/.agents/skills/triton-op-verifier/scripts
NPUKB=/path/AscendOpGenAgent/benchmarks/NPUKernelBench     # 真实数据集（参考 .py + 同名 .json）
IMG=openhands-triton-env:v1                                # 评测镜像（与 T2、worker judge 同一个）
# T3/T4 的 NPU 容器挂载（原样复用 remote_eval_worker.py:_run_judge 的那套）
ASCEND_MOUNTS=(-v /dev:/dev -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro -v /usr/local/dcmi:/usr/local/dcmi:ro
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro
  -v /usr/local/sbin:/usr/local/sbin:ro)
```

### T0 — 路由单测（开发机，无 NPU）
```bash
cd examples/openhands_sdk && python3 tests/test_op_route.py
# 期望：ALL 1 PASS（默认 triton + 大小写 + 显式 ascendc）
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

### 准备（T3–T5 共用）：真实 NPUKernelBench 算子 → 评测可读 task + 两份实现

NPUKernelBench 的 `{id}_{Name}.py` 只有 `Model`，输入 shape/dtype/attr 在同名 `.json`；而 `verify.py`
要 task 文件自带 `get_init_inputs()`+`get_input_groups()`（[verify.py:140](workspace/agent_workdir.triton/.agents/skills/triton-op-verifier/scripts/verify.py:140) / [:332](workspace/agent_workdir.triton/.agents/skills/triton-op-verifier/scripts/verify.py:332)）。用 `npukb_to_task.py`
把 json 的 case 烘焙成**自包含** task 文件（即数据集的 `task_code`）：

```bash
cd "$SDK"
# 真实算子 Abs（逐元素绝对值；单输入、空 init、含 f32/f16/bf16）——前 12 个小 case，跑得快
python3 npukb_to_task.py "$NPUKB/level1/4_Abs.py" --max-cases 12 -o /tmp/abs.py
#   去掉 --max-cases 跑全 50 case；--dtypes f32,f16,bf16 可筛 dtype

# 两份 ModelNew：good=正确 abs kernel；bad=identity kernel（AST 过、数值错，专测正确性闸门）
cat > /tmp/abs_impl_good.py <<'PY'
import torch, torch.nn as nn
import triton, triton.language as tl


@triton.jit
def abs_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.where(x >= 0, x, -x)
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        abs_kernel[grid](x, out, n, BLOCK_SIZE=1024)
        return out
PY
sed 's/tl.where(x >= 0, x, -x)/x/' /tmp/abs_impl_good.py > /tmp/abs_impl_bad.py   # identity = 数值错
# 开发机预检：good/bad 都应 valid:true（都有真 kernel；错的数值 AST 看不出，留给 NPU 正确性闸门）
python3 "$SD/validate_triton_impl.py" /tmp/abs_impl_good.py --json | grep -o '"valid": [a-z]*'
```

### T3 — 单算子手动评测（容器内，验证 pipeline 本身）

在评测容器直接跑固定入口：good 应 `success=True`；bad（identity）应 `ast_check_ok=True` 而 `correctness_ok=False`（正确性闸门拦住）。

```bash
WD=$(mktemp -d); cp -r "$AW" "$WD/agent_workdir"          # 一次性 agent_workdir（canonical 变体）
cp /tmp/abs.py "$WD/agent_workdir/src/abs.py"
mkdir -p "$WD/agent_workdir/output/submission"
run_pipe() {  # $1 = good|bad
  cp /tmp/abs_impl_$1.py "$WD/agent_workdir/output/submission/abs_impl.py"
  docker run --rm --network host --ipc host --shm-size 500g --privileged \
    -e EVAL_DEVICE_PREFIX=npu -e EVAL_DEVICE_COUNT=1 -e EVAL_DEVICE_IDS=0 \
    -e EVAL_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES "${ASCEND_MOUNTS[@]}" \
    -v "$WD:/opt/workspace" --add-host host.docker.internal:host-gateway --entrypoint bash "$IMG" \
    /opt/workspace/agent_workdir/tools/triton_eval_pipeline.sh --op_name abs \
      --impl /opt/workspace/agent_workdir/output/submission/abs_impl.py \
      --task /opt/workspace/agent_workdir/src/abs.py --out_dir /opt/workspace/agent_workdir/out
  python3 -c "import json;d=json.load(open('$WD/agent_workdir/out/metrics.json'));print('$1:',{k:d[k] for k in('success','ast_check_ok','correctness_ok')},'speedup=',(d.get('perf_data') or {}).get('speedup_vs_torch'))"
}
run_pipe good    # 期望 success=True, correctness_ok=True, speedup=<数>
run_pipe bad     # 期望 ast_check_ok=True, correctness_ok=False（闸门在拦）, success=False
```

### T4 — worker judge 隔离（容器内，验证 snapshot + judge 容器链路）

不跑整条 rollout，直接复刻 worker 的 judge 路径：**只**把 canonical 快照（[`_snapshot_canonical`](remote_eval_worker.py:243)：tools + verifier skill + src）+ 提交物放进 judge 目录，按 [`_run_judge`](remote_eval_worker.py:256) 的 docker 命令重跑，读回 `judge_out/metrics.json`（= 会被写回 `agent_workdir/judge_metrics.json` 供 reward 的那份）。

```bash
JR=$(mktemp -d)/judge
mkdir -p "$JR/agent_workdir/.agents/skills" "$JR/agent_workdir/src" "$JR/agent_workdir/output/submission"
cp -r "$AW/tools" "$JR/agent_workdir/tools"                                        # 固定入口 + env.sh + fixtures
cp -r "$AW/.agents/skills/triton-op-verifier" "$JR/agent_workdir/.agents/skills/"  # verify/benchmark/validate
cp /tmp/abs.py "$JR/agent_workdir/src/abs.py"                                      # 参考（canonical，agent 改不到）
cp /tmp/abs_impl_good.py "$JR/agent_workdir/output/submission/abs_impl.py"         # 提交物
docker run --rm --network host --ipc host --shm-size 500g --privileged \
  -e EVAL_LOCK_DIR=/shared/device-locks -e EVAL_DEVICE_PREFIX=npu \
  -e EVAL_DEVICE_COUNT=1 -e EVAL_DEVICE_IDS=0 -e EVAL_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES \
  "${ASCEND_MOUNTS[@]}" -v "$JR:/opt/workspace" -v /tmp/device-locks:/shared/device-locks \
  --add-host host.docker.internal:host-gateway --entrypoint bash "$IMG" \
  /opt/workspace/agent_workdir/tools/triton_eval_pipeline.sh --op_name abs \
    --impl /opt/workspace/agent_workdir/output/submission/abs_impl.py \
    --task /opt/workspace/agent_workdir/src/abs.py --out_dir /opt/workspace/agent_workdir/judge_out
cat "$JR/agent_workdir/judge_out/metrics.json"   # 期望 success=true —— judge 从最小快照独立重算，agent 碰不到
# 把 abs_impl_good 换成 abs_impl_bad 再跑一遍，应 correctness_ok=false（与 T3 bad 一致）
```

### T5 — 单 rollout 全栈冒烟（NPU + 完整栈，最终验收）

```bash
# 1) 造一条 triton 数据样本（task_code 已自包含，故不需要 task_json；operator_backend 显式 triton）
cd "$SDK" && python3 - <<'PY'
import pandas as pd
tc = open("/tmp/abs.py").read()
instr = "生成 Triton-Ascend 算子 abs：对输入张量逐元素取绝对值，类名 ModelNew。"
row = {"prompt": [{"role": "user", "content": instr}],
       "extra_info": {"instruction": instr, "scenario": "npu_operator",
                      "op_name": "abs", "arch": "ascend910b1",
                      "operator_backend": "triton", "task_code": tc}}
pd.DataFrame([row]).to_parquet("/tmp/abs_one.parquet", index=False)
print("wrote /tmp/abs_one.parquet")
PY
# 2) 用 train_openhands_qwen36_npu.sh 起 1 步：data.train_files/val_files=/tmp/abs_one.parquet、
#    rollout.n=1、OPENHANDS_MAX_ITERATIONS=3、trainer 只跑 1 步（total_training_steps=1 或等价）。
#    建议导出 KEEP_OPENHANDS_WORKSPACE=1，事后好翻 staging workspace。
# 3) 逐项核对（ws = staging workspace，在 $OPENHANDS_WORKSPACE_TEMP_HOST_DIR/trajectory-*/）：
grep -F "[openhands-triton] judge reward=" <训练日志>   # reward 来自 judge（openhands_agent.py:374），非异常兜底 0
ls   <ws>/agent_workdir/output/submission/abs_impl.py   # agent 写了提交物
ls   <ws>/agent_workdir/judge_metrics.json              # judge metrics 回传（_judge_and_record 写）
find <ws>/agent_workdir/.agents/skills -maxdepth 1 -type d   # 单路径纯净：只剩 triton-* + npu-arch，无 ascendc-*/tilelang
```

### 验收判据
- T0/T1 开发机即可过；**T2 是开训硬门槛**（pipeline 自己得先正确，你已过）。
- T3：good=`success` / bad=`correctness_ok:false` → 评测闸门方向正确；T4：判 judge 从**最小 canonical 快照**独立算分（agent 碰不到）。
- T5 看到 `[openhands-triton] judge reward=` 且 提交物 / 单路径纯净(无 ascendc) / judge_metrics 三项齐 → 链路打通，可放量。
- 失败定位顺序：T2（pipeline）→ T3（评测脚本/算子）→ T4（judge 容器/快照）→ T5（路由/产物/回传）。
- 换算子：改 `npukb_to_task.py` 的输入文件即可；`__init__` 带参数的算子（如 LayerNorm）需手补 `get_init_inputs()`，脚本会报错提示而非乱猜。
