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

> **执行位置（系统复杂，务必先看）：本节所有命令都在【宿主机】敲。** 标 `docker run …` 的也是在宿主机发起，它**自动在容器内**执行 pipeline/verify/benchmark（只有镜像里才有 conda+Ascend）。**绝不要先 `docker exec`／进容器再敲这些**——你在容器里再 `docker run -v /tmp/…` 是 DooD（docker-out-of-docker），`-v` 源按**宿主**解析 → 空挂载 → `triton_eval_pipeline.sh: No such file or directory`。T0/T1 例外：纯 AST、免 NPU、免容器，宿主/开发机直接跑。

下面每个阶段标了 **`【宿主机】`** 或 **`【宿主机→容器内】`**（= 宿主机敲、容器内执行）。变量都是**宿主机路径**（docker `-v` 源按宿主解析），按你机器改 `REPO`/`NPUKB`；同一个 shell 里按序跑（变量/数组沿用）：

```bash
# —— 全部在【宿主机】 ——
REPO=/workspace/rllm-071                                     # 宿主机上的 checkout 根
SDK=$REPO/examples/openhands_sdk
AW=$SDK/workspace/agent_workdir.triton                       # triton 纯净变体（canonical 源）
SD=$AW/.agents/skills/triton-op-verifier/scripts
NPUKB=/workspace/AscendOpGenAgent/benchmarks/NPUKernelBench  # 真实数据集（宿主机路径）
IMG=openhands-triton-env:v1                                  # 评测镜像（与 T2、worker judge 同一个）
ls "$AW/tools/triton_eval_pipeline.sh" "$NPUKB/level1/4_Abs.py"   # 先确认两个源路径在；不在就改上面
# T2–T4 容器的 NPU 挂载（原样复用 remote_eval_worker.py:_run_judge）。设备用 EVAL_DEVICE_IDS 指定；
# 若你镜像是直接读 ASCEND_RT_VISIBLE_DEVICES，把各 docker run 里 -e EVAL_DEVICE_IDS=0 换/加成 -e ASCEND_RT_VISIBLE_DEVICES=0
ASCEND_MOUNTS=(-v /dev:/dev -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro -v /usr/local/dcmi:/usr/local/dcmi:ro
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro
  -v /usr/local/sbin:/usr/local/sbin:ro)
```

### T0 — 路由单测　`【宿主机】`（无 NPU、无容器）
```bash
cd "$SDK" && python3 tests/test_op_route.py
# 期望：ALL 1 PASS（默认 triton + 大小写 + 显式 ascendc）
```

### T1 — AST 闸门　`【宿主机】`（纯 AST、无 NPU、无容器）
```bash
python3 "$SD/validate_triton_impl.py" "$AW/tools/fixtures/good/impl.py" --json   # 期望 valid:true
python3 "$SD/validate_triton_impl.py" "$AW/tools/fixtures/bad/impl.py"  --json   # 期望 valid:false, regression_type:1
```

### T2 — 固定入口 self-check　`【宿主机→容器内】`（开训前硬门槛）
宿主机发起 `docker run`，容器内跑自检（你已过）。直接挂 `$AW` 当 agent_workdir（self-check 只读 fixtures/脚本，临时输出写容器内 /tmp）：
```bash
docker run --rm --network host --ipc host --shm-size 500g --privileged \
  -e EVAL_DEVICE_PREFIX=npu -e EVAL_DEVICE_COUNT=1 -e EVAL_DEVICE_IDS=0 -e EVAL_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES \
  "${ASCEND_MOUNTS[@]}" -v "$AW:/opt/workspace/agent_workdir" \
  --add-host host.docker.internal:host-gateway --entrypoint bash "$IMG" \
  /opt/workspace/agent_workdir/tools/triton_eval_pipeline.sh --self-check
# 期望：good success/correctness=True True；bad correctness_ok=False；末行 [self-check] PASS
```

### 准备（T3–T5 共用）　`【宿主机】`：真实 NPUKernelBench 算子 → task + 两份实现

`verify.py` 要 task 文件**自包含**地给出 `get_init_inputs()` + `get_input_groups()`（[verify.py:140](workspace/agent_workdir.triton/.agents/skills/triton-op-verifier/scripts/verify.py:140) / [:332](workspace/agent_workdir.triton/.agents/skills/triton-op-verifier/scripts/verify.py:332)）。NPUKernelBench 的 `{op}.py` 有三态：① 只有 `Model`（输入在 `.json`）；② `get_input_groups()` 直接返回张量（已自包含）；③ **`get_input_groups()` 运行时 `open()` 读 sibling `{op}.json`**——③ 直接当 task_code 会在 pipeline 改名后 `FileNotFoundError`（NPU 实测踩到，见 §6）。`npukb_to_task.py --bake` 一律从 `.json` 重建自包含版（剥掉 ②③ 原输入函数），最稳，产物 `/tmp/abs.py`：

```bash
# —— 全部在【宿主机】 ——
# --bake：无视原有输入函数，从 sibling .json 强制烘焙成自包含 task_code（产物 /tmp/abs.py）
python3 "$SDK/npukb_to_task.py" --bake "$NPUKB/level1/4_Abs.py" --max-cases 12 -o /tmp/abs.py
#   去掉 --max-cases 跑全 50 case；--dtypes f32,f16,bf16 可筛 dtype
#   应急（适配器还没同步 --bake 时）：手写自包含 /tmp/abs.py（Model + get_init_inputs + get_input_groups 直接返回张量）

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

### T3 — 单算子手动评测　`【宿主机→容器内】`（验证 pipeline 本身）

宿主机发起 `docker run`，容器内跑固定入口（**别先进容器**）：good 应 `success=True`；bad（identity）应 `ast_check_ok=True` 而 `correctness_ok=False`（正确性闸门拦住）。`$WD` 在宿主机 `/tmp` 下，docker `-v` 才解析得到。

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

### T4 — worker judge 隔离　`【宿主机→容器内】`（验证 snapshot + judge 容器链路）

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

### T5 — 单 rollout 全栈冒烟　`【宿主机】`（trainer 起训、worker 自动起容器；最终验收）

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

---

## 6. NPU 实测发现与修复（2026-06-03，T2→T3 打通过程中）

逐项在 node-5-88 宿主机实测时踩到、并已修：

1. **执行位置（DooD 空挂载）**：评测必须在容器内（conda+Ascend 只在镜像 `openhands-triton-env:v1`），但 `docker run` 必须**从宿主机发起**——在容器里再 `docker run -v /tmp/…` 时 `-v` 源按宿主解析→空挂载→`triton_eval_pipeline.sh: No such file or directory`。**修复**：§5 全部命令标注 `【宿主机】`/`【宿主机→容器内】`，并把 T2 也改成宿主机发起。

2. **`env.sh` 在 `set -u` 下 source 厂商 `set_env.sh`**：vendor 脚本里 `export LD_LIBRARY_PATH=/cann/…:$LD_LIBRARY_PATH` 因 `$LD_LIBRARY_PATH`/`$CMAKE_PREFIX_PATH` 未定义而报 `unbound variable`，该 export 失败→Ascend/triton 库路径没拼上→verify 编译/运行 kernel 失败（good/bad 同样失败，像 kernel 问题其实是环境）。**修复**：[env.sh](workspace/agent_workdir.triton/tools/env.sh) 在 source 前预置二者为空、`set +u` 包裹 source、之后 `set -u` 恢复。

3. **task_code 必须自包含**（最关键）：用户的 NPUKernelBench 变体里 `4_Abs.py` 的 `get_input_groups()` **运行时 `open()` 读 sibling `4_Abs.json`**。pipeline 把 task 改名成 `abs_torch.py` 放进 `verify_tmp/`、那个 json 没跟过去 → `FileNotFoundError`；且 `op_name` 不能取 `4_Abs`（Python 模块名不能数字开头）。**修复**：`npukb_to_task.py` 加 `--bake`——剥掉运行时读文件的 `get_input_groups`/`get_init_inputs`，从 `.json` 重建自包含版；§5 准备改用 `--bake`。**含义**：RL 数据集的 `task_code` 一律走 `--bake` 产出自包含，不依赖运行时文件。

4. **`error_type` 误判**：`triton_eval_pipeline.sh` 的 `classify()` 用 `"ast" in l` 太宽，把上面的 `FileNotFoundError`（读 json）误标 `ast_check_failed`。**修复**：`ast` 判定收窄为 `ast退化`/`ast_check`/`ast check`，并新增 `input_load_failed`（命中 `get_input`/`.json` FileNotFound）。

> 验收现状：**T0–T3 已绿**（T3 用自包含 task：good `success=True` / bad `correctness_ok=False`，闸门方向正确）。下一步 T4（judge 隔离）→ T5（完整 rollout，数据须用 `--bake` 的自包含 task_code）。
