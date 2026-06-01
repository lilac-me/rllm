# Phase A 冒烟测试 Runbook — single-host AscendC plumbing

> 目的：用 **merge 后路线（`verl-main-merge`）+ Qwen3.6 + 本机 remote worker（NPU 8-15）**，
> 证明多步训练环路在功能上跑通：
> `rollout → HTTP worker → OpenHands 容器 → 产物回传 → reward → advantage → backward → optimizer step → 下一个 batch`。
>
> **范围**：只验 **plumbing（环路连通性）**，reward 不动。
> `OPENHANDS_MAX_ITERATIONS=1` 下 agent 产不出完整算子，reward 会默认 0.2 ——
> **这是预期且 OK 的**，不是失败。算子真正生成质量属于后续阶段，不在 Phase A 范围。
>
> 全文判断均带 `file:line` 引用（相对 repo 根 = `rllm-merge-tmp/`），便于核对/调试。

---

## 0. 阶段总览（A / B / C）

| 阶段 | 路线 | 源 | 状态 |
|---|---|---|---|
| **A**（本文档） | AscendC skills，单机 remote worker，验训练环路 | `verl-main-merge` | ⬅ 现在做 |
| B | 把 openhands-refactoring 的 triton skills 也纳入，env 驱动 ascendc/triton 路由切换 | `openhands-refactoring` | 待 A 通过后 |
| C | 切到最新 triton skills（在另一个仓） | 外部 repo（路径待定） | 待 B 通过后 |

A 通过的判据 = **训练环路不崩、能连续推进多个 step**（reward 数值不作要求）。

---

## 1. 拉什么分支 / 在哪跑

**训练必须在 NPU 宿主机（Linux + Ascend）上跑**；本地 Mac 只用于开发/规划，没有 NPU。

### 1.1 Phase A 分支

- 分支：**`verl-main-merge`**，已推送 HEAD = `865880cc`（`remove dood and dead path`）= `origin/verl-main` + 13 commit，**已与 `origin/verl-main-merge` 同步**。
- ⚠️ 本会话此后的工作区改动（launcher 孤儿清扫、MERGE/LOG 文档更新、本 runbook）**尚未 commit/push**——单靠 fetch 拿不到,要么先 commit+push,要么用 rsync。
- 远端：`origin git@github.com:lilac-me/rllm.git`。

**NPU host 取分支（任选其一）：**

```bash
# 方式 A：经远端（拿到 865880cc；不含上面那些未 push 的改动）
git fetch origin
git checkout verl-main-merge      # 或 git checkout -b verl-main-merge origin/verl-main-merge
git pull --ff-only
git log -1 --oneline              # 应为 865880cc remove dood and dead path

# 方式 B：直接同步本地工作树（会带上未 commit 的改动，含 launcher 孤儿清扫）
rsync -av --exclude '.git' /path/to/rllm-merge-tmp/ <npu-host>:/path/to/rllm/
```

### 1.2 后续阶段分支（先知道，别现在拉）

- Phase B：`openhands-refactoring`（`origin/openhands-refactoring` 已存在）。
- Phase C：另一个仓的最新 triton skills（路径待你提供）。

### 1.3 操作边界（重要）

- **绝不动** `/Users/yeji/Documents/Code/Python/Qwen36/rllm`（你正在进行中的工作仓，有未提交的 qwen36 改动）。所有验证都在 `rllm-merge-tmp` 的 `verl-main-merge` 上。
- **默认不 push、不做破坏性 git 操作**；commit 留本地待你 review。
- push / 改远端这类动作，我会先问你再做。

---

## 2. 架构数据流（验证过，便于排障）

```
trainer (--network host；走 remote 路径时本身不调 docker)
  ├─ 启动 LiteLLM proxy 子进程  :5000      ← rllm.sdk.proxy.mode=subprocess（train…npu.sh:480）
  ├─ rollout() → HTTP POST → http://127.0.0.1:16881/run   （openhands_agent.py:545）
  │                              │
  │                          remote_eval_worker.py（独立进程，**跑在 NPU 宿主机上裸起，非 DooD**）
  │                              └─ docker run 一个 OpenHands 容器（NPU 8-15，宿主机 dockerd）
  │                                   ├─ LLM_BASE_URL → host.docker.internal:5000 → LiteLLM proxy → vLLM
  │                                   ├─ 跑 AGENTS.md Phase 0-7（AscendC bundle）
  │                                   └─ 产物 tar 回传
  └─ _npu_operator_reward(task, ...) 读 trace.md → reward（openhands_agent.py:326-383）
```

关键事实（都核对过）：

- 单机拓扑下 `OPENHANDS_REMOTE_EVAL_URL` 非空（`config/qwen36.env:51`）→ 走 **HTTP worker** 路径。
  **同机 docker-run 兜底已删除**：URL 为空不再回退，而是直接 `raise`（`openhands_agent.py:518`）。
- **trainer 全程零 docker 调用**（rollout 只发 HTTP，`openhands_agent.py:705`）。
  真正 `docker run` OpenHands 容器的是 worker（`remote_eval_worker.py:151-194`）。docker 需求只在 worker——
  worker 固定**跑在宿主机上裸起**，用宿主 dockerd，是普通 docker、**不是 DooD**。trainer 不需要 docker，也不需要 docker.sock。
- `operator_backend` 对 Phase A **路由无影响**：worker 默认 `"triton"`（`remote_eval_worker.py:135`），
  trainer 把整个 `task` 透传进 payload（`openhands_agent.py:531`），其中若带 operator_backend 即覆盖；
  但 `cfg.operator_backend` **只在测试里被引用**（`workspace/tests/test_config.py:102,163`），
  runner/entrypoint 都不据它分流（`workspace/rllm_entrypoint/runner.py:84` 的 task_scope 是注释掉的）——
  agent 流程完全由挂载的 `agent_workdir/AGENTS.md`（AscendC）决定。**A 阶段无害；Phase B 接 triton 路由前需确保 payload 里 operator_backend 正确。**
- `max_iterations=1` 经 **payload** 从 trainer 传给 worker（`openhands_agent.py:534` → `remote_eval_worker.py:132`），
  覆盖 worker 自己的 env 默认值 30。
- LiteLLM proxy 由 trainer 自动起（子进程），**不用手动起**；端口 `PROXY_PORT=5000`（`config/qwen36.env:20`）。

---

## 3. Phase A 检测步骤

### §1 启动前宿主机检查（3 项）

```bash
# 1.1 mock parquet 必须存在（trainer 直接读，get_data() 返回 []）
ls -la examples/openhands_sdk/rl_single_ops.parquet      # train_openhands_qwen36_npu.py:52
python3 -c "import pandas as pd; d=pd.read_parquet('examples/openhands_sdk/rl_single_ops.parquet'); print(d.columns.tolist()); print(d.iloc[0].to_dict())"
#   确认每行带 prompt / extra_info(scenario=npu_operator)；task 里若有 op_name/arch 更好
#   （worker 缺省 op_name=operator, arch=ascend910b1，见 remote_eval_worker.py:133-134 —— 缺了不崩，只是 reward 可能找错算子目录）
#   ⚠️ 缺 parquet → verl 数据加载阶段直接崩

# 1.2 镜像存在（plumbing 只需能起容器跑 entrypoint.py；算子 toolchain 不必跑通）
docker image inspect openhands-triton-env:v1 >/dev/null && echo IMAGE_OK   # config/qwen36.env:25

# 1.3 NPU：16 张可见。trainer 用 0-7，worker 在 8-15 起容器，互不重叠
npu-smi info                                             # config/qwen36.env:68(0-7) vs :38(8-15)
```

### §2 启动 eval worker（用 launcher，端口已自动桥接）

worker 直接跑在 **NPU 宿主机**上（不进 dev 容器）。它用宿主机 dockerd `docker run` OpenHands 容器，
是普通 docker、**不是 DooD**。用 launcher 启动——它把 worker 的所有 scratch 状态（work + npu-locks）
都建在**启动目录**下，无 `/home/docker`、无 `/tmp` 硬编码（容器挂载 workspace 见 `remote_eval_worker.py:189`）：

```bash
cd <任意目录>          # 状态建在这里的 ./remote-eval-state/ 下
bash <repo>/examples/openhands_sdk/start_remote_worker.sh
# 可覆盖（都有默认值，无绝对路径硬编码）：
#   EVAL_WORKER_PORT=16881 OPENHANDS_EVAL_DEVICE_IDS=8,9,10,11,12,13,14,15 \
#   OPENHANDS_IMAGE=openhands-triton-env:v1 WORKER_STATE_DIR=/data/eval-state \
#   bash <repo>/examples/openhands_sdk/start_remote_worker.sh
```

launcher 替你处理（见 `examples/openhands_sdk/start_remote_worker.sh`）：
- **scratch 全在启动目录**：`$PWD/remote-eval-state/{work,npu-locks}`，导出 `OPENHANDS_EVAL_LOCK_DIR` 指过去。
  lock dir 能搬走依赖已改的 `remote_eval_worker.py:35`（读 env），bind 源复用同一常量做单一真相源（`:148`）。
- **桥接端口命名**：把 `EVAL_WORKER_PORT`（config 的变量名，默认 16881）传成 worker 的 `--port`。
  worker 自身默认 18880 且读 `OPENHANDS_REMOTE_EVAL_PORT`（`remote_eval_worker.py:337`），而 trainer POST 到 16881
  （`config/qwen36.env:31` → URL `:51`）——launcher 替你对齐，不会再 connection refused。
- **fail-fast 检查 Ascend 驱动路径**（`/usr/local/Ascend/driver`、`/usr/local/dcmi`、`npu-smi`、
  `/etc/ascend_install.info`、`/dev`）：宿主机 NPU 驱动、不可搬，缺了直接报错（容器挂载 `remote_eval_worker.py:175-181`）。
- **`--host 127.0.0.1`** 走 loopback：前提 trainer 以 `--network host` 跑（或也在宿主机上），127.0.0.1 才互通。

注意：
- worker 启动会打印 `[remote-eval] listening ...` 并**自动清全部 NPU 锁**（`remote_eval_worker.py:345-347`）——
  所以**每轮训练重启 worker 即可保证干净起点**，reset-locks 一般用不上（§3c 仅在复用长跑 worker 时才需要）。
- launcher 启动时还**自动清扫上轮崩溃残留的 `rllm-openhands-eval-*` 容器**（占着 NPU+锁的孤儿）——
  `WORKER_SKIP_ORPHAN_SWEEP=1` 可关；**单机跑多个 worker 时务必关**，否则会误杀兄弟 worker 在跑的容器。
- image / device_ids trainer 会在 payload 里覆盖（`remote_eval_worker.py:130,143`），launcher 设的只是 fallback。

### §3 safety net（(a)(b) 仍需手动；(c) 孤儿清扫已进 launcher）

`LITELLM_LOCAL_MODEL_COST_MAP` / `/health` 仍需手动（不在任何 `.sh` 里；`train_openhands_qwen36_npu.sh`
没继承 `debug_oom.sh` 当年的）。**孤儿容器清扫现已进 `start_remote_worker.sh`**——启动时自动扫
`rllm-openhands-eval-*` 残留并 `docker rm -f`（`WORKER_SKIP_ORPHAN_SWEEP=1` 可关；单机多 worker 慎用）。

```bash
# (a) KU8: LiteLLM 偶发去网上查 model cost map → flaky。proxy 跑在 trainer 进程，故启训前 export
export LITELLM_LOCAL_MODEL_COST_MAP=True

# (b) 启训前确认 worker 活着（fail-fast）
curl -fsS http://127.0.0.1:16881/health        # 期望 {"ok": true}（remote_eval_worker.py:251）

# (c) 孤儿容器清扫已自动进 launcher（start_remote_worker.sh 启动时做）。手动兜底（单机多 worker 慎用）：
#     docker ps -aq --filter "name=rllm-openhands-eval-" | xargs -r docker rm -f   # 前缀见 remote_eval_worker.py:129
#     若复用长跑 worker 跨多次重启训练，再清锁:
curl -fsS -X POST http://127.0.0.1:16881/admin/reset-locks   # remote_eval_worker.py:257
```

### §4 启动训练

```bash
export LITELLM_LOCAL_MODEL_COST_MAP=True
RLLM_TOPOLOGY=single bash examples/openhands_sdk/train_openhands_qwen36_npu.sh
```

### §5 看什么（pass / fail 信号）

| 现象 | 判定 |
|---|---|
| 越过第一个 rollout，打印 reward、loss/grad，**step 2 开始**，能连续推进多个 step | ✅ **plumbing 通了（A 阶段达标）** |
| reward 恒为 0.2 / 偏低 / 不变 | ✅ **预期**——`max_iter=1` agent 产不出 trace.md 成功串（`openhands_agent.py:349` 命中才 0.8，否则 `:339` 默认 0.2）。只验 plumbing，无所谓 |
| 第一个 rollout 卡很久 | ⏳ 正常——NPU vLLM/HCCL 初始化慢，超时设了 24h（`train…npu.sh:110-111`）；不是 hang |
| `empty DataProto` / "all trajectories dropped" 崩溃 | ❌ **真 plumbing 断了，别压制**——见下 |

**关于 empty-batch 崩溃（必须保持 fail-loud）**：F3 全 drop 守卫已删
（commit `1464f0c0`，BryanChen408；见 `MERGE_TO_VERL_MAIN_PLAN.md §7.1`）。
现在的行为：**有有效 step 但 reward=0 的 batch 能正常训**（advantage=0，不崩）；
**只有真·空 batch（0 有效 step）才崩**。所以这里一旦崩 = 真断了，典型根因：
每个 rollout 的 `docker run` 都失败 / worker 不可达 / 产物没回传。**查根因，不要绕过。**

**辅助观测点**：
- worker 端 `[remote-eval] ...` 日志；容器非零 exit 会被记（`openhands_agent.py:583`）。
- 每个 rollout 写 `agent_workdir/remote_eval_result.json`（`openhands_agent.py:571`）——看 exit_code / worker_error。
- 孤儿容器/锁堆积 → 下一轮抢不到 NPU，回 §3(c) 清理。

---

## 4. 注意事项 / 已知坑（汇总）

1. **分支已推 `origin/verl-main-merge`（HEAD `865880cc`）**；但本会话之后的 launcher 孤儿清扫 + 文档更新**未 commit/push**，NPU host 要这些得先 commit+push 或 rsync（§1.1）。
2. **端口双名**：worker 读 `OPENHANDS_REMOTE_EVAL_PORT`（默认 18880），config 用 `EVAL_WORKER_PORT=16881`。
   两者**不是同一个变量**——worker 必须 `--port 16881`（§2）。
3. **同机 DooD 兜底 + 预检已删除**：rollout 现在只走 remote HTTP worker；`OPENHANDS_REMOTE_EVAL_URL` 为空会直接
   `raise`（`openhands_agent.py:518`），不再有同机 docker-run 退路。train 脚本那段无条件 docker/overlay/bind-mount
   预检也一并删了 → trainer 不再需要本机 docker。trainer 暂存目录改为启动相对路径、Python 自动创建
   （`OPENHANDS_WORKSPACE_TEMP_HOST_DIR` 可覆盖，默认 `examples/openhands_sdk/workspace_temp`）。
4. **worker 固定宿主机裸起 → work-dir 无 bind-mount 约束**（§2）：worker 用宿主 dockerd，
   `-v {work-dir}/...:/opt/workspace` 直接落宿主机路径，任意可写路径即可。
   （"work-dir 必须在 bind-mount 下、否则子容器 `/opt/workspace` 空 → exit 127 → 100% drop" 这个坑
   只在把 worker 塞进 dev 容器时才出现——我们不那么做。）
5. **operator_backend 仅剩 worker 默认 `triton`**（`remote_eval_worker.py:135`；同机 `ascendc` 默认随同机路径一起删了）：
   A 阶段**无害**（仅测试引用、不分流，流程由 AGENTS.md 决定），但 **Phase B 接 triton 路由前**需确保 payload 的
   operator_backend 正确，否则路由会错。
6. **2 个 safety net 默认缺失**（§3 的 (a)(b)；(c) 孤儿清扫已进 launcher）：不补也可能跑通，但 KU8(LiteLLM flaky)/锁残留会偶发性卡跑，建议补。
7. **reward 低是预期**，不要据此判断失败（§5）。A 阶段唯一判据是环路是否连续推进。
8. **NPU 分配**：trainer 0-7、OpenHands 8-15，互斥（`config/qwen36.env:38,68`）。别让别的进程占 8-15。

---

## 5. Phase B / C 预告（不在本次执行）

- **Phase B**：从 `openhands-refactoring` 移植 triton skills bundle + metrics.json reward 读取（约 85 行）+ triton 数据集，
  加 env 驱动的 ascendc/triton 路由切换（先统一坑 #5），并重新评估 F3。详见 `MERGE_TO_VERL_MAIN_PLAN.md §4`。
- **Phase C**：切到另一个仓的最新 triton skills（路径待定）。

> 维护：本文档随 Phase A 执行结果更新；若某条 `file:line` 与代码对不上，以代码为准并回改本文档。
