# OpenHands triton-skills 能力评测（pass@k）

评测"**模型 + 当前这套 triton skills**"在 NPUKernelBench 上的能力。**复用训练同款 rollout + judge**：
每条轨迹跑一遍完整 skills 流程（Phase1/2/3、固定入口、R1 留 `.best`），worker 的 judge 在 canonical
快照上重跑固定入口给 `.best` 打分（agent 改不到）→ `judge_metrics.json`。eval 只做"遍历数据集 ×
N 轨迹 → 读 judge_metrics → 算 pass@k"。**不走 trainer/梯度**。

## 文件
- `eval_openhands.py` — 评测驱动（loader / rollout / 聚合 pass@k）。
- `run_eval.sh` — 一键：T=0 greedy(pass@1) + T>0 sampled(pass@k) + 步数标定，末尾出对比。

## 前置（NPU 宿主机）
1. **同步代码**：`git pull`（拿到 eval/ + runner.py 温度&run_meta + worker 温度透传）。
2. **重启 remote worker**：`remote_eval_worker._run_container` 的温度透传是宿主机进程改动，必须重启才生效
   （`runner.py` 的改动随 workspace tar 进容器，**不用重建镜像**）。
3. **vLLM**：`--max-model-len 262144`（256k）+ tool-calling：`--enable-auto-tool-choice --tool-call-parser qwen3_coder`（或 hermes）。
4. 镜像 + 设备：`OPENHANDS_IMAGE=openhands-triton-env:v1`、`OPENHANDS_EVAL_DEVICE_IDS=0,1,2,3`。

## 怎么定 `--max-iterations`（先小批量摸 step）
先用**宽松上限**跑几个算子，让轨迹自然收敛、不被截断，再读它实际用了多少步：
```bash
EVAL_MAX_CONCURRENT=4 OPENHANDS_REMOTE_EVAL_URL=http://127.0.0.1:18880 \
OPENHANDS_IMAGE=openhands-triton-env:v1 OPENHANDS_EVAL_DEVICE_IDS=0,1,2,3 \
LLM_BASE_URL=http://127.0.0.1:8003/v1 OPENHANDS_MODEL_NAME=qwen35 \
bash eval/run_eval.sh /home/c00937190/AscendOpGenAgent/benchmarks/NPUKernelBench \
    --levels 1 --max-ops 6 --max-iterations 40
```
看 `<out>/sample/summary.json → step_calibration`：
- `pct_hit_cap == 0` → 没被截断，正式跑设 `--max-iterations ≈ iterations.max × 1.3`。
- `pct_hit_cap > 0` → 40 不够，有轨迹被砍 → 调大（如 60）再摸一遍。

> 先验（旧 probe 真实轨迹）：首个 success≈9 步、含 Phase3 全程≈26 步、自然终止≈30。**40 作宽松上限够用**，标定后正式跑通常落在 ~30。

## 正式跑
```bash
EVAL_MAX_CONCURRENT=4 OPENHANDS_REMOTE_EVAL_URL=http://127.0.0.1:18880 \
OPENHANDS_IMAGE=openhands-triton-env:v1 OPENHANDS_EVAL_DEVICE_IDS=0,1,2,3 \
LLM_BASE_URL=http://127.0.0.1:8003/v1 OPENHANDS_MODEL_NAME=qwen35 \
EVAL_N=4 \
bash eval/run_eval.sh /home/c00937190/AscendOpGenAgent/benchmarks/NPUKernelBench \
    --levels 1,2 --max-iterations 30
```
等价直接调（只要 sampled、自定 K）：
```bash
python3 eval/eval_openhands.py --dataset <NPUKernelBench> --levels 1,2 --rollout remote_worker \
  --llm-base-url http://127.0.0.1:8003/v1 --model qwen35 \
  --n-trajectories 4 --k-values 1,4 --temperature 0.7 --max-iterations 30 \
  --judge-timeout 1800 --out-dir eval_runs/run1
```

## 输出
- `summary.json` — 全局：`pass_at_k`（**主指标**，按 `correctness_ok` 算）、`best_of_N_speedup`、`step_calibration`。
- `per_op/{task_id}.json` — 每算子 N 条轨迹：`correctness_ok` / `speedup_vs_torch` / `iterations` / `hit_cap` / `error_type`。
- `run_eval.sh` 末尾打印 greedy/sampled pass@k + 标定对比。
- 每轨迹归档 `{out}/{task_id}/traj_{i}/`：`judge_metrics.json` / `run_meta.json` / `{op}_impl.best.py` / `conversation.log`（复盘用）。

## 旋钮（env / flag）
| | 含义 | 默认 |
|---|---|---|
| `EVAL_MAX_CONCURRENT` / `--max-concurrent` | 在飞 rollout 数，设到 ≈ 可用 NPU 数 | 4 |
| `EVAL_N` / `--n-trajectories` | sampled 轨迹数（pass@k 的 N） | 4 |
| `EVAL_T` / `--temperature` | sampled 温度（>0 才有 pass@k 多样性） | 0.7 |
| `--levels` | 数据集 level（1 / 1,2 / …） | 1,2 |
| `--max-ops` / `--op-filter` | 限制算子数 / 子串过滤 | 全部 |
| `--judge-timeout` | judge 重跑超时 | 1800 |
| `EVAL_ROLLOUT=mock` | 无 NPU 干跑（验证 loader+聚合） | remote_worker |

## 注意
- **pass@k 必须 T>0**：T=0 贪心 N 条几乎同质，pass@k 退化。`run_eval.sh` 的 greedy 趟只为 pass@1 baseline。
- judge 评的是 `.best`（R1）→ 每条轨迹得分=这次能做到的最好正确版。
- 数据集：level1+2 共 **61 个算子全部可烘焙**（`npukb_to_task --bake`，自包含、无运行时读 json）。
- eval 直连 vLLM（`--llm-base-url`），**不走训练的 LiteLLM proxy / logprob 记录**。
- **eval 自包含**：只依赖 `op_route` + 标准库，**不 import `openhands_agent`/`rllm`** → 用任意 `python3` 跑即可，无需训练环境。
- 256k 上下文下不用 condenser、不用削 reference，模型跑全量 skills。
