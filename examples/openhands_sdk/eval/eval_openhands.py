#!/usr/bin/env python3
r"""eval_openhands.py — capability eval for the Triton-Ascend OpenHands skills flow.

Evaluates "what the model+skills can do" on NPUKernelBench, reusing the *exact*
training rollout + judge:

  load NPUKernelBench (level1[+level2]) --bake-> Task{op_name, src/{op}.py self-contained}
    for each op x N rollouts (temperature>0 for diversity):
        _setup_npu_operator_workspace(task)        # openhands_agent.py:247  (op_route -> agent_workdir.triton)
        POST /run -> remote_eval_worker            # rollout (Phase1/2/3, fixed entry, R1 keeps .best)
                                                   #   + judge: re-run pipeline on .best in a canonical snapshot
                                                   #   -> agent_workdir/judge_metrics.json   (independent score)
        read judge_metrics.json                    # schema v2: success / correctness_ok / perf_data.speedup_vs_torch
    aggregate: pass@k (PRIMARY) + best-of-N speedup -> per_op + summary

Why no separate scorer: the worker's judge already re-runs tools/triton_eval_pipeline.sh
on a clean snapshot the agent cannot touch (remote_eval_worker._judge_and_record), which IS
the "independent scoring" the design (EVAL_FRAMEWORK_DESIGN.md §0/§4.4) asked for.

Backends:
  --rollout mock           no NPU/worker; fabricates judge_metrics (dev-box smoke test)
  --rollout remote_worker  real path; needs OPENHANDS_REMOTE_EVAL_URL + image + vLLM (NPU host)

Concurrency: --max-concurrent, or env EVAL_MAX_CONCURRENT (env wins if --max-concurrent unset).

Dev-box smoke (no NPU):
  python eval/eval_openhands.py --dataset mock --rollout mock \
    --n-trajectories 6 --k-values 1,3,6 --out-dir /tmp/eval_out
  # real loader only (bakes NPUKernelBench, no rollout):
  python eval/eval_openhands.py --dataset /path/NPUKernelBench --levels 1 --rollout mock \
    --max-ops 5 --n-trajectories 4 --k-values 1,4 --out-dir /tmp/eval_out

Real run (NPU host):
  OPENHANDS_REMOTE_EVAL_URL=http://127.0.0.1:18880 OPENHANDS_IMAGE=openhands-triton-env:v1 \
  EVAL_MAX_CONCURRENT=4 \
  python eval/eval_openhands.py --dataset /path/NPUKernelBench --levels 1,2 \
    --rollout remote_worker --llm-base-url http://127.0.0.1:8003/v1 --model qwen35 \
    --n-trajectories 4 --k-values 1,4 --temperature 0.7 --out-dir eval_runs/run1
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OPENHANDS_DIR = _HERE.parent                       # examples/openhands_sdk
_NPUKB_TO_TASK = _OPENHANDS_DIR / "npukb_to_task.py"
_WORKSPACE_PKG = _OPENHANDS_DIR / "workspace"

# eval is decoupled from the training module (openhands_agent imports rllm at top, which
# isn't importable in a plain eval env). We reuse ONLY op_route (stdlib-only, no rllm) and
# reimplement the small workspace+tar helpers below — mirrors openhands_agent but self-contained.
sys.path.insert(0, str(_OPENHANDS_DIR))
from op_route import op_route as _op_route  # noqa: E402

# Kept in lockstep with openhands_agent._TRITON_INSTRUCTION_TEMPLATE (eval must match training).
_TRITON_INSTRUCTION_TEMPLATE = (
    "为算子 {op_name} 生成 Triton-Ascend 实现，参考算子在 src/{op_name}.py。\n"
    "严格按 AGENTS.md 的工作流执行（设计 → 生成 → 用固定入口自测迭代），把唯一提交物写到 output/submission/{op_name}_impl.py（类名 ModelNew），\n"
    "再用固定入口 bash tools/triton_eval_pipeline.sh --op_name {op_name} --impl output/submission/{op_name}_impl.py --task src/{op_name}.py 验证，\n"
    "读 metrics.json 按 error_type 迭代直到 success。未写出提交物、且未通过固定入口验证之前，不要结束任务。\n"
)


# --- workspace + tar helpers (mirror openhands_agent; no rllm dependency) ------------------ #
def _tar_directory_b64(path: str, arcname: str = ".") -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(path, arcname=arcname)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _manifest_directory(path: str, limit: int = 80) -> list[str]:
    root = Path(path)
    if not root.exists():
        return []
    items: list[str] = []
    for item in root.rglob("*"):
        items.append(item.relative_to(root).as_posix() + ("/" if item.is_dir() else ""))
        if len(items) >= limit:
            items.append("...")
            break
    return items


def _extract_tar_b64_into(encoded: str, destination: str) -> None:
    raw = base64.b64decode(encoded.encode("ascii"))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        dest = Path(destination).resolve()
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"unsafe tar member path: {member.name}")
        tar.extractall(destination)


def _setup_workspace(task: dict, trace_label: str) -> str:
    """Assemble a per-rollout workspace = shared infra + agent_workdir.{route} + src/{op}.(py|json).

    Faithful reimplementation of openhands_agent._setup_npu_operator_workspace (triton path)
    without importing that rllm-coupled module.
    """
    ws_temp_env = os.environ.get("OPENHANDS_WORKSPACE_TEMP_HOST_DIR", "").strip()
    workspace_temp = Path(ws_temp_env) if ws_temp_env else _OPENHANDS_DIR / "workspace_temp"
    workspace_temp.mkdir(parents=True, exist_ok=True)
    workspace = tempfile.mkdtemp(prefix=f"trajectory-{trace_label}-", dir=str(workspace_temp))

    op_name = task.get("op_name", "operator")
    route = _op_route(task)
    shutil.copytree(_WORKSPACE_PKG, workspace, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("agent_workdir.*"))
    shutil.copytree(_WORKSPACE_PKG / f"agent_workdir.{route}",
                    os.path.join(workspace, "agent_workdir"), dirs_exist_ok=True)

    with open(os.path.join(workspace, "agent_workdir", "INSTRUCTIONS.md"), "w", encoding="utf-8") as f:
        f.write(_TRITON_INSTRUCTION_TEMPLATE.format(op_name=op_name))

    task_code = task.get("task_code", "")
    if task_code:
        src_dir = os.path.join(workspace, "agent_workdir", "src")
        os.makedirs(src_dir, exist_ok=True)
        with open(os.path.join(src_dir, f"{op_name}.py"), "w", encoding="utf-8") as f:
            f.write(task_code)
        with open(os.path.join(src_dir, f"{op_name}.json"), "w", encoding="utf-8") as f:
            f.write(task.get("task_json", ""))
    return workspace

_TRITON_INSTRUCTION = (
    "为算子 `{op}` 生成 Triton-Ascend 实现。参考算子在 `src/{op}.py`。请严格按 `AGENTS.md` "
    "的工作流执行：把唯一提交物写到 `output/submission/{op}_impl.py`（类 `ModelNew`），"
    "用固定入口 `tools/triton_eval_pipeline.sh` 验证，按 metrics.json 迭代直到 success。"
)


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    task_id: str            # unique key for reporting, e.g. "L1_4_Abs"
    op_name: str            # python-module-safe operator name, e.g. "abs"
    arch: str
    reference_py: str       # baked, self-contained {op}.py (Model + get_input_groups)
    cases_json: str         # {op}.json (multi-case)
    task_code: str = ""     # contents of reference_py (injected into workspace src/{op}.py)
    task_json: str = ""     # contents of cases_json (injected into workspace src/{op}.json)
    instruction: str = ""


def _sanitize_op_name(stem: str) -> str:
    """NPUKernelBench '4_Abs' -> 'abs' (module-safe; verify.py imports {op}_torch)."""
    name = re.sub(r"^\d+[_-]*", "", stem).lower()
    name = re.sub(r"[^0-9a-z_]", "_", name).strip("_")
    if not name or name[0].isdigit():
        name = "op_" + (name or stem.lower())
    return name


# --------------------------------------------------------------------------- #
# 1. Loader — NPUKernelBench (real) + mock
# --------------------------------------------------------------------------- #
def _bake_task(py_path: Path, json_path: Path, staged_py: Path) -> str | None:
    """Run npukb_to_task.py --bake -> self-contained {op}.py. Returns reason-string on skip, None on ok."""
    staged_py.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(_NPUKB_TO_TASK), "--bake", "--json", str(json_path),
           "-o", str(staged_py), str(py_path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not staged_py.exists():
        return (p.stderr or p.stdout or f"exit {p.returncode}").strip().splitlines()[-1][:160]
    return None


def load_tasks(cfg: argparse.Namespace, staging: Path) -> list[Task]:
    if cfg.dataset == "mock":
        return _mock_tasks(cfg, staging)

    root = Path(cfg.dataset)
    if not root.is_dir():
        raise SystemExit(f"[eval] --dataset not a dir: {root}")
    levels = [s.strip() for s in str(cfg.levels).split(",") if s.strip()]
    tasks: list[Task] = []
    skipped: list[str] = []
    for lv in levels:
        ldir = root / f"level{lv}"
        if not ldir.is_dir():
            print(f"[eval] WARN level dir missing: {ldir}", file=sys.stderr)
            continue
        for py in sorted(ldir.glob("*.py")):
            if py.stem.endswith("_impl") or py.name.startswith("_"):
                continue
            jp = py.with_suffix(".json")
            if not jp.exists():
                skipped.append(f"{lv}/{py.stem}: no .json")
                continue
            op_name = _sanitize_op_name(py.stem)
            task_id = f"L{lv}_{py.stem}"
            if cfg.op_filter and cfg.op_filter not in task_id and cfg.op_filter not in op_name:
                continue
            staged = staging / task_id / f"{op_name}.py"
            reason = _bake_task(py, jp, staged)
            if reason:
                skipped.append(f"{task_id}: bake skip ({reason})")
                continue
            tasks.append(Task(
                task_id=task_id, op_name=op_name, arch=cfg.arch,
                reference_py=str(staged), cases_json=str(jp),
                task_code=staged.read_text(encoding="utf-8"),
                task_json=jp.read_text(encoding="utf-8"),
                instruction=_TRITON_INSTRUCTION.format(op=op_name),
            ))
    if skipped:
        print(f"[eval] loaded {len(tasks)} ops; skipped {len(skipped)}:", file=sys.stderr)
        for s in skipped[:40]:
            print(f"        - {s}", file=sys.stderr)
    if cfg.max_ops:
        tasks = tasks[: cfg.max_ops]
    return tasks


def _mock_tasks(cfg: argparse.Namespace, staging: Path) -> list[Task]:
    ops = {"softmax": "torch.softmax(x, dim=-1)", "relu": "torch.relu(x)",
           "add_scale": "x * 2.0 + 1.0", "gelu": "torch.nn.functional.gelu(x)"}
    tasks = []
    for name, body in ops.items():
        if cfg.op_filter and cfg.op_filter not in name:
            continue
        d = staging / name
        d.mkdir(parents=True, exist_ok=True)
        py = d / f"{name}.py"
        py.write_text(
            "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n"
            f"    def forward(self, x):\n        return {body}\n\n"
            "def get_init_inputs():\n    return []\n\n"
            "def get_input_groups():\n    return [[torch.randn(64, 64)]]\n", encoding="utf-8")
        jp = d / f"{name}.json"
        jp.write_text("\n".join(json.dumps({"inputs": [{"name": "x", "type": "tensor",
                      "dtype": "float16", "shape": [64, 64]}]}) for _ in range(4)), encoding="utf-8")
        tasks.append(Task(task_id=f"mock_{name}", op_name=name, arch=cfg.arch,
                          reference_py=str(py), cases_json=str(jp),
                          task_code=py.read_text(), task_json=jp.read_text(),
                          instruction=_TRITON_INSTRUCTION.format(op=name)))
    return tasks[: cfg.max_ops] if cfg.max_ops else tasks


# --------------------------------------------------------------------------- #
# 2. Rollout backends  ->  judge_metrics dict (schema v2)
# --------------------------------------------------------------------------- #
@dataclass
class TrajResult:
    task_id: str
    traj_idx: int
    judge: dict = field(default_factory=dict)   # judge_metrics.json (or fabricated)
    meta: dict = field(default_factory=dict)    # run_meta.json: total_llm_calls / hit_cap / exec_status
    exit_code: int = 0
    error: str | None = None


def run_rollout(task: Task, i: int, cfg: argparse.Namespace, out: Path) -> TrajResult:
    if cfg.rollout == "mock":
        rng = random.Random(f"{cfg.seed}-{task.task_id}-{i}")
        r = rng.random()
        iters = rng.randint(8, 30)
        meta = {"total_llm_calls": iters, "max_iterations": cfg.max_iterations,
                "hit_cap": iters >= cfg.max_iterations, "exec_status": "finished"}
        if r < 0.55:        # correct (+ a speedup)
            sp = round(0.2 + rng.random() * 1.4, 4)
            judge = {"success": sp >= 1.0, "ast_check_ok": True,
                     "correctness_ok": True, "perf_data": {"speedup_vs_torch": sp}}
        elif r < 0.8:       # ast ok, correctness fail
            judge = {"success": False, "ast_check_ok": True,
                     "correctness_ok": False, "perf_data": None, "error_type": "correctness_failed"}
        else:
            judge = {"success": False, "ast_check_ok": False,
                     "correctness_ok": False, "perf_data": None, "error_type": "ast_check_failed"}
        return TrajResult(task.task_id, i, judge, meta)

    if cfg.rollout == "remote_worker":
        return _rollout_remote_worker(task, i, cfg, out)
    raise ValueError(f"unknown --rollout: {cfg.rollout}")


def _rollout_remote_worker(task: Task, i: int, cfg: argparse.Namespace, out: Path) -> TrajResult:
    """Real path: build workspace, POST /run, read judge_metrics.json. Self-contained (no rllm)."""
    remote_url = (cfg.remote_eval_url or os.environ.get("OPENHANDS_REMOTE_EVAL_URL", "")).strip().rstrip("/")
    if not remote_url:
        return TrajResult(task.task_id, i, {}, {}, -1, "OPENHANDS_REMOTE_EVAL_URL/--remote-eval-url not set")

    task_dict = {"op_name": task.op_name, "arch": task.arch, "instruction": task.instruction,
                 "task_code": task.task_code, "task_json": task.task_json, "operator_backend": "triton"}
    try:
        ws = _setup_workspace(task_dict, trace_label=f"{task.task_id}-{i}")
    except Exception as e:
        return TrajResult(task.task_id, i, {}, {}, -1, f"setup_workspace failed: {e!r}")
    payload = {
        "workspace_tar_gz_b64": _tar_directory_b64(ws, arcname="workspace"),
        "workspace_manifest": _manifest_directory(ws),
        "proxied_url": cfg.llm_base_url or os.environ.get("LLM_BASE_URL", ""),
        "instruction": task.instruction,
        "task": task_dict,
        "image": cfg.image or os.environ.get("OPENHANDS_IMAGE", "openhands-triton-env:v1"),
        "model_name": cfg.model or os.environ.get("OPENHANDS_MODEL_NAME", "qwen35"),
        "max_iterations": cfg.max_iterations,
        "container_timeout": cfg.container_timeout,
        "judge_timeout": cfg.judge_timeout,
        "eval_device_ids": cfg.eval_device_ids or os.environ.get("OPENHANDS_EVAL_DEVICE_IDS", ""),
        "llm_temperature": str(cfg.temperature),
        "llm_max_output_tokens": str(cfg.max_output_tokens),
    }
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{remote_url}/run", data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    timeout = cfg.container_timeout + cfg.judge_timeout + 300
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return TrajResult(task.task_id, i, {}, {}, -1, f"worker POST failed: {e!r}")

    # Extract agent_workdir (contains judge_metrics.json) back into the staging workspace,
    # replacing the pre-rollout copy (mirrors openhands_agent._run_remote_eval_worker).
    if "agent_workdir_tar_gz_b64" in result:
        try:
            adir0 = Path(ws) / "agent_workdir"
            if adir0.exists():
                shutil.rmtree(adir0)
            _extract_tar_b64_into(result["agent_workdir_tar_gz_b64"], ws)
        except Exception as e:
            return TrajResult(task.task_id, i, {}, {}, -1, f"untar agent_workdir failed: {e!r}")
    judge_path = Path(ws) / "agent_workdir" / "judge_metrics.json"
    judge = {}
    if judge_path.exists():
        try:
            judge = json.loads(judge_path.read_text(encoding="utf-8"))
        except Exception as e:
            return TrajResult(task.task_id, i, {}, {}, -1, f"bad judge_metrics.json: {e!r}")
    meta_path = Path(ws) / "agent_workdir" / "run_meta.json"   # turn count for --max-iterations calibration
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Archive the trajectory for post-hoc inspection (submission/.best/conversation/self-report).
    if not cfg.no_archive:
        adir = out / task.task_id / f"traj_{i}"
        adir.mkdir(parents=True, exist_ok=True)
        for rel in ("judge_metrics.json", "run_meta.json", "metrics.json",
                    f"output/submission/{task.op_name}_impl.py",
                    f"output/submission/{task.op_name}_impl.best.py", "conversation.log"):
            src = Path(ws) / "agent_workdir" / rel
            if src.exists():
                dst = adir / Path(rel).name
                dst.write_bytes(src.read_bytes())
    # staging workspace is large (tar+extract); drop it unless asked to keep.
    if not cfg.keep_workdir:
        shutil.rmtree(ws, ignore_errors=True)

    return TrajResult(task.task_id, i, judge, meta, int(result.get("exit_code", -1)),
                      result.get("worker_error"))


# --------------------------------------------------------------------------- #
# 3. Metrics
# --------------------------------------------------------------------------- #
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al.): 1 - C(n-c,k)/C(n,k)."""
    if c <= 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else 0.0


def _pct(xs: list[float], p: float) -> float | None:
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 1)


def _judge_correct(j: dict) -> bool:
    return bool(j.get("correctness_ok", False))


def _judge_speedup(j: dict) -> float:
    pd = j.get("perf_data") or {}
    try:
        return float(pd.get("speedup_vs_torch") or 0.0)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OpenHands triton-skills capability eval (pass@k)")
    ap.add_argument("--dataset", default="mock", help="NPUKernelBench dir, or 'mock'")
    ap.add_argument("--levels", default="1,2", help="comma levels for real dataset, e.g. 1,2")
    ap.add_argument("--op-filter", default=None)
    ap.add_argument("--max-ops", type=int, default=None)
    ap.add_argument("--arch", default="ascend910b1")
    # trajectories / metrics
    ap.add_argument("--n-trajectories", type=int, default=4)
    ap.add_argument("--k-values", default="1,4")
    ap.add_argument("--temperature", type=float, default=0.7, help=">0 so N rollouts diverge for pass@k")
    ap.add_argument("--max-output-tokens", type=int, default=8192,
                    help="LLM max output tokens (8192 default; raise for reasoning models with long CoT)")
    # rollout / worker
    ap.add_argument("--rollout", choices=["mock", "remote_worker"], default="mock")
    ap.add_argument("--remote-eval-url", default=None)
    ap.add_argument("--llm-base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--max-iterations", type=int, default=30)
    ap.add_argument("--container-timeout", type=int, default=900)
    ap.add_argument("--judge-timeout", type=int, default=1800)
    ap.add_argument("--eval-device-ids", default=None)
    # run
    ap.add_argument("--max-concurrent", type=int, default=None,
                    help="in-flight rollouts; default env EVAL_MAX_CONCURRENT or 4")
    ap.add_argument("--out-dir", default="eval_runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-workdir", action="store_true")
    ap.add_argument("--no-archive", action="store_true")
    cfg = ap.parse_args()
    if cfg.max_concurrent is None:
        cfg.max_concurrent = int(os.environ.get("EVAL_MAX_CONCURRENT", "4"))
    return cfg


def main() -> int:
    cfg = parse_args()
    out = Path(cfg.out_dir).resolve()
    (out / "per_op").mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(cfg, out / "_staging")
    if not tasks:
        print("[eval] no tasks", file=sys.stderr)
        return 2
    K = [int(k) for k in str(cfg.k_values).split(",") if k.strip()]
    N = cfg.n_trajectories
    print(f"[eval] {len(tasks)} ops x N={N} | rollout={cfg.rollout} T={cfg.temperature} "
          f"k={K} concurrent={cfg.max_concurrent} | out={out}")

    per_traj: dict[str, dict[int, TrajResult]] = {t.task_id: {} for t in tasks}
    by_id = {t.task_id: t for t in tasks}
    with ThreadPoolExecutor(max_workers=cfg.max_concurrent) as ex:
        futs = {ex.submit(run_rollout, t, i, cfg, out): (t.task_id, i)
                for t in tasks for i in range(N)}
        done = 0
        for f in as_completed(futs):
            tid, i = futs[f]
            try:
                per_traj[tid][i] = f.result()
            except Exception as e:  # pragma: no cover
                per_traj[tid][i] = TrajResult(tid, i, {}, {}, -1, f"rollout raised: {e!r}")
            done += 1
            if done % max(1, len(futs) // 20) == 0 or done == len(futs):
                print(f"[eval] {done}/{len(futs)} rollouts", file=sys.stderr)

    # ---- per-op aggregation ----
    op_summaries = []
    for tid in (t.task_id for t in tasks):
        trajs = [per_traj[tid][i] for i in range(N)]
        correct = [_judge_correct(x.judge) for x in trajs]
        c = sum(correct)
        patk = {f"pass@{k}": round(pass_at_k(N, c, k), 4) for k in K}
        speeds = [_judge_speedup(x.judge) for x, ok in zip(trajs, correct) if ok]
        best = max(speeds, default=0.0)
        opd = {
            "task_id": tid, "op_name": by_id[tid].op_name, "arch": by_id[tid].arch,
            "n_trajectories": N, "num_correct": c, "pass_at_k": patk,
            "best_of_N_speedup": round(best, 4),
            "trajectories": [{"traj_idx": x.traj_idx, "correctness_ok": ok,
                              "speedup_vs_torch": _judge_speedup(x.judge),
                              "success": bool(x.judge.get("success", False)),
                              "error_type": x.judge.get("error_type"),
                              "iterations": x.meta.get("total_llm_calls"),
                              "hit_cap": bool(x.meta.get("hit_cap", False)),
                              "exit_code": x.exit_code, "error": x.error}
                             for x, ok in zip(trajs, correct)],
        }
        (out / "per_op" / f"{tid}.json").write_text(json.dumps(opd, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
        op_summaries.append(opd)

    # ---- global summary (pass@k is the headline) ----
    n = len(op_summaries)
    g_patk = {f"pass@{k}": round(sum(o["pass_at_k"][f"pass@{k}"] for o in op_summaries) / n, 4) for k in K}
    bests = [o["best_of_N_speedup"] for o in op_summaries]
    any_correct = sum(1 for o in op_summaries if o["num_correct"] > 0)
    # step calibration: how many agent turns rollouts actually used (size --max-iterations off this)
    all_meta = [x for tid in per_traj for x in per_traj[tid].values()]
    iters = [x.meta.get("total_llm_calls") for x in all_meta
             if isinstance(x.meta.get("total_llm_calls"), (int, float))]
    n_hit = sum(1 for x in all_meta if x.meta.get("hit_cap"))
    summary = {
        "n_ops": n, "n_trajectories": N, "k_values": K, "temperature": cfg.temperature,
        "rollout_backend": cfg.rollout, "model": cfg.model, "dataset": cfg.dataset, "levels": cfg.levels,
        "pass_at_k": g_patk,
        "solve_rate_any": round(any_correct / n, 4),                       # >=1/N correct
        "mean_correct_per_op": round(sum(o["num_correct"] for o in op_summaries) / n, 4),
        "best_of_N_speedup": {
            "geomean_over_correct_ops": round(geomean([b for b in bests if b > 0]), 4),
            "fast@1.0": round(sum(1 for b in bests if b >= 1.0) / n, 4),
            "fast@1.2": round(sum(1 for b in bests if b >= 1.2) / n, 4),
        },
        "step_calibration": {
            "max_iterations_set": cfg.max_iterations,
            "iterations_observed": {
                "max": max(iters) if iters else None, "p90": _pct(iters, 0.9),
                "median": _pct(iters, 0.5), "min": min(iters) if iters else None,
            },
            "pct_hit_cap": round(n_hit / len(all_meta), 4) if all_meta else 0.0,
            "n_with_meta": len(iters),
            "hint": "set --max-iterations ~1.3x of iterations.max; pct_hit_cap>0 ⇒ some "
                    "rollouts were truncated (raise --max-iterations and re-probe)",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[eval] per-op JSON in {out/'per_op'} ; summary in {out/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
