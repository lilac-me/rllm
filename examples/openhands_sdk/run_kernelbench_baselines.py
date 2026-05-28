#!/usr/bin/env python3
"""Run standalone OpenHands KernelBench baselines.

This script is intentionally separate from the RL training entrypoints. It
creates an isolated workspace per (baseline, operator), runs exactly one
OpenHands container/session for that pair, and records metrics for offline
baseline analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any


BASELINES = ("current", "ascend_full", "generator_verifier")


CURRENT_PROMPT = """Operator name: {op_name}
Target architecture: {arch}
Implementation file to create: `src/{op_name}_triton_ascend_impl.py`
Immediate action: your first file operation should create
`src/{op_name}_triton_ascend_impl.py` with complete executable Python code.
Do not write analysis or code in assistant messages. Use tools directly.
Do not inspect `tools/` or `tools/operator_pipeline.sh`.

Quick implementation rules, to use silently:
- Elementwise/broadcast: one vectorized tile kernel; `diag(A) @ B` is
  `C[i,j] = A[i] * B[i,j]`.
- Matmul/linear: tiled `tl.dot`; pass tensors directly, never `.data_ptr()`;
  do not cast dtype unless the reference does.
- Reductions: tile reduction, or partial-reduce plus final kernel for large axes.
- Layout/slice/transpose: write exact output indexing in Triton.
- On errors: read `metrics.json`; `int64 tl.load` means remove `.data_ptr()`;
  block mask pointer errors mean make the pointer expression block-shaped;
  `ub overflow` means reduce tile sizes.

Reference PyTorch code:
```python
{task_code}
```

Requirements:
1. Create `src/{op_name}_triton_ascend_impl.py`.
2. Define `ModelNew` with the same constructor and forward signature.
3. Keep core computation in module-scope Triton kernels.
4. Run:
   ```bash
   bash tools/operator_pipeline.sh --op_name {op_name}
   ```
5. Read `metrics.json` after every run. If success, do not edit again; run:
   ```bash
   cp src/{op_name}_triton_ascend_impl.py src/{op_name}_triton_ascend_impl_best.py
   cp metrics.json metrics_best.json
   ```
   Then finish.
"""


ASCEND_FULL_PROMPT = """Operator name: {op_name}
Target architecture: {arch}
Implementation file to create: `src/{op_name}_triton_ascend_impl.py`

Run the complete AscendOpGenAgent-style Triton flow inside this single
OpenHands session. Do not describe the stages in assistant messages; execute
them with file edits and terminal validation only.

Mandatory flow:
1. Silent design: infer exact executable semantics from `forward`,
   `get_inputs()`, and `get_init_inputs()`. Classify op type, shapes, dtype,
   strides, reductions, broadcasts, and layout transforms.
2. Generate: first file operation must create
   `src/{op_name}_triton_ascend_impl.py` with complete module-scope
   `@triton.jit` kernels and `ModelNew`.
3. Verify: run `bash tools/operator_pipeline.sh --op_name {op_name}`.
4. Repair: read `metrics.json` and any `error_file`; patch only the smallest
   failing block; rerun until success or an environment-only failure.
5. Success freeze: when `metrics.json` has `"success": true`, immediately copy
   the implementation and metrics to best files. Do not lose this best version.
{optimizer_instruction}

Implementation rules:
- Pass tensors directly to Triton kernels; never use `.data_ptr()`.
- Preserve reference dtype and output shape. Do not cast unless reference does.
- `diag(A) @ B` is `C[i,j] = A[i] * B[i,j]`; never materialize `diag(A)`.
- Matmul uses tiled `tl.dot`; reductions stay in Triton; scalar final outputs
  are also computed in Triton.
- If `ub overflow`, reduce tile sizes. If pointer/mask shape errors occur,
  make pointer expressions block-shaped.

Reference PyTorch code:
```python
{task_code}
```
"""


GENERATOR_VERIFIER_PROMPT = """Operator name: {op_name}
Target architecture: {arch}
Implementation file to create: `src/{op_name}_triton_ascend_impl.py`

Use a lightweight generator + verifier flow in this single OpenHands session.
Do not output analysis. First file operation must create the implementation.

Generate one simple correct Triton implementation, then run:
```bash
bash tools/operator_pipeline.sh --op_name {op_name}
```
Read `metrics.json`; repair only concrete failures and rerun. When success is
true, copy best files and finish. Do not perform performance tuning after first
success.

Reference PyTorch code:
```python
{task_code}
```
"""


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_records(parquet_path: Path, *, max_rows: int | None, start: int = 0) -> list[dict[str, Any]]:
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_path}. Run prepare_kernelbench_openhands_data.py first."
        )
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if start:
        df = df.iloc[start:]
    if max_rows is not None:
        df = df.head(max_rows)

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        extra = row.get("extra_info")
        if isinstance(extra, str):
            extra = json.loads(extra)
        if not isinstance(extra, dict):
            continue
        records.append(extra)
    return records


def _workspace_for_run(root: Path, baseline: str, op_name: str) -> Path:
    safe_op = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in op_name)[:120]
    return root / baseline / f"{safe_op}-{uuid.uuid4().hex[:8]}"


def _setup_workspace(workspace: Path, task: dict[str, Any], instruction: str) -> None:
    sdk_dir = _script_dir()
    workspace_pkg = sdk_dir / "workspace"
    shutil.copytree(workspace_pkg, workspace, dirs_exist_ok=True)

    lock_src = sdk_dir / "distributed_npu_lock.py"
    lock_dst = workspace / "agent_workdir" / "tools" / "scripts" / "distributed_npu_lock.py"
    if lock_src.exists():
        lock_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lock_src, lock_dst)

    agent_dir = workspace / "agent_workdir"
    src_dir = agent_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "INSTRUCTIONS.md").write_text(instruction, encoding="utf-8")
    (src_dir / f"{task['op_name']}.py").write_text(task["task_code"], encoding="utf-8")


def _prompt_for_baseline(
    baseline: str,
    task: dict[str, Any],
    *,
    full_enable_optimizer: bool,
    optimizer_attempts: int,
) -> str:
    values = {
        "op_name": task["op_name"],
        "arch": task.get("arch", "ascend910b1"),
        "task_code": task.get("task_code", ""),
    }
    if baseline == "current":
        return CURRENT_PROMPT.format(**values)
    if baseline == "generator_verifier":
        return GENERATOR_VERIFIER_PROMPT.format(**values)
    if baseline == "ascend_full":
        if full_enable_optimizer:
            optimizer_instruction = (
                f"6. Optional latency optimization: after best is saved, you may try at most "
                f"{optimizer_attempts} targeted performance edits. After each edit, rerun the "
                "pipeline and keep whichever metrics has the higher speedup. Finish with best "
                "files preserved."
            )
        else:
            optimizer_instruction = (
                "6. Performance tuning after first success is disabled for this run. Finish "
                "immediately after saving best files."
            )
        return ASCEND_FULL_PROMPT.format(optimizer_instruction=optimizer_instruction, **values)
    raise ValueError(f"unknown baseline: {baseline}")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _collect_result(
    *,
    baseline: str,
    task: dict[str, Any],
    workspace: Path,
    exit_code: int,
    started_at: float,
    finished_at: float,
) -> dict[str, Any]:
    agent_dir = workspace / "agent_workdir"
    metrics = _read_json(agent_dir / "metrics.json") or {}
    best_metrics = _read_json(agent_dir / "metrics_best.json")
    perf = metrics.get("perf_data") or {}
    best_perf = (best_metrics or {}).get("perf_data") or {}

    return {
        "baseline": baseline,
        "op_name": task.get("op_name"),
        "kernelbench_problem_id": task.get("kernelbench_problem_id"),
        "kernelbench_name": task.get("kernelbench_name"),
        "workspace": str(workspace),
        "exit_code": exit_code,
        "duration_s": round(finished_at - started_at, 3),
        "success": bool(metrics.get("success", False)),
        "ast_check_ok": bool(metrics.get("ast_check_ok", False)),
        "correctness_ok": bool(metrics.get("correctness_ok", False)),
        "speedup_vs_torch": perf.get("speedup_vs_torch"),
        "best_success": bool((best_metrics or {}).get("success", False)),
        "best_speedup_vs_torch": best_perf.get("speedup_vs_torch"),
        "error_type": metrics.get("error_type"),
        "error": str(metrics.get("error") or "")[:1000],
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for baseline in sorted({r["baseline"] for r in results}):
        rows = [r for r in results if r["baseline"] == baseline]
        total = len(rows)
        success = sum(1 for r in rows if r.get("success"))
        correctness = sum(1 for r in rows if r.get("correctness_ok"))
        ast_ok = sum(1 for r in rows if r.get("ast_check_ok"))
        speeds = [
            float(r["speedup_vs_torch"])
            for r in rows
            if r.get("speedup_vs_torch") is not None
        ]
        summary[baseline] = {
            "total": total,
            "ast_pass_rate": round(ast_ok / total, 4) if total else 0.0,
            "correctness_pass_rate": round(correctness / total, 4) if total else 0.0,
            "success_rate": round(success / total, 4) if total else 0.0,
            "speedup_mean": round(sum(speeds) / len(speeds), 4) if speeds else None,
            "speedup_gt_1_rate": round(sum(1 for s in speeds if s > 1.0) / len(speeds), 4) if speeds else None,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        default=os.environ.get(
            "OPENHANDS_KERNELBENCH_PARQUET",
            str(_script_dir() / "kernelbench_openhands.parquet"),
        ),
        help="KernelBench OpenHands parquet produced by prepare_kernelbench_openhands_data.py",
    )
    parser.add_argument("--work-dir", default=str(_script_dir() / "baseline_runs"))
    parser.add_argument("--prepare", action="store_true", help="Create/overwrite the parquet before running")
    parser.add_argument("--levels", default=os.environ.get("OPENHANDS_KERNELBENCH_LEVELS", "level_1"))
    parser.add_argument(
        "--hf-dataset",
        default=os.environ.get(
            "OPENHANDS_KERNELBENCH_DATASET",
            os.path.expanduser("~/.cache/huggingface/datasets/kernelbench"),
        ),
    )
    parser.add_argument(
        "--filter-mode",
        default=os.environ.get("OPENHANDS_KERNELBENCH_FILTER_MODE", "all"),
        help="Passed to prepare_kernelbench_openhands_data.py; use all for full level_1 baseline",
    )
    parser.add_argument("--baselines", default=",".join(BASELINES), help=f"Comma list from {BASELINES}")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--op-filter", default="", help="Substring filter on op_name/name/problem id")
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENHANDS_BASE_URL", os.environ.get("LLM_BASE_URL", "http://127.0.0.1:5000/v1")))
    parser.add_argument("--model", default=os.environ.get("OPENHANDS_MODEL_NAME", "openhands-model"))
    parser.add_argument("--image", default=os.environ.get("OPENHANDS_IMAGE", "openhands-triton-env:v1"))
    parser.add_argument("--remote-eval-url", default=os.environ.get("OPENHANDS_REMOTE_EVAL_URL", ""))
    parser.add_argument("--container-host-alias", default=os.environ.get("OPENHANDS_CONTAINER_HOST_ALIAS", "host.docker.internal"))
    parser.add_argument("--eval-device-ids", default=os.environ.get("OPENHANDS_EVAL_DEVICE_IDS", ""))
    parser.add_argument("--eval-device-count", default=os.environ.get("OPENHANDS_EVAL_DEVICE_COUNT", ""))
    parser.add_argument("--container-timeout", type=int, default=int(os.environ.get("OPENHANDS_CONTAINER_TIMEOUT", "1800")))
    parser.add_argument("--max-iterations", type=int, default=int(os.environ.get("OPENHANDS_BASELINE_MAX_ITERATIONS", os.environ.get("OPENHANDS_MAX_ITERATIONS", "12"))))
    parser.add_argument("--full-enable-optimizer", action="store_true", help="Allow ascend_full to optimize after first success while preserving best")
    parser.add_argument("--optimizer-attempts", type=int, default=2)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    selected_baselines = tuple(b.strip() for b in args.baselines.split(",") if b.strip())
    unknown = [b for b in selected_baselines if b not in BASELINES]
    if unknown:
        raise ValueError(f"unknown baseline(s): {unknown}; choices={BASELINES}")

    parquet_path = Path(args.parquet)
    if args.prepare or not parquet_path.is_file():
        from examples.openhands_sdk.prepare_kernelbench_openhands_data import _parse_levels, create_parquet

        os.environ["OPENHANDS_KERNELBENCH_FILTER_MODE"] = args.filter_mode
        create_parquet(
            str(parquet_path),
            levels=_parse_levels(args.levels),
            max_rows=None,
            hf_dataset=args.hf_dataset,
        )

    import examples.openhands_sdk.openhands_agent as agent_mod

    eval_device_count = args.eval_device_count or str(
        len([x for x in args.eval_device_ids.split(",") if x.strip()])
        if args.eval_device_ids
        else 1
    )
    agent_mod._OPENHANDS_IMAGE = args.image
    agent_mod._MODEL_NAME = args.model
    agent_mod._MAX_ITERATIONS = args.max_iterations
    agent_mod._CONTAINER_TIMEOUT = args.container_timeout
    agent_mod._OPENHANDS_REMOTE_EVAL_URL = args.remote_eval_url.strip().rstrip("/")
    agent_mod._OPENHANDS_CONTAINER_HOST_ALIAS = args.container_host_alias
    agent_mod._OPENHANDS_EVAL_DEVICE_IDS = args.eval_device_ids
    agent_mod._OPENHANDS_EVAL_DEVICE_COUNT = eval_device_count

    records = _load_records(parquet_path, max_rows=args.max_rows, start=args.start)
    if args.op_filter:
        needle = args.op_filter.lower()
        records = [
            r for r in records
            if needle in " ".join(
                str(r.get(k, "")) for k in ("op_name", "kernelbench_name", "kernelbench_problem_id")
            ).lower()
        ]

    run_root = Path(args.work_dir) / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    results_path = run_root / "results.jsonl"
    summary_path = run_root / "summary.json"

    print(f"[baseline] rows={len(records)} baselines={selected_baselines} run_root={run_root}")
    print(f"[baseline] llm_base_url={args.llm_base_url} model={args.model} max_iterations={args.max_iterations}")

    results: list[dict[str, Any]] = []
    with results_path.open("a", encoding="utf-8") as out:
        for task in records:
            for baseline in selected_baselines:
                instruction = _prompt_for_baseline(
                    baseline,
                    task,
                    full_enable_optimizer=args.full_enable_optimizer,
                    optimizer_attempts=args.optimizer_attempts,
                )
                workspace = _workspace_for_run(run_root, baseline, task["op_name"])
                _setup_workspace(workspace, task, instruction)

                started = time.time()
                print(f"[baseline] start baseline={baseline} op={task['op_name']} workspace={workspace}")
                try:
                    exit_code = agent_mod._run_remote_eval_worker(
                        str(workspace),
                        agent_mod._to_container_url(args.llm_base_url),
                        instruction,
                        task=task,
                    )
                except Exception as exc:
                    exit_code = -1
                    (workspace / "agent_workdir" / "baseline_exception.txt").write_text(
                        repr(exc),
                        encoding="utf-8",
                    )
                    print(f"[baseline] exception baseline={baseline} op={task['op_name']}: {exc!r}")

                finished = time.time()
                result = _collect_result(
                    baseline=baseline,
                    task=task,
                    workspace=workspace,
                    exit_code=int(exit_code),
                    started_at=started,
                    finished_at=finished,
                )
                results.append(result)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                print(
                    "[baseline] done "
                    f"baseline={baseline} op={task['op_name']} "
                    f"success={result['success']} correctness={result['correctness_ok']} "
                    f"speedup={result['speedup_vs_torch']} exit={exit_code}"
                )

    summary = _summarize(results)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[baseline] wrote {results_path}")
    print(f"[baseline] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
