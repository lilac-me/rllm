"""
Prepare real NPU operator data for stage1 training.

Converts source records (kernelgym-style JSONL, rl_single_ops-style JSON, or
HuggingFace dataset) into the parquet schema consumed by
``train_openhands_qwen36_npu.{py,sh}`` via verl's ``data.train_files``.

Target parquet schema (same as create_mock_npu_operator_data.py):

    prompt:     str  -- json-encoded list[{role, content}] (chat messages)
    extra_info: dict -- {
        instruction:  str        # user-facing problem statement
        scenario:     str        # 'npu_operator' / 'npu_ascend_operator'
                                 # rollout uses this to pick workspace setup
        op_name:      str        # short identifier for metrics aggregation
        arch:         str        # e.g. 'ascend910b1' / 'ascend910b'
        task_code:    str        # PyTorch reference implementation; written
                                 # into OpenHands workspace
        reward_model: dict       # optional {ground_truth, ...} for in-container
                                 # reward computation
    }

Supported input formats
-----------------------

1. ``--source local-jsonl`` — kernelgym-style JSONL, one record per line::

       {"task": {"problem_id": "...", "reference_code": "...",
                 "description": "...", "prompt": "...", "entry_point": "Model"},
        "backend": "triton"}

2. ``--source local-json`` — rl_single_ops-style JSON, a list of records::

       [{"prompt": [{"role":"user", "content":"..."}],
         "py_code": "...",
         "ops": "[\"nn.Conv2d\"]",
         "reward_model": {"ground_truth": "..."}}, ...]

3. ``--source hf`` — HuggingFace dataset path (local directory under
   ``hf_dataset/`` or remote Hub identifier); each row treated as
   kernelgym-style if ``task`` field present, else rl_single_ops-style.

Usage
-----

::

    # Kernelgym-style JSONL → parquet
    python3 -m examples.openhands_sdk.prepare_npu_operator_data \\
        --source local-jsonl \\
        --input ./data/drkernel_rl_data.jsonl \\
        --output ./examples/openhands_sdk/real_ops_train.parquet \\
        --scenario npu_ascend_operator \\
        --arch ascend910b

    # 切 train/val
    python3 -m examples.openhands_sdk.prepare_npu_operator_data \\
        --source local-jsonl \\
        --input ./data/drkernel_rl_data.jsonl \\
        --output-train ./examples/openhands_sdk/real_ops_train.parquet \\
        --output-val ./examples/openhands_sdk/real_ops_val.parquet \\
        --val-frac 0.05 \\
        --seed 42

    # HuggingFace 本地目录
    python3 -m examples.openhands_sdk.prepare_npu_operator_data \\
        --source hf --input ./hf_dataset/drkernel-rl-data --hf-split train \\
        --output ./examples/openhands_sdk/real_ops_train.parquet \\
        --scenario npu_ascend_operator --arch ascend910b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Source loaders — yield dicts in source-native shape
# ---------------------------------------------------------------------------


def _load_local_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} not valid JSON: {e}") from None


def _load_local_json(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"{path} must contain a JSON array of records at top level; got {type(data).__name__}"
        )
    yield from data


def _load_hf(input_id: str, split: str) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "--source hf needs `pip install datasets`. "
            "On the NPU env this should already be present (verl dep)."
        ) from None
    ds = load_dataset(input_id, split=split)
    for row in ds:
        yield dict(row)


# ---------------------------------------------------------------------------
# Schema normalizer — convert any source-native row into target schema
# ---------------------------------------------------------------------------


def _row_to_record(row: dict, *, scenario: str, arch: str) -> dict:
    """Normalize a single source record to the target stage1 schema."""

    # ── kernelgym-style: {"task": {...}, "backend": "..."} ──────────────
    if "task" in row and isinstance(row["task"], dict):
        task = row["task"]
        problem_id = task.get("problem_id") or "unknown_problem"
        instruction = task.get("prompt") or task.get("description") or ""
        reference_code = task.get("reference_code", "") or ""
        # Synthesize a short op_name (problem_id can be long like
        # "cuda_llm_763652_Conv2d+log2" — keep it for traceability)
        op_name = problem_id
        reward_model = {
            # Carry whole task as ground truth for the in-container reward
            # function to introspect (problem_id, entry_point, reference, etc.)
            "ground_truth": reference_code,
            "entry_point": task.get("entry_point", "Model"),
            "backend": row.get("backend", "triton"),
            "problem_id": problem_id,
        }

    # ── rl_single_ops-style: {"prompt": [...], "py_code": "...", "ops": "...",
    #                           "reward_model": {...}} ──────────────────────
    elif "prompt" in row and ("py_code" in row or "code" in row):
        prompt_list = row.get("prompt", [])
        instruction = prompt_list[0].get("content", "") if prompt_list else ""
        reference_code = row.get("py_code") or row.get("code") or ""

        ops_raw = row.get("ops", "[]")
        try:
            ops_list = json.loads(ops_raw) if isinstance(ops_raw, str) else ops_raw
            op_name = ops_list[0] if ops_list else "unknown"
        except Exception:
            op_name = str(ops_raw)

        reward_model = dict(row.get("reward_model") or {})

    else:
        raise ValueError(
            "Unrecognized source record schema; need either kernelgym-style "
            "(top-level 'task' dict) or rl_single_ops-style (top-level "
            f"'prompt' list + 'py_code'/'code'). Keys present: {sorted(row.keys())}"
        )

    # Chat messages used as the verl `prompt` column.
    prompt_messages = [{"role": "user", "content": instruction}]

    extra_info = {
        "instruction": instruction,
        "scenario": scenario,
        "op_name": op_name,
        "arch": arch,
        "task_code": reference_code,
        "reward_model": reward_model,
    }

    return {
        "prompt": json.dumps(prompt_messages, ensure_ascii=False),
        "extra_info": extra_info,
    }


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------


def _split_records(
    records: list[dict], val_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    if val_frac <= 0:
        return records, []
    if val_frac >= 1:
        return [], records
    import random

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(records) * val_frac)))
    val_idx = set(indices[:n_val])
    val = [records[i] for i in range(len(records)) if i in val_idx]
    train = [records[i] for i in range(len(records)) if i not in val_idx]
    return train, val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _write_parquet(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    print(f"  wrote {path}  rows={len(df)}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Prepare real NPU operator data into stage1 parquet schema."
    )
    p.add_argument(
        "--source",
        required=True,
        choices=["local-jsonl", "local-json", "hf"],
        help="Input source format.",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input path (file for local-{jsonl,json}, HF id or local dir for hf).",
    )
    p.add_argument(
        "--hf-split",
        default="train",
        help="HuggingFace split name when --source=hf (default: train).",
    )
    p.add_argument(
        "--scenario",
        default="npu_ascend_operator",
        choices=["npu_operator", "npu_ascend_operator"],
        help="Rollout-side workspace branch selector (default: npu_ascend_operator).",
    )
    p.add_argument(
        "--arch",
        default="ascend910b",
        help="Target hardware architecture string (default: ascend910b).",
    )

    p.add_argument("--output", help="Output parquet path (single file, no split).")
    p.add_argument("--output-train", help="Output parquet for train split.")
    p.add_argument("--output-val", help="Output parquet for val split.")
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.0,
        help="Fraction held out for validation (0..1). Default 0 = no split.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for split.")

    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only convert the first N records (smoke test).",
    )

    args = p.parse_args()

    # ── Argument sanity ─────────────────────────────────────────────────
    if args.val_frac > 0:
        if not (args.output_train and args.output_val):
            print(
                "[prepare] --val-frac > 0 requires both --output-train and --output-val",
                file=sys.stderr,
            )
            return 2
    else:
        if not args.output:
            print(
                "[prepare] need --output (single file) or --output-train + --output-val",
                file=sys.stderr,
            )
            return 2

    # ── Load source ─────────────────────────────────────────────────────
    print(f"[prepare] source={args.source} input={args.input}")
    if args.source == "local-jsonl":
        raw_iter = _load_local_jsonl(Path(args.input))
    elif args.source == "local-json":
        raw_iter = _load_local_json(Path(args.input))
    elif args.source == "hf":
        raw_iter = _load_hf(args.input, args.hf_split)
    else:
        raise AssertionError("unreachable")

    # ── Convert ────────────────────────────────────────────────────────
    records: list[dict] = []
    bad = 0
    for i, row in enumerate(raw_iter):
        if args.limit and i >= args.limit:
            break
        try:
            records.append(_row_to_record(row, scenario=args.scenario, arch=args.arch))
        except ValueError as e:
            bad += 1
            if bad <= 5:
                print(f"  [skip] row {i}: {e}", file=sys.stderr)
            continue
    if bad:
        print(f"[prepare] skipped {bad} unparseable rows")
    if not records:
        print("[prepare] no usable rows found", file=sys.stderr)
        return 1
    print(f"[prepare] converted {len(records)} records")

    # ── Write ──────────────────────────────────────────────────────────
    if args.val_frac > 0:
        train, val = _split_records(records, args.val_frac, args.seed)
        print(f"[prepare] split  train={len(train)}  val={len(val)}  seed={args.seed}")
        _write_parquet(train, Path(args.output_train))
        _write_parquet(val, Path(args.output_val))
    else:
        _write_parquet(records, Path(args.output))

    print("[prepare] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
