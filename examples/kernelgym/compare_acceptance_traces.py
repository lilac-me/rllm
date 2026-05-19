#!/usr/bin/env python3
"""Compare acceptance probe traces produced by AgentPPOTrainer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_trace(path: Path):
    meta = None
    steps = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item_type = item.get("type", "step")
            if item_type == "meta":
                meta = item
                continue
            if item_type != "step":
                raise ValueError(f"{path}:{lineno}: unknown type={item_type}")
            step = int(item["global_step"])
            steps[step] = item
    return meta, steps


def _resolve_trace_path(expr: str) -> Path:
    if any(ch in expr for ch in ["*", "?", "["]):
        matches = sorted(Path().glob(expr))
        if not matches:
            raise FileNotFoundError(f"no files matched glob: {expr}")
        if len(matches) > 1:
            matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    path = Path(expr)
    if path.is_dir():
        matches = sorted(path.glob("acceptance_trace_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            raise FileNotFoundError(f"no acceptance trace found in directory: {path}")
        return matches[0]
    return path


def _cmp_float(a: float, b: float, atol: float, rtol: float) -> bool:
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def _compare_meta(left_meta, right_meta):
    if left_meta is None or right_meta is None:
        return []
    keys = ["seed", "train_batch_size", "rollout_n", "n_parallel_agents"]
    diffs = []
    for key in keys:
        if left_meta.get(key) != right_meta.get(key):
            diffs.append(f"meta.{key} differs: {left_meta.get(key)} vs {right_meta.get(key)}")
    return diffs


def _compare_steps(left_steps, right_steps, atol: float, rtol: float):
    errors = []
    left_keys = sorted(left_steps.keys())
    right_keys = sorted(right_steps.keys())
    if left_keys != right_keys:
        errors.append(
            f"step ids differ: left={left_keys[:10]}{'...' if len(left_keys) > 10 else ''} "
            f"right={right_keys[:10]}{'...' if len(right_keys) > 10 else ''}"
        )
        return errors

    fp_fields = [
        "rollout_order_fp",
        "score_row_sums_fp",
        "reward_row_sums_fp",
        "adv_row_sums_fp",
        "ret_row_sums_fp",
        "oldlp_row_sums_fp",
    ]
    for step in left_keys:
        left = left_steps[step]
        right = right_steps[step]
        if int(left.get("batch_size", -1)) != int(right.get("batch_size", -1)):
            errors.append(f"step {step}: batch_size differs: {left.get('batch_size')} vs {right.get('batch_size')}")
            return errors

        for field in fp_fields:
            if left.get(field) != right.get(field):
                errors.append(f"step {step}: {field} differs: {left.get(field)} vs {right.get(field)}")
                return errors

        left_metrics = left.get("metrics", {})
        right_metrics = right.get("metrics", {})
        metric_keys = sorted(set(left_metrics.keys()) | set(right_metrics.keys()))
        for key in metric_keys:
            if key not in left_metrics or key not in right_metrics:
                errors.append(f"step {step}: metric key presence differs for {key}")
                return errors
            if not _cmp_float(left_metrics[key], right_metrics[key], atol=atol, rtol=rtol):
                errors.append(f"step {step}: metric {key} differs: {left_metrics[key]} vs {right_metrics[key]}")
                return errors
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    left_path = _resolve_trace_path(args.left)
    right_path = _resolve_trace_path(args.right)
    left_meta, left_steps = _load_trace(left_path)
    right_meta, right_steps = _load_trace(right_path)

    errors = []
    errors.extend(_compare_meta(left_meta, right_meta))
    errors.extend(_compare_steps(left_steps, right_steps, atol=args.atol, rtol=args.rtol))

    if errors:
        print("MISMATCH")
        for e in errors:
            print(e)
        return 1

    print(f"MATCH steps={len(left_steps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
