#!/usr/bin/env python3
"""Compare two rLLM debug rollout dumps for deterministic rollout checks."""

from __future__ import annotations

import argparse
import math
from typing import Any

torch = None


def _load_torch():
    global torch
    if torch is None:
        import torch as torch_module

        torch = torch_module


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _first_seq_diff(left: list, right: list) -> str | None:
    for idx, (left_value, right_value) in enumerate(zip(left, right, strict=False)):
        if left_value != right_value:
            return f"position {idx}: {left_value} vs {right_value}"
    if len(left) != len(right):
        common_len = min(len(left), len(right))
        left_next = left[common_len : common_len + 5]
        right_next = right[common_len : common_len + 5]
        return (
            f"length differs: {len(left)} vs {len(right)}; "
            f"common_prefix={common_len}; left_next={left_next}; right_next={right_next}"
        )
    return None


def _first_float_diff(left: list[float], right: list[float], atol: float, rtol: float) -> str | None:
    if len(left) != len(right):
        return f"length differs: {len(left)} vs {len(right)}"
    for idx, (left_value, right_value) in enumerate(zip(left, right, strict=False)):
        if not math.isclose(float(left_value), float(right_value), abs_tol=atol, rel_tol=rtol):
            return f"position {idx}: {left_value} vs {right_value}"
    return None


def _trajectory_sort_key(item: dict, fallback_idx: int = 0) -> tuple[int, str, int, int]:
    trajectory_key = item.get("trajectory_key")
    if trajectory_key not in (None, ""):
        return (0, str(trajectory_key), int(item.get("replica_idx", 0) or 0), int(item.get("idx", fallback_idx)))
    task_id = item.get("task_id") or item.get("problem_id")
    if task_id not in (None, ""):
        return (1, str(task_id), int(item.get("replica_idx", 0) or 0), int(item.get("idx", fallback_idx)))
    return (2, "", 0, int(item.get("idx", fallback_idx)))


def _trajectory_label(item: dict, fallback_idx: int = 0) -> str:
    idx = int(item.get("idx", fallback_idx))
    trajectory_key = item.get("trajectory_key")
    task_id = item.get("task_id") or item.get("problem_id")
    replica_idx = int(item.get("replica_idx", 0) or 0)
    if trajectory_key not in (None, ""):
        return f"key={trajectory_key} replica={replica_idx} idx={idx}"
    if task_id not in (None, ""):
        return f"task={task_id} replica={replica_idx} idx={idx}"
    return f"idx={idx}"


def _compare_step_trajectories(left: list[dict], right: list[dict], atol: float, rtol: float) -> list[str]:
    errors = []
    if len(left) != len(right):
        return [f"trajectory count differs: {len(left)} vs {len(right)}"]

    for traj_pos, (left_traj, right_traj) in enumerate(zip(left, right, strict=False)):
        left_label = _trajectory_label(left_traj, traj_pos)
        right_label = _trajectory_label(right_traj, traj_pos)
        left_sort = _trajectory_sort_key(left_traj, traj_pos)
        right_sort = _trajectory_sort_key(right_traj, traj_pos)
        if left_sort[:3] != right_sort[:3]:
            errors.append(f"trajectory identity differs: {left_label} vs {right_label}")
            break

        if not math.isclose(float(left_traj.get("trajectory_reward", 0.0)), float(right_traj.get("trajectory_reward", 0.0)), abs_tol=atol, rel_tol=rtol):
            errors.append(
                f"trajectory {left_label} reward differs: "
                f"{left_traj.get('trajectory_reward')} vs {right_traj.get('trajectory_reward')}"
            )
            break

        left_steps = left_traj.get("steps", [])
        right_steps = right_traj.get("steps", [])
        if len(left_steps) != len(right_steps):
            errors.append(f"trajectory {left_label} step count differs: {len(left_steps)} vs {len(right_steps)}")
            break

        for step_idx, (left_step, right_step) in enumerate(zip(left_steps, right_steps, strict=False)):
            for field in ("prompt_ids", "completion_ids"):
                diff = _first_seq_diff(_as_list(left_step.get(field)), _as_list(right_step.get(field)))
                if diff:
                    errors.append(f"trajectory {left_label} step={step_idx} {field} {diff}")
                    return errors
            diff = _first_float_diff(_as_list(left_step.get("logprobs")), _as_list(right_step.get("logprobs")), atol, rtol)
            if diff:
                errors.append(f"trajectory {left_label} step={step_idx} logprobs {diff}")
                return errors

    return errors


def _compare_token_trajectories(left: list[dict], right: list[dict], atol: float, rtol: float) -> list[str]:
    errors = []
    if len(left) != len(right):
        return [f"trajectory count differs: {len(left)} vs {len(right)}"]

    for traj_pos, (left_traj, right_traj) in enumerate(zip(left, right, strict=False)):
        left_label = _trajectory_label(left_traj, traj_pos)
        right_label = _trajectory_label(right_traj, traj_pos)
        left_sort = _trajectory_sort_key(left_traj, traj_pos)
        right_sort = _trajectory_sort_key(right_traj, traj_pos)
        if left_sort[:3] != right_sort[:3]:
            errors.append(f"trajectory identity differs: {left_label} vs {right_label}")
            break

        if not math.isclose(float(left_traj.get("trajectory_reward", 0.0)), float(right_traj.get("trajectory_reward", 0.0)), abs_tol=atol, rel_tol=rtol):
            errors.append(
                f"trajectory {left_label} reward differs: "
                f"{left_traj.get('trajectory_reward')} vs {right_traj.get('trajectory_reward')}"
            )
            break

        for field in ("prompt_tokens", "response_tokens", "response_masks"):
            diff = _first_seq_diff(_as_list(left_traj.get(field)), _as_list(right_traj.get(field)))
            if diff:
                errors.append(f"trajectory {left_label} {field} {diff}")
                return errors

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    _load_torch()

    left_payload = torch.load(args.left, map_location="cpu", weights_only=False)
    right_payload = torch.load(args.right, map_location="cpu", weights_only=False)

    if left_payload.get("mode") != right_payload.get("mode"):
        print(f"mode differs: {left_payload.get('mode')} vs {right_payload.get('mode')}")
        return 1

    left_trajectories = sorted(
        left_payload["trajectories"], key=lambda item: _trajectory_sort_key(item, item.get("idx", 0))
    )
    right_trajectories = sorted(
        right_payload["trajectories"], key=lambda item: _trajectory_sort_key(item, item.get("idx", 0))
    )
    mode = left_payload.get("mode")

    if mode == "step":
        errors = _compare_step_trajectories(left_trajectories, right_trajectories, args.atol, args.rtol)
    else:
        errors = _compare_token_trajectories(left_trajectories, right_trajectories, args.atol, args.rtol)

    if errors:
        print("MISMATCH")
        for error in errors:
            print(error)
        return 1

    print(f"MATCH mode={mode} trajectories={len(left_trajectories)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
