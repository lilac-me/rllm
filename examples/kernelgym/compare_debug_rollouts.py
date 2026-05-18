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
    if len(left) != len(right):
        return f"length differs: {len(left)} vs {len(right)}"
    for idx, (left_value, right_value) in enumerate(zip(left, right, strict=False)):
        if left_value != right_value:
            return f"position {idx}: {left_value} vs {right_value}"
    return None


def _first_float_diff(left: list[float], right: list[float], atol: float, rtol: float) -> str | None:
    if len(left) != len(right):
        return f"length differs: {len(left)} vs {len(right)}"
    for idx, (left_value, right_value) in enumerate(zip(left, right, strict=False)):
        if not math.isclose(float(left_value), float(right_value), abs_tol=atol, rel_tol=rtol):
            return f"position {idx}: {left_value} vs {right_value}"
    return None


def _compare_step_trajectories(left: list[dict], right: list[dict], atol: float, rtol: float) -> list[str]:
    errors = []
    if len(left) != len(right):
        return [f"trajectory count differs: {len(left)} vs {len(right)}"]

    for traj_pos, (left_traj, right_traj) in enumerate(zip(left, right, strict=False)):
        left_idx = left_traj.get("idx", traj_pos)
        right_idx = right_traj.get("idx", traj_pos)
        if left_idx != right_idx:
            errors.append(f"trajectory {traj_pos} idx differs: {left_idx} vs {right_idx}")
            break

        if not math.isclose(float(left_traj.get("trajectory_reward", 0.0)), float(right_traj.get("trajectory_reward", 0.0)), abs_tol=atol, rel_tol=rtol):
            errors.append(
                f"trajectory idx={left_idx} reward differs: "
                f"{left_traj.get('trajectory_reward')} vs {right_traj.get('trajectory_reward')}"
            )
            break

        left_steps = left_traj.get("steps", [])
        right_steps = right_traj.get("steps", [])
        if len(left_steps) != len(right_steps):
            errors.append(f"trajectory idx={left_idx} step count differs: {len(left_steps)} vs {len(right_steps)}")
            break

        for step_idx, (left_step, right_step) in enumerate(zip(left_steps, right_steps, strict=False)):
            for field in ("prompt_ids", "completion_ids"):
                diff = _first_seq_diff(_as_list(left_step.get(field)), _as_list(right_step.get(field)))
                if diff:
                    errors.append(f"trajectory idx={left_idx} step={step_idx} {field} {diff}")
                    return errors
            diff = _first_float_diff(_as_list(left_step.get("logprobs")), _as_list(right_step.get("logprobs")), atol, rtol)
            if diff:
                errors.append(f"trajectory idx={left_idx} step={step_idx} logprobs {diff}")
                return errors

    return errors


def _compare_token_trajectories(left: list[dict], right: list[dict], atol: float, rtol: float) -> list[str]:
    errors = []
    if len(left) != len(right):
        return [f"trajectory count differs: {len(left)} vs {len(right)}"]

    for traj_pos, (left_traj, right_traj) in enumerate(zip(left, right, strict=False)):
        left_idx = left_traj.get("idx", traj_pos)
        right_idx = right_traj.get("idx", traj_pos)
        if left_idx != right_idx:
            errors.append(f"trajectory {traj_pos} idx differs: {left_idx} vs {right_idx}")
            break

        if not math.isclose(float(left_traj.get("trajectory_reward", 0.0)), float(right_traj.get("trajectory_reward", 0.0)), abs_tol=atol, rel_tol=rtol):
            errors.append(
                f"trajectory idx={left_idx} reward differs: "
                f"{left_traj.get('trajectory_reward')} vs {right_traj.get('trajectory_reward')}"
            )
            break

        for field in ("prompt_tokens", "response_tokens", "response_masks"):
            diff = _first_seq_diff(_as_list(left_traj.get(field)), _as_list(right_traj.get(field)))
            if diff:
                errors.append(f"trajectory idx={left_idx} {field} {diff}")
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

    left_trajectories = sorted(left_payload["trajectories"], key=lambda item: item.get("idx", 0))
    right_trajectories = sorted(right_payload["trajectories"], key=lambda item: item.get("idx", 0))
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
