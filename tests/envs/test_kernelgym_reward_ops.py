"""Unit tests for KernelGYM hybrid reward ops (fully-async / rllm-071 alignment)."""

from __future__ import annotations

import pytest

from rllm.environments.kernelgym.kernelgym_reward_ops import (
    KernelGymHybridRewardParams,
    KernelGymRewardOps,
    preflight_validate,
)


def test_calculate_reward_like_kernel_completed() -> None:
    p = KernelGymHybridRewardParams()
    ops = KernelGymRewardOps(p)
    ret = ops.calculate_reward_like_kernel(
        {
            "status": "completed",
            "correctness": True,
            "compiled": True,
            "speedup": 2.0,
        }
    )
    assert ret["reward"] == pytest.approx(0.8)
    assert ret["score"] == pytest.approx(0.8)


def test_calculate_reward_like_kernel_not_completed() -> None:
    p = KernelGymHybridRewardParams()
    ops = KernelGymRewardOps(p)
    ret = ops.apply_reward(
        {"status": "failed", "error_message": "boom", "compiled": False}
    )
    assert ret["reward"] == pytest.approx(-1.0)


def test_from_kernel_mapping_reward_config_penalties() -> None:
    cfg = {
        "reward_config": {
            "penalties": {
                "compilation_fail": -0.9,
            }
        }
    }
    p = KernelGymHybridRewardParams.from_kernel_mapping(cfg)
    assert p.compilation_fail == pytest.approx(-0.9)


def test_preflight_validate_passes_minimal() -> None:
    ref = "import torch\nclass Model(torch.nn.Module):\n    pass\n"
    ker = "import torch\nclass ModelNew(torch.nn.Module):\n    pass\n"
    ok, miss = preflight_validate(ref, ker, "Model")
    assert ok is True
    assert miss == ""


def test_merge_apply_reward() -> None:
    p = KernelGymHybridRewardParams()
    ops = KernelGymRewardOps(p)
    merged = ops.apply_reward(
        {
            "status": "completed",
            "correctness": True,
            "compiled": True,
            "speedup": 1.0,
        }
    )
    assert ops.score_from_merged(merged) == pytest.approx(0.2)
