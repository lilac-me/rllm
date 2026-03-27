"""Unit tests for KernelGymEnv — no real KernelGYM server required.

All HTTP calls to the KernelGYM server are patched with unittest.mock.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rllm.environments.kernelgym.kernelgym_env import (
    KernelGymEnv,
    _compute_reward,
    _extract_kernel_code,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal task dict
# ---------------------------------------------------------------------------

REFERENCE_CODE = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)
"""

KERNEL_CODE = """\
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x):
        return torch.relu(x)
"""

SAMPLE_TASK = {
    "problem_id": "relu_test",
    "reference_code": REFERENCE_CODE,
    "description": "Optimise ReLU.",
    "entry_point": "Model",
    "backend": "cuda",
}


# ---------------------------------------------------------------------------
# _extract_kernel_code
# ---------------------------------------------------------------------------

class TestExtractKernelCode:
    def test_tagged(self):
        text = f"Here is my solution:\n<kernel>\n{KERNEL_CODE}\n</kernel>\n"
        out = _extract_kernel_code(text)
        assert "ModelNew" in out

    def test_fenced_python(self):
        text = f"My kernel:\n```python\n{KERNEL_CODE}\n```\n"
        out = _extract_kernel_code(text)
        assert "ModelNew" in out

    def test_fenced_last_wins(self):
        text = "First block:\n```python\nfoo\n```\nSecond block:\n```python\nbar\n```"
        out = _extract_kernel_code(text)
        assert out == "bar"

    def test_fallback(self):
        text = "plain text code"
        out = _extract_kernel_code(text)
        assert out == "plain text code"

    def test_tag_takes_priority_over_fence(self):
        text = "```python\nfence\n```\n<kernel>\ntag\n</kernel>"
        out = _extract_kernel_code(text)
        assert out == "tag"


# ---------------------------------------------------------------------------
# _compute_reward
# ---------------------------------------------------------------------------

class TestComputeReward:
    def test_all_fail(self):
        r = _compute_reward(False, None, None)
        assert r == 0.0

    def test_compile_only(self):
        r = _compute_reward(True, False, None)
        assert abs(r - 0.1) < 1e-6

    def test_compile_and_correct(self):
        r = _compute_reward(True, True, None)
        assert abs(r - 0.4) < 1e-6

    def test_speedup_clipped(self):
        # speedup=20 should be capped at 10 → 0.6 speedup contribution
        r = _compute_reward(True, True, speedup=20.0)
        assert abs(r - 1.0) < 1e-6

    def test_partial_speedup(self):
        # speedup=5 → 0.1 + 0.3 + 0.6*(5/10) = 0.7
        r = _compute_reward(True, True, speedup=5.0)
        assert abs(r - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# KernelGymEnv
# ---------------------------------------------------------------------------

def _mock_evaluate(result: dict):
    """Return a context-manager-compatible mock for httpx.Client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = result
    mock_response.raise_for_status = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post = MagicMock(return_value=mock_response)
    return mock_client


class TestKernelGymEnvReset:
    def test_reset_returns_task(self):
        env = KernelGymEnv(task=SAMPLE_TASK)
        obs, info = env.reset()
        assert obs == SAMPLE_TASK
        assert isinstance(info, dict)
        assert env.current_turn == 0
        assert env.done is False

    def test_reset_with_new_task(self):
        env = KernelGymEnv(task=SAMPLE_TASK)
        env.reset()
        new_task = {**SAMPLE_TASK, "problem_id": "new_problem"}
        obs, _ = env.reset(task=new_task)
        assert obs["problem_id"] == "new_problem"

    def test_reset_clears_state(self):
        env = KernelGymEnv(task=SAMPLE_TASK)
        env.reset()
        # Manually dirty the state
        env.history = ["dummy"]
        env.current_turn = 2
        env.reset()
        assert env.history == []
        assert env.current_turn == 0


class TestKernelGymEnvStep:
    @patch("httpx.Client")
    def test_step_compile_error(self, mock_cls):
        mock_cls.return_value = _mock_evaluate({
            "compiled": False,
            "correctness": False,
            "speedup": None,
            "error_message": "SyntaxError: invalid syntax",
        })
        env = KernelGymEnv(task=SAMPLE_TASK, max_turns=3)
        env.reset()
        obs, reward, done, info = env.step(f"<kernel>\n{KERNEL_CODE}\n</kernel>")
        assert reward == 0.0
        assert done is False
        assert "compilation" in obs.get("feedback", "").lower() or "Compilation" in obs.get("feedback", "")

    @patch("httpx.Client")
    def test_step_correct_fast(self, mock_cls):
        mock_cls.return_value = _mock_evaluate({
            "compiled": True,
            "correctness": True,
            "speedup": 4.0,
            "error_message": None,
        })
        env = KernelGymEnv(task=SAMPLE_TASK, max_turns=3)
        env.reset()
        _, reward, _, _ = env.step(f"<kernel>\n{KERNEL_CODE}\n</kernel>")
        # 0.1 + 0.3 + 0.6*(4/10) = 0.64
        assert abs(reward - 0.64) < 1e-4

    @patch("httpx.Client")
    def test_step_multi_turn_feedback(self, mock_cls):
        """Verify that the next_obs on failure contains feedback text."""
        mock_cls.return_value = _mock_evaluate({
            "compiled": True,
            "correctness": False,
            "speedup": None,
            "error_message": "Output mismatch",
        })
        env = KernelGymEnv(task=SAMPLE_TASK, max_turns=3)
        env.reset()
        obs, reward, done, _ = env.step(KERNEL_CODE)
        assert not done
        assert "feedback" in obs
        assert "Correctness" in obs["feedback"] or "correctness" in obs["feedback"].lower()

    @patch("httpx.Client")
    def test_max_turns_terminates(self, mock_cls):
        mock_cls.return_value = _mock_evaluate({
            "compiled": False,
            "correctness": False,
            "speedup": None,
            "error_message": None,
        })
        env = KernelGymEnv(task=SAMPLE_TASK, max_turns=2)
        env.reset()
        _, _, done1, _ = env.step(KERNEL_CODE)
        assert done1 is False
        _, _, done2, _ = env.step(KERNEL_CODE)
        assert done2 is True


class TestKernelGymEnvFromDict:
    def test_from_dict_basic(self):
        env = KernelGymEnv.from_dict({
            "task": SAMPLE_TASK,
            "max_turns": 2,
            "kernel_server_url": "http://myserver:9000",
        })
        assert isinstance(env, KernelGymEnv)
        assert env.max_turns == 2
        assert env.kernel_server_url == "http://myserver:9000"

    def test_is_multithread_safe(self):
        assert KernelGymEnv.is_multithread_safe() is True