"""
Stage1 layered-test mock rollout.

A drop-in replacement for ``examples.openhands_sdk.openhands_agent.rollout``
that returns a deterministic trajectory without touching docker, LiteLLM,
vllm, or the network. Used by the layered smoke pipeline to exercise rllm's
training step (Layer 5) in isolation from rollout infrastructure (Layer 4).

Activation: set environment variable ``STAGE1_MOCK_ROLLOUT=1`` before
launching ``train_openhands_qwen36_npu.{py,sh}``.

Returns a list with one trajectory dict in the schema expected by
``rllm.trainer.agent_trainer.AgentTrainer`` (mirrors the real ``rollout()``
contract in ``openhands_agent.py``).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


# Deterministic fake message exchange that exercises the parser / mask paths
# without depending on a real LLM. Two assistant turns (one tool call + one
# final answer) so multi-turn collapse logic gets exercised.
_FAKE_MESSAGES = [
    {"role": "system", "content": "You are a helpful coding agent."},
    {"role": "user", "content": "Implement a 1-d sum AscendC kernel."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "mock-tool-1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path":"sum.cce","content":"// stub"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "mock-tool-1",
        "content": '{"ok": true}',
    },
    {
        "role": "assistant",
        "content": "Done. File written.",
    },
]


def _fake_chat_completion() -> dict[str, Any]:
    """Mimic a vllm /v1/chat/completions response object (assistant tool call)."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "mock-qwen36",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "mock-tool-1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path":"sum.cce","content":"// stub"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 32, "completion_tokens": 16, "total_tokens": 48},
    }


def rollout(*args: Any, **kwargs: Any) -> list[dict]:
    """Mock replacement for ``openhands_agent.rollout``.

    Signature is intentionally ``(*args, **kwargs)`` to match the real one;
    extra_info / metadata kwargs are accepted and ignored.

    Returns:
        ``[{name, steps, reward, messages, chat_completions, extra_info}]``
        — same schema as the real rollout, with a deterministic 0.5 reward.
    """
    name = kwargs.get("name") or f"mock-task-{uuid.uuid4().hex[:8]}"
    logger.info("[mock_rollout] returning fake trajectory for task=%s", name)

    fake_completion = _fake_chat_completion()
    steps = [
        {
            "messages": _FAKE_MESSAGES,
            "chat_completions": [fake_completion],
            "reward": 0.5,
            "done": True,
            "info": {"mock": True},
        }
    ]

    return [
        {
            "name": name,
            "steps": steps,
            "reward": 0.5,
            "messages": _FAKE_MESSAGES,
            "chat_completions": [fake_completion],
            "extra_info": {"mock_rollout": True},
        }
    ]
