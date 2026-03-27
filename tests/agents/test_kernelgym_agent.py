"""Unit tests for KernelAgent."""

from __future__ import annotations

import pytest

from rllm.agents.kernelgym_agent import KernelAgent


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
    "description": "Optimise the ReLU operation.",
}


class TestKernelAgentInit:
    def test_default_init(self):
        agent = KernelAgent()
        assert agent.messages == []
        assert agent.trajectory.steps == []

    def test_custom_system_prompt(self):
        agent = KernelAgent(system_prompt="Custom prompt.")
        assert agent._system_prompt == "Custom prompt."


class TestKernelAgentReset:
    def test_reset_clears_state(self):
        agent = KernelAgent()
        agent.messages = [{"role": "user", "content": "test"}]
        agent.reset()
        assert agent.messages == []
        assert agent.trajectory.steps == []


class TestKernelAgentUpdateFromEnv:
    def test_initial_messages_contain_system_and_user(self):
        agent = KernelAgent()
        agent.update_from_env(SAMPLE_TASK, reward=0.0, done=False, info={})

        roles = [m["role"] for m in agent.messages]
        assert roles[0] == "system"
        assert "user" in roles
        # reference code should be in the user message
        assert REFERENCE_CODE.strip() in agent.messages[-1]["content"]

    def test_feedback_turn_appends_revision_message(self):
        agent = KernelAgent()
        # Turn 0
        agent.update_from_env(SAMPLE_TASK, reward=0.0, done=False, info={})
        # Simulate model response
        agent.update_from_model(f"<kernel>\n{KERNEL_CODE}\n</kernel>")
        # Turn 1: feedback
        feedback_obs = {
            "feedback": "❌ Compilation FAILED.\nSyntaxError",
            "compiled": False,
        }
        agent.update_from_env(feedback_obs, reward=0.0, done=False, info={})

        # Last user message should reference the feedback
        last_user = [m for m in agent.messages if m["role"] == "user"][-1]
        assert "Compilation FAILED" in last_user["content"]

    def test_done_does_not_add_new_step(self):
        agent = KernelAgent()
        agent.update_from_env(SAMPLE_TASK, reward=0.0, done=False, info={})
        agent.update_from_model(f"<kernel>\n{KERNEL_CODE}\n</kernel>")
        n_steps_before = len(agent.trajectory.steps)
        # done=True should not append a new step
        agent.update_from_env({}, reward=1.0, done=True, info={})
        assert len(agent.trajectory.steps) == n_steps_before


class TestKernelAgentUpdateFromModel:
    def test_action_extraction_tagged(self):
        agent = KernelAgent()
        agent.update_from_env(SAMPLE_TASK, reward=0.0, done=False, info={})
        response = f"Explanation.\n<kernel>\n{KERNEL_CODE}\n</kernel>"
        action = agent.update_from_model(response)
        assert "ModelNew" in action.action

    def test_action_extraction_think_block(self):
        agent = KernelAgent()
        agent.update_from_env(SAMPLE_TASK, reward=0.0, done=False, info={})
        response = f"<think>Some reasoning</think>\n<kernel>\n{KERNEL_CODE}\n</kernel>"
        action = agent.update_from_model(response)
        assert "ModelNew" in action.action
        step = agent.get_current_state()
        assert "Some reasoning" in step.thought

    def test_no_code_block_fallback(self):
        agent = KernelAgent()
        agent.update_from_env(SAMPLE_TASK, reward=0.0, done=False, info={})
        response = "I could not write a kernel."
        action = agent.update_from_model(response)
        assert action.action  # not empty

    def test_assistant_message_appended(self):
        agent = KernelAgent()
        agent.update_from_env(SAMPLE_TASK, reward=0.0, done=False, info={})
        agent.update_from_model("My kernel answer.")
        assert any(m["role"] == "assistant" for m in agent.messages)


class TestKernelAgentChatCompletions:
    def test_think_stripped_in_history(self):
        agent = KernelAgent(accumulate_thinking=False)
        agent.update_from_env(SAMPLE_TASK, reward=0.0, done=False, info={})
        agent.update_from_model("<think>hidden thought</think>answer text")

        # Force a second turn so the first assistant message is NOT the last
        feedback = {"feedback": "feedback text", "compiled": False}
        agent.update_from_env(feedback, reward=0.0, done=False, info={})

        completions = agent.chat_completions
        assistant_msgs = [m for m in completions if m["role"] == "assistant"]
        # All except last should have think stripped
        for msg in assistant_msgs[:-1]:
            assert "<think>" not in msg["content"]

    def test_think_kept_when_accumulate_thinking(self):
        agent = KernelAgent(accumulate_thinking=True)
        agent.update_from_env(SAMPLE_TASK, reward=0.0, done=False, info={})
        agent.update_from_model("<think>hidden thought</think>answer text")

        feedback = {"feedback": "x", "compiled": False}
        agent.update_from_env(feedback, reward=0.0, done=False, info={})

        completions = agent.chat_completions
        assistant_msgs = [m for m in completions if m["role"] == "assistant"]
        assert any("<think>" in m["content"] for m in assistant_msgs)