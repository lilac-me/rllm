"""KernelGYM agent for rllm.

Drives an LLM to iteratively write and refine CUDA/Triton kernels inside the
KernelGymEnv. Formats the initial system + user prompt and appends evaluation
feedback on subsequent turns so the model can revise its kernel implementation.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Optional

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer. Your task is to write a high-performance \
CUDA or Triton kernel that is functionally equivalent to the given PyTorch \
reference implementation, but runs faster.

**Instructions:**
1. Study the reference PyTorch implementation carefully.
2. Implement a custom CUDA or Triton kernel as a Python class named `ModelNew` \
   that has the same interface as the reference `Model` class.
3. Your implementation must pass correctness checks against the reference output.
4. Optimise for speed: minimise memory bandwidth, maximise compute utilisation.
5. Wrap your final kernel code inside `<kernel>` ... `</kernel>` tags so it can \
   be extracted automatically. Only include the full, runnable Python file inside \
   the tags — no explanatory text inside the tags.

**Allowed libraries:** torch, triton, CUDA C extensions (via torch.utils.cpp_extension).
"""

_INITIAL_USER_TEMPLATE = """\
Here is the reference PyTorch implementation you need to optimise:

```python
{reference_code}
```

{description}

Please write a high-performance `ModelNew` implementation. Remember to wrap your \
final code in `<kernel>` ... `</kernel>` tags.
"""

_REVISION_USER_TEMPLATE = """\
Your previous kernel submission was evaluated. Here is the feedback:

{feedback}

Please revise your `ModelNew` implementation to fix the issues above. \
Remember to wrap your final code in `<kernel>` ... `</kernel>` tags.
"""


class KernelAgent(BaseAgent):
    """Agent that iteratively writes and refines GPU kernels within KernelGymEnv.

    Args:
        accumulate_thinking: If True, keep ``<think>...</think>`` blocks from
            previous turns in the conversation history (default False).
        system_prompt: Custom system prompt. If None, uses the built-in kernel
            engineering prompt.
    """

    def __init__(
        self,
        accumulate_thinking: bool = False,
        system_prompt: Optional[str] = None,
    ):
        self.accumulate_thinking = accumulate_thinking
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._trajectory = Trajectory()
        self.messages: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset agent state at the start of a new episode."""
        self._trajectory = Trajectory()
        self.messages = []

    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: dict,
        **kwargs: Any,
    ) -> None:
        """Update internal state after an environment observation.

        Turn 0 (initial): builds system message + first user message from the
        task dict (reference code + description).
        Subsequent turns: appends the feedback string as a new user message.
        """
        if not self._trajectory.steps:
            # ── First turn: observation IS the task dict ──────────────────
            assert isinstance(observation, dict), (
                "Initial observation must be the task dict."
            )
            reference_code = observation.get("reference_code", "")
            description = observation.get("description", "")

            # Seed message history with the system prompt
            if not self.messages:
                self.messages.append(
                    {"role": "system", "content": self._system_prompt}
                )

            user_content = _INITIAL_USER_TEMPLATE.format(
                reference_code=reference_code,
                description=f"Additional context:\n{description}" if description else "",
            )
            self.messages.append({"role": "user", "content": user_content})

            new_step = Step(observation=observation)
            self._trajectory.steps.append(new_step)

        else:
            # ── Subsequent turns: observation is the feedback dict ────────
            # Update the previous step's reward / done flag
            cur_step = self.get_current_state()
            if cur_step is not None:
                cur_step.reward = reward
                cur_step.done = done
                cur_step.info = info

            if done:
                return

            # Extract human-readable feedback
            if isinstance(observation, dict) and "feedback" in observation:
                user_content = _REVISION_USER_TEMPLATE.format(
                    feedback=observation["feedback"]
                )
            elif isinstance(observation, str):
                user_content = _REVISION_USER_TEMPLATE.format(feedback=observation)
            else:
                user_content = _REVISION_USER_TEMPLATE.format(
                    feedback=str(observation)
                )

            self.messages.append({"role": "user", "content": user_content})

            new_step = Step(observation=observation)
            self._trajectory.steps.append(new_step)

    def update_from_model(self, response: str, **kwargs: Any) -> Action:
        """Update internal state with the model's response and return an Action.

        The action payload is the raw LLM response string (the environment will
        extract the kernel code from it via ``_extract_kernel_code``).
        """
        self.messages.append({"role": "assistant", "content": response})

        cur_step = self.get_current_state()
        if cur_step is not None:
            cur_step.chat_completions = list(self.chat_completions)
            cur_step.model_response = response

            # Separate thinking from answer if present
            if response.count("</think>") == 1:
                thought, sep, answer = response.partition("</think>")
                cur_step.thought = thought + sep
                action = Action(action=answer.strip())
            else:
                cur_step.thought = ""
                action = Action(action=response.strip())

            cur_step.action = action

        return action

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return the conversation history for model inference.

        When ``accumulate_thinking=False`` (default), ``<think>`` blocks in
        previous assistant turns are stripped to keep context concise.
        """
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def get_current_state(self) -> Step | None:
        if not self._trajectory.steps:
            return None
        return self._trajectory.steps[-1]