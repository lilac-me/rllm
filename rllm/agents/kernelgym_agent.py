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


##########
_SYSTEM_PROMPT = """\
You are looking at this PyTorch code and thinking it could be optimized with Triton. You need to create a Triton version with the `ModelNew`. This triton version must be execution on Ascend NPU platforms.

Please firstly analyze this code and think hard how you can optimize it. YOU MUST wrap your final code in a ```triton ... ``` code block. No other code block markers are acceptable.

**Please output and show your thinking, plan,
analysis etc., before your coding, which should be as
more as possible.**

Here's the PyTorch code:

"""

##########
_INITIAL_USER_TEMPLATE = """
```python
{reference_code}
```
"""

##########
_REVISION_USER_TEMPLATE = """\
Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):
{feedback}

Return an improved Triton implementation named `ModelNew` as a single ```python``` block. Let's think step by step.
"""

# TODO. fix err: "RuntimeError: grid should be less than 65536!"
# TODO. ValueError('Did you forget to add @triton.jit ? (`_builder` argument must be provided outside of JIT functions.)')
# TODO. RuntimeError: ModelNew requires x.device.type == 'ascend'
# TODO. ValueError('program_id axis must be 0, 1, or 2 but got 3')


def extract_kernel_code(solution_str: str) -> str:
    """Extract kernel code from model response (aligned with rllm-071)."""
    patterns = [
        r"# Kernel Implementation\s*\n(.*?)(?=# End|$)",
        r"```python\s*# Kernel\s*\n(.*?)```",
        r"# Your implementation:\s*\n(.*?)(?=# End|$)",
        r"# Generated kernel:\s*\n(.*?)(?=# End|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()
    code_blocks = re.findall(r"```(?:\w+)?\s*\n?(.*?)```", solution_str, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    return solution_str


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
        # 第一轮，需要初始化
        if not self._trajectory.steps:
            assert isinstance(observation, dict), ("Initial observation must be the task dict.")

            reference_code = observation.get("reference_code", "")
            # description = observation.get("description", "")

            # Seed message history with the system prompt
            if not self.messages:
                self.messages.append(
                    {"role": "system", "content": self._system_prompt}
                )

            user_content = _INITIAL_USER_TEMPLATE.format(
                reference_code=reference_code,
                # description=f"Additional context:\n{description}" if description else "",
            )
            self.messages.append({"role": "user", "content": user_content})

            new_step = Step(observation=observation)
            self._trajectory.steps.append(new_step)

        else:
            cur_step = self.get_current_state()
            if cur_step is not None:
                cur_step.reward = reward
                cur_step.done = done
                cur_step.info = info

            if done:
                return

            user_content = _REVISION_USER_TEMPLATE.format(feedback=observation)
            # if isinstance(observation, dict) and "feedback" in observation:
            #     user_content = _REVISION_USER_TEMPLATE.format(
            #         feedback=observation["feedback"]
            #     )
            # elif isinstance(observation, str):
            #     user_content = _REVISION_USER_TEMPLATE.format(feedback=observation)
            # else:
            #     user_content = _REVISION_USER_TEMPLATE.format(
            #         feedback=str(observation)
            #     )

            self.messages.append({"role": "user", "content": user_content})

            new_step = Step(observation=observation)
            self._trajectory.steps.append(new_step)


    def update_from_model(self, response: str, **kwargs: Any) -> Action:
        """Update internal state with the model's response and return an Action.

        Extracts kernel code using the same logic as rllm-071, adds the standard
        import prefix, and patches ``class ModelNew:`` → ``class ModelNew(nn.Module):``.
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
                raw = answer.strip()
            else:
                cur_step.thought = ""
                raw = response.strip()

            kernel_code = extract_kernel_code(raw)
            kernel_code = (
                "import triton\n"
                "import triton.language as tl\n"
                "import torch\n"
                "import torch.nn as nn\n"
                + kernel_code
            )
            if "class ModelNew:" in kernel_code:
                kernel_code = kernel_code.replace("class ModelNew:", "class ModelNew(nn.Module):")
            action = Action(action=kernel_code.strip())
            cur_step.action = action

        return action

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chat_completions(self) -> list[dict[str, str]]:
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
