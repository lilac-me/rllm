"""KernelGYM rollout function for fully-async training.

Wraps the multi-turn agent-environment interaction into a single
``async def rollout_fn(client, tokenizer, **kwargs) -> Trajectory``
that the ``RolloutExecutor`` can drive concurrently.

The function mirrors the conversation flow from ``KernelAgent`` +
``KernelGymEnv`` but talks to the LLM through ``RolloutClient``
(HTTP-based, version-aware) instead of the synchronous
``AgentExecutionEngine``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Optional

from rllm.experimental.fully_async.protocol import Trajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (aligned with kernelgym_agent.py)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are looking at this PyTorch code and thinking it could be optimized with Triton. You need to create a Triton version with the `ModelNew`. This triton version must be execution on Ascend NPU platforms.

Please firstly analyze this code and think hard how you can optimize it. YOU MUST wrap your final code in a ```triton ... ``` code block. No other code block markers are acceptable.

**Please output and show your thinking, plan,
analysis etc., before your coding, which should be as
more as possible.**

Here's the PyTorch code:

"""

_INITIAL_USER_TEMPLATE = """
```python
{reference_code}
```
"""

_REVISION_USER_TEMPLATE = """\
Your previous kernel submission was evaluated. Here is the feedback:

{feedback}

Please revise your `ModelNew` implementation to fix the issues above. \
Remember to wrap your **complete** final code in `<kernel>` ... `</kernel>` tags \
(the code inside tags must be substantive — at least a full kernel, not a few words).
"""


# ---------------------------------------------------------------------------
# Kernel code extraction
# ---------------------------------------------------------------------------

def _strip_thinking(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return a shallow copy of *messages* with ``<think>...</think>`` blocks
    removed from all assistant messages except the last one.

    This mirrors ``KernelAgent.chat_completions`` so the LLM context window
    is not polluted by reasoning traces from earlier turns.
    """
    import copy

    out = copy.deepcopy(messages)
    for msg in out[:-1]:
        if msg["role"] == "assistant":
            _, sep, after = msg["content"].partition("</think>")
            if sep:
                msg["content"] = after
    return out


def _content_for_kernel_extraction(response: str) -> str:
    """Align with ``KernelAgent.update_from_model`` (``kernelgym_agent.py``).

    When the model emits exactly one ``</redacted_thinking>`` marker, kernel code is
    taken from the suffix (answer) only — same string that would be passed to
    ``KernelGymEnv.step`` in the hybrid path.
    """
    if response.count("</redacted_thinking>") == 1:
        _, _, answer = response.partition("</redacted_thinking>")
        return answer.strip()
    return response.strip()


# KernelGYM server validates ``kernel_code`` length (e.g. >= 10). Prefer extractions that satisfy it.
_MIN_KERNEL_CODE_CHARS = 10


def _extract_kernel_code(text: str) -> str:
    """Extract kernel code from an LLM response.

    Collects candidates from ``<kernel>`` blocks and fenced code blocks, then prefers the
    **longest** segment that meets ``_MIN_KERNEL_CODE_CHARS`` so accidental tiny
    ``<kernel>and</kernel>`` matches do not beat a real ```triton``` block (first turn
    uses fenced code per system prompt; revision turns use ``<kernel>`` tags).
    """
    candidates: list[str] = []
    for m in re.finditer(r"<kernel>(.*?)</kernel>", text, re.DOTALL):
        s = m.group(1).strip()
        if s:
            candidates.append(s)

    fence_matches = re.findall(
        r"```(?:python|cuda|triton)?\n?(.*?)```", text, re.DOTALL
    )
    candidates.extend(s.strip() for s in fence_matches if s.strip())

    whole = text.strip()
    if whole:
        candidates.append(whole)

    if not candidates:
        return ""

    long_enough = [c for c in candidates if len(c) >= _MIN_KERNEL_CODE_CHARS]
    pool = long_enough if long_enough else candidates
    return max(pool, key=len)


# ---------------------------------------------------------------------------
# Feedback builder (same logic as KernelGymEnv._build_feedback)
# ---------------------------------------------------------------------------

def _build_feedback(
    compiled: bool,
    correctness: Optional[bool],
    speedup: Optional[float],
    error_message: Optional[str],
) -> str:
    lines = []
    if not compiled:
        lines.append("❌ Compilation FAILED.")
        if error_message:
            lines.append(f"Error:\n{error_message}")
        lines.append("Please fix the compilation error and resubmit your kernel.")
    elif not correctness:
        lines.append("✅ Compilation succeeded.")
        lines.append("❌ Correctness check FAILED — outputs do not match the reference.")
        if error_message:
            lines.append(f"Error:\n{error_message}")
        lines.append("Please fix the numerical correctness issue and resubmit.")
    else:
        speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
        lines.append("✅ Compilation succeeded.")
        lines.append("✅ Correctness check PASSED.")
        lines.append(f"Speedup over reference: {speedup_str}")
        lines.append("Can you further optimize the kernel?")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reward computation (lightweight fallback, same as kernelgym_env.py)
# ---------------------------------------------------------------------------

_W_COMPILED = 0.1
_W_CORRECT = 0.3
_W_SPEEDUP = 0.6
_SPEEDUP_CLIP_MAX = 10.0


def _compute_reward_simple(
    compiled: bool,
    correctness: Optional[bool],
    speedup: Optional[float],
) -> float:
    r = _W_COMPILED * float(compiled)
    if correctness:
        r += _W_CORRECT
    if speedup is not None and speedup > 0:
        clipped = min(speedup, _SPEEDUP_CLIP_MAX)
        r += _W_SPEEDUP * (clipped / _SPEEDUP_CLIP_MAX)
    return round(r, 4)


# ---------------------------------------------------------------------------
# Kernel evaluation via HTTP (non-blocking, uses httpx async)
# ---------------------------------------------------------------------------

_EVAL_SEMAPHORE = None

def _get_eval_semaphore() -> asyncio.Semaphore:
    global _EVAL_SEMAPHORE
    if _EVAL_SEMAPHORE is None:
        # 限制 KernelGYM 评测的最大并发数，防止发包过多导致 HTTP TimeOut
        _EVAL_SEMAPHORE = asyncio.Semaphore(32)
    return _EVAL_SEMAPHORE

async def _evaluate_kernel_async(
    task: dict,
    kernel_code: str,
    server_url: str,
    timeout: int = 300,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    toolkit: str = "kernelbench",
    backend_adapter: str = "kernelbench",
    backend: str = "cuda",
) -> dict[str, Any]:
    """POST to KernelGYM eval server asynchronously."""
    import uuid

    import httpx

    # KernelGYM service enforces task_id length <= 100.
    # Keep a deterministic prefix from problem_id and a random suffix for uniqueness.
    problem_id = str(task.get("problem_id", "task"))
    suffix = uuid.uuid4().hex[:16]
    max_task_id_len = 100
    # Reserve 1 char for "_" + suffix.
    prefix_max_len = max_task_id_len - (1 + len(suffix))
    safe_prefix = problem_id[:prefix_max_len] if prefix_max_len > 0 else "task"
    task_id = f"{safe_prefix}_{suffix}"
    kc = (kernel_code or "").strip()

    payload = {
        "task_id": task_id,
        "reference_code": task.get("reference_code", ""),
        "kernel_code": kc,
        "toolkit": task.get("toolkit", toolkit),
        "backend_adapter": task.get("backend_adapter", backend_adapter),
        "backend": task.get("backend", backend),
        "entry_point": task.get("entry_point", "Model"),
        "num_correct_trials": num_correct_trials,
        "num_perf_trials": num_perf_trials,
        "timeout": timeout,
        "priority": "normal",
        "workflow": task.get("workflow", "kernelbench"),
    }

    ref = payload.get("reference_code") or ""

    def _fail(msg: str, **extra: Any) -> dict[str, Any]:
        if extra:
            logger.warning("KernelGYM eval failed for %s: %s | %s", task_id, msg, extra)
        else:
            logger.warning("KernelGYM eval failed for %s: %s", task_id, msg)
        return {
            "compiled": False,
            "correctness": False,
            "speedup": None,
            "status": "failed",
            "error_message": msg,
        }

    if len(kc) < _MIN_KERNEL_CODE_CHARS:
        return _fail(
            f"kernel_code too short ({len(kc)} chars; min {_MIN_KERNEL_CODE_CHARS})",
            kernel_len=len(kc),
            ref_len=len(ref),
            skipped_http=True,
        )

    # Server-side ``timeout`` is only a hint inside the payload; the HTTP client must
    # allow long enough **read** time for queued work (many concurrent rollouts -> Kernel
    # backlog). A bare ``timeout + 60`` is easy to mis-tune; use explicit phases.
    read_s = max(float(timeout) * 3.0, float(timeout) + 300.0, 600.0)
    httpx_timeout = httpx.Timeout(connect=60.0, read=read_s, write=60.0, pool=read_s)

    try:
        sema = _get_eval_semaphore()
        async with sema:
            async with httpx.AsyncClient(timeout=httpx_timeout) as http_client:
                resp = await http_client.post(f"{server_url}/evaluate", json=payload)
            if resp.status_code >= 400:
                body = (resp.text or "")[:4000]
                return _fail(
                    f"HTTP {resp.status_code}",
                    http_status=resp.status_code,
                    body_preview=body,
                    kernel_len=len(kc),
                    ref_len=len(ref),
                    backend=payload.get("backend"),
                    kernel_empty=not kc.strip(),
                )
            try:
                return resp.json()
            except ValueError as exc:
                return _fail(f"invalid JSON from evaluate: {exc}", body_preview=(resp.text or "")[:2000])
    except httpx.RequestError as exc:
        return _fail(
            f"transport: {exc!r}",
            exc_type=type(exc).__name__,
            httpx_read_timeout_s=read_s,
            payload_timeout_s=timeout,
        )
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc!r}")


# ---------------------------------------------------------------------------
# Core rollout: multi-turn agent loop
# ---------------------------------------------------------------------------

async def kernelgym_rollout(
    client,
    task: dict,
    max_turns: int = 3,
    system_prompt: str | None = None,
    kernel_server_url: str = "http://localhost:8000",
    kernel_eval_timeout: int = 300,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    toolkit: str = "kernelbench",
    backend_adapter: str = "kernelbench",
    backend: str = "cuda",
    sampling_params: dict | None = None,
) -> dict[str, Any]:
    """Run a complete KernelGYM multi-turn episode.

    Returns a dict containing ``trajectory``, ``messages``, ``reward``,
    and various metrics — same shape the top-level ``rollout_fn`` expects.
    """
    system_prompt = system_prompt or _SYSTEM_PROMPT
    reference_code = task.get("reference_code", "")

    trajectory = Trajectory(sequences=[], reward=0, metadata={})
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _INITIAL_USER_TEMPLATE.format(reference_code=reference_code)},
    ]

    sampling_params = sampling_params or {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
    }

    total_completion_tokens = 0
    total_generation_time = 0.0
    total_eval_time = 0.0
    last_reward = 0.0
    best_reward = 0.0
    last_result: dict[str, Any] = {}
    overlong = False
    num_turns = 0

    try:
        for turn in range(max_turns):
            num_turns = turn + 1

            # --- LLM generation ---
            gen_start = time.time()
            chat_messages = _strip_thinking(messages) if turn > 0 else messages
            response_msg, output = await client.chat_completion(
                chat_messages, sampling_params=sampling_params
            )
            gen_time = time.time() - gen_start
            total_generation_time += gen_time

            content = response_msg.get("content", "") or ""
            messages.append({"role": "assistant", "content": content})
            trajectory.append(output.to_sequence())
            total_completion_tokens += output.num_output_tokens

            # --- Extract kernel code (same input scope as hybrid ``KernelAgent`` → ``env.step``) ---
            kernel_code = _extract_kernel_code(_content_for_kernel_extraction(content))

            # --- Evaluate ---
            eval_start = time.time()
            result = await _evaluate_kernel_async(
                task=task,
                kernel_code=kernel_code,
                server_url=kernel_server_url,
                timeout=kernel_eval_timeout,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                toolkit=toolkit,
                backend_adapter=backend_adapter,
                backend=backend,
            )
            eval_time = time.time() - eval_start
            total_eval_time += eval_time
            last_result = result

            compiled = result.get("compiled", False)
            correctness = result.get("correctness")
            speedup = result.get("speedup")
            error_msg = result.get("error_message")

            if "reward" in result and isinstance(result["reward"], (int, float)):
                turn_reward = float(result["reward"])
            else:
                turn_reward = _compute_reward_simple(compiled, correctness, speedup)

            last_reward = turn_reward
            best_reward = max(best_reward, turn_reward)

            is_last_turn = (turn == max_turns - 1)
            if is_last_turn:
                break

            if compiled and correctness:
                break

            # --- Append feedback and continue ---
            feedback = _build_feedback(compiled, correctness, speedup, error_msg)
            messages.append(
                {"role": "user", "content": _REVISION_USER_TEMPLATE.format(feedback=feedback)}
            )

    except Exception as exc:
        import traceback

        overlong = True
        logger.warning("KernelGYM rollout error for %s: %s\n%s",
                        task.get("problem_id", "?"), exc, traceback.format_exc())

    # Mask all responses when overlong (context exceeded / generation error)
    if overlong:
        for seq in trajectory.sequences:
            seq.response_masks = [0] * len(seq.response_masks)

    metrics = {
        "num_turns": num_turns,
        "total_generation_time": total_generation_time,
        "total_eval_time": total_eval_time,
        "total_completion_tokens": total_completion_tokens,
        "last_reward": last_reward,
        "best_reward": best_reward,
        "compiled": float(last_result.get("compiled", False)),
        "correctness": float(last_result.get("correctness", False) or False),
        "speedup": float(last_result.get("speedup") or 0.0),
        "problem_id": task.get("problem_id", "unknown"),
        "overlong": float(overlong),
        "merged_step": len(trajectory.merge()),
    }

    return {
        "trajectory": trajectory,
        "messages": messages,
        "reward": last_reward,
        "metrics": metrics,
    }
