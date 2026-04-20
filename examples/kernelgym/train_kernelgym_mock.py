"""KernelGYM mock training entry point.

Mock mode: bypasses the KernelGYM eval server entirely.
The LLM generation via vLLM is REAL — only the kernel evaluation
returns fake results so we can test the full async training pipeline
end-to-end without a running eval server.

Usage:
    python -m examples.kernelgym.train_kernelgym_mock [hydra overrides ...]
"""

from __future__ import annotations

import random
import time

import hydra

from rllm.experimental.fully_async.runner import AsyncAgentTrainer

from .kernelgym_rollout import kernelgym_rollout


def _ensure_mock_eval_patched_on_worker() -> None:
    """Apply mock eval in the **current** interpreter.

    ``make_mock_rollout_fn`` runs on the driver, but ``RolloutExecutor`` executes
    ``rollout_fn`` inside Ray workers — those processes never saw the driver-side
    monkeypatch, so they still called real ``_evaluate_kernel_async`` and hit
    ``http://mock:0`` (DNS ``Name or service not known``).
    """
    import importlib
    import sys

    mod_name = kernelgym_rollout.__module__
    mod = sys.modules.get(mod_name)
    if mod is None:
        mod = importlib.import_module(mod_name)
    mod._evaluate_kernel_async = _mock_evaluate_kernel
    kernelgym_rollout.__globals__["_evaluate_kernel_async"] = _mock_evaluate_kernel


# ---------------------------------------------------------------------------
# Mock eval function — replaces _evaluate_kernel_async
# ---------------------------------------------------------------------------

async def _mock_evaluate_kernel(task, kernel_code, **kwargs) -> dict:
    """Return a plausible fake evaluation result without hitting any server."""
    import asyncio
    await asyncio.sleep(0.05)

    has_code = len(kernel_code.strip()) > 20
    compiled = has_code and random.random() < 0.8
    correctness = compiled and random.random() < 0.6
    speedup = round(random.uniform(0.5, 3.0), 2) if correctness else None

    return {
        "compiled": compiled,
        "correctness": correctness,
        "speedup": speedup,
        "status": "mock",
        "error_message": "mock compile error" if not compiled else None,
    }


# ---------------------------------------------------------------------------
# Mock dataset — a handful of tiny "reference kernels"
# ---------------------------------------------------------------------------

_MOCK_PROBLEMS = [
    {
        "prompt": "Optimize this PyTorch matrix multiplication kernel using Triton.",
        "reference_code": (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def forward(self, a, b):\n"
            "        return torch.matmul(a, b)\n\n"
            "def get_inputs():\n"
            "    return [torch.randn(256, 256), torch.randn(256, 256)]\n\n"
            "def get_init_inputs():\n"
            "    return []\n"
        ),
        "problem_id": "mock_matmul_001",
    },
    {
        "prompt": "Optimize this PyTorch softmax kernel using Triton.",
        "reference_code": (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def forward(self, x):\n"
            "        return torch.softmax(x, dim=-1)\n\n"
            "def get_inputs():\n"
            "    return [torch.randn(32, 1024)]\n\n"
            "def get_init_inputs():\n"
            "    return []\n"
        ),
        "problem_id": "mock_softmax_002",
    },
    {
        "prompt": "Optimize this PyTorch layer normalization using Triton.",
        "reference_code": (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.ln = nn.LayerNorm(512)\n\n"
            "    def forward(self, x):\n"
            "        return self.ln(x)\n\n"
            "def get_inputs():\n"
            "    return [torch.randn(16, 128, 512)]\n\n"
            "def get_init_inputs():\n"
            "    return []\n"
        ),
        "problem_id": "mock_layernorm_003",
    },
    {
        "prompt": "Optimize this PyTorch ReLU + add fusion using Triton.",
        "reference_code": (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def forward(self, x, y):\n"
            "        return torch.relu(x + y)\n\n"
            "def get_inputs():\n"
            "    return [torch.randn(64, 256), torch.randn(64, 256)]\n\n"
            "def get_init_inputs():\n"
            "    return []\n"
        ),
        "problem_id": "mock_relu_add_004",
    },
]


def _register_mock_dataset():
    """Register a small mock dataset into DatasetRegistry so the framework can load it."""
    from rllm.data.dataset import DatasetRegistry

    for split in ("train", "test"):
        DatasetRegistry.register_dataset(
            name="kernelbench_mock",
            data=_MOCK_PROBLEMS * 8,
            split=split,
            source="mock",
            description="Mock KernelGYM dataset for pipeline testing",
            category="code",
        )
    print(f"[MOCK] Registered 'kernelbench_mock' with {len(_MOCK_PROBLEMS) * 8} examples per split")


# ---------------------------------------------------------------------------
# Patch & build rollout_fn
# ---------------------------------------------------------------------------

def make_mock_rollout_fn(kernel_cfg: dict):
    """Same as make_rollout_fn but monkey-patches eval to use mock."""
    _ensure_mock_eval_patched_on_worker()

    max_turns = int(kernel_cfg.get("max_turns", 3))
    system_prompt = kernel_cfg.get("system_prompt", None)

    async def rollout_fn(client, tokenizer, **kwargs):
        _ensure_mock_eval_patched_on_worker()
        start_time = time.time()
        param_version_start = client.cur_version

        task = dict(kwargs)
        result = await kernelgym_rollout(
            client=client,
            task=task,
            max_turns=max_turns,
            system_prompt=system_prompt,
            kernel_server_url="http://mock:0",
            kernel_eval_timeout=10,
        )

        trajectory = result["trajectory"]
        trajectory.reward = result["reward"]
        metrics = result["metrics"]

        end_time = time.time()
        param_version_end = client.cur_version

        metadata = {
            "processing_time": end_time - start_time,
            "param_version_start": param_version_start,
            "param_version_end": param_version_end,
            "param_version": param_version_end,
            "is_partial": param_version_start != param_version_end,
        }
        metadata.update(metrics)
        trajectory.metadata = metadata

        if random.random() < 0.1:
            print(
                f"\n[MOCK ROLLOUT] problem={task.get('problem_id', '?')} "
                f"turns={metrics.get('num_turns')} reward={result['reward']:.4f} "
                f"compiled={metrics.get('compiled')} correctness={metrics.get('correctness')}"
            )

        return trajectory

    return rollout_fn


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="pkg://rllm.experimental.fully_async.config",
    config_name="fully_async_ppo_megatron_trainer",
    version_base=None,
)
def main(config):
    _register_mock_dataset()

    kernel_cfg = config.get("kernel", {})
    rollout_fn = make_mock_rollout_fn(kernel_cfg)

    trainer = AsyncAgentTrainer(
        config=config,
        rollout_fn=rollout_fn,
        val_rollout_fn=rollout_fn,
    )
    trainer.train()


if __name__ == "__main__":
    main()
