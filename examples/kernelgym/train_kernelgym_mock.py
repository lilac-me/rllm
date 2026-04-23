"""KernelGYM mock training entry point.

Mock mode: ``kernel.mock_eval=true`` — no KernelGYM HTTP; eval returns random
plausible fields and normal ``KernelGymRewardOps`` scoring. vLLM generation is
unchanged.

Usage:
    python -m examples.kernelgym.train_kernelgym_mock [hydra overrides ...]

Or use the regular fully-async entry with override:
    python -m examples.kernelgym.train_kernelgym_fully_async kernel.mock_eval=true ...
"""

from __future__ import annotations

import hydra

from rllm.experimental.fully_async.runner import AsyncAgentTrainer

from .train_kernelgym_fully_async import make_rollout_fn


def _register_mock_dataset():
    """Register a small mock dataset into DatasetRegistry so the framework can load it."""
    import random

    from rllm.data.dataset import DatasetRegistry

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
    ]

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


@hydra.main(
    config_path="pkg://rllm.experimental.fully_async.config",
    config_name="fully_async_ppo_megatron_trainer",
    version_base=None,
)
def main(config):
    _register_mock_dataset()

    from omegaconf import OmegaConf

    kernel_cfg = config.get("kernel", {})
    try:
        kernel_cfg = OmegaConf.to_container(kernel_cfg, resolve=True, throw_on_missing=False)
    except Exception:
        pass
    if not isinstance(kernel_cfg, dict):
        kernel_cfg = dict(kernel_cfg)
    else:
        kernel_cfg = dict(kernel_cfg)
    kernel_cfg["mock_eval"] = True

    rollout_fn = make_rollout_fn(kernel_cfg)

    trainer = AsyncAgentTrainer(
        config=config,
        rollout_fn=rollout_fn,
        val_rollout_fn=rollout_fn,
    )
    trainer.train()


if __name__ == "__main__":
    main()
