"""
Training entry point: OpenHands + AgentSDKEngine/AgentSDKTrainer for NPU operators.

Uses a MOCK dataset of NPU operator tasks for the functional bring-up phase.
Once the end-to-end pipeline is verified, replace MockNPUOperatorDataset with
a real KernelBench / kernelGYM dataset loader.

The agent runs inside Docker containers via worker_server.py (sandbox mode).
SandboxOrchestrator manages container lifecycle, metadata slug injection,
and result collection.

Run via train_openhands.sh or directly:

    python3 openhands-npu/train_openhands.py \\
        actor_rollout_ref.model.path=<your_model> \\
        rllm.sdk.proxy.mode=subprocess \\
        rllm.sdk.store.path=${HOME}/rllm-traces.db

Mock Dataset format
-------------------
Each row has an extra_info field with::

    {
        "instruction": "Implement a VectorAdd Ascend C operator...",
    }
"""

import os
import warnings
from typing import Dict, List

import hydra

from openhands_agent import rollout
from rllm.trainer.agent_trainer import AgentTrainer


# ---------------------------------------------------------------------------
# Mock NPU dataset for functional bring-up
# ---------------------------------------------------------------------------

# A small set of mock NPU operator tasks. Each entry is a dict with at
# minimum an "instruction" field consumed by openhands_agent.rollout().
_MOCK_NPU_TASKS: List[Dict[str, str]] = [
    {
        "instruction": (
            "Implement an Ascend C custom operator 'VectorAdd' that adds two "
            "FP16 vectors element-wise. The input tensors are 1-D with size N. "
            "Optimise tiling for the Ascend AI Core pipeline."
        ),
    },
    {
        "instruction": (
            "Implement an Ascend C custom operator 'MatMul' that multiplies "
            "two FP16 matrices of size M×K and K×N. Handle tiling across "
            "the cube unit and vector unit properly."
        ),
    },
    {
        "instruction": (
            "Implement an Ascend C custom operator 'Softmax' that computes "
            "softmax over the last dimension of a 2-D FP16 tensor. Ensure "
            "numerical stability using the max-subtraction trick."
        ),
    },
    {
        "instruction": (
            "Implement an Ascend C custom operator 'LayerNorm' that performs "
            "layer normalisation on a 2-D FP16 tensor along the last dimension. "
            "Include learnable affine parameters gamma and beta."
        ),
    },
]


class MockNPUOperatorDataset:
    """Minimal mock dataset for NPU operator generation.

    Used during the functional bring-up phase when no real KernelBench
    dataset is available. Replace with ``DatasetRegistry.load_dataset(...)``
    once a proper dataset is registered.
    """

    def __init__(self, split: str = "train") -> None:
        self.split: str = split
        self._data: List[Dict[str, str]] = list(_MOCK_NPU_TASKS)

    # rllm.data.Dataset protocol requires get_data()
    def get_data(self) -> List[Dict[str, str]]:
        return self._data

    # AgentTrainer checks for get_verl_data_path() to wire up data.train_files
    def get_verl_data_path(self) -> str:
        # 指向你脚本里生成的那个 mock parquet
        return "/home/g00841271/rllm-071/examples/openhands-npu/mock_npu_data.parquet"


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config: object) -> None:  # DictConfig at runtime, but we avoid the import
    # ------------------------------------------------------------------
    # Load mock datasets
    # ------------------------------------------------------------------
    train_dataset = MockNPUOperatorDataset(split="train")
    val_dataset = MockNPUOperatorDataset(split="test")

    # ------------------------------------------------------------------
    # Build trainer with OpenHands rollout as the agent_run_func.
    #
    # The rollout function runs inside Docker containers managed by
    # SandboxOrchestrator. worker_server.py handles metadata slug
    # injection and result submission.
    # ------------------------------------------------------------------
    trainer = AgentTrainer(
        agent_run_func=rollout,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()