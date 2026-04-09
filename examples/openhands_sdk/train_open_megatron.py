"""
Training entry point: OpenHands + rllm (PPO/GRPO).

OpenHands runs entirely inside Docker containers — no openhands Python
package is needed in the training environment.

The rollout() function in openhands_agent.py:
  1. Calls _build_proxied_base_url() to embed rllm metadata into the LLM URL
  2. Launches an OpenHands container with LLM_BASE_URL set to the proxied URL
  3. Waits for completion and returns a trajectory dict

Run via train_openhands.sh or directly::

    python3 -m examples.openhands_sdk.train_openhands \\
        actor_rollout_ref.model.path=<model> \\
        rllm.sdk.proxy.mode=subprocess \\
        rllm.sdk.sandbox.enabled=False

算子 agent bring-up（mock 数据 + 容器内 mock profiling）::

    export OPENHANDS_DATASET=mock_npu
    python3 -m examples.openhands_sdk.train_openhands ...
"""

import os
import warnings

import hydra

from examples.openhands_sdk.openhands_agent import rollout
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

_EX_DIR = os.path.dirname(os.path.abspath(__file__))
_MOCK_NPU_PARQUET = os.path.join(_EX_DIR, "mock_npu_operator.parquet")


class _MockNPUOperatorParquetDataset:
    """占位数据集：仅提供 verl 所需的 parquet 路径（extra_info 含 scenario=npu_operator）。"""

    def __init__(self, split: str = "train") -> None:
        self.split = split

    def get_verl_data_path(self) -> str:
        return _MOCK_NPU_PARQUET

    def get_data(self) -> list:
        return []


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer_megatron",
    version_base=None,
)
def main(config):
    dataset_mode = os.environ.get("OPENHANDS_DATASET", "swe").strip().lower()

    if dataset_mode in ("mock_npu", "npu_operator", "npu"):
        from examples.openhands_sdk.create_mock_npu_operator_data import create_parquet

        if not os.path.isfile(_MOCK_NPU_PARQUET):
            create_parquet(_MOCK_NPU_PARQUET)
        train_dataset = _MockNPUOperatorParquetDataset("train")
        val_dataset = _MockNPUOperatorParquetDataset("test")
    else:
        try:
            train_dataset = DatasetRegistry.load_dataset("swe_bench", "train")
            val_dataset = DatasetRegistry.load_dataset("swe_bench", "test")
        except Exception:
            warnings.warn(
                "swe_bench not found; falling back to 'countdown' for smoke-test.",
                stacklevel=1,
            )
            train_dataset = DatasetRegistry.load_dataset("countdown", "train")
            val_dataset = DatasetRegistry.load_dataset("countdown", "test")

    assert train_dataset, "Train dataset not found."

    trainer = AgentTrainer(
        agent_run_func=rollout,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
