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
_DRKERNEL_RL_PARQUET = os.path.join(_EX_DIR, "drkernel_rl_operator.parquet")
_KERNELBENCH_PARQUET = os.path.join(_EX_DIR, "kernelbench_openhands.parquet")


class _MockNPUOperatorParquetDataset:
    """占位数据集：仅提供 verl 所需的 parquet 路径（extra_info 含 scenario=npu_operator）。"""

    def __init__(self, split: str = "train") -> None:
        self.split = split

    def get_verl_data_path(self) -> str:
        return _MOCK_NPU_PARQUET

    def get_data(self) -> list:
        return []


class _OpenHandsParquetDataset:
    """Minimal dataset wrapper for parquet files already in verl format."""

    def __init__(self, path: str, split: str = "train") -> None:
        self.path = path
        self.split = split

    def get_verl_data_path(self) -> str:
        return self.path

    def get_data(self) -> list:
        return []


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
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
    elif dataset_mode in ("drkernel", "drkernel_rl"):
        from examples.openhands_sdk.prepare_drkernel_rl_data import create_parquet

        drkernel_parquet = os.environ.get("OPENHANDS_DRKERNEL_PARQUET", _DRKERNEL_RL_PARQUET)
        if not os.path.isfile(drkernel_parquet):
            max_rows_env = os.environ.get("OPENHANDS_DRKERNEL_MAX_ROWS", "").strip()
            max_rows = int(max_rows_env) if max_rows_env else None
            hf_dataset = os.environ.get("OPENHANDS_DRKERNEL_DATASET", "hkust-nlp/drkernel-rl-data")
            create_parquet(drkernel_parquet, max_rows=max_rows, hf_dataset=hf_dataset)
        train_dataset = _OpenHandsParquetDataset(drkernel_parquet, "train")
        val_dataset = _OpenHandsParquetDataset(drkernel_parquet, "test")
    elif dataset_mode in ("kernelbench", "kb"):
        from examples.openhands_sdk.prepare_kernelbench_openhands_data import create_parquet, _parse_levels

        kernelbench_parquet = os.environ.get("OPENHANDS_KERNELBENCH_PARQUET", _KERNELBENCH_PARQUET)
        if not os.path.isfile(kernelbench_parquet):
            max_rows_env = os.environ.get("OPENHANDS_KERNELBENCH_MAX_ROWS", "").strip()
            max_rows = int(max_rows_env) if max_rows_env else None
            levels = _parse_levels(os.environ.get("OPENHANDS_KERNELBENCH_LEVELS"))
            hf_dataset = os.environ.get("OPENHANDS_KERNELBENCH_DATASET", "ScalingIntelligence/KernelBench")
            create_parquet(kernelbench_parquet, levels=levels, max_rows=max_rows, hf_dataset=hf_dataset)
        train_dataset = _OpenHandsParquetDataset(kernelbench_parquet, "train")
        val_dataset = _OpenHandsParquetDataset(kernelbench_parquet, "test")
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
