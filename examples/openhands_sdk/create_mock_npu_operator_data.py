"""
生成算子 agent bring-up 用的最小 parquet（verl / AgentTrainer 读 data.train_files）。

用法:
    python3 -m examples.openhands_sdk.create_mock_npu_operator_data

或在本目录:
    python3 create_mock_npu_operator_data.py

输出:
    与本脚本同目录下的 mock_npu_operator.parquet

每行 extra_info 含 instruction 与 scenario=npu_operator，rollout 侧据此布置
mock profiling 脚本与 NPU 风格 reward。
"""

from __future__ import annotations

import json
import os

import pandas as pd

_MOCK_TASKS = [
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


def create_parquet(output_path: str) -> None:
    rows = []
    for task in _MOCK_TASKS:
        instruction = task["instruction"]
        prompt = json.dumps([{"role": "user", "content": instruction}])
        extra_info = {
            "instruction": instruction,
            "scenario": "npu_operator",
        }
        rows.append({"prompt": prompt, "extra_info": extra_info})

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Created {output_path} ({len(df)} rows)")


def default_output_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_npu_operator.parquet")


if __name__ == "__main__":
    create_parquet(default_output_path())
