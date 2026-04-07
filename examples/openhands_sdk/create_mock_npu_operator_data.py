"""
生成算子 agent bring-up 用的最小 parquet（verl / AgentTrainer 读 data.train_files）。

用法:
    python3 -m examples.openhands_sdk.create_mock_npu_operator_data

或在本目录:
    python3 create_mock_npu_operator_data.py

输出:
    与本脚本同目录下的 mock_npu_operator.parquet

每行 extra_info 含 op_name / task_code / arch / instruction / scenario=npu_operator，
rollout 侧据此布置 workspace 并计算 reward。

完整字段说明见同目录 TRAINING_DATA.md。
"""

from __future__ import annotations

import json
import os
import textwrap

import pandas as pd

_MOCK_TASKS = [
    {
        "op_name": "vector_add",
        "arch": "ascend910b1",
        "instruction": (
            "Implement a Triton kernel 'vector_add' that adds two FP16 "
            "vectors element-wise on Ascend 910B."
        ),
        "task_code": textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    return a + b

            def get_inputs():
                return [torch.randn(4096, dtype=torch.float16), torch.randn(4096, dtype=torch.float16)]

            def get_init_inputs():
                return []
        """),
    },
    {
        "op_name": "matmul",
        "arch": "ascend910b1",
        "instruction": (
            "Implement a Triton kernel 'matmul' that multiplies two FP16 "
            "matrices of shape (M, K) and (K, N) on Ascend 910B."
        ),
        "task_code": textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    return torch.matmul(a, b)

            def get_inputs():
                return [torch.randn(512, 256, dtype=torch.float16), torch.randn(256, 512, dtype=torch.float16)]

            def get_init_inputs():
                return []
        """),
    },
    {
        "op_name": "softmax",
        "arch": "ascend910b1",
        "instruction": (
            "Implement a Triton kernel 'softmax' that computes softmax over "
            "the last dimension of a 2-D FP16 tensor on Ascend 910B."
        ),
        "task_code": textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return torch.softmax(x, dim=-1)

            def get_inputs():
                return [torch.randn(128, 1024, dtype=torch.float16)]

            def get_init_inputs():
                return []
        """),
    },
    {
        "op_name": "layer_norm",
        "arch": "ascend910b1",
        "instruction": (
            "Implement a Triton kernel 'layer_norm' that performs layer "
            "normalisation along the last dimension of a 2-D FP16 tensor "
            "on Ascend 910B."
        ),
        "task_code": textwrap.dedent("""\
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.ln = nn.LayerNorm(512)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.ln(x.float()).half()

            def get_inputs():
                return [torch.randn(64, 512, dtype=torch.float16)]

            def get_init_inputs():
                return []
        """),
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
            "op_name": task["op_name"],
            "arch": task["arch"],
            "task_code": task["task_code"],
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
