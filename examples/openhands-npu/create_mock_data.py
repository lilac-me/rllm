"""
Generate mock NPU operator dataset as a parquet file for verl.

verl's RayPPOTrainer expects data.train_files / data.val_files to point
to real parquet files. This script creates a minimal parquet with the
required schema.

Usage:
    python create_mock_data.py

Output:
    openhands-sdk/mock_npu_data.parquet
"""

import json
import os
import pandas as pd

_MOCK_NPU_TASKS = [
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
    """Create a verl-compatible parquet file from mock tasks."""
    rows = []
    for i, task in enumerate(_MOCK_NPU_TASKS):
        instruction = task["instruction"]
        
        # verl 往往期望 prompt 是 JSON 编码的消息列表或者直接是普通字符串
        # 既然之前这里没报错，我们保持不变
        prompt = json.dumps([
            {"role": "user", "content": instruction}
        ])
        
        # 【核心修复】：直接组装 Python 字典，不要用 json.dumps！
        # 并且为了满足 verl 源码的需求，显式塞入一个 "index" 字段
        extra_info = {
            "instruction": instruction,
        }
        
        rows.append({
            "prompt": prompt,
            "extra_info": extra_info,  # Parquet (通过 pyarrow) 会自动把它存为 Struct 类型
        })

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"Created mock parquet at: {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(script_dir, "mock_npu_data.parquet")
    create_parquet(output)