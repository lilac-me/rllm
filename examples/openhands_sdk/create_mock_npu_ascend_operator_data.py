"""
将 rl_single_ops.json 转换为 mock_npu_operator.parquet 同格式的 parquet，

用法:
    python3 convert_rl_single_ops_to_parquet.py

输出:
    与本脚本同目录下的 rl_single_ops.parquet
"""

from __future__ import annotations

import json
import os

import pandas as pd

_INPUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_single_ops.json")
_OUTPUT_PARQUET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_single_ops.parquet")


def convert(output_path: str = _OUTPUT_PARQUET, input_path: str = _INPUT_JSON) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    rows = []
    for item in raw_data:
        prompt_list = item.get("prompt", [])
        instruction = prompt_list[0].get("content", "") if prompt_list else ""

        # ops 字段是 JSON 字符串化的列表，例如 '["nn.FractionalMaxPool2d"]'
        ops_raw = item.get("ops", "[]")
        try:
            ops_list = json.loads(ops_raw) if isinstance(ops_raw, str) else ops_raw
            op_name = ops_list[0] if ops_list else "unknown"
        except Exception:
            op_name = str(ops_raw)

        extra_info = {
            "instruction": instruction,
            "scenario": "npu_ascend_operator",
            "op_name": op_name,
            "arch": "ascend910b",
            "task_code": item.get("py_code", ""),
            # 保留原始 reward_model/ground_truth 供 rollout 侧使用
            "reward_model": item.get("reward_model", {}),
        }

        # MUST be list[dict] not JSON string — verl _build_messages / doc2len
        # call tokenizer.apply_chat_template(doc[prompt_key], ...) directly
        # with no json.loads; a string crashes jinja with "No user query found".
        rows.append({
            "prompt": list(prompt_list),
            "extra_info": extra_info,
        })

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Created {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    convert()
