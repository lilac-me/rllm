"""Prepare KernelBench data for the OpenHands SDK rollout path.

The output parquet intentionally matches create_mock_npu_operator_data.py:

    prompt: JSON-encoded chat messages
    extra_info: dict consumed directly by openhands_agent.rollout()

The first OpenHands user message is generated from extra_info and includes the
full PyTorch reference code, so the agent does not need to inspect the mirrored
`src/{op_name}.py` task file before implementing `ModelNew`.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any

_HF_DATASET = os.environ.get(
    "OPENHANDS_KERNELBENCH_DATASET",
    os.path.expanduser("~/.cache/huggingface/datasets/kernelbench"),
)
_DEFAULT_LEVELS = ("level_1", "level_2")
_DEFAULT_ARCH = "ascend910b1"
_WARMUP_EXCLUDE_KEYWORDS = (
    "conv_transpose",
    "conv_transposed",
    "transpose3d",
    "transposed_3d",
    "conv3d",
    "3d_convolution",
    # "lstm",
    # "gru",
    # "rnn",
    "attention",
    "transformer",
    "conv2d",
    "conv3d",
    "conv_standard",
    # "convolution",
)


def _safe_name(value: Any, idx: int) -> str:
    raw = str(value or f"kernelbench_{idx}")
    raw = re.sub(r"[^0-9a-zA-Z_]+", "_", raw).strip("_")
    if not raw:
        raw = f"kernelbench_{idx}"
    if raw[0].isdigit():
        raw = f"task_{raw}"
    return raw[:96]


def _row_to_openhands_record(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    task_code = row.get("code") or row.get("reference_code") or ""
    if not task_code:
        return None

    level = row.get("level", "")
    problem_id = row.get("problem_id", idx)
    name = row.get("name", f"problem_{problem_id}")
    op_name = _safe_name(f"kernelbench_l{level}_{problem_id}_{name}", idx)

    instruction = (
        f"Implement KernelBench problem {problem_id}"
        f"{f' ({name})' if name else ''} as an Ascend NPU Triton operator."
    )

    messages = [{"role": "user", "content": instruction}]
    extra_info = {
        "instruction": instruction,
        "scenario": "npu_operator",
        "op_name": op_name,
        "arch": os.environ.get("OPENHANDS_KERNELBENCH_ARCH", os.environ.get("OPENHANDS_DRKERNEL_ARCH", _DEFAULT_ARCH)),
        "task_code": task_code,
        "operator_backend": os.environ.get("OPENHANDS_KERNELBENCH_OPERATOR_BACKEND", os.environ.get("OPENHANDS_DRKERNEL_OPERATOR_BACKEND", "triton")),
        "entry_point": "Model",
        "uid": op_name,
        "data_source": "kernelbench",
        "ability": "kernel_optimization",
        "kernelbench_level": str(level),
        "kernelbench_problem_id": str(problem_id),
        "kernelbench_name": str(name),
    }

    return {
        "prompt": json.dumps(messages, ensure_ascii=False),
        "extra_info": extra_info,
    }


def _parse_keywords(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def _row_search_text(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("name", "")),
        str(row.get("problem_id", "")),
        str(row.get("level", "")),
        str(row.get("code", row.get("reference_code", ""))),
    ]
    return "\n".join(parts).lower()


def _parse_optional_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def _kernelbench_filter_config() -> tuple[tuple[str, ...], tuple[str, ...], int | None, int | None, int | None, str]:
    mode = os.environ.get("OPENHANDS_KERNELBENCH_FILTER_MODE", "warmup").strip().lower()
    include = _parse_keywords(os.environ.get("OPENHANDS_KERNELBENCH_INCLUDE_KEYWORDS"))

    exclude_env = os.environ.get("OPENHANDS_KERNELBENCH_EXCLUDE_KEYWORDS")
    if exclude_env is not None:
        exclude = _parse_keywords(exclude_env)
    elif mode in ("all", "none", "off", "false", "0"):
        exclude = ()
    elif mode == "warmup":
        exclude = _WARMUP_EXCLUDE_KEYWORDS
    else:
        exclude = ()

    max_code_chars = _parse_optional_int_env("OPENHANDS_KERNELBENCH_MAX_CODE_CHARS")
    max_input_elements = _parse_optional_int_env("OPENHANDS_KERNELBENCH_MAX_INPUT_ELEMENTS")
    max_output_elements = _parse_optional_int_env("OPENHANDS_KERNELBENCH_MAX_OUTPUT_ELEMENTS")
    return include, exclude, max_code_chars, max_input_elements, max_output_elements, mode


def _eval_int_expr(node: ast.AST, names: dict[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _eval_int_expr(node.operand, names)
        return -value if value is not None else None
    if isinstance(node, ast.BinOp):
        left = _eval_int_expr(node.left, names)
        right = _eval_int_expr(node.right, names)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv) and right:
            return left // right
    return None


def _shape_numel(shape: tuple[int, ...] | None) -> int | None:
    if not shape:
        return None
    total = 1
    for value in shape:
        total *= value
    return total


def _torch_rand_shape(node: ast.AST, names: dict[str, int]) -> tuple[int, ...] | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in {"rand", "randn", "empty", "zeros", "ones"}:
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "torch":
        return None
    shape = tuple(_eval_int_expr(arg, names) for arg in node.args)
    if any(value is None for value in shape):
        return None
    return tuple(int(value) for value in shape if value is not None)


def _attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _matmul_output_shape(node: ast.AST, input_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...] | None:
    left: ast.AST | None = None
    right: ast.AST | None = None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        left, right = node.left, node.right
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch" and node.func.attr == "matmul" and len(node.args) >= 2:
            left, right = node.args[0], node.args[1]
    if left is None or right is None:
        return None

    left_shape = input_shapes.get(_attr_name(left) or "")
    right_shape: tuple[int, ...] | None = None
    if isinstance(right, ast.Attribute) and right.attr == "T":
        base_shape = input_shapes.get(_attr_name(right.value) or "")
        right_shape = tuple(reversed(base_shape)) if base_shape else None
    else:
        right_shape = input_shapes.get(_attr_name(right) or "")

    if not left_shape or not right_shape or len(left_shape) != 2 or len(right_shape) != 2:
        return None
    return (left_shape[0], right_shape[1])


def _infer_kernelbench_shape_limits(code: str) -> tuple[int | None, int | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None

    names: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _eval_int_expr(node.value, names)
            if value is not None:
                names[node.targets[0].id] = value

    input_shapes: dict[str, tuple[int, ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    shape = _torch_rand_shape(stmt.value, names)
                    if shape:
                        input_shapes[stmt.targets[0].id] = shape

    max_input = max((_shape_numel(shape) or 0 for shape in input_shapes.values()), default=0) or None

    output_numel: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return):
                    shape = _matmul_output_shape(stmt.value, input_shapes)
                    output_numel = _shape_numel(shape)
                    if output_numel is not None:
                        break
    return max_input, output_numel


def _keep_kernelbench_row(
    row: dict[str, Any],
    include_keywords: tuple[str, ...],
    exclude_keywords: tuple[str, ...],
    max_code_chars: int | None,
    max_input_elements: int | None,
    max_output_elements: int | None,
) -> bool:
    code = str(row.get("code", row.get("reference_code", "")))
    if max_code_chars is not None and len(code) > max_code_chars:
        return False

    if max_input_elements is not None or max_output_elements is not None:
        input_elements, output_elements = _infer_kernelbench_shape_limits(code)
        if max_input_elements is not None and input_elements is not None and input_elements > max_input_elements:
            return False
        if max_output_elements is not None and output_elements is not None and output_elements > max_output_elements:
            return False

    text = _row_search_text(row)
    if include_keywords and not any(keyword in text for keyword in include_keywords):
        return False
    if exclude_keywords and any(keyword in text for keyword in exclude_keywords):
        return False
    return True


def _apply_sample_filter(ds):
    include, exclude, max_code_chars, max_input_elements, max_output_elements, mode = _kernelbench_filter_config()
    if not include and not exclude and max_code_chars is None and max_input_elements is None and max_output_elements is None:
        print("[kernelbench] sample filter disabled")
        return ds

    before = len(ds)
    ds = ds.filter(lambda row: _keep_kernelbench_row(row, include, exclude, max_code_chars, max_input_elements, max_output_elements))
    print(
        "[kernelbench] sample filter: "
        f"mode={mode}, before={before}, after={len(ds)}, "
        f"include={include or 'ALL'}, exclude={exclude or 'NONE'}, "
        f"max_code_chars={max_code_chars or 'NONE'}, "
        f"max_input_elements={max_input_elements or 'NONE'}, "
        f"max_output_elements={max_output_elements or 'NONE'}"
    )
    return ds


def _parse_levels(levels: str | None) -> tuple[str, ...]:
    if not levels:
        return _DEFAULT_LEVELS
    parsed = tuple(part.strip() for part in levels.split(",") if part.strip())
    return parsed or _DEFAULT_LEVELS


def _level_to_int(level: str) -> int | None:
    text = str(level).strip()
    if text.startswith("level_"):
        text = text[len("level_") :]
    try:
        return int(text)
    except ValueError:
        return None


def _local_kernelbench_files(source: str, levels: tuple[str, ...]) -> list[str]:
    path = Path(source)
    if path.is_file():
        return [str(path)]

    if not path.is_dir():
        return []

    parquet_files = sorted(path.rglob("*.parquet"))
    if not parquet_files:
        return []

    level_names = set(levels)
    matched = [
        file
        for file in parquet_files
        if file.stem in level_names or file.name in {f"{level}.parquet" for level in level_names}
    ]
    return [str(file) for file in (matched or parquet_files)]


def _filter_by_level_if_present(ds, levels: tuple[str, ...]):
    target_levels = {value for value in (_level_to_int(level) for level in levels) if value is not None}
    if not target_levels:
        return ds

    if not hasattr(ds, "column_names") or "level" not in ds.column_names:
        return ds

    return ds.filter(lambda row: int(row.get("level", -1)) in target_levels)


def _load_kernelbench_levels(hf_dataset: str, levels: tuple[str, ...]):
    from datasets import concatenate_datasets, load_dataset

    local_files = _local_kernelbench_files(hf_dataset, levels)
    if local_files:
        ds = load_dataset("parquet", data_files=local_files, split="train")
        return _filter_by_level_if_present(ds, levels)

    loaded = []
    direct_load_failed = False
    for level in levels:
        try:
            loaded.append(load_dataset(hf_dataset, split=level))
        except ValueError as exc:
            if "Unknown split" not in str(exc) and "unknown split" not in str(exc):
                raise
            direct_load_failed = True
            break

    if loaded and not direct_load_failed:
        return loaded[0] if len(loaded) == 1 else concatenate_datasets(loaded)

    ds = load_dataset(hf_dataset, split="train")
    return _filter_by_level_if_present(ds, levels)


def create_parquet(
    output_path: str,
    *,
    levels: tuple[str, ...] | None = None,
    max_rows: int | None = None,
    hf_dataset: str = _HF_DATASET,
) -> None:
    import pandas as pd

    ds = _load_kernelbench_levels(hf_dataset, levels)
    ds = _apply_sample_filter(ds)

    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))

    rows = []
    skipped = 0
    for idx, row in enumerate(ds):
        record = _row_to_openhands_record(row, idx)
        if record is None:
            skipped += 1
            continue
        rows.append(record)

    if not rows:
        raise ValueError(f"No usable rows found in {hf_dataset!r} levels {levels!r}")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"Created {output_path} ({len(rows)} rows, skipped {skipped}, levels={levels})")


def default_output_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernelbench_openhands.parquet")


if __name__ == "__main__":
    max_rows_env = os.environ.get("OPENHANDS_KERNELBENCH_MAX_ROWS", "").strip()
    max_rows = int(max_rows_env) if max_rows_env else None
    levels = _parse_levels(os.environ.get("OPENHANDS_KERNELBENCH_LEVELS"))
    dataset = os.environ.get("OPENHANDS_KERNELBENCH_DATASET", _HF_DATASET)
    create_parquet(default_output_path(), levels=levels, max_rows=max_rows, hf_dataset=dataset)
