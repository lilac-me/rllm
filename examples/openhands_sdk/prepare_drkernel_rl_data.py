"""Prepare DR.Kernel RL data for the OpenHands SDK rollout path.

This intentionally writes the same lightweight parquet shape as
create_mock_npu_operator_data.py:

    prompt: JSON-encoded chat messages
    extra_info: dict consumed directly by openhands_agent.rollout()

Do not register this through DatasetRegistry for this AgentSdkEngine branch.
The trainer reads non_tensor_batch["extra_info"] from verl, so extra_info must
already contain instruction/op_name/task_code at the top level.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_HF_DATASET = "hkust-nlp/drkernel-rl-data"
_DEFAULT_ARCH = "ascend910b1"
_SUPPORTED_DATA_SUFFIXES = (".parquet", ".json", ".jsonl", ".arrow")


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _as_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [{"role": "user", "content": value}]

    if isinstance(value, list):
        messages = []
        for item in value:
            if isinstance(item, dict):
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))
                messages.append({"role": role, "content": content})
        if messages:
            return messages

    return [{"role": "user", "content": str(value)}]


def _last_user_content(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return messages[-1].get("content", "") if messages else ""


def _safe_op_name(value: Any, idx: int) -> str:
    raw = str(value or f"drkernel_{idx}")
    raw = re.sub(r"[^0-9a-zA-Z_]+", "_", raw).strip("_")
    if not raw:
        raw = f"drkernel_{idx}"
    if raw[0].isdigit():
        raw = f"task_{raw}"
    return raw[:80]


def _row_to_openhands_record(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    reward_model = _as_dict(row.get("reward_model"))
    extra = _as_dict(row.get("extra_info"))

    task_code = reward_model.get("ground_truth") or row.get("ground_truth") or ""
    if not task_code:
        return None

    messages = _as_messages(row.get("prompt", ""))
    instruction = _last_user_content(messages)
    uid = extra.get("uuid") or f"drkernel_{idx}"
    op_name = _safe_op_name(uid, idx)

    extra_info = {
        "instruction": instruction,
        "scenario": "npu_operator",
        "op_name": op_name,
        "arch": os.environ.get("OPENHANDS_DRKERNEL_ARCH", _DEFAULT_ARCH),
        "task_code": task_code,
        "operator_backend": os.environ.get("OPENHANDS_DRKERNEL_OPERATOR_BACKEND", "triton"),
        "entry_point": extra.get("entry_point", "Model"),
        "uid": str(uid),
        "data_source": row.get("data_source", "cuda_llm"),
        "ability": row.get("ability", "kernel_optimization"),
        "drkernel_extra": json.dumps(extra, ensure_ascii=False),
    }

    return {
        "prompt": json.dumps(messages, ensure_ascii=False),
        "extra_info": extra_info,
    }


def _data_files_for_directory(path: str, split: str) -> list[str]:
    root = Path(path)

    split_dir = root / split
    search_roots = [split_dir] if split_dir.is_dir() else [root]

    files: list[str] = []
    for search_root in search_roots:
        for suffix in _SUPPORTED_DATA_SUFFIXES:
            files.extend(str(p) for p in sorted(search_root.rglob(f"*{suffix}")))
        if files:
            break
    return files


def _load_source_dataset(source: str, split: str):
    from datasets import load_dataset, load_from_disk

    if os.path.isdir(source):
        try:
            loaded = load_from_disk(source)
            if hasattr(loaded, "keys"):
                if split in loaded:
                    return loaded[split]
                if "train" in loaded:
                    return loaded["train"]
                first_split = next(iter(loaded.keys()))
                return loaded[first_split]
            return loaded
        except Exception:
            pass

        files = _data_files_for_directory(source, split)
        if not files:
            raise FileNotFoundError(f"No supported dataset files found under {source!r}")

        suffix = Path(files[0]).suffix.lower()
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=files, split="train")
        if suffix in (".json", ".jsonl"):
            return load_dataset("json", data_files=files, split="train")
        if suffix == ".arrow":
            return load_dataset("arrow", data_files=files, split="train")
        raise ValueError(f"Unsupported dataset file suffix: {suffix}")

    return load_dataset(source, split=split)


def create_parquet(
    output_path: str,
    *,
    split: str = "train",
    max_rows: int | None = None,
    hf_dataset: str = _HF_DATASET,
) -> None:
    import pandas as pd

    ds = _load_source_dataset(hf_dataset, split)
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
        raise ValueError(f"No usable rows found in {hf_dataset!r} split {split!r}")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(rows).to_parquet(output_path, index=False)
    print(f"Created {output_path} ({len(rows)} rows, skipped {skipped})")


def default_output_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "drkernel_rl_operator.parquet")


if __name__ == "__main__":
    max_rows_env = os.environ.get("OPENHANDS_DRKERNEL_MAX_ROWS", "").strip()
    max_rows = int(max_rows_env) if max_rows_env else None
    dataset = os.environ.get("OPENHANDS_DRKERNEL_DATASET", _HF_DATASET)
    create_parquet(default_output_path(), max_rows=max_rows, hf_dataset=dataset)
