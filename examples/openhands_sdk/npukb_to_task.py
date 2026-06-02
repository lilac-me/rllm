#!/usr/bin/env python3
"""NPUKernelBench ``{op}.py`` + ``{op}.json`` → verify 格式 task 文件（= 数据集条目的 task_code）。

为什么需要它：RL 评测固定入口 ``triton_eval_pipeline.sh`` 调的 ``verify.py``/``benchmark.py``
要求 task 文件自带三件套——``class Model(nn.Module)``、``get_init_inputs()``、
``get_inputs()``/``get_input_groups()``（见 verify.py:140 / :332）。而 NPUKernelBench 的原始
``{id}_{Name}.py`` 只有 ``Model``，输入的 shape/dtype/attr 全在同名 ``.json``（JSONL，每行一个 case）。
cannbot 旧流程靠 agent 端的 op-task-extractor 在环内手工补这两个函数；RL 侧不跑 extractor，
所以这里用一个**确定性**脚本把 json 的 case 烘焙进 ``get_input_groups()``，产出**自包含**、可被
verify.py 直接 import 的 task 文件。

确定性、仅依赖标准库（把 torch 代码当文本生成，自身不 import torch），可在开发机跑。
产物即 ``create_mock_npu_operator_data.py`` 里 ``extra_info['task_code']`` 的内容
（同时把原 ``.json`` 作为 ``extra_info['task_json']`` 携带即可，见 WORK_LOG §5 T5）。

用法：
    python3 npukb_to_task.py <op.py> [--json <op.json>] [-o out.py]
                            [--max-cases N] [--dtypes f32,f16,bf16]
    # --json 缺省=同名 <op>.json；-o 缺省=stdout
例：
    python3 npukb_to_task.py \
      /path/AscendOpGenAgent/benchmarks/NPUKernelBench/level1/4_Abs.py \
      --max-cases 12 -o abs.py
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

# JSON dtype 字符串 → torch.dtype 字面量
_DTYPE = {
    "float32": "torch.float32", "float": "torch.float32", "fp32": "torch.float32",
    "float16": "torch.float16", "half": "torch.float16", "fp16": "torch.float16",
    "bfloat16": "torch.bfloat16", "bf16": "torch.bfloat16",
    "int64": "torch.int64", "long": "torch.int64",
    "int32": "torch.int32", "int": "torch.int32",
    "int16": "torch.int16", "int8": "torch.int8", "uint8": "torch.uint8",
    "bool": "torch.bool",
}
_FLOAT = {"float32", "float", "fp32", "float16", "half", "fp16", "bfloat16", "bf16"}
# --dtypes 简写 → 规范 json dtype
_DTYPE_SHORT = {"f32": "float32", "f16": "float16", "bf16": "bfloat16",
                "i64": "int64", "i32": "int32", "i8": "int8", "bool": "bool"}


def _norm_dtypes(spec: str) -> set[str]:
    out: set[str] = set()
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        out.add(_DTYPE_SHORT.get(tok, tok))
    return out


def _inspect(src: str, path: str) -> dict:
    """探测 task 契约：Model/forward 必须有；返回已自带哪些输入函数 + __init__ 必填参数数。"""
    tree = ast.parse(src)
    model = next((n for n in ast.walk(tree)
                  if isinstance(n, ast.ClassDef) and n.name == "Model"), None)
    if model is None:
        sys.exit(f"[npukb] {path}: 未找到 class Model —— 不是 KernelBench 风格的参考文件")
    if not any(isinstance(b, ast.FunctionDef) and b.name == "forward" for b in model.body):
        sys.exit(f"[npukb] {path}: Model 缺 forward()")
    init = next((b for b in model.body
                 if isinstance(b, ast.FunctionDef) and b.name == "__init__"), None)
    init_required = 0
    if init is not None:  # 去掉 self；带默认值的可不传，只数“必填”
        init_required = len(init.args.args[1:]) - len(init.args.defaults)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    return {
        "has_init": "get_init_inputs" in funcs,
        "has_inputs": "get_inputs" in funcs,
        "has_groups": "get_input_groups" in funcs,
        "init_required": init_required,
    }


def _write(body: str, out: str | None, note: str) -> None:
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"[npukb] {note} -> {out}", file=sys.stderr)
    else:
        sys.stdout.write(body)
        print(f"[npukb] {note}", file=sys.stderr)


def _tensor_expr(dtype: str, shape: list[int]) -> str:
    if dtype not in _DTYPE:
        sys.exit(f"[npukb] 不认识的 dtype: {dtype!r}（请在 _DTYPE 里补充）")
    if dtype in _FLOAT:
        return f"torch.randn({shape!r}, dtype={_DTYPE[dtype]})"
    if dtype == "bool":
        return f"(torch.randn({shape!r}) > 0)"
    return f"torch.randint(0, 100, {shape!r}, dtype={_DTYPE[dtype]})"


def _group_expr(inputs: list[dict]) -> str:
    parts: list[str] = []
    for item in inputs:
        if item.get("type") == "tensor":
            parts.append(_tensor_expr(str(item["dtype"]).lower(), list(item["shape"])))
        else:  # attr：标量/布尔/字符串/列表 —— 原样取 value
            parts.append(repr(item.get("value")))
    return "[" + ", ".join(parts) + "]"


def _case_dtypes(inputs: list[dict]) -> set[str]:
    return {str(i["dtype"]).lower() for i in inputs if i.get("type") == "tensor"}


def main() -> None:
    ap = argparse.ArgumentParser(description="NPUKernelBench → verify 格式 task 文件")
    ap.add_argument("op_py", help="NPUKernelBench 参考文件，如 level1/4_Abs.py")
    ap.add_argument("--json", default=None, help="同名 .json（缺省=同目录同名）")
    ap.add_argument("-o", "--out", default=None, help="输出路径（缺省=stdout）")
    ap.add_argument("--max-cases", type=int, default=0, help="只取前 N 个 case（0=全部）")
    ap.add_argument("--dtypes", default="", help="只保留这些 dtype 的 case，如 f32,f16,bf16")
    args = ap.parse_args()

    with open(args.op_py, encoding="utf-8") as f:
        head = f.read()
    info = _inspect(head, args.op_py)

    # 有的 NPUKernelBench 变体的 {op}.py 已是 verify 格式（自带 get_init_inputs + get_inputs/
    # get_input_groups）——原样透传，不读 json、不覆盖。只自带其一视为不完整，报错让人工补。
    if info["has_init"] and (info["has_inputs"] or info["has_groups"]):
        _write(head, args.out, "已是 verify 格式，原样透传（无需 json）")
        return
    present = [k for k, v in (("get_init_inputs", info["has_init"]),
                              ("get_inputs", info["has_inputs"]),
                              ("get_input_groups", info["has_groups"])) if v]
    if present:
        sys.exit(f"[npukb] {args.op_py}: 自带 {present} 但不完整"
                 f"（需 get_init_inputs + get_inputs/get_input_groups）；请手工补全后再用。")
    if info["init_required"] > 0:
        sys.exit(f"[npukb] {args.op_py}: Model.__init__ 需要 {info['init_required']} 个参数，"
                 f"但 json 不携带 init 参数；请手工写 get_init_inputs() 后再用。")

    json_path = args.json or os.path.splitext(args.op_py)[0] + ".json"
    if not os.path.isfile(json_path):
        sys.exit(f"[npukb] 找不到 json: {json_path}（用 --json 指定）")
    allow = _norm_dtypes(args.dtypes) if args.dtypes else None

    groups: list[str] = []
    with open(json_path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                inputs = json.loads(line)["inputs"]
            except (ValueError, KeyError) as exc:
                sys.exit(f"[npukb] {json_path}:{ln} 解析失败: {exc}")
            if allow is not None and not (_case_dtypes(inputs) <= allow):
                continue
            groups.append(_group_expr(inputs))
            if args.max_cases and len(groups) >= args.max_cases:
                break

    if not groups:
        sys.exit(f"[npukb] {json_path}: 没有可用 case（dtype 过滤后为空？）")

    body = head if head.endswith("\n") else head + "\n"
    body += "\n\ndef get_init_inputs():\n    return []\n"
    body += "\n\ndef get_input_groups():\n    return [\n"
    body += "".join(f"        {g},\n" for g in groups)
    body += "    ]\n"
    _write(body, args.out, f"baked {len(groups)} cases")


if __name__ == "__main__":
    main()
