# 当前任务

（此文件由 host 侧动态生成，运行时会被覆写为具体的任务内容。以下为格式说明。）

## 任务格式（KernelBench）

任务文件以 `src/{op_name}.py` 形式存在，包含 `Model`（PyTorch 参考）、`get_inputs()`、`get_init_inputs()`。

## 要求

1. 在 `src/{op_name}_triton_ascend_impl.py` 中实现 `ModelNew` 类。
2. 运行 `bash tools/operator_pipeline.sh --op_name <op_name>` 验证。
3. 迭代修复直到 `metrics.json` 报 `"success": true`。
