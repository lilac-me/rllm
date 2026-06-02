# 当前任务

（此文件由 host 侧动态生成，运行时会被覆写为具体的任务内容。以下为格式说明。）

## 任务格式（AGENTS.md）

任务文件以 `src/{op_name}.py` 形式存在，包含 `Model`（PyTorch 参考）、`get_inputs()`、`get_init_inputs()`。

## 要求

1. 阅读 `AGENTS.md` 了解全局约定和工作流。
2. 按照 AGENTS.md 的 Phase 0-7 流程执行。
3. 在 `output/{op_name}/` 目录下生成 `model_new_ascendc.py`。
4. 根据 `trace.md` 修复错误直到成功。
