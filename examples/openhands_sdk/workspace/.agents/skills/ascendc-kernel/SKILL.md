---
name: ascendc-kernel
description: >
  AscendC 算子：CANN / 内存层次约定；编译与运行通过 tools/operator_pipeline.sh。
  在用户提到 ascend、ascendc、npu kernel 时使用。
triggers:
  - ascendc
  - ascend
  - npu
---

# AscendC 算子

1. 遵守 UB / L1 等内存与同步约定；对齐 `INSTRUCTIONS.md` 中的接口形状。
2. **不要**在对话里手动切换 conda；统一执行 `bash tools/operator_pipeline.sh`。
3. 将真实编译器调用写在 `tools/compile.sh` 的 `ascendc` 分支（当前为占位）；`ASCEND_HOME` 等见 `tools/env.sh`。
