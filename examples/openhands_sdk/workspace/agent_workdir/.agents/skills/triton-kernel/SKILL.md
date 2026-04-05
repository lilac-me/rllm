---
name: triton-kernel
description: >
  Triton 算子：tile、mask、dtype、与 reference 对齐；编译与评测须通过 tools/operator_pipeline.sh。
  在用户提到 triton、tile、kernel 优化时使用。
triggers:
  - triton
  - tile
---

# Triton 算子

1. 先固定 reference（同 shape / dtype），再写 `@triton.jit` kernel。
2. 边界：0 长度、非连续 tensor、最后一个 block 的 mask。
3. **不要**在对话里手动切换 conda；统一执行 `bash tools/operator_pipeline.sh`。
4. 占位编译命令在 `tools/compile.sh` 的 `triton` 分支中替换为真实的 `triton` / `python -m` 构建方式。
