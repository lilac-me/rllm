---
name: latency-optimizer
description: >
  Triton Ascend 性能优化——分析代码特征，应用优化策略，
  确保优化前后功能一致、精度一致。
triggers:
  - optimize
  - performance
---

# Latency Optimizer Skill

你是 Ascend NPU 平台上 Triton 算子性能优化专家。分析已通过验证的 Triton 代码，
识别代码特征，查阅对应优化策略文档，生成优化后的代码。
**必须确保优化前后功能一致、精度一致。**

## 参考文档索引

路径前缀：`.agents/skills/latency-optimizer/references/`

### 必选（每次优化都加载）

| 优化模式 | 文档 |
|----------|------|
| 入参静态化 | `constexpr_parameters.md` |
| Int32 向量加法 | `int32_vector_add.md` |
| Load 指令重排序 | `load-order.md` |

### 按代码特征选择

| 识别特征 | 文档 |
|----------|------|
| 涉及数值比较操作（整数索引与数据比较、`tl.where` 等） | `vector_compare.md` |
