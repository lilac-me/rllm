# 当前任务

（由 host 侧写入或环境变量 `TASK_INSTRUCTION` 覆盖；此处为默认占位。）

## 默认步骤

1. 在 `src/triton/` 或 `src/ascendc/`（见任务指定后端）完成算子实现。
2. 运行统一流水线：

   ```bash
   bash tools/operator_pipeline.sh
   ```

3. 确认 `metrics.json` 中 `"success": true`（且 `compile_ok`、`correctness_ok` 为 true）。
4. 失败时根据终端输出与 JSON 中的 `error` 字段迭代修复。

## 后端切换

设置环境变量 `OPERATOR_BACKEND` 为 `triton` 或 `ascendc`（可在 `tools/env.sh` 或容器 `docker run -e` 中设置）。未设置时脚本内默认值见 `tools/env.sh`。
