# 算子 workspace 全局约定（容器内挂载路径一般为 `/opt/workspace`）

## 环境与工具

- **OpenHands SDK** 使用镜像自带的 venv；**不要**在运行 Agent 的进程里切换 conda。
- **编译 / 运行 / 性能采集** 必须通过 `tools/` 下脚本；脚本内部会 `source tools/env.sh` 并（若已安装）激活算子专用 conda 环境。路径占位符见 `tools/env.sh`。
- **禁止修改** `tools/` 目录下的脚本（仅允许改 `src/` 与任务要求的实现文件）。

## 目录布局

| 路径 | 说明 |
|------|------|
| `INSTRUCTIONS.md` | 当前任务说明（优先于环境变量中的摘要）。 |
| `src/triton/` | Triton 实现（默认入口文件见 `tools/env.sh` 中 `OPERATOR_TRITON_FILE`）。 |
| `src/ascendc/` | AscendC 实现（默认 `OPERATOR_ASCENDC_FILE`）。 |
| `refs/` | 可选：参考实现或 golden 数据。 |
| `metrics.json` | **由工具链写入**：机读指标（schema 见 `.agents/skills/operator-metrics/SKILL.md`）。 |
| `profiling_results.json` | 与 `metrics.json` 同步写入的兼容子集，供训练侧 reward 解析。 |

## 工作流（必须遵守）

1. 阅读 `INSTRUCTIONS.md`，在约定路径完成实现。
2. 执行：**`bash tools/operator_pipeline.sh`**（内部顺序：编译 → 正确性 → 性能）。
3. 读取 `metrics.json`（及 `profiling_results.json`）；失败则根据 `error` / 日志修复并重复步骤 2。
4. 数值与 reference 对比须注明 dtype、rtol/atol；禁止无说明的精度降级。

## 遗留 CUDA 扩展示例

若任务要求优化本镜像自带的 PyTorch CUDA 扩展，仍可使用根目录的 `model.py`、`kernels/`、`utils/` 与根目录 `SKILL.md` 中的流程；与 `src/triton` / `src/ascendc` 算子流可并存，以 `INSTRUCTIONS.md` 为准。
