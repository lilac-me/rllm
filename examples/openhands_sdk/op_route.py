"""算子路径路由键：按 task['operator_backend'] 选 triton / ascendc（默认 triton）。

仅依赖标准库，便于在开发机上单元测试。见 EXTERNAL_SCORER_DESIGN.md §7。
源工作目录已**物理分成** `agent_workdir.triton/` 与 `agent_workdir.ascendc/`（各自纯净），
dispatch 时由 `openhands_agent.py:_setup_npu_operator_workspace` 按本路由键拷对应变体
装配成运行时 `agent_workdir`——不再需要运行时裁剪（旧 apply_route 已移除）。
"""
from __future__ import annotations

from typing import Any


def op_route(task: dict[str, Any]) -> str:
    """评测/技能路由键：'triton' 或 'ascendc'。

    默认/fallback = **'triton'**（当前主线）；仅当 task['operator_backend'] 显式为
    'ascendc' 时才走 AscendC。数据集条目通过该字段选择路径。
    """
    return "ascendc" if str(task.get("operator_backend", "triton")).lower() == "ascendc" else "triton"
