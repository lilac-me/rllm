#!/usr/bin/env bash
# 性能采集占位：替换为 npu-smi / msprof / triton benchmark 等。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"
cd "${ROOT}"

operator_activate_conda

# 简单占位计时（不依赖硬件；避免使用 GNU date 专有 %N）
ELAPSED_MS="$(python3 -c "import time; t=time.perf_counter(); time.sleep(0); print(f'{(time.perf_counter()-t)*1000:.3f}')")"

BW=$((RANDOM % 500 + 100))
export PROFILE_LATENCY_MS="${ELAPSED_MS}"
export PROFILE_BANDWIDTH_GBPS="${BW}.0"

echo "[tools/profile] placeholder latency_ms=${PROFILE_LATENCY_MS} bandwidth_gbps=${PROFILE_BANDWIDTH_GBPS}"
exit 0
