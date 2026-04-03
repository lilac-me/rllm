#!/usr/bin/env bash
# 统一流水线：编译 → 正确性 → 性能 → 写入 metrics.json / profiling_results.json
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"
cd "${ROOT}"

COMPILE_OK=1
CORRECTNESS_OK=1
PROFILE_OK=1
ERROR_MSG=""

run_compile() {
  if bash "${SCRIPT_DIR}/compile.sh"; then
    COMPILE_OK=1
  else
    COMPILE_OK=0
    ERROR_MSG="compile failed"
    return 1
  fi
}

run_correctness() {
  if bash "${SCRIPT_DIR}/run_correctness.sh"; then
    CORRECTNESS_OK=1
  else
    CORRECTNESS_OK=0
    ERROR_MSG="correctness failed"
    return 1
  fi
}

if ! run_compile; then
  LATENCY_MS=""
  BANDWIDTH_GBPS="0.0"
  "${OPERATOR_PYTHON}" "${SCRIPT_DIR}/render_metrics.py" \
    --workspace "${ROOT}" \
    --compile-ok false \
    --correctness-ok false \
    --profile-ok false \
    --backend "${OPERATOR_BACKEND}" \
    --error "${ERROR_MSG}"
  exit 1
fi

if ! run_correctness; then
  "${OPERATOR_PYTHON}" "${SCRIPT_DIR}/render_metrics.py" \
    --workspace "${ROOT}" \
    --compile-ok true \
    --correctness-ok false \
    --profile-ok false \
    --backend "${OPERATOR_BACKEND}" \
    --error "${ERROR_MSG}"
  exit 1
fi

if bash "${SCRIPT_DIR}/profile.sh"; then
  PROFILE_OK=1
else
  PROFILE_OK=0
  ERROR_MSG="profile failed"
fi
LATENCY_MS="${PROFILE_LATENCY_MS:-}"
BANDWIDTH_GBPS="${PROFILE_BANDWIDTH_GBPS:-0.0}"

if [[ "${PROFILE_OK}" -eq 1 ]]; then
  _POK="true"
else
  _POK="false"
fi

"${OPERATOR_PYTHON}" "${SCRIPT_DIR}/render_metrics.py" \
  --workspace "${ROOT}" \
  --compile-ok true \
  --correctness-ok true \
  --profile-ok "${_POK}" \
  --backend "${OPERATOR_BACKEND}" \
  --latency-ms "${LATENCY_MS}" \
  --bandwidth-gbps "${BANDWIDTH_GBPS}" \
  --error "${ERROR_MSG}"

echo "[operator_pipeline] wrote ${ROOT}/metrics.json and profiling_results.json"
