#!/usr/bin/env bash
# 正确性检查占位：替换为与 refs/ 或 PyTorch reference 的对比脚本。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"
cd "${ROOT}"

operator_activate_conda

backend="${OPERATOR_BACKEND:-triton}"
echo "[tools/run_correctness] backend=${backend}"

case "${backend}" in
  triton)
    # 占位：若实现了 forward_reference 以外的逻辑，可在此调用 pytest 或自定义脚本
    if grep -q "NotImplementedError" "${ROOT}/${OPERATOR_TRITON_FILE}" 2>/dev/null; then
      echo "[tools/run_correctness] operator still placeholder (NotImplementedError)" >&2
      exit 1
    fi
    echo "[tools/run_correctness] triton: placeholder pass"
    ;;
  ascendc)
    if grep -q "Placeholder" "${ROOT}/${OPERATOR_ASCENDC_FILE}" 2>/dev/null; then
      echo "[tools/run_correctness] ascendc kernel still placeholder" >&2
      exit 1
    fi
    echo "[tools/run_correctness] ascendc: placeholder pass"
    ;;
  mock)
    echo "[tools/run_correctness] mock: pass"
    ;;
  *)
    echo "[tools/run_correctness] unknown backend" >&2
    exit 1
    ;;
esac

exit 0
