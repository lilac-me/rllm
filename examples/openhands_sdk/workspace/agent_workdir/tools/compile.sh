#!/usr/bin/env bash
# 编译步骤：在 tools/env.sh 中填真实命令；此处保留占位与最小检查。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"
cd "${ROOT}"

operator_activate_conda

backend="${OPERATOR_BACKEND:-triton}"
echo "[tools/compile] backend=${backend} ROOT=${ROOT}"

case "${backend}" in
  triton)
    rel="${OPERATOR_TRITON_FILE}"
    if [[ ! -f "${ROOT}/${rel}" ]]; then
      echo "[tools/compile] missing ${rel}" >&2
      exit 1
    fi
    # 占位：语法检查（不依赖 triton 包）
    if ! "${OPERATOR_PYTHON}" -m py_compile "${ROOT}/${rel}"; then
      echo "[tools/compile] py_compile failed for ${rel}" >&2
      exit 1
    fi
    # TODO: 替换为真实命令，例如 triton 编译或 setup.py build
    # eval "${TRITON_COMPILE_CMD_PLACEHOLDER}"
    ;;
  ascendc)
    rel="${OPERATOR_ASCENDC_FILE}"
    if [[ ! -f "${ROOT}/${rel}" ]]; then
      echo "[tools/compile] missing ${rel}" >&2
      exit 1
    fi
    # TODO: 替换为 atc / bisheng / 项目规定编译命令，例如：
    # "${ASCEND_HOME}/compiler/ccec_compiler/bin/ccec" ...  # 占位
    echo "[tools/compile] ascendc: placeholder OK (file present)"
    ;;
  mock)
    echo "[tools/compile] mock backend: skip"
    ;;
  *)
    echo "[tools/compile] unknown OPERATOR_BACKEND=${backend}" >&2
    exit 1
    ;;
esac

exit 0
