# shellcheck shell=bash
# 算子工具链环境配置
# OpenHands Agent 禁止手动 conda activate。

# TODO(遗留): 在 NPU 上真正编译/执行 Triton-Ascend 算子时，按需在此补充（或 source 厂商提供的 setenv）：
#   CANN/Ascend 工具链：如 ASCEND_HOME、ASCEND_TOOLKIT_HOME、PATH、LD_LIBRARY_PATH；
#   Triton Ascend 编译相关：各版本要求的额外 env（以 CANN + torch_npu + triton 发布说明为准）；
#   调试：如 ASCEND_LAUNCH_BLOCKING 等。
# 上述变量须在 OPERATOR_PYTHON 跑 verify.py / benchmark.py 之前生效（本文件由 operator_pipeline.sh 最先 source）。

# --- Conda ---
: "${CONDA_BASE:=/opt/conda}"
: "${OPERATOR_CONDA_ENV:=evaluator-py311}"

_CONDA_PYTHON="${CONDA_BASE}/envs/${OPERATOR_CONDA_ENV}/bin/python"
if [[ -x "${_CONDA_PYTHON}" ]]; then
  export OPERATOR_PYTHON="${_CONDA_PYTHON}"
elif [[ -n "${OPERATOR_PYTHON:-}" ]]; then
  echo "[env.sh] WARN: conda python not found at ${_CONDA_PYTHON}, using OPERATOR_PYTHON=${OPERATOR_PYTHON}" >&2
else
  echo "[env.sh] ERROR: conda python not found at ${_CONDA_PYTHON} and OPERATOR_PYTHON not set." >&2
  echo "[env.sh] Set CONDA_BASE or OPERATOR_PYTHON in the container environment." >&2
  exit 1
fi

# validate_triton_impl.py 是纯 AST，不需要 torch；用 venv python 即可
export AST_CHECK_PYTHON="${AST_CHECK_PYTHON:-python3}"

# --- Workspace ---
: "${WORKSPACE_BASE:=/opt/workspace}"


rm -rf /root/.triton/cache/
 
source /opt/conda/envs/evaluator-py311/Ascend/ascend-toolkit/set_env.sh

export TRITON_DEBUG=1
export TRITON_ALLWAYS_COMPILE=1
export PATH=/opt/conda/envs/evaluator-py311/Ascend/cann-8.5.0/tools/bishengir/bin:$PATH