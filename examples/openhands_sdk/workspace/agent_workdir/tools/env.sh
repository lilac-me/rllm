# shellcheck shell=bash
# 算子工具链环境占位符 — 在镜像构建或 docker run 时改为真实路径。
# OpenHands Agent 请勿在交互 shell 中手动 conda activate；由本文件与各脚本统一处理。

# --- Conda（占位：替换为你的安装根目录与 env 名）---
: "${CONDA_BASE:=/path/to/conda}"
: "${OPERATOR_CONDA_ENV:=operator-build}"

# 优先使用 conda env 内的 Python；不存在时回退系统 python3
if [[ -x "${CONDA_BASE}/envs/${OPERATOR_CONDA_ENV}/bin/python" ]]; then
  export OPERATOR_PYTHON="${CONDA_BASE}/envs/${OPERATOR_CONDA_ENV}/bin/python"
else
  export OPERATOR_PYTHON="${OPERATOR_PYTHON:-python3}"
fi

# --- Ascend（占位）---
: "${ASCEND_HOME:=/path/to/Ascend/ascend-toolkit/latest}"
# 若使用 setenv.sh：在 compile.sh 中取消注释并填写路径
# shellcheck disable=SC1091
# source "${ASCEND_HOME}/bin/setenv.sh"

# --- 后端与入口文件（可被环境变量覆盖）---
: "${OPERATOR_BACKEND:=triton}"
: "${OPERATOR_TRITON_FILE:=src/triton/operator.py}"
: "${OPERATOR_ASCENDC_FILE:=src/ascendc/kernel.cpp}"

# --- Triton / CUDA 占位（按需替换）---
: "${TRITON_COMPILE_CMD_PLACEHOLDER:=echo [placeholder] set TRITON_COMPILE_CMD in tools/compile.sh}"
: "${CUDA_HOME_PLACEHOLDER:=/path/to/cuda}"

operator_activate_conda() {
  if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${OPERATOR_CONDA_ENV}" 2>/dev/null || true
  fi
}
