#!/usr/bin/env bash
# 统一流水线：AST 退化检查 → 数值正确性 → 性能采集 → 写入 metrics.json
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"
cd "${ROOT}"

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
OP_NAME=""
WARMUP=5
REPEATS=50
IMPL_NAME="triton_ascend_impl"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --op_name)   OP_NAME="$2";    shift 2 ;;
    --warmup)    WARMUP="$2";     shift 2 ;;
    --repeats)   REPEATS="$2";    shift 2 ;;
    --impl_name) IMPL_NAME="$2";  shift 2 ;;
    *) echo "[pipeline] unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${OP_NAME}" ]]; then
  echo "[pipeline] --op_name is required" >&2
  exit 1
fi

IMPL_FILE="${ROOT}/src/${OP_NAME}_${IMPL_NAME}.py"
TASK_FILE="${ROOT}/src/${OP_NAME}.py"
VERIFY_DIR="${ROOT}/src"

# ---------------------------------------------------------------------------
# 检查文件
# ---------------------------------------------------------------------------
if [[ ! -f "${IMPL_FILE}" ]]; then
  echo "[pipeline] implementation not found: ${IMPL_FILE}" >&2
  write_metrics false false false "" "" "" "implementation file missing"
  exit 1
fi

if [[ ! -f "${TASK_FILE}" ]]; then
  TASK_FILE_ALT="${ROOT}/${OP_NAME}.py"
  if [[ -f "${TASK_FILE_ALT}" ]]; then
    cp "${TASK_FILE_ALT}" "${ROOT}/src/${OP_NAME}.py"
    TASK_FILE="${ROOT}/src/${OP_NAME}.py"
  else
    echo "[pipeline] task file not found: ${TASK_FILE}" >&2
    write_metrics false false false "" "" "" "task file missing"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 辅助：写 metrics
# ---------------------------------------------------------------------------
write_metrics() {
  local ast_ok="$1" corr_ok="$2" success="$3"
  local fw_lat="$4" impl_lat="$5" speedup="$6" error="$7"

  local perf_json="null"
  if [[ -n "${fw_lat}" && -n "${impl_lat}" && -n "${speedup}" ]]; then
    perf_json="{\"framework_latency_ms\":${fw_lat},\"impl_latency_ms\":${impl_lat},\"speedup_vs_torch\":${speedup}}"
  fi

  local err_json="null"
  if [[ -n "${error}" ]]; then
    err_json="\"$(echo "${error}" | head -c 500 | tr '"' "'" | tr '\n' ' ')\""
  fi

  cat > "${ROOT}/metrics.json" <<EOJSON
{
  "schema_version": 2,
  "op_name": "${OP_NAME}",
  "success": ${success},
  "ast_check_ok": ${ast_ok},
  "correctness_ok": ${corr_ok},
  "perf_data": ${perf_json},
  "error": ${err_json}
}
EOJSON

  # 兼容训练侧 reward
  cat > "${ROOT}/profiling_results.json" <<EOJSON2
{
  "success": ${success},
  "bandwidth_gbps": 0.0,
  "execution_time_ms": ${impl_lat:-null},
  "speedup_vs_torch": ${speedup:-null},
  "error": ${err_json}
}
EOJSON2
}

# ---------------------------------------------------------------------------
# Step 1: AST 退化检查（venv Python，不需要 NPU）
# ---------------------------------------------------------------------------
echo "[pipeline] Step 1: AST check (validate_triton_impl.py)"
AST_OUTPUT=""
if AST_OUTPUT=$("${AST_CHECK_PYTHON}" "${SCRIPT_DIR}/scripts/validate_triton_impl.py" \
    "${IMPL_FILE}" --json 2>&1); then
  echo "[pipeline] AST check passed"
else
  echo "[pipeline] AST check FAILED"
  write_metrics false false false "" "" "" "AST退化检查失败: ${AST_OUTPUT}"
  echo "${AST_OUTPUT}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: 数值正确性（conda Python + NPU）
# ---------------------------------------------------------------------------
echo "[pipeline] Step 2: correctness (verify.py)"

# 创建 verify 目录并准备文件（verify.py 要求特定文件名）
VERIFY_WORK="${ROOT}/verify_tmp"
rm -rf "${VERIFY_WORK}"
mkdir -p "${VERIFY_WORK}"
cp "${TASK_FILE}" "${VERIFY_WORK}/${OP_NAME}_torch.py"
cp "${IMPL_FILE}" "${VERIFY_WORK}/${OP_NAME}_${IMPL_NAME}.py"

VERIFY_ERR=""
if VERIFY_ERR=$("${OPERATOR_PYTHON}" "${SCRIPT_DIR}/scripts/verify.py" \
    --op_name "${OP_NAME}" \
    --verify_dir "${VERIFY_WORK}" \
    --triton_impl_name "${IMPL_NAME}" \
    --timeout 900 2>&1); then
  echo "[pipeline] correctness passed"
else
  echo "[pipeline] correctness FAILED"
  rm -rf "${VERIFY_WORK}"
  write_metrics true false false "" "" "" "数值验证失败: ${VERIFY_ERR}"
  echo "${VERIFY_ERR}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: 性能测试（conda Python + torch_npu.profiler）
# ---------------------------------------------------------------------------
echo "[pipeline] Step 3: benchmark (benchmark.py)"

PERF_JSON="${ROOT}/perf_result.json"
BENCH_ERR=""
if BENCH_ERR=$("${OPERATOR_PYTHON}" "${SCRIPT_DIR}/scripts/benchmark.py" \
    --op_name "${OP_NAME}" \
    --verify_dir "${VERIFY_WORK}" \
    --triton_impl_name "${IMPL_NAME}" \
    --warmup "${WARMUP}" \
    --repeats "${REPEATS}" \
    --output "${PERF_JSON}" 2>&1); then
  echo "[pipeline] benchmark completed"
else
  echo "[pipeline] benchmark FAILED (correctness was OK)"
  rm -rf "${VERIFY_WORK}"
  write_metrics true true false "" "" "" "性能测试失败: ${BENCH_ERR}"
  echo "${BENCH_ERR}"
  exit 1
fi

rm -rf "${VERIFY_WORK}"

# ---------------------------------------------------------------------------
# Step 4: 解析 perf_result.json → metrics.json
# ---------------------------------------------------------------------------
FW_LAT=$(python3 -c "import json; d=json.load(open('${PERF_JSON}')); print(d['framework']['avg_latency_ms'])" 2>/dev/null || echo "")
IMPL_LAT=$(python3 -c "import json; d=json.load(open('${PERF_JSON}')); print(d['implementation']['avg_latency_ms'])" 2>/dev/null || echo "")
SPEEDUP=$(python3 -c "import json; d=json.load(open('${PERF_JSON}')); print(d['speedup_vs_torch'])" 2>/dev/null || echo "")

write_metrics true true true "${FW_LAT}" "${IMPL_LAT}" "${SPEEDUP}" ""

echo "[pipeline] done — metrics.json and profiling_results.json written"
echo "[pipeline] speedup_vs_torch = ${SPEEDUP}"
