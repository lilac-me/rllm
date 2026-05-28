#!/usr/bin/env bash
# 统一流水线：AST 退化检查 → 数值正确性 → 性能采集 → 写入 metrics.json
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"
cd "${ROOT}"

run_with_eval_lock() {
  local lock_script="${SCRIPT_DIR}/scripts/distributed_npu_lock.py"
  if [[ "${EVAL_USE_DEVICE_LOCK:-1}" != "1" || ! -f "${lock_script}" ]]; then
    "$@"
    return
  fi

  local lock_dir="${EVAL_LOCK_DIR:-/shared/device-locks}"
  local device_prefix="${EVAL_DEVICE_PREFIX:-npu}"
  local device_count="${EVAL_DEVICE_COUNT:-1}"
  local env_name="${EVAL_ENV_NAME:-ASCEND_RT_VISIBLE_DEVICES}"
  local retry_interval="${EVAL_RETRY_INTERVAL:-1.0}"
  local timeout="${EVAL_TIMEOUT:-}"

  local args=(
    python3 "${lock_script}"
    --lock-dir "${lock_dir}"
    --device-prefix "${device_prefix}"
    --device-count "${device_count}"
    --env-name "${env_name}"
    --retry-interval "${retry_interval}"
  )
  if [[ -n "${EVAL_DEVICE_IDS:-}" ]]; then
    args+=(--device-ids "${EVAL_DEVICE_IDS}")
  fi
  if [[ -n "${timeout}" && "${timeout}" != "None" && "${timeout}" != "none" ]]; then
    args+=(--timeout "${timeout}")
  fi
  if [[ "${EVAL_VERBOSE:-0}" == "1" || "${EVAL_VERBOSE:-}" == "true" || "${EVAL_VERBOSE:-}" == "True" ]]; then
    args+=(--verbose)
  fi
  args+=(-- "$@")
  "${args[@]}"
}

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
# 辅助：写 metrics
# ---------------------------------------------------------------------------
write_metrics() {
  local ast_ok="$1" corr_ok="$2" success="$3"
  local fw_lat="$4" impl_lat="$5" speedup="$6" error="$7"

  if [[ -n "${error}" ]]; then
    printf "%s" "${error}" > "${ROOT}/metrics_error.log"
  else
    : > "${ROOT}/metrics_error.log"
  fi

  AST_OK="$ast_ok" \
  CORR_OK="$corr_ok" \
  SUCCESS="$success" \
  FW_LAT="$fw_lat" \
  IMPL_LAT="$impl_lat" \
  SPEEDUP="$speedup" \
  ERROR_FILE="${ROOT}/metrics_error.log" \
  ROOT="$ROOT" \
  OP_NAME="$OP_NAME" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])

def to_num(x):
    return None if x == "" else float(x)

def to_bool(x):
    return str(x).lower() in ("1", "true", "yes")

fw_lat = os.environ.get("FW_LAT", "")
impl_lat = os.environ.get("IMPL_LAT", "")
speedup = os.environ.get("SPEEDUP", "")
error_file = Path(os.environ.get("ERROR_FILE", ""))
err_full = error_file.read_text(encoding="utf-8", errors="replace") if error_file.exists() else ""
err = err_full[:4000] if err_full else None

def classify_error(text: str) -> str | None:
    if not text:
        return None
    lower = text.lower()
    if "implementation file missing" in lower:
        return "implementation_missing"
    if "task file missing" in lower:
        return "task_missing"
    if "ast" in lower:
        return "ast_check_failed"
    if "ub overflow" in lower:
        return "triton_ub_overflow"
    if "failed to run bishenghir pipeline" in lower or "failed to run bishengir pipeline" in lower:
        return "triton_compile_failed"
    if "tritontostructured" in lower or "pointer analysis" in lower:
        return "triton_lowering_failed"
    if "correctness" in lower or "mismatch" in lower or "数值验证失败" in text:
        return "correctness_failed"
    if "benchmark" in lower or "性能测试失败" in text:
        return "benchmark_failed"
    return "unknown"

perf_data = None
if fw_lat and impl_lat and speedup:
    perf_data = {
        "framework_latency_ms": float(fw_lat),
        "impl_latency_ms": float(impl_lat),
        "speedup_vs_torch": float(speedup),
    }

metrics = {
    "schema_version": 2,
    "op_name": os.environ.get("OP_NAME", ""),
    "success": to_bool(os.environ["SUCCESS"]),
    "ast_check_ok": to_bool(os.environ["AST_OK"]),
    "correctness_ok": to_bool(os.environ["CORR_OK"]),
    "perf_data": perf_data,
    "error": err,
    "error_type": classify_error(err_full),
    "error_file": "metrics_error.log" if err_full else None,
    "error_truncated": bool(err_full and len(err_full) > len(err or "")),
}

(root / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

profiling = {
    "success": to_bool(os.environ["SUCCESS"]),
    "bandwidth_gbps": 0.0,
    "execution_time_ms": to_num(impl_lat),
    "speedup_vs_torch": to_num(speedup),
    "error": err,
    "error_type": classify_error(err_full),
    "error_file": "metrics_error.log" if err_full else None,
}

(root / "profiling_results.json").write_text(
    json.dumps(profiling, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY
}

print_error_preview() {
  local label="$1"
  local text="$2"
  local max_chars="${OPENHANDS_PIPELINE_ERROR_PREVIEW_CHARS:-2000}"

  echo "[pipeline] ${label}; full error saved to metrics_error.log"
  ERROR_PREVIEW_TEXT="$text" ERROR_PREVIEW_MAX="$max_chars" python3 - <<'PY'
import os

text = os.environ.get("ERROR_PREVIEW_TEXT", "")
try:
    max_chars = int(os.environ.get("ERROR_PREVIEW_MAX", "2000"))
except ValueError:
    max_chars = 2000

if len(text) <= max_chars:
    print(text)
else:
    head = max_chars // 2
    tail = max_chars - head
    print(text[:head])
    print(f"\n... [truncated {len(text) - max_chars} chars; read metrics_error.log for full error] ...\n")
    print(text[-tail:])
PY
}

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
  print_error_preview "AST check FAILED" "${AST_OUTPUT}"
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
if VERIFY_ERR=$(run_with_eval_lock "${OPERATOR_PYTHON}" "${SCRIPT_DIR}/scripts/verify.py" \
    --op_name "${OP_NAME}" \
    --verify_dir "${VERIFY_WORK}" \
    --triton_impl_name "${IMPL_NAME}" \
    --timeout 900 2>&1); then
  echo "[pipeline] correctness passed"
else
  echo "[pipeline] correctness FAILED"
  rm -rf "${VERIFY_WORK}"
  write_metrics true false false "" "" "" "数值验证失败: ${VERIFY_ERR}"
  print_error_preview "correctness FAILED" "${VERIFY_ERR}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: 性能测试（conda Python + torch_npu.profiler）
# ---------------------------------------------------------------------------
echo "[pipeline] Step 3: benchmark (benchmark.py)"

PERF_JSON="${ROOT}/perf_result.json"
BENCH_ERR=""
if BENCH_ERR=$(run_with_eval_lock "${OPERATOR_PYTHON}" "${SCRIPT_DIR}/scripts/benchmark.py" \
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
  print_error_preview "benchmark FAILED" "${BENCH_ERR}"
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
