#!/usr/bin/env bash
# =============================================================================
# triton_eval_pipeline.sh — Triton 固定评测入口（advisor 与 judge 共用同一份）
#
# 设计见 examples/openhands_sdk/EXTERNAL_SCORER_DESIGN.md §3.1。
# 流程：AST 退化检查 → 数值正确性(verify.py) → 性能(benchmark.py) → 单一 metrics.json
#
# 与 agent 的关系：
#   - in-loop（advisor）：agent 在 workspace 内调它自测迭代。
#   - reward（judge）：worker 在 agent 写不到的快照上调同一份，算权威 reward。
#
# 可信约束（对应 EXTERNAL_SCORER_DESIGN.md §2 C4）：
#   warmup/repeats/timeout 全部写死；**绝不**向 benchmark.py 传
#   --skip_framework / --framework_latency_ms / --verify_not_required
#   （这三个能任意抬 speedup / 绕过正确性闸门）。framework 基线一律实测。
#
# 用法：
#   bash triton_eval_pipeline.sh --op_name <op> --impl <impl.py> --task <ref_{op}.py> [--json <{op}.json>] [--out_dir <dir>]
#   bash triton_eval_pipeline.sh --self-check        # 跑 fixtures 自检，正式 RL 前的 precondition gate
# =============================================================================
set -euo pipefail

# ----- 固定参数（agent 不可改）-----
readonly WARMUP=5
readonly REPEATS=50
readonly VERIFY_TIMEOUT=900
readonly IMPL_NAME="triton_ascend_impl"   # verify/benchmark 据此推 {op}_{IMPL_NAME}.py + verify_result.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"        # = agent_workdir
VERIFIER_SCRIPTS="${ROOT}/.agents/skills/triton-op-verifier/scripts"

# 解释器：沿用 env.sh（AST_CHECK_PYTHON 走 venv 免 NPU；OPERATOR_PYTHON 走 conda+NPU）
# 缺省回落 python3，便于无 env.sh 的环境（如 self-check 的纯静态用例）。
# shellcheck source=/dev/null
[[ -f "${SCRIPT_DIR}/env.sh" ]] && source "${SCRIPT_DIR}/env.sh"
AST_CHECK_PYTHON="${AST_CHECK_PYTHON:-python3}"
OPERATOR_PYTHON="${OPERATOR_PYTHON:-python3}"

# NOTE(设备锁)：本分支 operator_pipeline.sh 不在 pipeline 内抢锁，scripts 直调；
# distributed_npu_lock.py 在 examples/openhands_sdk/ 是 host/worker 级、容器内不可见。
# 所以这里也直调，NPU 设备分配交给启动方（容器的 ASCEND_RT_VISIBLE_DEVICES / worker）。
# 若将来多 rollout 共享 NPU 需要 in-pipeline 锁，再在 worker 层引入。

# ----- 写 metrics.json（schema_version 2，字段对齐 openhands_agent.py:_reward_from_metrics）-----
write_metrics() {
  local ast_ok="$1" corr_ok="$2" success="$3" fw_lat="$4" impl_lat="$5" speedup="$6" error="$7"
  if [[ -n "${error}" ]]; then printf "%s" "${error}" > "${OUT_DIR}/metrics_error.log"; else : > "${OUT_DIR}/metrics_error.log"; fi
  AST_OK="$ast_ok" CORR_OK="$corr_ok" SUCCESS="$success" FW_LAT="$fw_lat" IMPL_LAT="$impl_lat" \
  SPEEDUP="$speedup" ERROR_FILE="${OUT_DIR}/metrics_error.log" OUT_DIR="$OUT_DIR" OP_NAME="$OP_NAME" \
  python3 - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_DIR"])
def b(x): return str(x).lower() in ("1","true","yes")
def n(x): return None if x == "" else float(x)
fw, impl, sp = os.environ.get("FW_LAT",""), os.environ.get("IMPL_LAT",""), os.environ.get("SPEEDUP","")
ef = Path(os.environ.get("ERROR_FILE",""))
full = ef.read_text(encoding="utf-8", errors="replace") if ef.exists() else ""
err = full[:4000] if full else None
def classify(t):
    if not t: return None
    l = t.lower()
    if "implementation file missing" in l: return "implementation_missing"
    if "task file missing" in l: return "task_missing"
    if "ast" in l: return "ast_check_failed"
    if "ub overflow" in l: return "triton_ub_overflow"
    if "bishenghir" in l or "bishengir" in l: return "triton_compile_failed"
    if "tritontostructured" in l or "pointer analysis" in l: return "triton_lowering_failed"
    if "correctness" in l or "mismatch" in l or "数值" in t: return "correctness_failed"
    if "benchmark" in l or "性能" in t: return "benchmark_failed"
    return "unknown"
perf = {"framework_latency_ms": float(fw), "impl_latency_ms": float(impl), "speedup_vs_torch": float(sp)} if (fw and impl and sp) else None
metrics = {
    "schema_version": 2, "op_name": os.environ.get("OP_NAME",""),
    "success": b(os.environ["SUCCESS"]), "ast_check_ok": b(os.environ["AST_OK"]),
    "correctness_ok": b(os.environ["CORR_OK"]), "perf_data": perf,
    "error": err, "error_type": classify(full),
    "error_file": "metrics_error.log" if full else None,
    "error_truncated": bool(full and len(full) > len(err or "")),
}
(out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

# ----- 参数解析（只接受这些；没有 warmup/repeats/skip_framework 等危险开关）-----
OP_NAME="" IMPL_FILE="" TASK_FILE="" JSON_FILE="" OUT_DIR="" SELF_CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --op_name)   OP_NAME="$2";   shift 2 ;;
    --impl)      IMPL_FILE="$2"; shift 2 ;;
    --task)      TASK_FILE="$2"; shift 2 ;;
    --json)      JSON_FILE="$2"; shift 2 ;;
    --out_dir)   OUT_DIR="$2";   shift 2 ;;
    --self-check) SELF_CHECK=1;  shift ;;
    *) echo "[triton-eval] unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ----- self-check：正式 RL 前确认 pipeline 行为正确（决策 #1 硬性要求）-----
if [[ "${SELF_CHECK}" == "1" ]]; then
  FX="${SCRIPT_DIR}/fixtures"
  # 约定：fixtures/{good,bad}/ 各含 {op}.py(参考) + impl.py(实现)；good 期望 success=true，bad 期望 correctness_ok=false。
  # TODO(待补): 放一个真实小算子的 good/bad fixture（需在 NPU 上验证过其期望值）。
  if [[ ! -d "${FX}/good" || ! -d "${FX}/bad" ]]; then
    echo "[self-check] 缺少 fixtures（${FX}/good 与 ${FX}/bad）——pipeline 自检未就绪，禁止开训。" >&2
    exit 3
  fi
  fail=0
  for kind in good bad; do
    op=$(basename "$(ls "${FX}/${kind}"/*.py | grep -v impl | head -1)" .py)
    tmp=$(mktemp -d); bash "${BASH_SOURCE[0]}" --op_name "${op}" \
      --impl "${FX}/${kind}/impl.py" --task "${FX}/${kind}/${op}.py" --out_dir "${tmp}" || true
    ok=$(python3 -c "import json;d=json.load(open('${tmp}/metrics.json'));print(d['success'],d['correctness_ok'])" 2>/dev/null || echo "ERR")
    echo "[self-check] ${kind}: success/correctness = ${ok}"
    [[ "${kind}" == "good" && "${ok}" != "True True" ]] && { echo "  ✗ good 应 success=True"; fail=1; }
    [[ "${kind}" == "bad"  && "${ok##* }" != "False" ]] && { echo "  ✗ bad 应 correctness_ok=False"; fail=1; }
    rm -rf "${tmp}"
  done
  [[ "${fail}" == "0" ]] && echo "[self-check] PASS — pipeline 可用" || { echo "[self-check] FAIL"; exit 3; }
  exit 0
fi

# ----- 正常评测 -----
[[ -z "${OP_NAME}" ]] && { echo "[triton-eval] --op_name required" >&2; exit 1; }
OUT_DIR="${OUT_DIR:-${ROOT}}"; mkdir -p "${OUT_DIR}"

if [[ ! -f "${IMPL_FILE}" ]]; then
  echo "[triton-eval] implementation not found: ${IMPL_FILE}" >&2
  write_metrics false false false "" "" "" "implementation file missing"; exit 1
fi
if [[ ! -f "${TASK_FILE}" ]]; then
  echo "[triton-eval] task(reference) not found: ${TASK_FILE}" >&2
  write_metrics false false false "" "" "" "task file missing"; exit 1
fi

# verify_dir：verify.py/benchmark.py 按 import 名加载 {op}_torch / {op}_{IMPL_NAME}
VERIFY_DIR="${OUT_DIR}/verify_tmp"; rm -rf "${VERIFY_DIR}"; mkdir -p "${VERIFY_DIR}"
cp "${TASK_FILE}" "${VERIFY_DIR}/${OP_NAME}_torch.py"
cp "${IMPL_FILE}" "${VERIFY_DIR}/${OP_NAME}_${IMPL_NAME}.py"
# 多 case：若打包了 {op}.json（其 get_input_groups 可能读同名 json），放进 verify_dir。
# NOTE(待 self-check 校验): 实际文件名/是否被 get_input_groups 读取，取决于数据集 {op}.py 的写法。
[[ -n "${JSON_FILE}" && -f "${JSON_FILE}" ]] && cp "${JSON_FILE}" "${VERIFY_DIR}/${OP_NAME}.json"

# Step 1: AST 退化预检查（venv Python，免 NPU）
echo "[triton-eval] Step1 AST check"
if ! AST_OUT=$("${AST_CHECK_PYTHON}" "${VERIFIER_SCRIPTS}/validate_triton_impl.py" "${IMPL_FILE}" --json 2>&1); then
  rm -rf "${VERIFY_DIR}"; write_metrics false false false "" "" "" "AST退化检查失败: ${AST_OUT}"
  echo "[triton-eval] AST FAILED"; exit 1
fi

# Step 2: 数值正确性（NPU，抢设备锁）
echo "[triton-eval] Step2 verify"
if ! VERIFY_ERR=$("${OPERATOR_PYTHON}" "${VERIFIER_SCRIPTS}/verify.py" \
      --op_name "${OP_NAME}" --verify_dir "${VERIFY_DIR}" --triton_impl_name "${IMPL_NAME}" \
      --timeout "${VERIFY_TIMEOUT}" 2>&1); then
  rm -rf "${VERIFY_DIR}"; write_metrics true false false "" "" "" "数值验证失败: ${VERIFY_ERR}"
  echo "[triton-eval] verify FAILED"; exit 1
fi

# Step 3: 性能（NPU，抢锁）—— 注意：不传 --skip_framework/--framework_latency_ms/--verify_not_required
echo "[triton-eval] Step3 benchmark"
PERF_JSON="${OUT_DIR}/perf_result.json"
if ! BENCH_ERR=$("${OPERATOR_PYTHON}" "${VERIFIER_SCRIPTS}/benchmark.py" \
      --op_name "${OP_NAME}" --verify_dir "${VERIFY_DIR}" --triton_impl_name "${IMPL_NAME}" \
      --warmup "${WARMUP}" --repeats "${REPEATS}" --output "${PERF_JSON}" 2>&1); then
  rm -rf "${VERIFY_DIR}"; write_metrics true true false "" "" "" "性能测试失败: ${BENCH_ERR}"
  echo "[triton-eval] benchmark FAILED"; exit 1
fi
rm -rf "${VERIFY_DIR}"

# Step 4: perf_result.json → metrics.json（speedup_vs_torch = 全 shape 几何平均，由 benchmark.py 算好）
FW=$(python3 -c "import json;print(json.load(open('${PERF_JSON}'))['framework']['avg_latency_ms'])" 2>/dev/null || echo "")
IMPL_LAT=$(python3 -c "import json;print(json.load(open('${PERF_JSON}'))['implementation']['avg_latency_ms'])" 2>/dev/null || echo "")
SP=$(python3 -c "import json;print(json.load(open('${PERF_JSON}'))['speedup_vs_torch'])" 2>/dev/null || echo "")
write_metrics true true true "${FW}" "${IMPL_LAT}" "${SP}" ""
echo "[triton-eval] done — metrics.json written; speedup_vs_torch=${SP}"
