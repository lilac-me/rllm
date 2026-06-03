#!/usr/bin/env bash
# test_best_snapshot.sh — R1 pipeline side: keep the best-correct submission across re-runs.
#
# Runs the REAL tools/triton_eval_pipeline.sh end-to-end with verify.py / benchmark.py
# stubbed (no NPU): the AST check is real, the perf speedup is injected via STUB_SP,
# and correctness can be forced to fail via STUB_FAIL=1. Asserts:
#   - first success      -> {op}_impl.best.py created
#   - slower-but-correct -> best NOT overwritten (kept faster)
#   - faster             -> best updated
#   - correctness FAIL   -> best untouched (pipeline exits before the snapshot block)
# This is exactly the Phase-3 regression the probe hit (success then optimization broke it).
#
# Run on a dev machine (no NPU): bash tests/test_best_snapshot.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_AW="$(cd "${HERE}/../workspace/agent_workdir.triton" && pwd)"
TMP="$(mktemp -d)"; WS="${TMP}/agent_workdir"
trap 'rm -rf "${TMP}"' EXIT
cp -R "${SRC_AW}" "${WS}"

# Isolate from Ascend/conda env.sh on a dev box → pipeline falls back to python3.
rm -f "${WS}/tools/env.sh"

SCR="${WS}/.agents/skills/triton-op-verifier/scripts"
# Stub verify.py: correctness passes (exit 0), or fails when STUB_FAIL=1.
cat > "${SCR}/verify.py" <<'PY'
import os, sys
sys.exit(1 if os.environ.get("STUB_FAIL") == "1" else 0)
PY
# Stub benchmark.py: write perf_result.json with speedup from env STUB_SP.
cat > "${SCR}/benchmark.py" <<'PY'
import json, os, sys
a = sys.argv
out = next((a[i + 1] for i, x in enumerate(a) if x == "--output" and i + 1 < len(a)), None)
sp = float(os.environ.get("STUB_SP", "1.0"))
json.dump({"framework": {"avg_latency_ms": 1.0},
           "implementation": {"avg_latency_ms": 1.0 / sp},
           "speedup_vs_torch": sp}, open(out, "w"))
PY

mkdir -p "${WS}/src" "${WS}/output/submission"
cat > "${WS}/src/softmax.py" <<'PY'
import torch, torch.nn as nn
class Model(nn.Module):
    def forward(self, x): return torch.softmax(x, -1)
def get_inputs(): return [torch.randn(4, 4)]
def get_init_inputs(): return []
PY

IMPL="${WS}/output/submission/softmax_impl.py"
write_impl() {  # $1 = version marker — minimal AST-valid Triton impl
  cat > "${IMPL}" <<PY
# VERSION $1
import torch, torch.nn as nn, triton, triton.language as tl
@triton.jit
def k(p, o, n, B: tl.constexpr):
    c = tl.arange(0, B); m = c < n
    tl.store(o + c, tl.load(p + c, mask=m), mask=m)
class ModelNew(nn.Module):
    def forward(self, x):
        o = torch.empty_like(x)
        k[(1,)](x, o, x.numel(), 16)
        return o
PY
}

run() {  # $1 = STUB_SP, $2 = version ; success path
  write_impl "$2"
  STUB_SP="$1" bash "${WS}/tools/triton_eval_pipeline.sh" \
    --op_name softmax --impl "${IMPL}" --task "${WS}/src/softmax.py" >/dev/null 2>&1
}
run_fail() {  # $1 = version ; correctness fails → pipeline exits before snapshot
  write_impl "$1"
  STUB_FAIL=1 bash "${WS}/tools/triton_eval_pipeline.sh" \
    --op_name softmax --impl "${IMPL}" --task "${WS}/src/softmax.py" >/dev/null 2>&1 || true
}

best_sp()  { python3 -c "import json;print(json.load(open('${WS}/metrics.best.json'))['perf_data']['speedup_vs_torch'])"; }
best_ver() { grep -o 'VERSION [0-9]*' "${WS}/output/submission/softmax_impl.best.py" | head -1; }

fail=0
chk() { if [[ "$2" == "$3" ]]; then echo "  PASS $1"; else echo "  FAIL $1: got '$2' want '$3'"; fail=1; fi; }

run 0.30 1; chk "first success → best=v1@0.3"           "$(best_ver)|$(best_sp)" "VERSION 1|0.3"
run 0.10 2; chk "slower-but-correct → best STAYS v1@0.3" "$(best_ver)|$(best_sp)" "VERSION 1|0.3"
run 0.85 3; chk "faster → best=v3@0.85"                  "$(best_ver)|$(best_sp)" "VERSION 3|0.85"

# ① hash short-circuit: re-run the SAME impl (v3, unchanged) → reuse metrics, skip NPU.
# STUB_SP=0.99 would update best IF benchmark ran; short-circuit means it does NOT.
sc_out=$(STUB_SP=0.99 bash "${WS}/tools/triton_eval_pipeline.sh" \
         --op_name softmax --impl "${IMPL}" --task "${WS}/src/softmax.py" 2>&1)
if echo "${sc_out}" | grep -q "未改动"; then echo "  PASS ① re-run unchanged → short-circuit"; else echo "  FAIL ① no short-circuit msg"; fail=1; fi
chk "① short-circuit skips re-eval → best untouched (not 0.99)" "$(best_ver)|$(best_sp)" "VERSION 3|0.85"

run_fail 4; chk "correctness FAIL → best STAYS v3@0.85"  "$(best_ver)|$(best_sp)" "VERSION 3|0.85"

[[ "${fail}" == "0" ]] && echo "ALL PASS" || { echo "FAILED"; exit 1; }
