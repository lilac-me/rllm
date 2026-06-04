#!/usr/bin/env bash
# run_eval.sh — one-shot eval: T=0 greedy (pass@1 baseline) + T>0 sampled (pass@k),
#               with step calibration (how many agent turns rollouts actually use).
#
# STEP CALIBRATION first (small batch, generous cap):
#   EVAL_MAX_CONCURRENT=4 OPENHANDS_REMOTE_EVAL_URL=http://127.0.0.1:18880 \
#   OPENHANDS_IMAGE=openhands-triton-env:v1 OPENHANDS_EVAL_DEVICE_IDS=0,1,2,3 \
#   LLM_BASE_URL=http://127.0.0.1:8003/v1 OPENHANDS_MODEL_NAME=qwen35 \
#   bash eval/run_eval.sh /path/NPUKernelBench --levels 1 --max-ops 6 --max-iterations 40
#   # then read <out>/sample/summary.json -> step_calibration.iterations_observed.max
#   #   set --max-iterations ~= max * 1.3 ; if pct_hit_cap > 0, raise it and re-probe.
#
# FULL RUN (after you picked --max-iterations):
#   ... bash eval/run_eval.sh /path/NPUKernelBench --levels 1,2 --max-iterations 30
#
# Knobs (env): EVAL_N (sampled trajectories, default 4), EVAL_T (sampled temperature, 0.7),
#   EVAL_OUT_DIR, EVAL_MAX_CONCURRENT, plus the worker env above. Anything after <dataset>
#   is passed through to eval_openhands.py (--levels/--max-ops/--max-iterations/--op-filter/...).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:?usage: run_eval.sh <NPUKernelBench dir|mock> [extra eval args...]}"; shift || true
N="${EVAL_N:-4}"
T="${EVAL_T:-0.7}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${EVAL_OUT_DIR:-eval_runs/${STAMP}}"
COMMON=(--dataset "${DATASET}" --rollout "${EVAL_ROLLOUT:-remote_worker}" --judge-timeout 1800 "$@")

echo "================================================================"
echo "[run_eval] dataset=${DATASET}  N=${N}  T=${T}  out=${OUT}"
echo "[run_eval] concurrency=${EVAL_MAX_CONCURRENT:-(default 4)}  worker=${OPENHANDS_REMOTE_EVAL_URL:-UNSET}"
echo "================================================================"

echo "[run_eval] === pass A: greedy baseline (T=0, N=1 -> pass@1) ==="
python3 "${HERE}/eval_openhands.py" "${COMMON[@]}" \
    --temperature 0 --n-trajectories 1 --k-values 1 --out-dir "${OUT}/greedy"

echo "[run_eval] === pass B: sampled (T=${T}, N=${N} -> pass@1,pass@${N}) ==="
python3 "${HERE}/eval_openhands.py" "${COMMON[@]}" \
    --temperature "${T}" --n-trajectories "${N}" --k-values "1,${N}" --out-dir "${OUT}/sample"

echo "================================================================"
echo "[run_eval] COMPARISON"
python3 - "${OUT}" "${N}" <<'PY'
import json, sys
out, N = sys.argv[1], sys.argv[2]
def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return {"_error": repr(e)}
g = load(f"{out}/greedy/summary.json")
s = load(f"{out}/sample/summary.json")
print(f"  greedy  pass@1            : {g.get('pass_at_k', {}).get('pass@1')}   solve_any={g.get('solve_rate_any')}")
print(f"  sampled pass@1 / pass@{N}    : {s.get('pass_at_k', {}).get('pass@1')} / {s.get('pass_at_k', {}).get(f'pass@{N}')}")
print(f"  sampled best-of-N speedup  : geomean={s.get('best_of_N_speedup', {}).get('geomean_over_correct_ops')} "
      f"fast@1.0={s.get('best_of_N_speedup', {}).get('fast@1.0')}")
sc = s.get("step_calibration", {})
io = sc.get("iterations_observed", {})
print(f"  STEP CALIBRATION (sampled) : iterations max={io.get('max')} p90={io.get('p90')} median={io.get('median')} "
      f"| pct_hit_cap={sc.get('pct_hit_cap')} (set --max-iterations={sc.get('max_iterations_set')})")
if sc.get("pct_hit_cap", 0):
    print("  ⚠ some rollouts hit the cap — raise --max-iterations and re-probe.")
print(f"\n  full JSON: {out}/greedy/summary.json , {out}/sample/summary.json")
PY
