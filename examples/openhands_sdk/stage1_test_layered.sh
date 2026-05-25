#!/usr/bin/env bash
# ==============================================================================
# Stage1 layered smoke orchestrator
#
# Runs the stage1 training pipeline in escalating fidelity, fail-fast at the
# first failing layer so you know exactly which subsystem is broken.
#
# Layers:
#   L1  preflight              hydra config + import + dataset + AgentTrainer ctor
#                              (no Ray, no NPU, no docker, no LLM)
#   L2a mock_llm_server smoke  curl :MOCK_LLM_PORT /v1/chat/completions, verify
#                              200 + tool_calls schema
#   L2b mock_rollout dry-step  STAGE1_MOCK_ROLLOUT=1 + STAGE1_DRY_STEPS=1
#                              → exercises full rllm trainer step (Ray + Megatron
#                              load + 1 ppo iter) with deterministic fake rollout
#                              (no docker, no LLM); failures here are 100% rllm
#                              trainer / Megatron / NPU issues
#   L3  real-rollout dry-step  STAGE1_DRY_STEPS=1 only → real OpenHands docker +
#                              LiteLLM proxy + vllm-ascend + 1 ppo iter; failures
#                              here are Layer 4 (rollout chain) issues
#   L4  full training          no hatches; full run
#
# Usage::
#
#     bash examples/openhands_sdk/stage1_test_layered.sh                  # run all layers L1..L4
#     bash examples/openhands_sdk/stage1_test_layered.sh L1 L2a L2b       # subset
#     LAYERS="L1 L2b" bash examples/openhands_sdk/stage1_test_layered.sh  # same via env
#
# Env knobs:
#   MOCK_LLM_PORT         default 18000
#   STAGE1_LOG_DIR        where to dump per-layer logs (default /tmp/stage1-layered)
#   STAGE1_HALT_ON_FAIL   1 (default) = fail-fast; 0 = run all even on fail
# ==============================================================================

set -u  # don't `set -e`; we manage step-level pass/fail explicitly

LAYERS_DEFAULT="L1 L2a L2b L3 L4"
LAYERS_ARGS="${*:-}"
LAYERS_ENV="${LAYERS:-}"
LAYERS="${LAYERS_ARGS:-${LAYERS_ENV:-$LAYERS_DEFAULT}}"

MOCK_LLM_PORT="${MOCK_LLM_PORT:-18000}"
STAGE1_LOG_DIR="${STAGE1_LOG_DIR:-/tmp/stage1-layered}"
STAGE1_HALT_ON_FAIL="${STAGE1_HALT_ON_FAIL:-1}"

mkdir -p "${STAGE1_LOG_DIR}"

_now() { date +'%H:%M:%S'; }
_log() { echo "[$(_now)] $*"; }
_hdr() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

# Plain variables (bash 3.2 compat — macOS default). Keys: L1 L2a L2b L3 L4
LAYER_RESULT_L1=""
LAYER_RESULT_L2a=""
LAYER_RESULT_L2b=""
LAYER_RESULT_L3=""
LAYER_RESULT_L4=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_set_result() {
    # _set_result <layer> <value>
    case "$1" in
        L1)  LAYER_RESULT_L1="$2" ;;
        L2a) LAYER_RESULT_L2a="$2" ;;
        L2b) LAYER_RESULT_L2b="$2" ;;
        L3)  LAYER_RESULT_L3="$2" ;;
        L4)  LAYER_RESULT_L4="$2" ;;
    esac
}
_get_result() {
    case "$1" in
        L1)  echo "$LAYER_RESULT_L1" ;;
        L2a) echo "$LAYER_RESULT_L2a" ;;
        L2b) echo "$LAYER_RESULT_L2b" ;;
        L3)  echo "$LAYER_RESULT_L3" ;;
        L4)  echo "$LAYER_RESULT_L4" ;;
    esac
}

_run_layer() {
    local layer="$1"
    local label="$2"
    shift 2
    _hdr "$layer  $label"
    _log "log file: ${STAGE1_LOG_DIR}/${layer}.log"

    local start_ts end_ts elapsed rc
    start_ts=$(date +%s)
    ( set -o pipefail; "$@" ) 2>&1 | tee -i "${STAGE1_LOG_DIR}/${layer}.log"
    rc=${PIPESTATUS[0]}
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    if [[ $rc -eq 0 ]]; then
        _set_result "$layer" "PASS (${elapsed}s)"
        _log "$layer ${label}: PASS in ${elapsed}s"
    else
        _set_result "$layer" "FAIL rc=$rc (${elapsed}s)"
        _log "$layer ${label}: FAIL rc=$rc after ${elapsed}s; tail of log:"
        tail -n 30 "${STAGE1_LOG_DIR}/${layer}.log" | sed 's/^/    | /'
        if [[ "${STAGE1_HALT_ON_FAIL}" != "0" ]]; then
            _summarize
            exit $rc
        fi
    fi
}

# --- L1: preflight -----------------------------------------------------------
_l1_preflight() {
    PREFLIGHT_ONLY=1 bash "${SCRIPT_DIR}/train_openhands_qwen36_npu.sh"
}

# --- L2a: mock_llm_server curl smoke ----------------------------------------
_l2a_mock_llm_server() {
    # Start mock server in background
    _log "starting mock_llm_server on port ${MOCK_LLM_PORT}"
    python3 -m examples.openhands_sdk.mock_llm_server --port "${MOCK_LLM_PORT}" \
        > "${STAGE1_LOG_DIR}/mock_llm_server.log" 2>&1 &
    local server_pid=$!
    trap "kill ${server_pid} 2>/dev/null || true" EXIT

    # Wait up to 15s for /health
    local waited=0
    while ! curl -fsS "http://localhost:${MOCK_LLM_PORT}/health" >/dev/null 2>&1; do
        sleep 1
        waited=$((waited + 1))
        if [[ $waited -ge 15 ]]; then
            _log "mock_llm_server did not become ready in 15s"
            kill ${server_pid} 2>/dev/null || true
            return 1
        fi
    done
    _log "mock_llm_server ready in ${waited}s"

    # 1) Plain prompt smoke
    _log "smoke 1/2 — plain prompt"
    curl -fsS -X POST "http://localhost:${MOCK_LLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"mock","messages":[{"role":"user","content":"hi"}]}' \
        | python3 -c "import sys,json; r=json.load(sys.stdin); assert r['choices'][0]['message']['content'] is not None, r; print('  → content OK:', r['choices'][0]['message']['content'])"
    local rc1=$?

    # 2) Tool-call smoke
    _log "smoke 2/2 — tool call"
    curl -fsS -X POST "http://localhost:${MOCK_LLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"mock","messages":[{"role":"user","content":"list /tmp"}],"tools":[{"type":"function","function":{"name":"list_files","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]}' \
        | python3 -c "import sys,json; r=json.load(sys.stdin); m=r['choices'][0]['message']; assert m.get('tool_calls'), r; tc=m['tool_calls'][0]; assert tc['function']['name']=='list_files', tc; print('  → tool_call OK:', tc['function']['name'], tc['function']['arguments'])"
    local rc2=$?

    kill ${server_pid} 2>/dev/null || true
    trap - EXIT

    if [[ $rc1 -ne 0 || $rc2 -ne 0 ]]; then
        return 1
    fi
    return 0
}

# --- L2b: mock rollout + dry-step training -----------------------------------
_l2b_mock_dry_step() {
    STAGE1_MOCK_ROLLOUT=1 STAGE1_DRY_STEPS=1 \
        bash "${SCRIPT_DIR}/train_openhands_qwen36_npu.sh"
}

# --- L3: real rollout + dry-step training ------------------------------------
_l3_real_dry_step() {
    STAGE1_DRY_STEPS=1 \
        bash "${SCRIPT_DIR}/train_openhands_qwen36_npu.sh"
}

# --- L4: full training -------------------------------------------------------
_l4_full() {
    bash "${SCRIPT_DIR}/train_openhands_qwen36_npu.sh"
}

# --- summary -----------------------------------------------------------------
_summarize() {
    _hdr "Stage1 layered smoke summary"
    local v
    for L in L1 L2a L2b L3 L4; do
        v="$(_get_result "$L")"
        if [[ -n "$v" ]]; then
            printf "  %-4s %s\n" "$L" "$v"
        else
            printf "  %-4s %s\n" "$L" "(skipped)"
        fi
    done
    echo "logs: ${STAGE1_LOG_DIR}/"
}

# --- main --------------------------------------------------------------------
_log "layered smoke starting; layers=[${LAYERS}]"
for L in $LAYERS; do
    case "$L" in
        L1)  _run_layer L1  "preflight (config + import + dataset + ctor)"  _l1_preflight ;;
        L2a) _run_layer L2a "mock_llm_server curl smoke"                    _l2a_mock_llm_server ;;
        L2b) _run_layer L2b "mock_rollout + STAGE1_DRY_STEPS=1 training"    _l2b_mock_dry_step ;;
        L3)  _run_layer L3  "real rollout + STAGE1_DRY_STEPS=1 training"    _l3_real_dry_step ;;
        L4)  _run_layer L4  "full training (no hatches)"                    _l4_full ;;
        *)   _log "unknown layer: $L (valid: L1 L2a L2b L3 L4)"; exit 2 ;;
    esac
done
_summarize
