#!/usr/bin/env bash
# =============================================================================
# Portable launcher for the OpenHands remote eval worker.
#
# Goal: maximum portability. ALL worker-side scratch state (work dir + NPU lock
# dir) is created under the LAUNCH DIR ($PWD) — no /home/docker, no /tmp
# hardcoding. Copy the repo anywhere, run this, and everything it needs is made
# right next to where you launched from.
#
# Hard floor (cannot be relocated): the host's Ascend NPU driver paths. The
# worker's containers bind-mount them (remote_eval_worker.py:175-181) to reach
# the NPU — they ARE the driver install, not something we can create in a launch
# dir. We fail-fast with a clear message if they're missing.
#
# Note: there is NO "content" to pre-seed on the worker side — the actual
# workspace/skills bundle is shipped by the trainer in each /run POST payload
# (openhands_agent.py:524). Worker state is just empty scratch dirs.
#
# Usage:
#   bash examples/openhands_sdk/start_remote_worker.sh
#   EVAL_WORKER_PORT=16881 OPENHANDS_EVAL_DEVICE_IDS=8,9,10,11,12,13,14,15 \
#       bash examples/openhands_sdk/start_remote_worker.sh
#   WORKER_STATE_DIR=/data/eval-state bash .../start_remote_worker.sh   # override
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_PY="$SCRIPT_DIR/remote_eval_worker.py"
[[ -f "$WORKER_PY" ]] || { echo "ERROR: remote_eval_worker.py not found next to this script: $WORKER_PY" >&2; exit 1; }

# --- tunables (all overridable; none are absolute host paths) ----------------
HOST="${HOST:-127.0.0.1}"                                  # single-host loopback
EVAL_WORKER_PORT="${EVAL_WORKER_PORT:-16881}"              # MUST match config/qwen36.env:31
OPENHANDS_IMAGE="${OPENHANDS_IMAGE:-openhands-triton-env:v1}"
OPENHANDS_EVAL_DEVICE_IDS="${OPENHANDS_EVAL_DEVICE_IDS:-8,9,10,11,12,13,14,15}"

# --- worker scratch state under the launch dir (fully portable) --------------
WORKER_STATE_DIR="${WORKER_STATE_DIR:-$PWD/remote-eval-state}"
WORK_DIR="$WORKER_STATE_DIR/work"
LOCK_DIR="$WORKER_STATE_DIR/npu-locks"
mkdir -p "$WORK_DIR" "$LOCK_DIR"

# --- Ascend driver paths CANNOT be relocated (host NPU install). Fail fast. --
# docker -v auto-creates a MISSING source as an empty dir, so a missing driver
# path would NOT fail `docker run` — it'd silently give the container an empty
# mount and NPU access breaks deep inside the eval. Catch it here instead.
missing=0
for p in /usr/local/Ascend/driver /usr/local/dcmi /usr/local/bin/npu-smi /etc/ascend_install.info /dev; do
    [[ -e "$p" ]] || { echo "ERROR: required Ascend host path missing: $p" >&2; missing=1; }
done
if [[ "$missing" != 0 ]]; then
    echo "  ^ These are the NPU driver install; worker containers mount them (remote_eval_worker.py:175-181)." >&2
    echo "    They are not relocatable into the launch dir. Run this on a real Ascend NPU host." >&2
    exit 1
fi
[[ -e /mnt/pipeline-data ]] || echo "WARN: /mnt/pipeline-data absent — docker will auto-create it empty (harmless for mock/plumbing; remote_eval_worker.py:184)." >&2

# --- launch ------------------------------------------------------------------
# OPENHANDS_EVAL_LOCK_DIR is honored by BOTH the bind-mount source AND the
# startup-clear / reset-locks logic (remote_eval_worker.py:31 + _run_request),
# so the relocated lock dir is consistent end-to-end.
export OPENHANDS_EVAL_LOCK_DIR="$LOCK_DIR"
export OPENHANDS_IMAGE OPENHANDS_EVAL_DEVICE_IDS

# --- clear orphan eval containers from a crashed prior worker run ------------
# Each rollout container (rllm-openhands-eval-*, remote_eval_worker.py:129) is
# normally `docker rm -f`'d in the worker's per-request finally (:236). A worker
# KILLED mid-rollout leaves orphans that still hold NPUs + file locks → the next
# run collides. Worker startup = clean slate (no rollouts in flight), so sweep
# them here — mirrors the worker's startup lock-clear (remote_eval_worker.py:345-347).
# ASSUMES ONE WORKER PER HOST: a blanket prefix sweep also kills a sibling
# worker's in-flight containers. Set WORKER_SKIP_ORPHAN_SWEEP=1 to skip.
# Best-effort: docker absent / rm failures never abort the launch (|| true).
if [[ "${WORKER_SKIP_ORPHAN_SWEEP:-0}" != "1" ]]; then
    orphans="$(docker ps -aq --filter "name=rllm-openhands-eval-" 2>/dev/null || true)"
    if [[ -n "$orphans" ]]; then
        echo "[start-worker] removing $(echo "$orphans" | wc -l | tr -d ' ') orphan eval container(s)"
        echo "$orphans" | xargs docker rm -f >/dev/null 2>&1 || true
    fi
fi

echo "[start-worker] state dir : $WORKER_STATE_DIR"
echo "[start-worker] work_dir  : $WORK_DIR"
echo "[start-worker] lock_dir  : $LOCK_DIR"
echo "[start-worker] listen    : http://$HOST:$EVAL_WORKER_PORT"
echo "[start-worker] image     : $OPENHANDS_IMAGE   devices: $OPENHANDS_EVAL_DEVICE_IDS"

exec python3 "$WORKER_PY" --host "$HOST" --port "$EVAL_WORKER_PORT" --work-dir "$WORK_DIR"
