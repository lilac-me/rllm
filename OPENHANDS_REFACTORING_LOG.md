# OpenHands refactoring log (branch: `openhands-refactoring`)

> Track decisions and known-unsolved issues for the OpenHands RL training stack
> as it's lifted out of `pqy/debug_oom` into a cleaner form on top of stage1's
> `5daa24c0` diverge point.

Branch base: `5daa24c0` (offload buffer, last common ancestor with stage1).

## Commit timeline

| # | Hash | Theme |
|---|---|---|
| 1 | `978539bf` | Port `examples/openhands_sdk/*` from `pqy/debug_oom@fbd7dbe3` (squash) |
| 2 | `eeb5ccbb` | Port `rllm/engine/agent_sdk_engine.py` changes (trajectory_id naming + debug instrumentation, latter gated in #3) |
| 3 | `93898fa5` | Env-driven config layer: `examples/openhands_sdk/config/{.env.example,qwen30b.env}`, `RLLM_LOG_DIR` / `OPENHANDS_KERNELBENCH_DATASET` for hardcoded paths, `RLLM_ENGINE_DEBUG` env gate for `agent_sdk_engine.py` prints |
| 4 | `d4ed3328` | Single-host (16-NPU) config + orphan-container sweep at sh launch |
| 5 | TBD | HTTP-only routing (delete same-machine docker-run path) + worker-side GC + worker health check in sh + single/multi unification |

## Active design decisions

### D1. All OpenHands rollouts route through `remote_eval_worker.py` over HTTP

- **Why**: under DooD (dev container with host docker.sock mounted), `docker run -v <dev_container_path>:/...` mounts an empty directory because host dockerd looks the path up on the host filesystem. This silently breaks `--entrypoint /opt/workspace/entrypoint.py` (binary not present in mounted empty dir) → container stuck in `Created` → trainer hangs on `docker wait`.
- **Fix**: HTTP path anchors all `-v` source paths on the worker host's real filesystem. Trainer doesn't care whether the worker is on the same host or a remote host — only the URL changes.
- **Single-host config**: `OPENHANDS_REMOTE_EVAL_URL=http://127.0.0.1:16881`, worker runs on the host (not in dev container).
- **Multi-host config**: `OPENHANDS_REMOTE_EVAL_URL=http://<eval-host-ip>:16881`.

The legacy "same-machine docker run" branch in [openhands_agent.py:626-627](examples/openhands_sdk/openhands_agent.py:626) (`_run_openhands_container`) is **dead code** in any DooD deployment. Kept for backwards compat with non-DooD setups; can be deleted once we're sure no one needs it.

### D2. Worker GC at startup + per-trainer-restart lock reset

Two cleanup mechanisms with different policies:

**Workdir GC (mtime-based, conservative)**: `_gc_stale_workdirs()` removes
`openhands-remote-eval-*` workdirs older than 1 hour at worker startup. Older
than 1h almost certainly means SIGKILL / OOM. Newer workdirs are kept so
`OPENHANDS_REMOTE_KEEP_WORKDIR=1` debug runs are not silently wiped.

**NPU lock clear (unconditional, aggressive)**: `_clear_npu_locks()` removes
*all* files under `/tmp/shared_npu_lock/`. Triggered:
  1. At worker startup (worker starting = no rollouts in flight)
  2. On `POST /admin/reset-locks` (called by `debug_oom.sh` after worker health
     check passes — this gives the trainer-restart-without-worker-restart case
     a way to clean stale locks)

Per-rollout cleanup of *one* lock is the OpenHands container's job. These two
mechanisms are the cross-rollout / cross-trainer-restart safety nets.

Stuck `Created` containers (dockerd state-machine bug) need a daemon restart —
we fail soft, see KU2.

### D3. Worker health check in `debug_oom.sh`, fail-fast

Sh-level `curl -sf -m 3 ${OPENHANDS_REMOTE_EVAL_URL}/health` before any training setup. If worker isn't running, training exits at second 3 with a clear error pointing at the worker launch cmd, instead of hanging on a 2100s HTTP timeout at first rollout.

The worker launch itself is **manual** (`python3 remote_eval_worker.py ...` on the worker host, by the user). Reasons:
  - Multi-host can't auto-spawn from trainer sh (it's on a different machine)
  - Single = multi consistency: same operator workflow regardless of topology
  - User has full control over worker logs / restart policy

### D4. HTTP timeout chain

- Trainer side ([openhands_agent.py:653](examples/openhands_sdk/openhands_agent.py:653)): `urlopen(timeout=container_timeout + 300)`
- Worker side ([remote_eval_worker.py:158](examples/openhands_sdk/remote_eval_worker.py:158)): `docker wait` `timeout=container_timeout`
- 300s buffer covers tar unpack + docker start + docker logs + agent_workdir tar + HTTP response

If `container_timeout` is changed, both sides honor it. No drift possible.

## Known-unsolved / deferred (record only)

### KU1. Worker resilience — single point of failure

If `remote_eval_worker.py` crashes, all in-flight rollouts hang on HTTP timeout. Currently mitigated by D3 (fail-fast at sh launch), but no auto-recovery during training.

**Considered**: supervisord / systemd unit / docker-compose health check + restart. **Defer**: not blocking phase 1, and the dev container deployment doesn't have systemd anyway. Revisit if worker crashes are observed in long runs.

### KU2. dockerd "Created" stuck — independent failure mode

Both 51 (multi-host, BS=3 step 42) and 65 (single-host, step 1 before DooD fix) saw `docker run` produce containers stuck in `Created` state, with `docker rm -f` reporting "No such container" and `docker inspect` hanging. Root cause not identified — candidates:
  - 16-way concurrent `--privileged --device-cgroup-rule` triggering containerd shim deadlock
  - `/dev/davinci_manager` device cgroup contention when trainer already holds NPUs
  - Long-running dockerd state accumulation from rollout container churn

**Mitigations in place**:
  - `debug_oom.sh` startup sweep of `rllm-openhands-*` containers + `docker container prune -f`
  - `docker rm -f` in trainer-side `finally` ([openhands_agent.py:848](examples/openhands_sdk/openhands_agent.py:848)) and worker-side `finally` ([remote_eval_worker.py:175](examples/openhands_sdk/remote_eval_worker.py:175))

**Not yet done**:
  - Reproduce in a controlled experiment (single `docker run` of the rollout cmd, no trainer) once the DooD path issue is fixed and base rollouts work. Will tell whether it's still a real failure mode or was only triggered by the path-mismatch retry storm.
  - If real: bisect docker cmd flags (`--privileged` / `--shm-size 500g` / `-v /dev:/dev` / `--device-cgroup-rule`) to isolate the trigger.

### KU3. `base + cp` workspace optimization — not needed

Initial idea: worker decompresses workspace tar once into `/tmp/openhands_workspace_base`, then `cp -r base trajectory-X` per rollout instead of re-decompressing.

**Why deferred**: measured workspace tar.gz = 97 KB, base64 = 132 KB. HTTP transfer < 0.1s, unpack < 50ms per rollout. Over a minute-scale LLM rollout this is < 0.1% of total time. **Not worth the implementation cost.**

If workspace ever balloons to GB scale (e.g. embedded model weights), revisit.

### KU4. `docker container prune -f` has no name filter

[debug_oom.sh:40](examples/openhands_sdk/debug_oom.sh:40) prunes all stopped containers on the host, not just `rllm-openhands-*`. Risk: collides with user dev containers in `Exited` state. **Defer**: add `--filter "label=rllm-openhands"` once trainer adds a label to launched containers (small follow-up).

### KU5. NPU device count verification

`OPENHANDS_EVAL_DEVICE_IDS=8,9,...,15` assumes 16 NPUs. Not programmatically verified in trainer. If a single-host machine has only 8 NPUs, OpenHands containers will try to lock non-existent devices and `entrypoint.py` will hang at NPU probe.

**Mitigation**: user verifies `npu-smi info` before launch (operator runbook).

### KU6. `OPENHANDS_REMOTE_KEEP_WORKDIR` debug toggle

[remote_eval_worker.py:261](examples/openhands_sdk/remote_eval_worker.py:261) checks this env to skip the per-rollout `rmtree`. Useful for postmortem inspection of failed rollouts. Documented here, no automated lifecycle.

## Verified facts (don't re-debate)

- `--network host` on dev container is required for trainer ↔ host-worker on `127.0.0.1` to work (verified via `ip route` showing host LAN as default).
- `/tmp/shared_npu_lock` must be created **on the host** by the worker, not by the trainer in the dev container, because the `docker run -v` source is interpreted on the host fs.
- HTTP body size for workspace transfer is ~130 KB base64-encoded; not a bottleneck.
- `_run_openhands_container` in `openhands_agent.py` (the legacy non-HTTP path) **never worked under DooD**. 51-step BS=2 runs succeeded only because they went via HTTP (worker on host 51, trainer on host 65) — multi-host topology accidentally avoided the DooD path bug.

## Rejected approaches (don't propose again)

- ❌ Mount `/workspace` from host into dev container — breaks dev container isolation requirement
- ❌ `docker container prune` triggered by training loop — race with in-flight rollouts
- ❌ `--rm` flag on `docker run -d` — racy with `docker logs` retrieval after `docker wait`
- ❌ Reduce `n_parallel_tasks` to dodge dockerd state issues — hides the real bug
- ❌ Reduce `--shm-size` to "fix" Created stuck — no evidence shm is the cause
- ❌ Restart dockerd as a workaround in `debug_oom.sh` — too disruptive, kills user dev containers
