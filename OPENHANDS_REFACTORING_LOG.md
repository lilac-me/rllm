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
| 5 | `f6d3db64` | HTTP-only routing + worker-side GC + worker health check in sh |
| 6 | `40a48c2e` | Unconditional NPU lock clear (worker startup + `/admin/reset-locks` from trainer) |
| 7 | TBD | Collapse `qwen30b-singlehost.env` into `qwen30b.env` via `RLLM_TOPOLOGY` switch |

## Active design decisions

### D1. All OpenHands rollouts route through `remote_eval_worker.py` over HTTP

- **Why**: under DooD (dev container with host docker.sock mounted), `docker run -v <dev_container_path>:/...` mounts an empty directory because host dockerd looks the path up on the host filesystem. This silently breaks `--entrypoint /opt/workspace/entrypoint.py` (binary not present in mounted empty dir) → container stuck in `Created` → trainer hangs on `docker wait`.
- **Fix**: HTTP path anchors all `-v` source paths on the worker host's real filesystem. Trainer doesn't care whether the worker is on the same host or a remote host — only the URL changes.
- **Single-host config**: `OPENHANDS_REMOTE_EVAL_URL=http://127.0.0.1:16881`, worker runs on the host (not in dev container).
- **Multi-host config**: `OPENHANDS_REMOTE_EVAL_URL=http://<eval-host-ip>:16881`.

~~The legacy "same-machine docker run" branch (`_run_openhands_container`) is **dead code** in any DooD deployment. Kept for backwards compat with non-DooD setups; can be deleted once we're sure no one needs it.~~ **DELETED 2026-06-01**: `_run_openhands_container` and the same-host DooD path were removed; rollouts are now remote-worker-only and an empty `OPENHANDS_REMOTE_EVAL_URL` fails fast (`openhands_agent.py:518`).

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

### D4b. One config file, topology via env switch

`config/qwen30b.env` is the single config; `RLLM_TOPOLOGY` env decides
single-host vs multi-host. Default is `multi` (preserves prior behavior).

```
RLLM_TOPOLOGY=multi  bash debug_oom.sh   # trainer 65, eval worker 51
RLLM_TOPOLOGY=single bash debug_oom.sh   # both on this host
```

The differences between modes are only 4 variables:
  - `TRAIN_HOST_IP` / `EVAL_WORKER_IP` (which IP)
  - `OPENHANDS_CONTAINER_HOST_ALIAS` (TRAIN_HOST_IP in multi, host.docker.internal in single)
  - `OPENHANDS_EVAL_DEVICE_IDS` (0-3 in multi for eval host, 8-15 in single to stay disjoint from trainer)

Bogus `RLLM_TOPOLOGY` value fails fast with a clear error.

`qwen30b-singlehost.env` is **deleted**. Don't recreate it; add another
branch to the `if` if a new topology comes up.

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

### KU7. dynamic batch alignment (FIXED via padding, not truncation)

verl asserts `batch.batch_size[0] % mini_batch_size == 0` at
[verl/protocol.py:815](https://github.com/volcengine/verl/blob/main/verl/protocol.py#L815)
for megatron data-parallel rank alignment. Under rllm's
`stepwise_advantage.enable=True`, the batch row count is
`sum(valid_steps over trajectories)` — `valid_steps` varies independently
per trajectory (OpenHands agents finish early / hit MAX_ITERATIONS /
get filtered by overlong prompts at the per-step level).

So **even `train_batch_size % ppo_mini_batch_size == 0` is NOT enough**.

This is **rllm upstream issue #350** (open, no comments, no fix as of
2026-05): https://github.com/rllm-org/rllm/issues/350

**Observed failures**:
- `train=2 ppo_mini=4 rollout_n=8`: 144 % 32 != 0 (avg_steps=9)
- `train=4 ppo_mini=4 rollout_n=8`: 296 % 32 != 0 (varied step counts)

**Fix landed in this branch (padding, not truncation)**:
The existing `_pad_dataproto_to_world_size` in
[`agent_sdk_trainer.py:720`](rllm/trainer/verl/agent_sdk_trainer.py) already
pads to DP `world_size` via `pad_dataproto_to_divisor`, with the padded
rows tagged `is_pad_step=True` / `is_last_step=False` / `is_valid=False`.
We just fold the effective verl `mini_batch_size = ppo_mini_batch_size *
rollout.n` into the LCM so a single padding call satisfies both
constraints. One added line:

```python
world_sizes.append(self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                   * self.config.actor_rollout_ref.rollout.n)
world_size = reduce(math.lcm, world_sizes)
```

**Why padding instead of truncation (first attempt was wrong)**:
- Sparse reward setting (KernelBench tasks): most trajectories have
  reward=0; truncating trailing rows risks dropping the few valuable
  reward>0 step rows. `for episode in enumerate(episodes)` is
  deterministic order, so truncation has systematic bias toward dropping
  the same tail tasks. Hurts both data efficiency and GRPO group-mean/std
  advantage normalization.
- The framework already wires `is_pad_step` masking through the trainer
  (verl_backend.py / agent_workflow_trainer.py / agent_ppo_trainer.py
  / agent_sdk_trainer.py all call `_remove_padding` before
  advantage compute), so padded rows don't pollute gradients in
  `stepwise_advantage.mode=broadcast`.

**Provenance of the fix**:
- AgentPPOTrainer got the same fix on this fork's
  `openhands-observability` branch (commit `f3763717`, by ZhihaoSun,
  2026-04-30). We mirrored it here for AgentSdkTrainer; AgentPPOTrainer
  on this branch is not touched (no need yet).
- Upstream rllm PR #506 fixes the same class of bug but only in
  `rllm/experimental/verl/verl_backend.py` (fully-async path), not
  AgentSdkTrainer/AgentPPOTrainer.

**Verify after long run**: pad rows in PPO loss should contribute zero
gradient (advantage tensor zero-padded by `pad_dataproto_to_divisor`).
If reward/loss curve looks off after 50+ steps, double-check that pad
rows in the `advantages` tensor are indeed zero (not copy-of-last-row).

**Upstream contribution opportunity**: open a PR to rllm-org/rllm
mirroring `openhands-observability`'s AgentPPOTrainer fix +
AgentSdkTrainer fix, referencing issue #350.

### KU6. `OPENHANDS_REMOTE_KEEP_WORKDIR` debug toggle

[remote_eval_worker.py:261](examples/openhands_sdk/remote_eval_worker.py:261) checks this env to skip the per-rollout `rmtree`. Useful for postmortem inspection of failed rollouts. Documented here, no automated lifecycle.

### KU8. LiteLLM proxy startup non-determinism on flaky network

**Symptom**: `bash debug_oom.sh` sometimes works, sometimes fails with
`TimeoutError: Proxy server did not start within 30.0s` at
[proxy_manager.py:244](rllm/sdk/proxy/proxy_manager.py:244). Same machine,
same config, back-to-back runs — pure non-determinism.

**Root cause**: LiteLLM synchronously fetches
`https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json`
inside `get_model_cost_map.py:271` during startup. The fetch result is
non-essential (LiteLLM has a local fallback), but the fetch **blocks
startup**. On flaky outbound network the timing varies:
- DNS hit + TCP connect ok → fetch < 5s → server ready inside rllm's 30s window ✓
- DNS slow / TCP timeout to GitHub → fetch blocks > 30s → rllm
  `_wait_for_server_start` times out before LiteLLM finishes init ✗

Trainer log evidence (3 occurrences across two attempts):
```
LiteLLM:WARNING: get_model_cost_map.py:271 — LiteLLM: Failed to fetch
remote model cost map ... [Errno 101] Network is unreachable. Falling
back to local backup.
```

**Fix (operator-side, no code change)**: set in shell / env before
`debug_oom.sh`:
```bash
export LITELLM_LOCAL_MODEL_COST_MAP="True"
```
This makes LiteLLM skip the remote fetch entirely and use the bundled
local map. Startup drops to < 5s, fully deterministic.

**Verify env name matches your LiteLLM version**:
```bash
LITELLM_DIR=$(python3 -c 'import litellm, os; print(os.path.dirname(litellm.__file__))')
grep -nE 'LITELLM_LOCAL|os.environ' $LITELLM_DIR/litellm_core_utils/get_model_cost_map.py | head
```

**Long-term framework fixes** (not done in this branch):
1. rllm `proxy_manager.py:_wait_for_server_start` hardcodes
   `timeout=30.0`. Should be env-configurable (`RLLM_PROXY_START_TIMEOUT`)
   so slow networks aren't outright blocked. ~2 line change.
2. rllm `proxy_manager.py` could set
   `LITELLM_LOCAL_MODEL_COST_MAP=True` by default when launching the
   subprocess, removing the trap entirely. Even cleaner: just `env=...`
   to the `subprocess.Popen` call.

Recorded as an upstream contribution candidate.

## Verified facts (don't re-debate)

- `--network host` on dev container is required for trainer ↔ host-worker on `127.0.0.1` to work (verified via `ip route` showing host LAN as default).
- `/tmp/shared_npu_lock` must be created **on the host** by the worker, not by the trainer in the dev container, because the `docker run -v` source is interpreted on the host fs.
- HTTP body size for workspace transfer is ~130 KB base64-encoded; not a bottleneck.
- `_run_openhands_container` in `openhands_agent.py` (the legacy non-HTTP path) **never worked under DooD**. 51-step BS=2 runs succeeded only because they went via HTTP (worker on host 51, trainer on host 65) — multi-host topology accidentally avoided the DooD path bug. **(Deleted 2026-06-01 — remote-worker-only now.)**

## Rejected approaches (don't propose again)

- ❌ Mount `/workspace` from host into dev container — breaks dev container isolation requirement
- ❌ `docker container prune` triggered by training loop — race with in-flight rollouts
- ❌ `--rm` flag on `docker run -d` — racy with `docker logs` retrieval after `docker wait`
- ❌ Reduce `n_parallel_tasks` to dodge dockerd state issues — hides the real bug
- ❌ Reduce `--shm-size` to "fix" Created stuck — no evidence shm is the cause
- ❌ Restart dockerd as a workaround in `debug_oom.sh` — too disruptive, kills user dev containers
