# Merge plan: openhands-refactoring ↔ verl-main

> Status: **PLAN ONLY, no code changes yet.** Reconcile two parallel OpenHands
> integrations. Not a one-way merge — both branches have unique value.

## 0. Branch state (as of 2026-05-29)

| | commits since 5daa24c0 | role |
|---|---|---|
| `origin/verl-main` | 72 | Qwen3.6 + AscendC, stage1 main line (active) |
| `origin/openhands-refactoring` | 9 | Qwen3-Coder + triton, pqy lineage (this work) |
| merge-base | `5daa24c0` | last common ancestor |

verl-main independently built a full OpenHands integration (DooD fix, padding
fix, cleanup, DataProto bridge, Dr.GRPO, NPUKernelBench AscendC data). This is
**NOT** "add openhands-refactoring on top of verl-main" — it's reconciling two
implementations that solved overlapping problems differently.

## 1. Architectural decision: keep the HTTP worker (corrected)

Earlier assessment called the HTTP worker "redundant" because verl-main's
bind-mount DooD fix is simpler. **That was a single-host-only view and is
wrong.** Decision: **HTTP worker is the target rollout architecture.**

| dimension | verl-main bind-mount (in-process) | openhands-refactoring HTTP worker |
|---|---|---|
| multi-host | ❌ impossible — `-v /home/docker/openhands_workspace:...` is a host-local path; a remote eval host's dockerd can't see the trainer host's fs | ✅ native — worker runs on whichever host owns the eval NPUs |
| code unity | single-host only; multi-host needs a second mechanism (tech debt) | single = worker@localhost, multi = worker@remote, one `_run_remote_eval_worker` path |
| workspace transfer | relies on host fs path equality (fragile, the DooD trap) | HTTP tar, path-independent |
| cost | no extra process | worker process + HTTP serialize (~132KB/rollout, negligible) |

bind-mount is the `worker=localhost` degenerate case of the HTTP worker, minus
multi-host. Keep HTTP worker; bind-mount may stay as an optional same-host fast
path or be dropped.

## 2. Fine-grained content audit

Legend: **A** = verl-main already has equal-or-better, drop ours · **B** =
both changed, line-by-line reconcile · **C** = openhands-refactoring unique &
valuable, must port.

### 2.1 Rollout / engine / trainer (code)

| file | class | diff vs verl-main | notes |
|---|---|---|---|
| `openhands_agent.py` `_run_remote_eval_worker` + `_tar_directory_b64` + `_manifest_directory` + `_extract_tar_b64_into` | **C** | unique | HTTP worker entry + helpers. verl-main has none. |
| `openhands_agent.py` `_impl_newer_than_metrics` + `_snapshot_success_as_best` | **C** | unique | reward anti-stale + best-impl snapshot. **Easy to miss in a naive merge.** |
| `openhands_agent.py` `_npu_operator_reward` / `_setup_npu_operator_workspace` / `rollout` / `_run_openhands_container` | **B** | large | both diverged from different pqy baselines; reconcile reward + workspace logic carefully |
| `remote_eval_worker.py` | **C** | unique (absent in verl-main) | HTTP worker server + GC + /health + /admin/reset-locks |
| `agent_sdk_engine.py` `_ENGINE_DEBUG`/`_dprint` gate | **C** | part of 131 | env-gated debug prints (RLLM_ENGINE_DEBUG) |
| `agent_sdk_engine.py` rollout_logprobs padding | **B/C** | part of 131 | logprob support (cf. openhands-observability "增加rollout_logprobs"); verify verl-main equivalent |
| `agent_sdk_engine.py` other | **A** | part of 131 | verl-main has +78 lines incl DataProto path; verl-main likely newer |
| `agent_sdk_trainer.py` LCM mini_batch padding | **A** | — | **byte-identical both sides** (verl-main:998). drop ours. |
| `agent_sdk_trainer.py` DataProto→TensorDict bridge, Dr.GRPO, stage2 audit | **A** | part of 329 | verl-main unique & newer; keep verl-main |
| `agent_sdk_trainer.py` config reads (ours) | **B** | part of 329 | only if we keep config/ layer |
| `distributed_npu_lock.py` | **B** | 37 | both diverged; reconcile |
| `runner.py` | **B** | 33 | small; reconcile |
| `create_mock_npu_operator_data.py` | **B** | 7 | trivial |

### 2.2 Config / infra (openhands-refactoring unique)

| file | class | notes |
|---|---|---|
| `config/.env.example`, `config/qwen30b.env` + `RLLM_TOPOLOGY` switch | **C** | env-driven config, decouples machine/path/topology. verl-main hardcodes in train_*.sh |
| `debug_oom.sh` (worker health check + reset-locks + cleanup) | **C** | verl-main has its own train_openhands_qwen36_npu.sh; reconcile or keep both |
| `OPENHANDS_REFACTORING_LOG.md` (KU1-8) | **C** | knowledge: DooD, NPU-lock, batch-align, LiteLLM flaky, etc. Port as reference. |

### 2.3 verl-main unique (keep, don't touch)

- DataProto→TensorDict bridge (`35f12439`)
- Dr.GRPO + std0 metric (`50bc9f96`)
- stage2 audit + observability metrics (`f258836b`)
- NPUKernelBench AscendC dataset (level2/3/4, ~500 files) — **AscendC, see §4**
- bind-mount DooD fix (`34cfe866`) — superseded by HTTP worker but harmless

## 3. Current-phase execution plan (code merge, NO skills)

Goal: land openhands-refactoring's **C-class code/infra** onto verl-main,
**reconcile B-class**, drop A-class. Skills deferred to §4.

Recommended base: **branch off `origin/verl-main`** (e.g. `verl-main-merge`),
do not touch the user's working verl-main.

Order (low-risk first):

1. **C-class clean adds** (no verl-main conflict):
   - `remote_eval_worker.py` (whole file)
   - `config/.env.example` + `config/qwen30b.env`
   - `OPENHANDS_REFACTORING_LOG.md`
2. **C-class into openhands_agent.py** (port unique functions):
   - HTTP worker: `_run_remote_eval_worker` + tar helpers + route in `rollout`
   - reward: `_impl_newer_than_metrics` + `_snapshot_success_as_best`
   - **decide**: HTTP worker as primary path, bind-mount as fallback or removed
3. **B-class reconcile** (line-by-line, needs care):
   - `openhands_agent.py` reward / workspace functions
   - `agent_sdk_engine.py` rollout_logprobs + _dprint (keep verl-main's newer parts)
   - `distributed_npu_lock.py`
   - `runner.py`, `create_mock_npu_operator_data.py`
4. **A-class**: take verl-main as-is (padding, DataProto bridge, Dr.GRPO).
5. **Validate** (no training): py_compile + bash -n + single-rollout smoke with
   worker on localhost.

Estimate: **3-5 person-days** (most cost is B-class reconcile in
openhands_agent.py + agent_sdk_engine.py).

## 4. Next phase (deferred): triton skills migration

**Out of scope for current phase.** Recorded here so it isn't lost.

- skills source: **user's dedicated triton skills repo (Claude Code format,
  `.claude/skills/`)** — NOT the pqy 4-skill set in openhands-refactoring, NOT
  the verl-main AscendC 5-skill set. **Repo path TBD (ask user).**
- verl-main current skills (AscendC): ascendc-translator, tilelang-designer,
  case-simplifier, performance-analyzer, trace-recorder
- migration tasks (next phase):
  1. format adapt: `.claude/skills/` → OpenHands `agent_workdir/.agents/skills/`
  2. `AGENTS.md` / `INSTRUCTIONS.md`: AscendC Phase flow → triton flow
  3. `operator_pipeline.sh` (155 diff lines AscendC↔triton): swap eval backend
     evaluate_ascendc.sh / evaluate_tilelang.sh → triton validation
  4. reward: `metrics.json` schema ↔ `_npu_operator_reward` alignment for triton
  5. dataset: verl-main NPUKernelBench (AscendC) → triton KernelBench data
     (openhands-refactoring `prepare_kernelbench_openhands_data.py` is triton)
  6. `validate_triton_impl.py`, `ascend_op_gen_agent_triton.md` port

## 5. Risks / open questions

1. **B-class reconcile is the real cost** — openhands_agent.py and
   agent_sdk_engine.py diverged from different pqy baselines; cannot blindly
   pick one side. Each shared function needs a 3-way look (5daa24c0 base +
   verl-main + openhands-refactoring).
2. **rollout_logprobs**: openhands-refactoring has logprob padding; verify
   whether verl-main has an equivalent before porting (avoid double impl).
3. **HTTP worker vs bind-mount coexistence**: decide if bind-mount stays as a
   same-host fast path or is removed. Keeping both = two code paths to test.
4. **verl-main has uncommitted work + in-progress merge** (user's qwen36) —
   work on a fresh branch off origin/verl-main, never the user's local
   verl-main.
5. **distributed_npu_lock.py diverged (37 lines)** — both sides changed; needs
   a real diff review, not assumed-identical.

## 6. What NOT to do (don't repeat the earlier mistakes)

- ❌ Don't drop the HTTP worker as "redundant" — it's the multi-host architecture
- ❌ Don't `git merge` openhands-refactoring into verl-main wholesale — 15
  conflicting files, most engine/trainer changes are superseded; you'd reconcile
  329+131 diff lines of noise
- ❌ Don't assume verl-main's newer = strictly better on every function — reward
  improvements (_snapshot_success_as_best) and HTTP worker are ours and valuable
- ❌ Don't migrate skills in this phase — separate concern, needs the user's
  triton skills repo
