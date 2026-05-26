#!/usr/bin/env python3
"""
W2.23 一次性诊断：trace store ↔ OpenHands session_uid 是否对齐？

定位 L3 "no valid trajectories, dropping" 的根因。
不依赖训练栈，只读 SQLite + docker inspect。

跑法（在 main container 内 / 跟 train script 同一环境）：

    python3 examples/openhands_sdk/diagnose_trace_store.py

可选 env：
    TRACE_DB       默认 /workspace/results/rllm-openhands-traces.db
    CONTAINER_NAME 指定要解码哪个保留容器（默认自动挑最新的 rllm-openhands-*）
"""
from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import subprocess
import sys
from typing import Any


DB = os.environ.get("TRACE_DB", "/workspace/results/rllm-openhands-traces.db")
FORCE_CONTAINER = os.environ.get("CONTAINER_NAME", "").strip()


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def run(cmd: list[str]) -> str:
    """Run shell command, return stdout (empty string on error)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return r.stdout
    except Exception as e:
        print(f"  [warn] command failed: {' '.join(cmd)} → {e}", file=sys.stderr)
        return ""


def decode_slug(url: str) -> dict[str, Any] | None:
    """Extract /meta/rllm1:<base64-url-safe> from URL and decode."""
    m = re.search(r"/meta/rllm1:([A-Za-z0-9_\-]+)", url)
    if not m:
        return None
    raw = m.group(1)
    # url-safe base64 → standard base64
    raw = raw.replace("-", "+").replace("_", "/")
    pad = (-len(raw)) % 4
    raw += "=" * pad
    try:
        return json.loads(base64.b64decode(raw).decode("utf-8"))
    except Exception as e:
        print(f"  [warn] slug decode failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 1. DB schema + 表统计
# ---------------------------------------------------------------------------
section(f"1. DB schema + counts  ({DB})")

if not os.path.exists(DB):
    print(f"  ✗ DB not found: {DB}")
    sys.exit(1)

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# tables
tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()]
print(f"  Tables: {tables}")

has_traces = "traces" in tables
has_trace_sessions = "trace_sessions" in tables

if has_traces:
    n_traces = cur.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    print(f"  traces           rows: {n_traces}")
else:
    print("  ✗ NO `traces` TABLE — schema migration issue?")
    sys.exit(1)

if has_trace_sessions:
    n_ts = cur.execute("SELECT COUNT(*) FROM trace_sessions").fetchone()[0]
    print(f"  trace_sessions   rows: {n_ts}")
else:
    print("  ✗ NO `trace_sessions` TABLE — schema migration issue? proxy writes traces but no junction")
    n_ts = 0


# ---------------------------------------------------------------------------
# 2. traces 内容 sample
# ---------------------------------------------------------------------------
section("2. Latest traces (id / context_type / namespace / metadata head)")
rows = cur.execute(
    "SELECT id, context_type, namespace, "
    "substr(metadata, 1, 400) AS metadata_head, created_at "
    "FROM traces ORDER BY created_at DESC LIMIT 3"
).fetchall()
if not rows:
    print("  (no rows)")
else:
    for r in rows:
        print(f"  id={r['id']}")
        print(f"    context_type={r['context_type']}  namespace={r['namespace']}")
        print(f"    created_at={r['created_at']}")
        # try to pretty-print metadata
        try:
            md = json.loads(r["metadata_head"]) if r["metadata_head"] else {}
            keys = list(md.keys())
            print(f"    metadata keys: {keys}")
            for k in ("session_uids", "session_name"):
                if k in md:
                    print(f"      {k}: {md[k]!r}")
        except json.JSONDecodeError:
            print(f"    metadata (raw head): {r['metadata_head']}")


# ---------------------------------------------------------------------------
# 3. trace_sessions 内容 sample
# ---------------------------------------------------------------------------
section("3. Latest trace_sessions rows + distinct session_uids")
db_session_uids: set[str] = set()
db_session_names: set[str] = set()
if has_trace_sessions and n_ts > 0:
    rows = cur.execute(
        "SELECT trace_id, session_uid, created_at "
        "FROM trace_sessions ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    for r in rows:
        print(f"  trace_id={r['trace_id'][:32]}  session_uid={r['session_uid']}  at={r['created_at']}")
        db_session_uids.add(r["session_uid"])

    distinct = cur.execute(
        "SELECT session_uid, COUNT(*) c FROM trace_sessions "
        "GROUP BY session_uid ORDER BY c DESC LIMIT 20"
    ).fetchall()
    print("\n  Distinct session_uids in DB (top 20):")
    for r in distinct:
        print(f"    {r['session_uid']!r:60s} count={r['c']}")
elif has_trace_sessions:
    print("  ✗ trace_sessions table EXISTS but EMPTY")
    print("    → traces wrote OK but junction did NOT — likely litellm_callbacks 漏 session_uids")
else:
    print("  ✗ no junction table; can't compare")

# also pull session_names embedded in traces.metadata for cross-check
md_rows = cur.execute(
    "SELECT metadata FROM traces ORDER BY created_at DESC LIMIT 20"
).fetchall()
for r in md_rows:
    try:
        md = json.loads(r["metadata"]) if r["metadata"] else {}
    except Exception:
        continue
    if "session_name" in md:
        db_session_names.add(md["session_name"])
    if "session_uids" in md:
        v = md["session_uids"]
        if isinstance(v, list):
            for u in v:
                db_session_uids.add(u)

conn.close()


# ---------------------------------------------------------------------------
# 4. OpenHands 保留容器的 LLM_BASE_URL slug 解码
# ---------------------------------------------------------------------------
section("4. Decode LLM_BASE_URL slug from kept OpenHands containers")

if FORCE_CONTAINER:
    containers = [FORCE_CONTAINER]
else:
    # list kept containers, latest first
    out = run(["docker", "ps", "-a", "--filter", "name=rllm-openhands-",
               "--format", "{{.Names}}"])
    containers = [c.strip() for c in out.splitlines() if c.strip()][:4]

if not containers:
    print("  ✗ no kept rllm-openhands-* containers found — already cleaned up?")
else:
    print(f"  Inspecting {len(containers)} container(s)")

container_uids: set[str] = set()
container_names: set[str] = set()

for name in containers:
    print(f"\n  --- container {name} ---")
    env_out = run(["docker", "inspect", name,
                   "--format", "{{range .Config.Env}}{{println .}}{{end}}"])
    llm_url = ""
    for line in env_out.splitlines():
        if line.startswith("LLM_BASE_URL="):
            llm_url = line.split("=", 1)[1]
            break
    if not llm_url:
        print("    ✗ no LLM_BASE_URL env found")
        continue
    print(f"    LLM_BASE_URL: {llm_url[:120]}...")
    slug = decode_slug(llm_url)
    if slug is None:
        print("    ✗ slug decode failed")
        continue
    sn = slug.get("session_name")
    suids = slug.get("session_uids", [])
    print(f"    session_name: {sn!r}")
    print(f"    session_uids: {suids!r}")
    if sn:
        container_names.add(sn)
    if isinstance(suids, list):
        for u in suids:
            container_uids.add(u)


# ---------------------------------------------------------------------------
# 5. 对齐判定
# ---------------------------------------------------------------------------
section("5. Verdict: DB vs container session identifiers")

print(f"\n  session_uids:")
print(f"    in DB        : {sorted(db_session_uids) or '(none)'}")
print(f"    in containers: {sorted(container_uids) or '(none)'}")

overlap_uid = db_session_uids & container_uids
only_in_db = db_session_uids - container_uids
only_in_cont = container_uids - db_session_uids
print(f"    overlap      : {sorted(overlap_uid) or '(none)'}")
print(f"    only in DB   : {sorted(only_in_db) or '(none)'}")
print(f"    only in cont : {sorted(only_in_cont) or '(none)'}")

print(f"\n  session_names:")
print(f"    in DB.metadata : {sorted(db_session_names) or '(none)'}")
print(f"    in containers  : {sorted(container_names) or '(none)'}")


# ---------------------------------------------------------------------------
# 6. 结论 + 下一步
# ---------------------------------------------------------------------------
section("6. Diagnosis")

if not has_trace_sessions:
    print("  ✗ ROOT CAUSE: trace_sessions table missing — schema not migrated")
    print("    Fix: rm the DB, restart training, schema is created on first proxy init")
elif n_traces == 0:
    print("  ✗ ROOT CAUSE: traces table empty — proxy never wrote anything")
    print("    Fix: check proxy logs, callback wiring, db path env")
elif n_ts == 0:
    print("  ✗ ROOT CAUSE: traces written but trace_sessions junction empty")
    print("    Fix: litellm_callbacks.py is dropping session_uids before insert")
    print("    Check: rllm/sdk/proxy/litellm_callbacks.py:99-112")
elif container_uids and not overlap_uid:
    print("  ✗ ROOT CAUSE: name mismatch — DB and containers use disjoint session_uids")
    print("    → rllm queries by container_uids → 0 results → 'no valid trajectories'")
    print("    Fix: trace metadata_slug.py / litellm_callbacks.py session_uid derivation")
elif overlap_uid:
    print("  ✓ session_uids align between DB and containers")
    print("  → not a DB/name mismatch issue; problem is elsewhere:")
    print("    - LLM call recorded but step extraction returns empty (parse issue)")
    print("    - MAX_ITERATIONS=1 means agent had no step to commit")
    print("    - episode→trajectory mapping uses different key than session_uid")
    print("    Next: check rllm/engine/agent_sdk_engine.py how trajectory.steps is built")
else:
    print("  ⚠ inconclusive — both DB and containers have data but couldn't auto-correlate")
    print("    Manually compare the printed sets above")
