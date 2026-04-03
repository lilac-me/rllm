"""
rllm Training Dashboard — FastAPI Backend

Standalone read-only dashboard server.  Does NOT import any rllm trainer code;
it only reads:
  - ~/.rllm/traces.db   (SQLite WAL, opened read-only)
  - ~/.rllm/datasets/registry.json + parquet/arrow files

Run with:
    python -m rllm.tools.dashboard [options]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("rllm.dashboard")

# ---------------------------------------------------------------------------
# Globals set at startup
# ---------------------------------------------------------------------------
DB_PATH: str = ""
DATASET_DIR: str = ""
METRICS_FILE: str = ""

# ---------------------------------------------------------------------------
# Helper: read-only SQLite connection
# ---------------------------------------------------------------------------

async def _ro_conn(db_path: str) -> aiosqlite.Connection:
    """Open a read-only aiosqlite connection (WAL-safe)."""
    uri = f"file:{db_path}?mode=ro"
    conn = await aiosqlite.connect(uri, uri=True)
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = aiosqlite.Row
    return conn


async def _db_exists() -> bool:
    return os.path.exists(DB_PATH)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("rllm dashboard started  db=%s  datasets=%s", DB_PATH, DATASET_DIR)
    yield
    logger.info("rllm dashboard stopped")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="rllm Training Dashboard", lifespan=lifespan)

# Allow cross-origin requests (useful when running on a remote machine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static HTML served from embedded string (no build step)
# ---------------------------------------------------------------------------

def _html() -> str:
    """Return the single-page dashboard HTML."""
    static_dir = Path(__file__).parent / "static"
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return "<h1>Dashboard HTML not found</h1>"


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(_html())


# Serve static assets if present
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# API: global stats
# ---------------------------------------------------------------------------

@app.get("/api/stats")
async def api_stats():
    """Aggregate statistics for the overview panel."""
    if not await _db_exists():
        return JSONResponse({"db_exists": False})

    conn = await _ro_conn(DB_PATH)
    try:
        # Total traces
        async with conn.execute("SELECT COUNT(*) FROM traces") as cur:
            row = await cur.fetchone()
            total_traces = row[0] if row else 0

        # Namespaces
        async with conn.execute(
            "SELECT namespace, COUNT(*) as cnt FROM traces GROUP BY namespace ORDER BY cnt DESC LIMIT 20"
        ) as cur:
            namespaces = [{"namespace": r[0], "count": r[1]} for r in await cur.fetchall()]

        # Latest trace time
        async with conn.execute("SELECT MAX(created_at) FROM traces") as cur:
            row = await cur.fetchone()
            latest_ts = row[0] if row else None

        # Finish-reason distribution (from JSON)
        async with conn.execute(
            "SELECT json_extract(data,'$.output.finish_reason') as fr, COUNT(*) as cnt "
            "FROM traces GROUP BY fr"
        ) as cur:
            finish_reasons = {r[0] or "unknown": r[1] for r in await cur.fetchall()}

        # Token stats (last 1000)
        async with conn.execute(
            "SELECT json_extract(data,'$.tokens.prompt') as p, "
            "       json_extract(data,'$.tokens.completion') as c "
            "FROM traces ORDER BY created_at DESC LIMIT 1000"
        ) as cur:
            rows = await cur.fetchall()
            prompt_tokens = [r[0] for r in rows if r[0]]
            completion_tokens = [r[1] for r in rows if r[1]]

        def _safe_avg(lst):
            return round(sum(lst) / len(lst), 1) if lst else 0

        return JSONResponse({
            "db_exists": True,
            "total_traces": total_traces,
            "namespaces": namespaces,
            "latest_ts": latest_ts,
            "finish_reasons": finish_reasons,
            "avg_prompt_tokens": _safe_avg(prompt_tokens),
            "avg_completion_tokens": _safe_avg(completion_tokens),
        })
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# API: trace list
# ---------------------------------------------------------------------------

@app.get("/api/traces")
async def api_traces(
    namespace: str = Query(default=""),
    since: float = Query(default=0),
    limit: int = Query(default=50, ge=1, le=500),
    session_name: str = Query(default=""),
    offset: int = Query(default=0, ge=0),
):
    """Paginated trace list with optional filters."""
    if not await _db_exists():
        return JSONResponse({"traces": [], "total": 0})

    conn = await _ro_conn(DB_PATH)
    try:
        where: list[str] = []
        params: list[Any] = []

        if namespace:
            where.append("namespace = ?")
            params.append(namespace)
        if since > 0:
            where.append("created_at >= ?")
            params.append(since)
        if session_name:
            where.append("json_extract(data,'$.session_name') LIKE ?")
            params.append(f"%{session_name}%")

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        # Count
        async with conn.execute(
            f"SELECT COUNT(*) FROM traces {where_sql}", params
        ) as cur:
            count_row = await cur.fetchone()
            total = count_row[0] if count_row else 0

        # Fetch page
        query = f"""
            SELECT id, namespace, created_at,
                   json_extract(data,'$.session_name') as session_name,
                   json_extract(data,'$.model') as model,
                   json_extract(data,'$.tokens.prompt') as prompt_tokens,
                   json_extract(data,'$.tokens.completion') as completion_tokens,
                   json_extract(data,'$.tokens.total') as total_tokens,
                   json_extract(data,'$.latency_ms') as latency_ms,
                   json_extract(data,'$.output.finish_reason') as finish_reason
            FROM traces {where_sql}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        async with conn.execute(query, params + [limit, offset]) as cur:
            rows = await cur.fetchall()

        traces = []
        for r in rows:
            traces.append({
                "id": r[0],
                "namespace": r[1],
                "created_at": r[2],
                "session_name": r[3],
                "model": r[4],
                "prompt_tokens": r[5],
                "completion_tokens": r[6],
                "total_tokens": r[7],
                "latency_ms": r[8],
                "finish_reason": r[9],
            })

        return JSONResponse({"traces": traces, "total": total})
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# API: single session detail
# ---------------------------------------------------------------------------

@app.get("/api/traces/session/{session_name:path}")
async def api_session_detail(session_name: str):
    """Return all traces for a given session_name, ordered by timestamp."""
    if not await _db_exists():
        raise HTTPException(404, "Database not found")

    conn = await _ro_conn(DB_PATH)
    try:
        query = """
            SELECT id, namespace, created_at, data
            FROM traces
            WHERE json_extract(data,'$.session_name') = ?
            ORDER BY created_at ASC
        """
        async with conn.execute(query, [session_name]) as cur:
            rows = await cur.fetchall()

        steps = []
        for r in rows:
            d = json.loads(r[3])
            # data is Trace.model_dump() → output is LLMOutput (flat, not choices array):
            #   output.message.content, output.message.reasoning, output.finish_reason
            out = d.get("output") or {}
            msg = out.get("message") or {}
            steps.append({
                "id": r[0],
                "namespace": r[1],
                "created_at": r[2],
                "session_name": d.get("session_name"),
                "model": d.get("model"),
                "latency_ms": d.get("latency_ms"),
                "tokens": d.get("tokens", {}),
                "input_messages": (d.get("input") or {}).get("messages", []),
                "output_content": msg.get("content", ""),
                "output_reasoning": msg.get("reasoning"),
                "finish_reason": out.get("finish_reason"),
            })

        return JSONResponse({"session_name": session_name, "steps": steps})
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# API: episodes (View 3 — Episode-level aggregation)
# ---------------------------------------------------------------------------

@app.get("/api/episodes")
async def api_episodes(
    since: float = Query(default=0),
    limit: int = Query(default=200, ge=1, le=2000),
):
    """Aggregate episode statistics from recent traces."""
    if not await _db_exists():
        return JSONResponse({"episodes": [], "global": {}})

    conn = await _ro_conn(DB_PATH)
    try:
        since_clause = "AND created_at >= ?" if since > 0 else ""
        params: list[Any] = [since] if since > 0 else []
        params.append(limit)

        query = f"""
            SELECT json_extract(data,'$.session_name') as sn,
                   json_extract(data,'$.tokens.prompt') as pt,
                   json_extract(data,'$.tokens.completion') as ct,
                   json_extract(data,'$.latency_ms') as lat,
                   json_extract(data,'$.output.finish_reason') as fr,
                   created_at
            FROM traces
            WHERE 1=1 {since_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        async with conn.execute(query, params) as cur:
            rows = list(await cur.fetchall())

        # Group by task_id (first segment of session_name before ':')
        from collections import defaultdict
        tasks: dict[str, list] = defaultdict(list)
        for r in rows:
            sn = r[0] or ""
            task_id = sn.split(":")[0] if sn else "unknown"
            tasks[task_id].append({
                "session_name": sn,
                "prompt_tokens": r[1],
                "completion_tokens": r[2],
                "latency_ms": r[3],
                "finish_reason": r[4],
                "created_at": r[5],
            })

        episodes = []
        finish_reason_global: dict[str, int] = defaultdict(int)
        for task_id, steps in list(tasks.items())[:100]:
            finish_reasons_local: dict[str, int] = defaultdict(int)
            for s in steps:
                fr = s["finish_reason"] or "unknown"
                finish_reasons_local[fr] += 1
                finish_reason_global[fr] += 1

            ct_list = [s["completion_tokens"] for s in steps if s["completion_tokens"]]
            episodes.append({
                "task_id": task_id,
                "step_count": len(steps),
                "finish_reasons": dict(finish_reasons_local),
                "avg_completion_tokens": round(sum(ct_list) / len(ct_list), 1) if ct_list else 0,
                "max_completion_tokens": max(ct_list) if ct_list else 0,
                "last_created_at": max(s["created_at"] for s in steps),
            })

        # Sort episodes by recency
        episodes.sort(key=lambda x: x["last_created_at"], reverse=True)

        return JSONResponse({
            "episodes": episodes,
            "global": {
                "finish_reason_dist": dict(finish_reason_global),
                "total_steps": len(rows),
                "total_tasks": len(tasks),
            }
        })
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# API: sampling → training intermediate (View 4)
# ---------------------------------------------------------------------------

@app.get("/api/sampling")
async def api_sampling(
    limit: int = Query(default=500, ge=1, le=5000),
    since: float = Query(default=0),
):
    """Latency timeline, token histograms for recent traces."""
    if not await _db_exists():
        return JSONResponse({"timeline": [], "token_hist": []})

    conn = await _ro_conn(DB_PATH)
    try:
        since_clause = "AND created_at >= ?" if since > 0 else ""
        params: list[Any] = [since] if since > 0 else []
        params.append(limit)

        query = f"""
            SELECT created_at,
                   json_extract(data,'$.latency_ms') as lat,
                   json_extract(data,'$.tokens.completion') as ct,
                   json_extract(data,'$.tokens.prompt') as pt,
                   json_extract(data,'$.output.finish_reason') as fr,
                   json_extract(data,'$.session_name') as sn
            FROM traces
            WHERE 1=1 {since_clause}
            ORDER BY created_at ASC
            LIMIT ?
        """
        async with conn.execute(query, params) as cur:
            rows = await cur.fetchall()

        timeline = []
        completion_tokens = []
        prompt_tokens = []
        for r in rows:
            timeline.append({
                "ts": r[0],
                "latency_ms": r[1],
                "completion_tokens": r[2],
                "prompt_tokens": r[3],
                "finish_reason": r[4],
                "session_name": r[5],
            })
            if r[2] is not None:
                completion_tokens.append(r[2])
            if r[3] is not None:
                prompt_tokens.append(r[3])

        def _histogram(values: list[int], bins: int = 20) -> list[dict]:
            if not values:
                return []
            lo, hi = min(values), max(values)
            if lo == hi:
                return [{"bin_start": lo, "bin_end": hi, "count": len(values)}]
            width = (hi - lo) / bins
            counts = [0] * bins
            for v in values:
                idx = min(int((v - lo) / width), bins - 1)
                counts[idx] += 1
            return [
                {"bin_start": round(lo + i * width), "bin_end": round(lo + (i + 1) * width), "count": counts[i]}
                for i in range(bins)
            ]

        return JSONResponse({
            "timeline": timeline,
            "completion_token_hist": _histogram(completion_tokens),
            "prompt_token_hist": _histogram(prompt_tokens),
        })
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# API: datasets
# ---------------------------------------------------------------------------

def _registry_path() -> str:
    return os.path.join(DATASET_DIR, "registry.json")


@app.get("/api/datasets")
async def api_datasets():
    """List all registered datasets from registry.json."""
    reg_path = _registry_path()
    if not os.path.exists(reg_path):
        return JSONResponse({"datasets": []})

    with open(reg_path) as f:
        registry = json.load(f)

    datasets = []
    for name, info in registry.get("datasets", {}).items():
        splits_summary = {}
        for split_name, split_info in info.get("splits", {}).items():
            splits_summary[split_name] = {
                "num_examples": split_info.get("num_examples"),
                "fields": split_info.get("fields", []),
                "path": split_info.get("path"),
            }
        datasets.append({
            "name": name,
            "metadata": info.get("metadata", {}),
            "splits": splits_summary,
        })

    return JSONResponse({"datasets": datasets})


@app.get("/api/datasets/{name}/{split}")
async def api_dataset_split(
    name: str,
    split: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    search: str = Query(default=""),
):
    """Preview rows from a dataset split."""
    reg_path = _registry_path()
    if not os.path.exists(reg_path):
        raise HTTPException(404, "Dataset registry not found")

    with open(reg_path) as f:
        registry = json.load(f)

    ds_info = registry.get("datasets", {}).get(name)
    if not ds_info:
        raise HTTPException(404, f"Dataset '{name}' not found")

    split_info = ds_info.get("splits", {}).get(split)
    if not split_info:
        raise HTTPException(404, f"Split '{split}' not found in dataset '{name}'")

    rel_path = split_info["path"]
    abs_path = os.path.join(DATASET_DIR, rel_path)
    if not os.path.exists(abs_path):
        raise HTTPException(404, f"Data file not found: {abs_path}")

    try:
        import pandas as pd
        if abs_path.endswith(".parquet"):
            df = pd.read_parquet(abs_path)
        elif abs_path.endswith(".arrow"):
            df = pd.read_feather(abs_path)
        else:
            raise HTTPException(400, f"Unsupported file format: {abs_path}")

        # Drop binary columns for JSON serialization
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().head(1)
                if len(sample) > 0 and isinstance(sample.iloc[0], (bytes, bytearray)):
                    df[col] = "[binary]"

        # Search filter (simple string match across all text columns)
        if search:
            mask = df.apply(
                lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1
            )
            df = df[mask]

        total = len(df)
        page = df.iloc[offset : offset + limit]

        # Field stats
        stats = {}
        for col in df.columns:
            s = df[col]  # type: ignore
            non_null = int(s.notna().sum())  # type: ignore
            dtype_str = str(s.dtype)
            info: dict[str, Any] = {
                "dtype": dtype_str,
                "non_null_rate": round(non_null / len(df), 3) if len(df) > 0 else 0,
            }
            if dtype_str in ("object", "string"):
                lens = s.dropna().astype(str).str.len()  # type: ignore
                if len(lens) > 0:  # type: ignore
                    info["mean_len"] = round(float(lens.mean()), 1)  # type: ignore
                    info["max_len"] = int(lens.max())  # type: ignore
            stats[col] = info

        # Convert to JSON-serializable records
        records = json.loads(page.to_json(orient="records", default_handler=str))

        return JSONResponse({
            "name": name,
            "split": split,
            "total": total,
            "offset": offset,
            "limit": limit,
            "fields": list(df.columns),
            "rows": records,
            "field_stats": stats,
        })
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# API: training metrics (View 5)
# ---------------------------------------------------------------------------

@app.get("/api/metrics")
async def api_metrics():
    """Read metrics from ~/.rllm/metrics.jsonl if it exists."""
    if not os.path.exists(METRICS_FILE):
        return JSONResponse({"steps": []})

    steps = []
    try:
        with open(METRICS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    steps.append(json.loads(line))
    except Exception as e:
        logger.warning("Failed to read metrics file: %s", e)

    return JSONResponse({"steps": steps})


@app.post("/api/metrics")
async def api_ingest_metrics(body: dict):
    """Accept a single metrics dict from trainer and append to metrics.jsonl."""
    try:
        with open(METRICS_FILE, "a") as f:
            f.write(json.dumps(body) + "\n")
    except Exception as e:
        raise HTTPException(500, str(e))
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    global DB_PATH, DATASET_DIR, METRICS_FILE

    parser = argparse.ArgumentParser(description="rllm Training Dashboard")
    parser.add_argument("--db-path", default=os.path.expanduser("~/rllm-openhands-traces.db"))
    parser.add_argument("--dataset-dir", default=os.path.expanduser("~/.rllm/datasets"))
    parser.add_argument("--metrics-file", default=os.path.expanduser("~/.rllm/metrics.jsonl"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    DB_PATH = os.path.expanduser(args.db_path)
    DATASET_DIR = os.path.expanduser(args.dataset_dir)
    METRICS_FILE = os.path.expanduser(args.metrics_file)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    uvicorn.run(
        "rllm.tools.dashboard.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
