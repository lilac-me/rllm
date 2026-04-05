"""
observer_gateway.py — Standalone REST API server that receives telemetry
from rllm OpenHands containers.

Run with:
    python observer_gateway.py [--host HOST] [--port PORT] [--db PATH]

Default: http://0.0.0.0:8858   db: ./observer_data.db

─────────────────────────────────────────────────────────────────────────────
REST API
─────────────────────────────────────────────────────────────────────────────

Sessions
  GET    /api/v1/sessions                       List all sessions (summary)
  DELETE /api/v1/sessions/{id}                  Delete session + all data

Per-session: live state (latest in-memory)
  GET    /api/v1/sessions/{id}/state            Latest state snapshot
  POST   /api/v1/sessions/{id}/state            Upload state (from container)

Per-session: historical snapshots (persisted)
  GET    /api/v1/sessions/{id}/snapshots        Snapshot index (metadata only)
         ?limit=N&since=<unix_ts>
  GET    /api/v1/sessions/{id}/snapshots/{snap_id}
                                                Full payload of one snapshot

Per-session: events (persisted)
  GET    /api/v1/sessions/{id}/events           Event list
         ?limit=N&offset=N&type=<event_type>&since=<unix_ts>
  POST   /api/v1/sessions/{id}/events           Upload single event (from container)
  GET    /api/v1/sessions/{id}/events/{db_id}/payload
                                                Full raw payload of one event

Per-session: control (in-memory + persisted log)
  GET    /api/v1/sessions/{id}/control          Current pause/resume flags
  POST   /api/v1/sessions/{id}/control          Set {"pause":bool}|{"resume":bool}
  GET    /api/v1/sessions/{id}/control/log      Control action history
         ?limit=N&since=<unix_ts>

Per-session: request log (persisted)
  GET    /api/v1/sessions/{id}/requests         HTTP request log for this session
         ?limit=N&since=<unix_ts>&method=GET|POST|...

Per-session: stats
  GET    /api/v1/sessions/{id}/stats            Counts by category + event breakdown

Global
  GET    /api/v1/stats                          Global counts
  GET    /api/v1/requests                       All request log entries
         ?limit=N&since=<unix_ts>&method=GET|POST|...

─────────────────────────────────────────────────────────────────────────────
Dependencies: stdlib only. No external packages required.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger(__name__)

# Module-level DB handle (set by serve())
_db: Any = None  # GatewayDB instance

# ---------------------------------------------------------------------------
# In-memory live-state store (fast reads; DB has the full history)
# ---------------------------------------------------------------------------

_SESSION_LOCK = threading.Lock()

# session_id → {state, events (recent), control, updated_at, created_at}
_SESSIONS: dict[str, dict[str, Any]] = {}
_MAX_MEM_EVENTS = 500   # events kept in RAM per session


def _get_or_create(session_id: str) -> dict[str, Any]:
    with _SESSION_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = {
                "state": {},
                "events": [],
                "control": {"pause": False, "resume": False},
                "updated_at": time.time(),
                "created_at": time.time(),
            }
        return _SESSIONS[session_id]


# ---------------------------------------------------------------------------
# URL routing
# ---------------------------------------------------------------------------

_R_SESSIONS   = re.compile(r"^/api/v1/sessions$")
_R_SESSION    = re.compile(r"^/api/v1/sessions/([^/]+)$")
_R_STATE      = re.compile(r"^/api/v1/sessions/([^/]+)/state$")
_R_SNAPSHOTS  = re.compile(r"^/api/v1/sessions/([^/]+)/snapshots$")
_R_SNAPSHOT1  = re.compile(r"^/api/v1/sessions/([^/]+)/snapshots/(\d+)$")
_R_EVENTS     = re.compile(r"^/api/v1/sessions/([^/]+)/events$")
_R_EVENT1     = re.compile(r"^/api/v1/sessions/([^/]+)/events/(\d+)/payload$")
_R_CONTROL    = re.compile(r"^/api/v1/sessions/([^/]+)/control$")
_R_CTRL_LOG   = re.compile(r"^/api/v1/sessions/([^/]+)/control/log$")
_R_REQ_SESS   = re.compile(r"^/api/v1/sessions/([^/]+)/requests$")
_R_STATS_SESS = re.compile(r"^/api/v1/sessions/([^/]+)/stats$")
_R_STATS_GLOB = re.compile(r"^/api/v1/stats$")
_R_REQS_GLOB  = re.compile(r"^/api/v1/requests$")


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ObserverHandler(BaseHTTPRequestHandler):
    """HTTP handler for the observer gateway."""

    server_version = "ObserverGateway/2.0"

    def log_message(self, fmt, *args):  # type: ignore[override]
        pass   # We log via the request_log table

    # ── Helpers ───────────────────────────────────────────────────────────

    def _send_json(self, code: int, body: Any, *, _ts_start: float | None = None) -> None:
        data = json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)
        self._response_code = code
        self._ts_end = time.time()

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def _parse_json_body(self) -> dict[str, Any] | None:
        try:
            raw = self._read_body()
            self._body_bytes = len(raw)
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return None

    def _qp(self, name: str, default: str = "") -> str:
        """Parse a single query parameter."""
        if "?" in self.path:
            qs = self.path.split("?", 1)[1]
            for part in qs.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    if k == name:
                        return v
        return default

    def _qp_int(self, name: str, default: int) -> int:
        try:
            return int(self._qp(name, str(default)))
        except ValueError:
            return default

    def _qp_float(self, name: str, default: float | None = None) -> float | None:
        v = self._qp(name)
        if not v:
            return default
        try:
            return float(v)
        except ValueError:
            return default

    def _client_ip(self) -> str:
        return self.client_address[0] if self.client_address else ""

    # ── Lifecycle: log every request to DB ────────────────────────────────

    def handle_one_request(self) -> None:
        self._ts_start = time.time()
        self._body_bytes = 0
        self._response_code = 0
        self._ts_end = self._ts_start
        self._session_id_for_log: str | None = None
        # DEBUG: log every inbound request immediately
        logger.debug(
            "[gateway] --> %s %s  client=%s",
            getattr(self, "command", "?"),
            self.path,
            self._client_ip(),
        )
        try:
            super().handle_one_request()
        finally:
            elapsed_ms = (self._ts_end - self._ts_start) * 1000
            logger.debug(
                "[gateway] <-- %s %s  status=%d  body=%db  %.1fms  client=%s",
                getattr(self, "command", "?"),
                self.path.split("?")[0],
                self._response_code,
                self._body_bytes,
                elapsed_ms,
                self._client_ip(),
            )
            if _db is not None:
                try:
                    _db.log_request(
                        method=self.command or "",
                        path=self.path.split("?")[0],
                        client_ip=self._client_ip(),
                        received_at=self._ts_start,
                        status_code=self._response_code,
                        body_bytes=self._body_bytes,
                        response_ms=elapsed_ms,
                        session_id=self._session_id_for_log,
                    )
                except Exception:
                    pass

    # ── OPTIONS (CORS) ────────────────────────────────────────────────────

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self._response_code = 204
        self._ts_end = time.time()

    # ── GET ───────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: C901
        path = self.path.split("?")[0]

        # GET /api/v1/sessions
        if _R_SESSIONS.match(path):
            self._handle_list_sessions()
            return

        # GET /api/v1/stats
        if _R_STATS_GLOB.match(path):
            stats = _db.get_global_stats() if _db else {"error": "no db"}
            self._send_json(200, stats)
            return

        # GET /api/v1/requests
        if _R_REQS_GLOB.match(path):
            self._handle_list_requests(session_id=None)
            return

        m = _R_STATE.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_get_state(m.group(1))
            return

        m = _R_SNAPSHOTS.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_list_snapshots(m.group(1))
            return

        m = _R_SNAPSHOT1.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_get_snapshot(m.group(1), int(m.group(2)))
            return

        m = _R_EVENTS.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_list_events(m.group(1))
            return

        m = _R_EVENT1.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_get_event_payload(m.group(1), int(m.group(2)))
            return

        m = _R_CONTROL.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_get_control(m.group(1))
            return

        m = _R_CTRL_LOG.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_list_control_log(m.group(1))
            return

        m = _R_REQ_SESS.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_list_requests(session_id=m.group(1))
            return

        m = _R_STATS_SESS.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            stats = _db.get_session_stats(m.group(1)) if _db else {}
            self._send_json(200, stats)
            return

        self._send_json(404, {"error": "not found"})

    # ── POST ──────────────────────────────────────────────────────────────

    def do_POST(self) -> None:  # noqa: C901
        path = self.path.split("?")[0]

        m = _R_STATE.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_post_state(m.group(1))
            return

        m = _R_EVENTS.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_post_event(m.group(1))
            return

        m = _R_CONTROL.match(path)
        if m:
            self._session_id_for_log = m.group(1)
            self._handle_post_control(m.group(1))
            return

        self._send_json(404, {"error": "not found"})

    # ── DELETE ────────────────────────────────────────────────────────────

    def do_DELETE(self) -> None:
        path = self.path.split("?")[0]
        m = _R_SESSION.match(path)
        if m:
            sid = m.group(1)
            self._session_id_for_log = sid
            # Remove from memory
            with _SESSION_LOCK:
                _SESSIONS.pop(sid, None)
            # Remove from DB
            ok = _db.delete_session(sid) if _db else False
            if ok:
                logger.info("[gateway] Session %s deleted", sid)
                self._send_json(200, {"ok": True})
            else:
                self._send_json(404, {"error": "session not found"})
            return
        self._send_json(404, {"error": "not found"})

    # ─────────────────────────────────────────────────────────────────────
    # Handler implementations
    # ─────────────────────────────────────────────────────────────────────

    def _handle_list_sessions(self) -> None:
        # Merge in-memory + DB sessions
        if _db:
            db_sessions = {s["session_id"]: s for s in _db.list_sessions()}
        else:
            db_sessions = {}

        with _SESSION_LOCK:
            mem_ids = set(_SESSIONS.keys())

        all_ids = set(db_sessions.keys()) | mem_ids
        result = []
        for sid in all_ids:
            mem = _SESSIONS.get(sid, {})
            db_s = db_sessions.get(sid, {})
            state = mem.get("state", {})
            result.append({
                "session_id": sid,
                "label": db_s.get("label", state.get("session_label", "")),
                "is_running": state.get("is_running", False),
                "phase": state.get("phase", "unknown"),
                "iteration": state.get("iteration", 0),
                "uptime_seconds": state.get("uptime_seconds", 0),
                "mem_events_count": len(mem.get("events", [])),
                "db_events_count": db_s.get("event_count", 0) if db_s else 0,
                "created_at": db_s.get("created_at", mem.get("created_at", 0)),
                "updated_at": mem.get("updated_at", db_s.get("last_seen_at", 0)),
                "in_memory": sid in mem_ids,
            })
        result.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        self._send_json(200, result)

    # ── State ─────────────────────────────────────────────────────────────

    def _handle_get_state(self, session_id: str) -> None:
        with _SESSION_LOCK:
            mem = _SESSIONS.get(session_id)
        if mem is not None:
            self._send_json(200, mem["state"])
            return
        # Fall back to DB latest snapshot
        if _db:
            state = _db.get_latest_state(session_id)
            if state is not None:
                self._send_json(200, state)
                return
        self._send_json(404, {"error": "session not found"})

    def _handle_post_state(self, session_id: str) -> None:
        body = self._parse_json_body()
        if body is None:
            self._send_json(400, {"error": "invalid JSON"})
            return

        client_ip = self._client_ip()
        now = time.time()

        # DEBUG: summarise incoming state
        phase = body.get("phase", "?")
        iteration = body.get("iteration", "?")
        is_running = body.get("is_running", "?")
        new_events = body.get("events", [])
        logger.debug(
            "[gateway] POST state  session=%s  phase=%s  iter=%s  running=%s  embedded_events=%d  from=%s",
            session_id, phase, iteration, is_running, len(new_events), client_ip,
        )

        # Update in-memory live state
        sess = _get_or_create(session_id)
        with _SESSION_LOCK:
            sess["state"] = body
            sess["updated_at"] = now
            # Merge events into RAM ring buffer
            if new_events:
                existing_ids = {e.get("event_id") for e in sess["events"]}
                for ev in new_events:
                    if ev.get("event_id") not in existing_ids:
                        sess["events"].append(ev)
                        existing_ids.add(ev.get("event_id"))
                if len(sess["events"]) > _MAX_MEM_EVENTS:
                    sess["events"] = sess["events"][-_MAX_MEM_EVENTS:]

        # Persist to DB
        if _db:
            label = body.get("session_label", "")
            meta = body.get("extra_metadata", {})
            _db.upsert_session(
                session_id, label=label, metadata=meta, first_state=True
            )
            _db.save_state(session_id, body, client_ip=client_ip)
            # Also persist events
            if new_events:
                _db.save_events(session_id, new_events, client_ip=client_ip)

        self._send_json(200, {"ok": True})

    def _handle_list_snapshots(self, session_id: str) -> None:
        if not _db:
            self._send_json(501, {"error": "persistence not enabled"})
            return
        limit = self._qp_int("limit", 50)
        since = self._qp_float("since")
        snaps = _db.list_state_snapshots(session_id, limit=limit, since=since)
        self._send_json(200, snaps)

    def _handle_get_snapshot(self, session_id: str, snap_id: int) -> None:
        if not _db:
            self._send_json(501, {"error": "persistence not enabled"})
            return
        snap = _db.get_state_snapshot_by_id(snap_id)
        if snap is None or snap.get("session_id") != session_id:
            self._send_json(404, {"error": "snapshot not found"})
            return
        self._send_json(200, snap)

    # ── Events ────────────────────────────────────────────────────────────

    def _handle_list_events(self, session_id: str) -> None:
        limit = self._qp_int("limit", 100)
        offset = self._qp_int("offset", 0)
        event_type = self._qp("type") or None
        since = self._qp_float("since")

        # Try DB first
        if _db:
            events = _db.list_events(
                session_id,
                limit=limit,
                event_type=event_type,
                since=since,
                offset=offset,
            )
            total = _db.count_events(session_id, event_type=event_type)
            self._send_json(200, {"total": total, "items": events})
            return

        # Fall back to in-memory
        with _SESSION_LOCK:
            sess = _SESSIONS.get(session_id)
        if sess is None:
            self._send_json(404, {"error": "session not found"})
            return
        items = sess["events"]
        if event_type:
            items = [e for e in items if e.get("event_type") == event_type]
        items = items[-limit:]
        self._send_json(200, {"total": len(items), "items": items})

    def _handle_post_event(self, session_id: str) -> None:
        body = self._parse_json_body()
        if body is None:
            self._send_json(400, {"error": "invalid JSON"})
            return

        # DEBUG: show every inbound event
        event_type = body.get("event_type", "?")
        summary = body.get("summary", "")
        client_ip = self._client_ip()
        logger.debug(
            "[gateway] POST event  session=%s  type=%s  summary=%s  from=%s",
            session_id, event_type, summary[:120], client_ip,
        )

        sess = _get_or_create(session_id)
        with _SESSION_LOCK:
            sess["events"].append(body)
            sess["updated_at"] = time.time()
            if len(sess["events"]) > _MAX_MEM_EVENTS:
                sess["events"] = sess["events"][-_MAX_MEM_EVENTS:]

        if _db:
            _db.save_events(session_id, [body], client_ip=client_ip)

        self._send_json(200, {"ok": True})

    def _handle_get_event_payload(self, session_id: str, db_id: int) -> None:
        if not _db:
            self._send_json(501, {"error": "persistence not enabled"})
            return
        payload = _db.get_event_payload(db_id)
        if payload is None:
            self._send_json(404, {"error": "event not found"})
            return
        self._send_json(200, payload)

    # ── Control ───────────────────────────────────────────────────────────

    def _handle_get_control(self, session_id: str) -> None:
        sess = _get_or_create(session_id)
        with _SESSION_LOCK:
            ctrl = dict(sess["control"])
        # DEBUG: log control poll result
        logger.debug(
            "[gateway] GET control  session=%s  pause=%s  resume=%s  from=%s",
            session_id, ctrl.get("pause"), ctrl.get("resume"), self._client_ip(),
        )
        # Log as an outbound poll
        if _db:
            _db.log_control(
                session_id, direction="outbound", action="poll",
                client_ip=self._client_ip(),
            )
        self._send_json(200, ctrl)

    def _handle_post_control(self, session_id: str) -> None:
        body = self._parse_json_body()
        if body is None:
            self._send_json(400, {"error": "invalid JSON"})
            return

        sess = _get_or_create(session_id)
        client_ip = self._client_ip()

        with _SESSION_LOCK:
            ctrl = sess["control"]
            if "pause" in body:
                ctrl["pause"] = bool(body["pause"])
                if ctrl["pause"]:
                    ctrl["resume"] = False
                logger.info("[gateway] Session %s → pause=%s (from %s)",
                            session_id, ctrl["pause"], client_ip)
                if _db:
                    _db.log_control(session_id, "inbound", "pause",
                                    value=ctrl["pause"], client_ip=client_ip)
            if "resume" in body:
                ctrl["resume"] = bool(body["resume"])
                if ctrl["resume"]:
                    ctrl["pause"] = False
                logger.info("[gateway] Session %s → resume=%s (from %s)",
                            session_id, ctrl["resume"], client_ip)
                if _db:
                    _db.log_control(session_id, "inbound", "resume",
                                    value=ctrl["resume"], client_ip=client_ip)

        self._send_json(200, {"ok": True, "control": dict(sess["control"])})

    def _handle_list_control_log(self, session_id: str) -> None:
        if not _db:
            self._send_json(501, {"error": "persistence not enabled"})
            return
        limit = self._qp_int("limit", 100)
        since = self._qp_float("since")
        rows = _db.list_control_log(session_id, limit=limit, since=since)
        self._send_json(200, rows)

    # ── Requests ──────────────────────────────────────────────────────────

    def _handle_list_requests(self, session_id: str | None) -> None:
        if not _db:
            self._send_json(501, {"error": "persistence not enabled"})
            return
        limit = self._qp_int("limit", 100)
        since = self._qp_float("since")
        method = self._qp("method") or None
        rows = _db.list_requests(
            session_id=session_id, limit=limit, since=since, method=method
        )
        self._send_json(200, rows)


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def serve(
    host: str = "0.0.0.0",
    port: int = 8858,
    db_path: str = "observer_data.db",
    log_level: str = "INFO",
) -> None:
    global _db

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.debug(
        "[gateway] Log level set to %s — DEBUG output is active", log_level.upper()
    )

    # Initialise persistence
    from persistence import GatewayDB
    _db = GatewayDB(db_path)

    server = HTTPServer((host, port), ObserverHandler)
    logger.info("Observer gateway listening on http://%s:%d", host, port)
    logger.info("DB: %s", db_path)
    logger.info("─" * 60)
    logger.info("REST API overview:")
    logger.info("  Sessions          GET  /api/v1/sessions")
    logger.info("  Live state        GET  /api/v1/sessions/{id}/state")
    logger.info("  Snapshot index    GET  /api/v1/sessions/{id}/snapshots?limit=N&since=T")
    logger.info("  Single snapshot   GET  /api/v1/sessions/{id}/snapshots/{snap_id}")
    logger.info("  Events            GET  /api/v1/sessions/{id}/events?limit=N&type=T&since=T")
    logger.info("  Event payload     GET  /api/v1/sessions/{id}/events/{db_id}/payload")
    logger.info("  Control           GET  /api/v1/sessions/{id}/control")
    logger.info("  Set control       POST /api/v1/sessions/{id}/control")
    logger.info("  Control log       GET  /api/v1/sessions/{id}/control/log")
    logger.info("  Request log       GET  /api/v1/sessions/{id}/requests")
    logger.info("  Session stats     GET  /api/v1/sessions/{id}/stats")
    logger.info("  Global stats      GET  /api/v1/stats")
    logger.info("  All requests      GET  /api/v1/requests")
    logger.info("─" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Gateway stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rllm OpenHands Observer Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8858, help="Bind port (default: 8858)")
    parser.add_argument(
        "--db", default="observer_data.db",
        help="SQLite database path (default: observer_data.db in CWD)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )
    args = parser.parse_args()
    serve(host=args.host, port=args.port, db_path=args.db, log_level=args.log_level)
