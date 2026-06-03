"""Unit test: judge scores the best-correct submission (R1 judge side).

Exercises the real remote_eval_worker._judge_and_record: it must score
{op}_impl.best.py when present (the pipeline keeps it on each success), and
only fall back to {op}_impl.py when no best exists. This is what guarantees a
Phase-3 optimization that breaks _impl.py cannot lower reward below the
achieved best.

Dep-light (stdlib only) so it runs on a dev machine without NPU/rllm.
Run: `python3 tests/test_judge_best_selection.py`  or  `pytest tests/test_judge_best_selection.py`.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import remote_eval_worker as w  # noqa: E402


def _setup_ws(tmp, *, final=None, best=None):
    """Build workspace/agent_workdir/output/submission with optional _impl.py / _impl.best.py."""
    sub = Path(tmp) / "workspace" / "agent_workdir" / "output" / "submission"
    sub.mkdir(parents=True, exist_ok=True)
    if final is not None:
        (sub / "softmax_impl.py").write_text(final, encoding="utf-8")
    if best is not None:
        (sub / "softmax_impl.best.py").write_text(best, encoding="utf-8")
    workspace = str(Path(tmp) / "workspace")
    judge_root = str(Path(tmp) / "judge")
    (Path(judge_root) / "agent_workdir").mkdir(parents=True, exist_ok=True)
    agent_dir = Path(tmp) / "agent_out"
    return workspace, judge_root, agent_dir


def _run(tmp, **kw):
    """Run _judge_and_record with _run_judge monkeypatched to echo the file it would score."""
    workspace, judge_root, agent_dir = _setup_ws(tmp, **kw)
    captured = {}

    def fake_run_judge(jr, request):
        scored = Path(jr) / "agent_workdir" / "output" / "submission" / "softmax_impl.py"
        captured["scored"] = scored.read_text(encoding="utf-8") if scored.exists() else None
        return {"success": True, "scored_marker": captured["scored"]}

    orig = w._run_judge
    w._run_judge = fake_run_judge
    try:
        w._judge_and_record(workspace, judge_root, {"task": {"op_name": "softmax"}}, agent_dir)
    finally:
        w._run_judge = orig
    metrics = json.loads((agent_dir / "judge_metrics.json").read_text(encoding="utf-8"))
    return captured.get("scored"), metrics


def test_prefers_best_when_present():
    # Phase 3 left _impl.py broken; _impl.best.py holds the achieved-best correct version.
    with tempfile.TemporaryDirectory() as tmp:
        scored, _ = _run(tmp, final="# FINAL broken by Phase 3", best="# BEST correct version")
        assert scored == "# BEST correct version", f"judge must score .best, got: {scored!r}"


def test_falls_back_to_final_when_no_best():
    with tempfile.TemporaryDirectory() as tmp:
        scored, _ = _run(tmp, final="# FINAL only")
        assert scored == "# FINAL only", f"judge must score _impl.py when no .best, got: {scored!r}"


def test_submission_missing_when_neither():
    with tempfile.TemporaryDirectory() as tmp:
        scored, metrics = _run(tmp)  # neither file present
        assert scored is None, "judge must not run when there is nothing to score"
        assert metrics.get("error_type") == "submission_missing", f"expected submission_missing, got {metrics}"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in fns:
        f()
        print(f"PASS {n}")
    print(f"ALL {len(fns)} PASS")
