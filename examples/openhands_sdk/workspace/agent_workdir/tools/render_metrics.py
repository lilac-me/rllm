#!/usr/bin/env python3
"""Write metrics.json + profiling_results.json (stdlib only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Render operator metrics JSON files.")
    p.add_argument("--workspace", required=True, type=Path)
    p.add_argument("--compile-ok", choices=("true", "false"), required=True)
    p.add_argument("--correctness-ok", choices=("true", "false"), required=True)
    p.add_argument("--profile-ok", choices=("true", "false"), default="true")
    p.add_argument("--backend", default="unknown")
    p.add_argument("--latency-ms", default="")
    p.add_argument("--bandwidth-gbps", default="0.0")
    p.add_argument("--error", default="")
    args = p.parse_args()

    compile_ok = args.compile_ok == "true"
    correctness_ok = args.correctness_ok == "true"
    profile_ok = args.profile_ok == "true"
    err = (args.error or "").strip() or None

    latency_val: float | None
    if str(args.latency_ms).strip() == "":
        latency_val = None
    else:
        try:
            latency_val = float(args.latency_ms)
        except ValueError:
            latency_val = None

    try:
        bw = float(args.bandwidth_gbps)
    except ValueError:
        bw = 0.0

    success = bool(compile_ok and correctness_ok and profile_ok)
    if err and "fail" in err.lower():
        success = False

    metrics = {
        "schema_version": 1,
        "backend": args.backend,
        "compile_ok": compile_ok,
        "correctness_ok": correctness_ok,
        "profile_ok": profile_ok,
        "success": success,
        "latency_ms": latency_val,
        "execution_time_ms": latency_val,
        "bandwidth_gbps": bw,
        "error": err,
    }

    profiling = {
        "success": success,
        "bandwidth_gbps": bw,
        "execution_time_ms": latency_val,
        "error": err,
    }

    root = args.workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (root / "profiling_results.json").write_text(
        json.dumps(profiling, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
