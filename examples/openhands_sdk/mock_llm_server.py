"""
Stage1 layered-test mock OpenAI-compatible LLM server.

A minimal FastAPI server that mimics the subset of the vllm-ascend
``/v1/chat/completions`` API we exercise in stage1, returning deterministic
``tool_calls`` and (optionally) ``logprobs`` / ``token_ids``. Used to isolate
Layer 2 (LiteLLM proxy + OpenHands container LLM endpoint) from Layer 3
(vllm-ascend + Qwen3.6 model load) when a smoke is failing somewhere in
between.

Usage::

    python3 -m examples.openhands_sdk.mock_llm_server --port 18000

Verify with curl::

    curl -X POST http://localhost:18000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "mock-qwen36",
            "messages": [{"role":"user","content":"List files in /tmp"}],
            "tools": [{"type":"function","function":{"name":"list_files",
                       "parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],
            "chat_template_kwargs": {"enable_thinking": false}
        }'

The server returns immediately (no model load, no NPU). Use it as the
backend for LiteLLM proxy during layered smoke to confirm the proxy +
metadata-slug routing path works without depending on vllm.
"""

from __future__ import annotations

import argparse
import time
import uuid
from typing import Any

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as e:
    raise SystemExit(
        f"mock_llm_server needs fastapi + pydantic: pip install fastapi uvicorn ({e})"
    ) from None


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    sampling_params: dict[str, Any] | None = None
    logprobs: int | bool | None = None
    max_tokens: int | None = None
    # Permissive: accept whatever else clients send and ignore it.
    model_config = {"extra": "allow"}


app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "server": "mock_llm_server", "ts": int(time.time())}


@app.get("/v1/models")
def list_models() -> dict:
    return {
        "object": "list",
        "data": [{"id": "mock-qwen36", "object": "model", "created": int(time.time()), "owned_by": "mock"}],
    }


def _build_tool_call_response(req: ChatRequest) -> dict:
    """Return a deterministic tool_call if request has tools, else plain text."""
    msg: dict[str, Any] = {"role": "assistant"}
    finish_reason = "stop"

    if req.tools:
        # Pick the first tool, call it with empty args (or with `path: /tmp` if param exists).
        tool = req.tools[0]
        fn = tool.get("function", {})
        name = fn.get("name", "unknown_tool")
        props = fn.get("parameters", {}).get("properties", {})
        # Synthesize a minimal valid argument dict from declared properties.
        args = {}
        for k, spec in props.items():
            t = spec.get("type")
            if t == "string":
                args[k] = "/tmp"
            elif t == "integer":
                args[k] = 0
            elif t == "number":
                args[k] = 0.0
            elif t == "boolean":
                args[k] = True
            else:
                args[k] = None
        import json as _json

        msg["content"] = None
        msg["tool_calls"] = [
            {
                "id": f"chatcmpl-tool-{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {"name": name, "arguments": _json.dumps(args)},
            }
        ]
        finish_reason = "tool_calls"
    else:
        msg["content"] = "This is a deterministic mock reply."

    # Honor enable_thinking=false (we don't emit reasoning ever anyway, but keep field).
    msg["reasoning"] = None

    choice: dict[str, Any] = {
        "index": 0,
        "message": msg,
        "finish_reason": finish_reason,
        "logprobs": None,
        "stop_reason": None,
    }

    # If client asked for token_ids/logprobs, return small deterministic stub.
    if req.logprobs:
        choice["token_ids"] = [1234, 5678, 9012]
        choice["logprobs"] = {
            "content": [
                {"token": "mock", "logprob": -0.1, "top_logprobs": []},
                {"token": "tool", "logprob": -0.2, "top_logprobs": []},
                {"token": "call", "logprob": -0.3, "top_logprobs": []},
            ]
        }

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [choice],
        "usage": {"prompt_tokens": 16, "completion_tokens": 8, "total_tokens": 24},
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> dict:
    return _build_tool_call_response(req)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=18000)
    p.add_argument("--log-level", default="warning")
    args = p.parse_args()

    try:
        import uvicorn
    except ImportError:
        raise SystemExit("mock_llm_server needs uvicorn: pip install uvicorn")

    print(f"[mock_llm_server] listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
