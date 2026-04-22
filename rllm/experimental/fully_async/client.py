import asyncio
import itertools
from typing import Any

import httpx

from rllm.experimental.fully_async.protocol import OutputChunk, OutputWithVersion
from rllm.parser.tool_parser import ToolParser

# ── vLLM /v1/completions: these are top-level fields on CompletionRequest (not nested extra_body).
# ``extra_body`` is for the OpenAI *Python* client SDK, not raw JSON to the HTTP API.
_VLLM_COMPLETIONS_ROOT_KEYS = {"top_k", "repetition_penalty", "min_p", "min_tokens"}


class RolloutClient:
    def __init__(
        self,
        router_url: str,
        tokenizer=None,
        max_concurrency: int = 4096,
        max_tokens=32768,
        backend: str = "sglang",
        model_name: str | None = None,
    ):
        # Support multiple backend URLs (comma-separated) for vLLM multi-replica
        # round-robin load balancing.  Single-URL and SGLang cases are unaffected.
        raw_urls = [u.strip() for u in router_url.split(",") if u.strip()]
        self._urls: list[str] = raw_urls if raw_urls else [router_url]
        self._url_cycle = itertools.cycle(self._urls)
        # Kept for backward compat (abort/resume helpers, logging, etc.)
        self.router_url = self._urls[0]

        self.tokenizer = tokenizer
        self.parser = ToolParser.get_parser(tokenizer)
        self._max_concurrency = max_concurrency
        self.backend = backend
        # vLLM OpenAI-compatible endpoints require a served model identifier.
        # Use configured model path/name when available; keep "default" as fallback.
        self.model_name = model_name or "default"

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self._max_concurrency,
                max_keepalive_connections=min(self._max_concurrency, 1000),
            ),
            timeout=httpx.Timeout(None),
        )

        self.cur_version = 0
        self.max_tokens = max_tokens
        self.resume_event = asyncio.Event()
        self.resume_event.set()

    def _next_url(self) -> str:
        """Return the next backend URL in round-robin order.

        For a single-URL client this always returns the same URL.
        For multi-replica vLLM deployments this distributes requests evenly
        across all replicas without external infrastructure changes.
        """
        return next(self._url_cycle)

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    def set_version(self, version: int):
        self.cur_version = version

    async def _post(self, url: str, payload: dict):
        await self.resume_event.wait()
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def resume(self):
        self.resume_event.set()

    def pause(self):
        self.resume_event.clear()

    # ========== Low-Level API ==========

    async def generate(self, prompt_ids: list[int], sampling_params: dict) -> OutputWithVersion:
        """Generate with token IDs directly (low-level API)."""
        output = OutputWithVersion(prompt_ids=prompt_ids, output_chunks=[])

        while True:
            await self.resume_event.wait()
            output, sampling_params = await self._generate(output, sampling_params)
            if output.finish_reason == "abort":
                continue
            else:
                return output

    async def _generate(self, output: OutputWithVersion, sampling_params: dict):
        """Internal generate that dispatches to backend-specific implementation."""
        if self.backend == "sglang":
            return await self._generate_sglang(output, sampling_params)
        else:
            return await self._generate_vllm(output, sampling_params)

    # ── SGLang backend ──

    async def _generate_sglang(self, output: OutputWithVersion, sampling_params: dict):
        old_version = self.cur_version
        payload = {
            "input_ids": output.all_tokens(),
            "sampling_params": sampling_params,
            "return_logprob": True,
        }

        response = await self._post(self._next_url() + "/generate", payload)

        finish_reason_obj = response["meta_info"].get("finish_reason")
        output.finish_reason = finish_reason_obj["type"] if finish_reason_obj else "unknown"

        output_token_logprobs = response["meta_info"].get("output_token_logprobs", [])
        logprob_values = [float(log_prob) for log_prob, token_id, _ in output_token_logprobs]

        output_ids = [token_id for _, token_id, _ in output_token_logprobs]
        assert output_ids == response["output_ids"], f"output_ids mismatch, {output_ids} != {response['output_ids']}"

        chunk = OutputChunk(
            response_ids=response["output_ids"],
            response_logprobs=logprob_values,
            version=old_version if output.finish_reason == "abort" else self.cur_version,
        )
        output.append(chunk)

        max_tokens_val = sampling_params.get("max_new_tokens") or sampling_params.get("max_tokens")
        if max_tokens_val is None:
            return output, sampling_params

        sampling_params = sampling_params.copy()
        remaining = max_tokens_val - len(chunk.response_ids)
        if "max_new_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = remaining
        else:
            sampling_params["max_tokens"] = remaining

        return output, sampling_params

    # ── vLLM (OpenAI-compatible) backend ──

    async def _generate_vllm(self, output: OutputWithVersion, sampling_params: dict):
        old_version = self.cur_version

        sp = sampling_params.copy()

        # Translate max_new_tokens -> max_tokens (OpenAI naming)
        if "max_new_tokens" in sp:
            sp["max_tokens"] = sp.pop("max_new_tokens")

        # Construct OpenAI /v1/completions request
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": output.all_tokens(),
            "logprobs": 1,
            "echo": False,
        }
        # vLLM-specific sampling params (CompletionRequest top-level; do not wrap in extra_body)
        for key in list(sp.keys()):
            if key in _VLLM_COMPLETIONS_ROOT_KEYS:
                payload[key] = sp.pop(key)

        # Map standard sampling params
        for key in ("temperature", "top_p", "max_tokens", "stop", "n", "frequency_penalty", "presence_penalty"):
            if key in sp:
                payload[key] = sp[key]

        response = await self._post(self._next_url() + "/v1/completions", payload)

        choice = response["choices"][0]
        finish_reason_raw = choice.get("finish_reason", "unknown") or "unknown"
        output.finish_reason = finish_reason_raw

        # Parse token IDs and logprobs from vLLM response
        response_text = choice.get("text", "")
        logprobs_obj = choice.get("logprobs")

        if logprobs_obj and logprobs_obj.get("tokens") is not None:
            token_logprobs = logprobs_obj.get("token_logprobs", [])
            tokens = logprobs_obj.get("tokens", [])
            logprob_values = [float(lp) if lp is not None else 0.0 for lp in token_logprobs]
            # vLLM /v1/completions returns text tokens; decode prompt+response to get IDs
            if self.tokenizer is not None:
                response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
            else:
                response_ids = []
        else:
            logprob_values = []
            if self.tokenizer is not None:
                response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
            else:
                response_ids = []

        # Align logprobs length with response_ids (encode might tokenize differently)
        if len(logprob_values) != len(response_ids):
            min_len = min(len(logprob_values), len(response_ids))
            logprob_values = logprob_values[:min_len]
            response_ids = response_ids[:min_len]

        chunk = OutputChunk(
            response_ids=response_ids,
            response_logprobs=logprob_values,
            version=old_version if output.finish_reason == "abort" else self.cur_version,
        )
        output.append(chunk)

        # Adjust max_tokens for continuation
        orig_max = sampling_params.get("max_new_tokens") or sampling_params.get("max_tokens")
        if orig_max is None:
            return output, sampling_params

        sampling_params = sampling_params.copy()
        remaining = orig_max - len(chunk.response_ids)
        if "max_new_tokens" in sampling_params:
            sampling_params["max_new_tokens"] = remaining
        else:
            sampling_params["max_tokens"] = remaining

        return output, sampling_params

    # ========== High-Level Chat API ==========
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], OutputWithVersion]:
        """Generate chat completion and parse response into OpenAI message format."""
        from rllm.experimental.fully_async.message_utils import parse_response

        if self.tokenizer is None:
            raise ValueError("tokenizer required for chat_completion")

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)

        sampling_params = sampling_params or {}
        if sampling_params.get("max_new_tokens", None) is None:
            sampling_params["max_new_tokens"] = self.max_tokens - len(prompt_ids)

        output = await self.generate(prompt_ids, sampling_params)

        message = parse_response(self.tokenizer, self.parser, output.all_response_ids())
        return message, output

    async def close(self):
        await self.client.aclose()
