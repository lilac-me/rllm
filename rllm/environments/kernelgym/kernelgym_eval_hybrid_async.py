"""Async KernelGYM evaluate client (submit + poll + results), aligned with rllm-071 ``_HybridHttpWorker``."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_submit_semaphores: dict[str, asyncio.Semaphore] = {}


def _get_submit_semaphore(key: str, rate_limit: int) -> asyncio.Semaphore:
    if key not in _submit_semaphores:
        n = max(1, int(rate_limit))
        _submit_semaphores[key] = asyncio.Semaphore(n)
    return _submit_semaphores[key]


def _backoff(attempt: int, base: int = 2, cap: int = 30) -> float:
    return float(min(base**attempt, cap))


async def async_submit_and_poll(
    server_url: str,
    task_data: Dict[str, Any],
    *,
    default_timeout: int,
    client_timeout: int,
    max_retries: Optional[int],
    acquire_timeout: int,
    rate_limit: int,
) -> Dict[str, Any]:
    """Mirror ``_HybridHttpWorker.submit_and_poll`` from rllm-071 using ``httpx.AsyncClient``."""
    base = server_url.rstrip("/")
    sem_key = f"{base}@{max(1, rate_limit)}"
    sem = _get_submit_semaphore(sem_key, rate_limit)
    to = httpx.Timeout(
        connect=10.0, read=float(default_timeout), write=10.0, pool=5.0
    )
    limits = httpx.Limits(
        max_keepalive_connections=64, max_connections=128, keepalive_expiry=30.0
    )
    start_ts = time.time()

    async with httpx.AsyncClient(
        base_url=base, timeout=to, limits=limits, headers={"Content-Type": "application/json"}
    ) as client:
        attempt = 0
        unlimited = max_retries is None or max_retries == -1
        submit_ok = False

        while unlimited or attempt < (max_retries or 0):
            try:
                await asyncio.wait_for(sem.acquire(), timeout=float(acquire_timeout))
            except asyncio.TimeoutError:
                logger.warning(
                    "[AsyncHybrid] acquire timeout (rate limiter) url=%s task_id=%s",
                    base,
                    task_data.get("task_id", ""),
                )
                return {"status": "failed", "error_message": "rate limiter acquire timeout"}

            if attempt == 0:
                logger.debug(
                    "[AsyncHybrid] POST /evaluate task_id=%s url=%s",
                    task_data.get("task_id", ""),
                    base,
                )

            try:
                resp = await client.post("/evaluate", json=task_data)
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                try:
                    sem.release()
                except Exception:  # noqa: BLE001
                    pass
                if unlimited or attempt < (max_retries or 0) - 1:
                    await asyncio.sleep(_backoff(attempt))
                    attempt += 1
                    continue
                return {"status": "failed", "error_message": str(e)}
            except Exception as e:  # noqa: BLE001
                try:
                    sem.release()
                except Exception:  # noqa: BLE001
                    pass
                return {"status": "failed", "error_message": str(e)}

            try:
                sem.release()
            except Exception:  # noqa: BLE001
                pass

            try:
                logger.debug(
                    "[AsyncHybrid] POST /evaluate resp=%s task_id=%s",
                    resp.status_code,
                    task_data.get("task_id", ""),
                )
            except Exception:  # noqa: BLE001
                pass

            if resp.status_code == 200:
                try:
                    j = json.loads(resp.content.decode("utf-8") or "{}")
                    logger.debug(
                        "[AsyncHybrid] POST body errcode: %s",
                        j.get("error_code", None),
                    )
                except Exception:  # noqa: BLE001
                    pass
                submit_ok = True
                break

            if resp.status_code in (429, 503):
                await asyncio.sleep(
                    _backoff(attempt, base=2 if resp.status_code == 429 else 5)
                )
                attempt += 1
                continue

            try:
                resp.raise_for_status()
            except Exception as e:  # noqa: BLE001
                return {"status": "failed", "error_message": str(e)}

        if not submit_ok:
            return {"status": "failed", "error_message": "POST /evaluate did not return HTTP 200"}

        task_id = str(task_data.get("task_id", ""))
        last_status: str | None = None
        _poll_failures = 0
        while time.time() - start_ts < float(client_timeout):
            try:
                s = await client.get(f"/status/{task_id}")
                _poll_failures = 0  # reset on any successful HTTP exchange
                if s.status_code == 200:
                    data = s.json()
                    status = str(data.get("status", "unknown"))
                    if status != last_status:
                        last_status = status
                        logger.debug(
                            "[AsyncHybrid] STATUS task_id=%s -> %s", task_id, status
                        )
                    if status in ("completed", "failed", "timeout", "cancelled"):
                        r = await client.get(f"/results/{task_id}")
                        if r.status_code == 200:
                            result = r.json()
                            result["status"] = status
                            logger.debug(
                                "[AsyncHybrid] PULL result task_id=%s status=%s",
                                task_id,
                                status,
                            )
                            return result
                        return {
                            "status": status,
                            "error_message": f"Failed to fetch results: HTTP {r.status_code}",
                        }
            except Exception as _poll_exc:  # noqa: BLE001
                _poll_failures += 1
                logger.warning(
                    "[AsyncHybrid] poll error #%d task_id=%s: %s",
                    _poll_failures,
                    task_id,
                    _poll_exc,
                )
            # Exponential backoff: 1s → 2s → 4s → 8s → 16s → capped at 30s
            _sleep_s = min(1.0 * (2 ** min(_poll_failures, 4)), 30.0)
            await asyncio.sleep(_sleep_s)

        return {
            "status": "timeout",
            "error_message": f"Task timeout after {client_timeout}s (client-side)",
        }
