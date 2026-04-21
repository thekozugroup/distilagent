# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-model async rate limiter for AutoReason.

Provides a dual token-bucket (RPM + RPD) implementation suitable for wrapping
async LLM calls against providers that enforce both per-minute and per-day
request limits (e.g. OpenRouter free-tier models).
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

# Exposed as a module-level constant so tests can monkeypatch it to a small
# value (e.g. 0.1) to exercise RPD refill behaviour without waiting 24 hours.
_DAY_SECONDS: float = 86400.0

_MINUTE_SECONDS: float = 60.0


def _loop_time() -> float:
    """Return a monotonic clock tied to the running event loop."""
    return asyncio.get_event_loop().time()


class AsyncTokenBucket:
    """Dual token bucket enforcing both RPM (minute) and RPD (day) limits."""

    def __init__(self, rpm: int, rpd: Optional[int] = None, name: str = "default"):
        self.name = name
        self._lock = asyncio.Lock()
        self._set_limits(rpm, rpd)

    def _set_limits(self, rpm: int, rpd: Optional[int]) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self.rpm = rpm
        self.rpd = rpd
        now = _loop_time()
        # RPM bucket: continuous refill at rpm/60 tokens/sec.
        self._rpm_tokens: float = float(rpm)
        self._rpm_last_refill: float = now
        # RPD bucket: refills fully every _DAY_SECONDS.
        self._rpd_tokens: Optional[float] = float(rpd) if rpd is not None else None
        self._rpd_window_start: float = now

    def update_limits(self, rpm: int, rpd: Optional[int]) -> None:
        """Update this bucket's limits in-place. Clamps outstanding tokens."""
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self.rpm = rpm
        self.rpd = rpd
        # Clamp current RPM tokens to new capacity.
        self._rpm_tokens = min(self._rpm_tokens, float(rpm))
        if rpd is None:
            self._rpd_tokens = None
        else:
            if self._rpd_tokens is None:
                self._rpd_tokens = float(rpd)
                self._rpd_window_start = _loop_time()
            else:
                self._rpd_tokens = min(self._rpd_tokens, float(rpd))

    def _refill_rpm(self, now: float) -> None:
        elapsed = now - self._rpm_last_refill
        if elapsed > 0:
            rate = self.rpm / _MINUTE_SECONDS
            self._rpm_tokens = min(float(self.rpm), self._rpm_tokens + elapsed * rate)
            self._rpm_last_refill = now

    def _refill_rpd(self, now: float) -> None:
        if self._rpd_tokens is None or self.rpd is None:
            return
        if now - self._rpd_window_start >= _DAY_SECONDS:
            # Advance by whole windows to keep alignment stable.
            elapsed = now - self._rpd_window_start
            windows = int(elapsed // _DAY_SECONDS)
            self._rpd_window_start += windows * _DAY_SECONDS
            self._rpd_tokens = float(self.rpd)

    async def acquire(self, n: int = 1) -> None:
        """Block until n tokens available in both buckets, then consume."""
        if n <= 0:
            return
        async with self._lock:
            while True:
                now = _loop_time()
                self._refill_rpm(now)
                self._refill_rpd(now)

                rpm_ok = self._rpm_tokens >= n
                rpd_ok = self._rpd_tokens is None or self._rpd_tokens >= n

                if rpm_ok and rpd_ok:
                    self._rpm_tokens -= n
                    if self._rpd_tokens is not None:
                        self._rpd_tokens -= n
                    return

                # Compute the shortest wait needed to satisfy whichever
                # bucket is currently short.
                waits = []
                if not rpm_ok:
                    deficit = n - self._rpm_tokens
                    rate = self.rpm / _MINUTE_SECONDS
                    waits.append(deficit / rate)
                if not rpd_ok:
                    assert self._rpd_tokens is not None and self.rpd is not None
                    waits.append(
                        max(0.0, _DAY_SECONDS - (now - self._rpd_window_start))
                    )
                sleep_for = max(0.0, min(waits)) if waits else 0.0
                # Guard against zero-sleep spin loops from floating-point dust.
                if sleep_for <= 0:
                    sleep_for = 0.001
                await asyncio.sleep(sleep_for)

    @property
    def state(self) -> dict:
        """Return a snapshot of bucket state for debugging."""
        now = _loop_time()
        # Pure read — don't mutate.
        rpm_elapsed = now - self._rpm_last_refill
        rpm_rate = self.rpm / _MINUTE_SECONDS
        rpm_tokens = min(float(self.rpm), self._rpm_tokens + rpm_elapsed * rpm_rate)
        if rpm_tokens >= self.rpm:
            next_rpm_refill_s = 0.0
        else:
            next_rpm_refill_s = max(0.0, (1.0 - (rpm_tokens - int(rpm_tokens))) / rpm_rate)

        rpd_remaining: Optional[float]
        next_rpd_refill_s: Optional[float]
        if self._rpd_tokens is None or self.rpd is None:
            rpd_remaining = None
            next_rpd_refill_s = None
        else:
            if now - self._rpd_window_start >= _DAY_SECONDS:
                rpd_remaining = float(self.rpd)
                next_rpd_refill_s = 0.0
            else:
                rpd_remaining = float(self._rpd_tokens)
                next_rpd_refill_s = max(
                    0.0, _DAY_SECONDS - (now - self._rpd_window_start)
                )

        return {
            "rpm_remaining": rpm_tokens,
            "rpd_remaining": rpd_remaining,
            "next_rpm_refill_s": next_rpm_refill_s,
            "next_rpd_refill_s": next_rpd_refill_s,
        }


# Process-global registry keyed by name.
_REGISTRY: Dict[str, AsyncTokenBucket] = {}


def get_limiter(
    name: str, rpm: int, rpd: Optional[int] = None
) -> AsyncTokenBucket:
    """Singleton-per-name registry.

    Returns the same bucket for the same name. If called twice with different
    rpm/rpd for the same name, UPDATES the existing bucket's limits rather
    than raising — helpful when config changes during a run.
    """
    existing = _REGISTRY.get(name)
    if existing is None:
        bucket = AsyncTokenBucket(rpm=rpm, rpd=rpd, name=name)
        _REGISTRY[name] = bucket
        return bucket
    if existing.rpm != rpm or existing.rpd != rpd:
        existing.update_limits(rpm, rpd)
    return existing


def _default_is_rate_limit_error(exc: BaseException) -> bool:
    # Status-code style: openai / httpx / openrouter client exceptions.
    for attr in ("status_code", "status"):
        val = getattr(exc, attr, None)
        if val in (429, 503):
            return True
    msg = str(exc).lower()
    if "429" in msg or "503" in msg:
        return True
    if "rate" in msg:
        return True
    if "too many requests" in msg:
        return True
    # Local server backpressure (MLX Metal working-set guard, vLLM queue full, etc.)
    if "working-set" in msg or "working set" in msg:
        return True
    if "service unavailable" in msg:
        return True
    if "overloaded" in msg or "server overloaded" in msg:
        return True
    return False


async def rate_limited_call(
    limiter: AsyncTokenBucket,
    fn: Callable[..., Awaitable[Any]],
    *args: Any,
    max_retries: int = 5,
    base_backoff: float = 1.0,
    max_backoff: float = 60.0,
    is_rate_limit_error: Optional[Callable[[BaseException], bool]] = None,
    **kwargs: Any,
) -> Any:
    """Acquire tokens, call fn, retry with exponential backoff on 429s.

    - Default `is_rate_limit_error`: matches "429" / "rate" / "too many requests"
      in the message (case-insensitive), or a `status_code`/`status` attr == 429.
    - On a detected rate-limit error: sleep `min(base_backoff * 2**attempt,
      max_backoff)` and retry (attempt counts up from 0).
    - On any other error: re-raise immediately.
    - After `max_retries` rate-limit errors, re-raise the last exception.
    """
    detector = is_rate_limit_error or _default_is_rate_limit_error
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries):
        await limiter.acquire(1)
        try:
            return await fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — we re-raise non-matching errors
            if not detector(exc):
                raise
            last_exc = exc
            backoff = min(base_backoff * (2 ** attempt), max_backoff)
            await asyncio.sleep(backoff)
    assert last_exc is not None
    raise last_exc
