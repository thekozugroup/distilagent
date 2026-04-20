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

import asyncio
import time

import pytest

from distilabel.steps.tasks.autoreason import rate_limit as rl
from distilabel.steps.tasks.autoreason.rate_limit import (
    AsyncTokenBucket,
    get_limiter,
    rate_limited_call,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _clear_registry():
    rl._REGISTRY.clear()
    yield
    rl._REGISTRY.clear()


# -----------------------------
# 1. Basic RPM enforcement
# -----------------------------
async def test_rpm_basic_enforcement():
    # rpm=60 -> 1 token/sec refill, capacity 60. 60 rapid acquires succeed
    # immediately; the 61st must wait ~1s.
    bucket = AsyncTokenBucket(rpm=60)
    t0 = time.perf_counter()
    for _ in range(60):
        await bucket.acquire(1)
    fast_elapsed = time.perf_counter() - t0
    assert fast_elapsed < 0.5, f"60 immediate acquires took {fast_elapsed:.3f}s"

    t1 = time.perf_counter()
    await bucket.acquire(1)
    wait_elapsed = time.perf_counter() - t1
    # Refill rate = 1/sec, so we need ~1s for a single token.
    assert 0.5 <= wait_elapsed <= 1.8, (
        f"61st acquire waited {wait_elapsed:.3f}s (expected ~1s)"
    )


# -----------------------------
# 2. RPM refill over time
# -----------------------------
async def test_rpm_refill_over_time():
    bucket = AsyncTokenBucket(rpm=60)  # 1 token/sec
    # Exhaust the bucket.
    for _ in range(60):
        await bucket.acquire(1)
    # Wait ~1s and one more acquire should succeed quickly.
    await asyncio.sleep(1.1)
    t0 = time.perf_counter()
    await bucket.acquire(1)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.3, f"post-refill acquire took {elapsed:.3f}s"


# -----------------------------
# 3. RPD enforcement
# -----------------------------
async def test_rpd_enforcement(monkeypatch):
    # Shrink the "day" window so the test runs fast.
    monkeypatch.setattr(rl, "_DAY_SECONDS", 0.3)
    bucket = AsyncTokenBucket(rpm=1000, rpd=3)

    # First 3 acquires are immediate.
    t0 = time.perf_counter()
    for _ in range(3):
        await bucket.acquire(1)
    fast = time.perf_counter() - t0
    assert fast < 0.1

    # State shows rpd exhausted.
    st = bucket.state
    assert st["rpd_remaining"] == 0
    assert st["next_rpd_refill_s"] is not None and st["next_rpd_refill_s"] > 0

    # 4th acquire blocks until the shortened day window rolls over.
    t1 = time.perf_counter()
    await bucket.acquire(1)
    waited = time.perf_counter() - t1
    assert waited >= 0.1, f"4th acquire waited {waited:.3f}s (expected ~0.3s)"
    assert waited <= 1.5


# -----------------------------
# 4. get_limiter singleton behaviour
# -----------------------------
async def test_get_limiter_singleton_and_update():
    a = get_limiter("model-x", rpm=60, rpd=100)
    b = get_limiter("model-x", rpm=60, rpd=100)
    assert a is b

    c = get_limiter("model-y", rpm=60)
    assert c is not a

    # Same name, new limits -> updates existing.
    d = get_limiter("model-x", rpm=120, rpd=200)
    assert d is a
    assert a.rpm == 120
    assert a.rpd == 200


# -----------------------------
# 5. rate_limited_call — no error
# -----------------------------
async def test_rate_limited_call_success_no_retries():
    bucket = AsyncTokenBucket(rpm=600)
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        return "ok"

    result = await rate_limited_call(bucket, fn)
    assert result == "ok"
    assert calls["n"] == 1


# -----------------------------
# 6. rate_limited_call — retries on 429 then succeeds
# -----------------------------
async def test_rate_limited_call_retries_on_429_then_succeeds():
    bucket = AsyncTokenBucket(rpm=600)
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise Exception("429 Too Many Requests")
        return "success"

    result = await rate_limited_call(
        bucket, fn, max_retries=5, base_backoff=0.01, max_backoff=0.05
    )
    assert result == "success"
    assert calls["n"] == 3  # 2 failures + 1 success


# -----------------------------
# 7. rate_limited_call — gives up after max_retries
# -----------------------------
async def test_rate_limited_call_gives_up_after_max_retries():
    bucket = AsyncTokenBucket(rpm=600)
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise Exception("429 rate limit hit")

    with pytest.raises(Exception) as excinfo:
        await rate_limited_call(
            bucket, fn, max_retries=3, base_backoff=0.01, max_backoff=0.02
        )
    assert "429" in str(excinfo.value)
    assert calls["n"] == 3


# -----------------------------
# 8. rate_limited_call — non-rate-limit error propagates immediately
# -----------------------------
async def test_rate_limited_call_non_rate_limit_error_propagates():
    bucket = AsyncTokenBucket(rpm=600)
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await rate_limited_call(
            bucket, fn, max_retries=5, base_backoff=0.01, max_backoff=0.02
        )
    assert calls["n"] == 1


# -----------------------------
# Bonus: status_code=429 attr detection
# -----------------------------
async def test_rate_limited_call_detects_status_code_attr():
    bucket = AsyncTokenBucket(rpm=600)
    calls = {"n": 0}

    class FakeHTTPError(Exception):
        def __init__(self):
            super().__init__("oops")
            self.status_code = 429

    async def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise FakeHTTPError()
        return 42

    result = await rate_limited_call(
        bucket, fn, max_retries=3, base_backoff=0.01, max_backoff=0.02
    )
    assert result == 42
    assert calls["n"] == 2
