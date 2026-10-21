"""M2 基础设施测试：限流、缓存、HTTP 包装、标识与错误映射。

这些部件是四个 adapter 的共同地基，出问题会一次性影响全部来源，
所以单独测、测细一点。

场景划分：正常 / 边界 / 失败 / 安全。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_evidence_qa.ids import deterministic_id, random_id, sha256_hex
from paper_evidence_qa.schemas import ErrorCode
from paper_evidence_qa.sources.cache import CachePolicy, ResponseCache
from paper_evidence_qa.sources.errors import (
    OnlineDisabledError,
    RateLimitedError,
    ResponseShapeError,
    UpstreamError,
)
from paper_evidence_qa.sources.http import (
    CachingHttpClient,
    HttpResponse,
    build_cache_key,
    parse_json,
)
from paper_evidence_qa.sources.ratelimit import RateLimiter

from fakes import FakeHttpClient, json_response, text_response

NOW = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
URL = "https://example.org/api"

# 缓存键在契约上是 64 位十六进制摘要（见 M1 的 Sha256），
# 所以测试也要用真实形态的键，不能用 "k1" 这种占位串。
KEY_A = sha256_hex("cache-key-a")
KEY_MISSING = sha256_hex("cache-key-missing")
KEY_OLD = sha256_hex("cache-key-old")
KEY_NEW = sha256_hex("cache-key-new")


class FakeClock:
    """可注入的假时钟：睡眠会推进时间，于是"等待"不需要真的发生。"""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# ---------------------------------------------------------------- 限流器


def test_rate_limiter_first_call_does_not_wait() -> None:
    clock = FakeClock()
    limiter = RateLimiter(3.0, clock=clock.monotonic, sleeper=clock.sleep)
    assert limiter.acquire() == 0.0
    assert clock.sleeps == []


def test_rate_limiter_enforces_minimum_interval() -> None:
    clock = FakeClock()
    limiter = RateLimiter(3.0, clock=clock.monotonic, sleeper=clock.sleep)
    limiter.acquire()
    waited = limiter.acquire()
    assert waited == pytest.approx(3.0)
    assert clock.sleeps == [pytest.approx(3.0)]
    assert limiter.call_count == 2


def test_rate_limiter_does_not_wait_when_enough_time_passed() -> None:
    clock = FakeClock()
    limiter = RateLimiter(1.0, clock=clock.monotonic, sleeper=clock.sleep)
    limiter.acquire()
    clock.now += 5.0
    assert limiter.acquire() == 0.0


def test_rate_limiter_with_zero_interval_never_waits() -> None:
    clock = FakeClock()
    limiter = RateLimiter(0.0, clock=clock.monotonic, sleeper=clock.sleep)
    for _ in range(5):
        assert limiter.acquire() == 0.0


def test_rate_limiter_rejects_negative_interval() -> None:
    with pytest.raises(ValueError):
        RateLimiter(-1.0)


def test_rate_limiter_reset_clears_history() -> None:
    clock = FakeClock()
    limiter = RateLimiter(2.0, clock=clock.monotonic, sleeper=clock.sleep)
    limiter.acquire()
    limiter.reset()
    assert limiter.call_count == 0
    assert limiter.acquire() == 0.0


def test_arxiv_style_interval_is_actually_enforced() -> None:
    """arXiv 要求 3 秒间隔，这里用真实适配器常量验证，防止有人把它调小。"""
    from paper_evidence_qa.sources.arxiv import MIN_INTERVAL_SECONDS

    assert MIN_INTERVAL_SECONDS >= 3.0
    clock = FakeClock()
    limiter = RateLimiter(
        MIN_INTERVAL_SECONDS, clock=clock.monotonic, sleeper=clock.sleep
    )
    limiter.acquire()
    assert limiter.acquire() >= 3.0


# ---------------------------------------------------------------- 缓存


def test_cache_roundtrip(tmp_path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.put(
        cache_key=KEY_A,
        url=URL,
        status_code=200,
        content_type="application/json",
        body='{"a": 1}',
        fetched_at=NOW,
    )
    entry = cache.get(KEY_A)
    assert entry is not None
    assert entry.body == '{"a": 1}'
    assert entry.url == URL
    assert cache.count() == 1
    cache.close()


def test_cache_returns_none_for_missing_key(tmp_path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    assert cache.get(KEY_MISSING) is None
    cache.close()


def test_cache_freshness_respects_ttl(tmp_path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.put(
        cache_key=KEY_A,
        url=URL,
        status_code=200,
        content_type="application/json",
        body="{}",
        fetched_at=NOW,
    )
    entry = cache.get(KEY_A)
    assert entry is not None
    assert entry.is_fresh(3600, NOW + timedelta(minutes=30)) is True
    assert entry.is_fresh(3600, NOW + timedelta(hours=2)) is False


def test_cache_ttl_zero_means_never_expires(tmp_path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.put(
        cache_key=KEY_A,
        url=URL,
        status_code=200,
        content_type="application/json",
        body="{}",
        fetched_at=NOW,
    )
    entry = cache.get(KEY_A)
    assert entry is not None
    assert entry.is_fresh(0, NOW + timedelta(days=3650)) is True


def test_cache_purge_expired_removes_only_old_entries(tmp_path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.put(
        cache_key=KEY_OLD,
        url=URL,
        status_code=200,
        content_type="application/json",
        body="{}",
        fetched_at=NOW - timedelta(days=3),
    )
    cache.put(
        cache_key=KEY_NEW,
        url=URL,
        status_code=200,
        content_type="application/json",
        body="{}",
        fetched_at=NOW,
    )
    removed = cache.purge_expired(86400, now=NOW)
    assert removed == 1
    assert cache.get(KEY_OLD) is None
    assert cache.get(KEY_NEW) is not None
    cache.close()


def test_cache_clear_empties_everything(tmp_path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.put(
        cache_key=KEY_A,
        url=URL,
        status_code=200,
        content_type="application/json",
        body="{}",
        fetched_at=NOW,
    )
    cache.clear()
    assert cache.count() == 0
    cache.close()


def test_cache_persists_across_reopen(tmp_path) -> None:
    path = tmp_path / "cache.sqlite"
    with ResponseCache(path) as cache:
        cache.put(
            cache_key=KEY_A,
            url=URL,
            status_code=200,
            content_type="application/json",
            body='{"persisted": true}',
            fetched_at=NOW,
        )
    with ResponseCache(path) as reopened:
        entry = reopened.get(KEY_A)
        assert entry is not None
        assert entry.body == '{"persisted": true}'


# ---------------------------------------------------------------- 缓存键


def test_cache_key_is_parameter_order_independent() -> None:
    first = build_cache_key(URL, {"b": "2", "a": "1"})
    second = build_cache_key(URL, {"a": "1", "b": "2"})
    assert first == second
    assert len(first) == 64


def test_cache_key_ignores_none_valued_parameters() -> None:
    assert build_cache_key(URL, {"a": "1"}) == build_cache_key(URL, {"a": "1", "b": None})


def test_cache_key_differs_for_different_requests() -> None:
    assert build_cache_key(URL, {"a": "1"}) != build_cache_key(URL, {"a": "2"})
    assert build_cache_key(URL) != build_cache_key("https://example.org/other")


# ---------------------------------------------------------------- 缓存客户端


def _caching(inner, cache, *, ttl=3600, replay_only=False, enabled=True):
    return CachingHttpClient(
        inner,
        cache,
        CachePolicy(ttl_seconds=ttl, enabled=enabled, replay_only=replay_only),
        now=lambda: NOW,
    )


def test_caching_client_stores_then_serves_from_cache(tmp_path) -> None:
    inner = FakeHttpClient({URL: '{"ok": true}'})
    cache = ResponseCache(tmp_path / "cache.sqlite")
    client = _caching(inner, cache)

    first = client.get(URL, params={"a": "1"})
    assert first.from_cache is False
    assert inner.call_count == 1

    second = client.get(URL, params={"a": "1"})
    assert second.from_cache is True
    assert second.text == '{"ok": true}'
    assert inner.call_count == 1, "命中缓存时不得再发请求"
    cache.close()


def test_caching_client_refetches_when_expired(tmp_path) -> None:
    inner = FakeHttpClient({URL: '{"n": 1}'})
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.put(
        cache_key=build_cache_key(URL, {"a": "1"}),
        url=URL,
        status_code=200,
        content_type="application/json",
        body='{"n": 0}',
        fetched_at=NOW - timedelta(days=10),
    )
    client = _caching(inner, cache, ttl=3600)
    response = client.get(URL, params={"a": "1"})
    assert response.from_cache is False
    assert response.text == '{"n": 1}'
    cache.close()


def test_caching_client_replay_only_serves_stale_cache(tmp_path) -> None:
    """离线重放时允许用过期条目——这是"不联网也能把链路跑通"的关键。"""
    inner = FakeHttpClient({})
    cache = ResponseCache(tmp_path / "cache.sqlite")
    cache.put(
        cache_key=build_cache_key(URL, None),
        url=URL,
        status_code=200,
        content_type="application/json",
        body='{"replayed": true}',
        fetched_at=NOW - timedelta(days=999),
    )
    client = _caching(inner, cache, ttl=60, replay_only=True)
    response = client.get(URL)
    assert response.from_cache is True
    assert response.text == '{"replayed": true}'
    assert inner.call_count == 0
    cache.close()


def test_caching_client_replay_only_refuses_network_on_miss(tmp_path) -> None:
    inner = FakeHttpClient({})
    cache = ResponseCache(tmp_path / "cache.sqlite")
    client = _caching(inner, cache, replay_only=True)
    with pytest.raises(OnlineDisabledError) as excinfo:
        client.get(URL)
    assert excinfo.value.to_error_info().code is ErrorCode.ONLINE_DISABLED
    assert inner.call_count == 0
    cache.close()


def test_caching_client_disabled_bypasses_cache_entirely(tmp_path) -> None:
    inner = FakeHttpClient({URL: "{}"})
    cache = ResponseCache(tmp_path / "cache.sqlite")
    client = _caching(inner, cache, enabled=False)
    client.get(URL)
    client.get(URL)
    assert inner.call_count == 2
    assert cache.count() == 0
    cache.close()


def test_caching_client_does_not_cache_error_responses(tmp_path) -> None:
    """偶发的 500 不能被固化，否则后续所有请求都会被这条脏缓存污染。"""
    inner = FakeHttpClient({URL: text_response(URL, "boom", status_code=500)})
    cache = ResponseCache(tmp_path / "cache.sqlite")
    client = _caching(inner, cache)
    client.get(URL)
    assert cache.count() == 0
    cache.close()


# ---------------------------------------------------------------- JSON 解析


def test_parse_json_returns_payload() -> None:
    assert parse_json(json_response(URL, {"a": 1}), where="test") == {"a": 1}


def test_parse_json_raises_structured_error_on_html_error_page() -> None:
    response = text_response(URL, "<html>502 Bad Gateway</html>", status_code=200)
    with pytest.raises(ResponseShapeError) as excinfo:
        parse_json(response, where="test.body")
    info = excinfo.value.to_error_info()
    assert info.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert info.details["where"] == "test.body"


# ---------------------------------------------------------------- 错误映射


def test_error_codes_map_to_m1_enum() -> None:
    assert (
        RateLimitedError("too many", attempts=3).to_error_info().code
        is ErrorCode.RATE_LIMITED
    )
    assert (
        UpstreamError("boom", status_code=503).to_error_info().code
        is ErrorCode.UPSTREAM_UNAVAILABLE
    )
    assert (
        OnlineDisabledError("offline").to_error_info().code
        is ErrorCode.ONLINE_DISABLED
    )


def test_rate_limited_error_keeps_retry_metadata() -> None:
    info = RateLimitedError(
        "limited", retry_after_seconds=12.0, attempts=2
    ).to_error_info()
    assert info.details["retry_after_seconds"] == 12.0
    assert info.details["attempts"] == 2


def test_response_shape_error_records_location() -> None:
    error = ResponseShapeError("bad shape", where="openalex.results", url=URL)
    info = error.to_error_info()
    assert info.details["where"] == "openalex.results"
    assert info.details["url"] == URL


def test_error_message_is_clipped_to_contract_limit() -> None:
    info = UpstreamError("x" * 5000).to_error_info()
    assert len(info.message) <= 1000


# ---------------------------------------------------------------- 标识


def test_random_id_is_prefixed_and_unique() -> None:
    values = {random_id("rcd") for _ in range(200)}
    assert len(values) == 200
    assert all(value.startswith("rcd_") for value in values)


def test_deterministic_id_is_stable_and_part_sensitive() -> None:
    assert deterministic_id("ppr", "a", "b") == deterministic_id("ppr", "a", "b")
    assert deterministic_id("ppr", "a", "b") != deterministic_id("ppr", "b", "a")
    assert deterministic_id("ppr", "ab") != deterministic_id("ppr", "a", "b")


def test_deterministic_id_rejects_empty_parts() -> None:
    with pytest.raises(ValueError):
        deterministic_id("ppr", "a", "   ")
    with pytest.raises(ValueError):
        deterministic_id("ppr")


def test_sha256_hex_is_full_length() -> None:
    assert len(sha256_hex("abc")) == 64


def test_http_response_ok_flag() -> None:
    assert HttpResponse(url=URL, status_code=200).ok is True
    assert HttpResponse(url=URL, status_code=429).ok is False
