"""HTTP 客户端（M2）。

三层，各自只解决一件事：

1. ``HttpResponse``：把响应压成统一形态（正文、状态码、是否来自缓存、重试次数）。
2. ``HttpxHttpClient``：真正发请求。负责礼貌请求头、超时、以及 429/5xx 的退避重试。
3. ``CachingHttpClient``：包在任一客户端外面，加缓存与"只读缓存"模式。

**适配器只依赖 ``HttpClient`` 协议**，所以测试可以注入一个假客户端，
整个 M2 就能在完全离线的条件下被验证——这也正是课程要求的
"测试不依赖公网"。

关于重试的取舍：重试结束后**不抛异常而是把响应原样返回**，
由适配器统一翻译成错误码。这样"重试 3 次后仍 429"和"第 1 次就 400"
在最终错误码上能被正确区分，而不是都退化成一句"请求失败"。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, Protocol

import httpx
from pydantic import Field, StringConstraints

from paper_evidence_qa.ids import sha256_hex
from paper_evidence_qa.schemas import BaseSchema
from paper_evidence_qa.sources.cache import CachePolicy, ResponseCache
from paper_evidence_qa.sources.errors import OnlineDisabledError, ResponseShapeError
from paper_evidence_qa.sources.ratelimit import DEFAULT_SLEEP

DEFAULT_BACKOFF_BASE_SECONDS = 1.0
DEFAULT_MAX_BACKOFF_SECONDS = 30.0
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class HttpResponse(BaseSchema):
    """统一的响应形态。"""

    url: str
    status_code: int
    content_type: Annotated[str, StringConstraints(max_length=128)] = ""
    text: str = ""
    from_cache: bool = False
    retry_count: Annotated[int, Field(ge=0)] = 0

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


class HttpClient(Protocol):
    """适配器依赖的客户端协议。实现它即可替换网络层。"""

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:  # pragma: no cover - 协议声明
        ...


def build_cache_key(url: str, params: dict[str, Any] | None = None) -> str:
    """由 URL 与查询参数派生缓存键。

    参数按键名排序，保证"参数顺序不同但语义相同"的请求命中同一条缓存。
    """
    if not params:
        return sha256_hex(f"GET {url}")
    normalized = "&".join(
        f"{key}={params[key]}" for key in sorted(params) if params[key] is not None
    )
    return sha256_hex(f"GET {url}?{normalized}")


def parse_json(response: HttpResponse, *, where: str) -> Any:
    """解析 JSON 正文。失败时抛 ``ResponseShapeError``，不返回 ``None`` 蒙混过关。

    上游返回 HTML 错误页、被网关拦截、或直接改了格式，都会在这里被拦住，
    而不是等到下游某个 KeyError 才暴露。
    """
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as exc:
        preview = response.text[:200].replace("\n", " ")
        raise ResponseShapeError(
            f"响应不是合法 JSON：{exc.msg}",
            where=where,
            url=response.url,
            details={"preview": preview, "status_code": response.status_code},
        ) from exc


class HttpxHttpClient:
    """基于 httpx 的同步客户端。

    ``transport`` 可注入，便于在需要时把请求导向本地桩服务；
    正常使用时不传，走真实网络。
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        sleeper: Callable[[float], None] | None = None,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries 不得为负数")
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._sleep = sleeper or DEFAULT_SLEEP
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            transport=transport,
        )

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        attempt = 0
        last_error: Exception | None = None
        while True:
            try:
                response = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise ResponseShapeError(
                        f"网络请求失败：{exc}",
                        where="transport",
                        url=url,
                        details={"attempts": attempt + 1},
                    ) from exc
                self._sleep(self._backoff_delay(attempt, None))
                attempt += 1
                continue

            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt < self.max_retries
            ):
                self._sleep(self._backoff_delay(attempt, response))
                attempt += 1
                continue

            return HttpResponse(
                url=str(response.url),
                status_code=response.status_code,
                content_type=response.headers.get("content-type", "")[:128],
                text=response.text,
                from_cache=False,
                retry_count=attempt,
            )

    def _backoff_delay(
        self, attempt: int, response: httpx.Response | None
    ) -> float:
        """优先听上游的 Retry-After，否则指数退避。"""
        if response is not None:
            header = response.headers.get("retry-after")
            if header:
                try:
                    return min(float(header), self.max_backoff_seconds)
                except ValueError:
                    pass
        return min(
            self.backoff_base_seconds * (2**attempt), self.max_backoff_seconds
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpxHttpClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class CachingHttpClient:
    """给任意客户端加缓存与"只读缓存"模式。"""

    def __init__(
        self,
        inner: HttpClient,
        cache: ResponseCache,
        policy: CachePolicy | None = None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._policy = policy or CachePolicy()
        self._now = now or (lambda: datetime.now(timezone.utc))

    @property
    def policy(self) -> CachePolicy:
        return self._policy

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        if not self._policy.enabled and not self._policy.replay_only:
            return self._inner.get(url, params=params, headers=headers)

        cache_key = build_cache_key(url, params)
        cached = self._cache.get(cache_key)
        if cached is not None:
            usable = self._policy.replay_only or cached.is_fresh(
                self._policy.ttl_seconds, self._now()
            )
            if usable:
                return HttpResponse(
                    url=cached.url,
                    status_code=cached.status_code,
                    content_type=cached.content_type,
                    text=cached.body,
                    from_cache=True,
                )

        if self._policy.replay_only:
            raise OnlineDisabledError(
                "离线重放模式下无可用缓存条目，拒绝发起网络请求",
                details={"url": url, "cache_key": cache_key},
            )

        response = self._inner.get(url, params=params, headers=headers)
        if response.ok:
            self._cache.put(
                cache_key=cache_key,
                url=response.url,
                status_code=response.status_code,
                content_type=response.content_type,
                body=response.text,
                fetched_at=self._now(),
            )
        return response


__all__ = [
    "DEFAULT_BACKOFF_BASE_SECONDS",
    "DEFAULT_MAX_BACKOFF_SECONDS",
    "RETRYABLE_STATUS_CODES",
    "CachingHttpClient",
    "HttpClient",
    "HttpResponse",
    "HttpxHttpClient",
    "build_cache_key",
    "parse_json",
]
