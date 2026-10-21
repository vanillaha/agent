"""M2 测试用的假 HTTP 客户端与样本加载器。

**这些测试完全离线**：不访问任何真实接口，不使用真实邮箱与密钥。
所有响应都来自 `fixtures/sources/` 下的脱敏样本。

假客户端刻意做得"笨"——它不做任何解析和容错，只负责：
按地址返回预置响应、记录每一次调用。这样测试里"到底发了几个请求、
请求参数长什么样"都是可断言的，而不是靠肉眼猜。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from paper_evidence_qa.sources import HttpResponse
from paper_evidence_qa.sources.arxiv import API_URL as ARXIV_URL
from paper_evidence_qa.sources.crossref import API_URL as CROSSREF_URL
from paper_evidence_qa.sources.openalex import API_URL as OPENALEX_URL
from paper_evidence_qa.sources.semantic_scholar import API_URL as S2_URL

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sources"

ALL_SOURCE_URLS: tuple[str, ...] = (OPENALEX_URL, CROSSREF_URL, ARXIV_URL, S2_URL)

FIXTURE_FILES: Mapping[str, str] = {
    OPENALEX_URL: "openalex_works.json",
    CROSSREF_URL: "crossref_works.json",
    ARXIV_URL: "arxiv_feed.xml",
    S2_URL: "semantic_scholar_search.json",
}

TEST_MAILTO = "fixture@example.org"
"""测试用联系邮箱。不是真实邮箱，也不该出现在生产配置里。"""

TEST_API_KEY = "fixture-api-key-not-real"
"""测试用密钥。明确写成假值，避免被误当成真实凭据。"""


def load_fixture_text(name: str) -> str:
    """读取脱敏样本原文。"""
    path = FIXTURES_DIR / name
    if not path.exists():  # pragma: no cover - 样本缺失属于测试环境问题
        raise FileNotFoundError(f"缺少测试样本：{path}")
    return path.read_text(encoding="utf-8")


def load_fixture_json(name: str) -> dict[str, Any]:
    """读取 JSON 样本并解析成对象（供测试断言字段用）。"""
    return json.loads(load_fixture_text(name))


@dataclass
class FakeCall:
    """一次被记录的调用。"""

    url: str
    params: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)


def json_response(
    url: str, payload: Any, *, status_code: int = 200
) -> HttpResponse:
    """构造一个 JSON 响应。"""
    return HttpResponse(
        url=url,
        status_code=status_code,
        content_type="application/json",
        text=json.dumps(payload, ensure_ascii=False),
    )


def text_response(
    url: str, body: str, *, status_code: int = 200, content_type: str = "text/plain"
) -> HttpResponse:
    return HttpResponse(
        url=url,
        status_code=status_code,
        content_type=content_type,
        text=body,
    )


class FakeHttpClient:
    """按地址返回预置响应的假客户端。

    ``routes`` 的值可以是：

    - ``str``：当作 200 的正文返回
    - ``HttpResponse``：原样返回（可用来伪造 429 / 500）
    - ``Exception``：抛出（可用来伪造网络中断）
    - ``list``：按顺序消费，用于"先失败后成功"的重试用例
    """

    def __init__(
        self,
        routes: Mapping[str, Any] | None = None,
        *,
        default: Any = None,
    ) -> None:
        self._routes: dict[str, Any] = dict(routes or {})
        self._default = default
        self.calls: list[FakeCall] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(
            FakeCall(url=url, params=dict(params or {}), headers=dict(headers or {}))
        )
        route = self._routes.get(url, self._default)
        if route is None:
            raise AssertionError(
                f"测试未为地址预置响应：{url}（可用地址：{sorted(self._routes)}）"
            )
        if isinstance(route, list):
            if not route:
                raise AssertionError(f"地址 {url} 的响应队列已耗尽")
            route = route.pop(0) if len(route) > 1 else route[0]
        if isinstance(route, BaseException):
            raise route
        if isinstance(route, str):
            return text_response(url, route)
        return route

    # ------------------------------------------------------------ 断言辅助

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def urls(self) -> list[str]:
        return [call.url for call in self.calls]

    def calls_for(self, url: str) -> Sequence[FakeCall]:
        return [call for call in self.calls if call.url == url]

    def last_params_for(self, url: str) -> dict[str, Any]:
        matched = self.calls_for(url)
        if not matched:
            raise AssertionError(f"没有发往 {url} 的调用")
        return matched[-1].params

    def last_headers_for(self, url: str) -> dict[str, str]:
        matched = self.calls_for(url)
        if not matched:
            raise AssertionError(f"没有发往 {url} 的调用")
        return matched[-1].headers


def default_routes() -> dict[str, Any]:
    """四个源的默认样本响应。"""
    return {
        url: load_fixture_text(name) for url, name in FIXTURE_FILES.items()
    }


def empty_routes() -> dict[str, Any]:
    """四个源都返回"检索成功但没有结果"的空响应。"""
    return {
        OPENALEX_URL: json.dumps({"meta": {"count": 0}, "results": []}),
        CROSSREF_URL: json.dumps({"message": {"items": []}}),
        ARXIV_URL: (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        ),
        S2_URL: json.dumps({"total": 0, "data": []}),
    }


def failing_routes(
    *, error: BaseException | None = None, status_code: int = 500
) -> dict[str, Any]:
    """四个源都失败的响应。用于验证"全挂也不崩"。"""

    def make(url: str) -> Any:
        if error is not None:
            return error
        return text_response(
            url, "upstream exploded", status_code=status_code
        )

    return {url: make(url) for url in ALL_SOURCE_URLS}


__all__ = [
    "ALL_SOURCE_URLS",
    "ARXIV_URL",
    "CROSSREF_URL",
    "FIXTURE_FILES",
    "FIXTURES_DIR",
    "OPENALEX_URL",
    "S2_URL",
    "TEST_API_KEY",
    "TEST_MAILTO",
    "FakeCall",
    "FakeHttpClient",
    "default_routes",
    "empty_routes",
    "failing_routes",
    "json_response",
    "load_fixture_json",
    "load_fixture_text",
    "text_response",
]
