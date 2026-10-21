"""Crossref 适配器（M2）—— 骨干层，负责校验与合规信息。

**它的独特价值只有一个，但不可替代**：Crossref 是 DOI 的登记机构本身，
并且返回 ``license`` 字段（许可证链接）。M3 要判断"这篇能不能合法下载"，
唯一可靠的依据就是这个字段——Semantic Scholar 只给全文链接不给许可条款，
arXiv 只对自家论文有说明。

**它不做主检索**：Crossref 在 2025-12-01 调整了限流，列表查询（含检索）
在 polite pool 下只有 3 req/s，而单条 DOI 查询有 10 req/s。所以它的正确用法
是"按 DOI 定点查询"，而不是"拿研究问题去搜"。本适配器仍实现检索能力
（便于补齐元数据），但默认建议由编排层优先把它用作定点查询。

**它不判定开放获取**：Crossref 不提供可靠的 OA 判定，本适配器一律写 ``unknown``。
OA 判定交给 OpenAlex——避免两个源给出互相矛盾的 ``open``/``closed``，
那正是 M4 最难清理的一类脏数据。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from paper_evidence_qa.schemas import SourceName
from paper_evidence_qa.sources.base import (
    ParsedPage,
    SourceAdapter,
    SourceCandidate,
    SourceQuery,
    SourceRequest,
    build_candidate,
)
from paper_evidence_qa.sources.errors import ResponseShapeError
from paper_evidence_qa.sources.http import HttpResponse, parse_json
from paper_evidence_qa.sources.normalize import (
    classify_crossref_type,
    doc_types_to_crossref_filter,
    extract_arxiv_id_from_urls,
    resolve_oa_status,
)
from paper_evidence_qa.sources.payload import (
    as_mapping,
    as_sequence,
    as_str,
    first_mapping_in,
    first_string_in,
    join_names,
    strings_in,
)
from paper_evidence_qa.sources.query import doc_types_of

API_URL = "https://api.crossref.org/works"

SELECT_FIELDS = ",".join(
    (
        "DOI",
        "title",
        "author",
        "issued",
        "published",
        "container-title",
        "type",
        "license",
        "is-referenced-by-count",
        "abstract",
        "URL",
        "link",
    )
)


class CrossrefAdapter(SourceAdapter):
    """Crossref ``/works`` 适配器。"""

    name = SourceName.CROSSREF
    display_name = "Crossref"
    base_url = API_URL
    default_min_interval_seconds = 0.4
    """polite pool 列表查询限 3 req/s，取 2.5 req/s 留余量。"""

    default_page_size = 20
    max_page_size = 100
    requires_mailto = True
    supports_open_access_filter = False
    """Crossref 不支持按开放获取筛选，只能在本地过滤并记告警。"""

    def build_request(self, query: SourceQuery) -> SourceRequest:
        params: dict[str, Any] = {
            "query.bibliographic": query.query_text,
            "rows": self.limit_for(query),
            "select": SELECT_FIELDS,
            "sort": "relevance",
        }
        filter_parts = self._filter_parts(query)
        if filter_parts:
            params["filter"] = ",".join(filter_parts)
        if self.settings.mailto:
            params["mailto"] = self.settings.mailto
        return SourceRequest(
            url=self.base_url, params=params, headers=self.polite_headers()
        )

    def _filter_parts(self, query: SourceQuery) -> list[str]:
        parts: list[str] = []
        year_from = query.filters.get("year_from")
        year_to = query.filters.get("year_to")
        if year_from:
            parts.append(f"from-pub-date:{int(year_from)}-01-01")
        if year_to:
            parts.append(f"until-pub-date:{int(year_to)}-12-31")

        # Crossref 的 OR 语义靠"重复同名过滤器"表达，不是用竖线。
        for value in doc_types_to_crossref_filter(doc_types_of(query)):
            parts.append(f"type:{value}")
        return parts

    # ---------------------------------------------------------------- 解析

    def parse_response(
        self, response: HttpResponse, query: SourceQuery
    ) -> ParsedPage:
        payload = parse_json(response, where="crossref.root")
        if not isinstance(payload, Mapping):
            raise ResponseShapeError(
                "Crossref 响应顶层不是对象",
                where="crossref.root",
                url=response.url,
            )
        message = payload.get("message")
        if not isinstance(message, Mapping):
            raise ResponseShapeError(
                "Crossref 响应缺少 message 对象",
                where="crossref.message",
                url=response.url,
            )
        items = message.get("items")
        if not isinstance(items, list):
            raise ResponseShapeError(
                "Crossref 响应缺少 message.items 数组",
                where="crossref.message.items",
                url=response.url,
            )

        candidates: list[SourceCandidate] = []
        warnings = []
        for raw in items:
            if not isinstance(raw, Mapping):
                continue
            candidate, extra, _reason = self._build_one(raw)
            warnings.extend(extra)
            if candidate is not None:
                candidates.append(candidate)

        return ParsedPage(
            candidates=candidates, raw_count=len(items), warnings=warnings
        )

    def _build_one(self, raw: Mapping[str, Any]) -> tuple[Any, list[Any], str | None]:
        authors = [_author_name(item) for item in as_sequence(raw.get("author"))]

        # Crossref 不给可靠的开放获取判定，一律按未核查处理。
        oa = resolve_oa_status(is_open=None, oa_url=None, source="crossref")

        link_urls = strings_in(
            [as_mapping(item).get("URL") for item in as_sequence(raw.get("link"))]
        )

        return build_candidate(
            source_name=self.name,
            title=first_string_in(raw.get("title")),
            oa=oa,
            now=self.now(),
            authors=authors,
            year=as_mapping(raw.get("issued")).get("date-parts")
            or as_mapping(raw.get("published")).get("date-parts"),
            venue=first_string_in(raw.get("container-title")),
            doi=as_str(raw.get("DOI")),
            arxiv_raw=extract_arxiv_id_from_urls(
                [as_str(raw.get("URL")), *link_urls]
            ),
            abstract=as_str(raw.get("abstract")),
            doc_type=classify_crossref_type(as_str(raw.get("type"))),
            external_ids={"crossref": as_str(raw.get("DOI"))},
            citation_count=raw.get("is-referenced-by-count"),
            upstream_id=as_str(raw.get("DOI")),
            license_value=as_str(first_mapping_in(raw.get("license")).get("URL")),
        )


def _author_name(raw: Any) -> str | None:
    """Crossref 作者可能是 ``{given, family}`` 或直接 ``{name}``（机构作者）。"""
    if not isinstance(raw, Mapping):
        return None
    return as_str(raw.get("name")) or join_names(
        [as_str(raw.get("given")), as_str(raw.get("family"))]
    )


__all__ = ["API_URL", "SELECT_FIELDS", "CrossrefAdapter"]
