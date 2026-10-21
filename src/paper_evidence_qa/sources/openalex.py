"""OpenAlex 适配器（M2）—— 骨干层，主检索。

**为什么它是主检索**：一次请求就能填满 ``PaperRecord`` 的绝大部分字段
（DOI、标题、作者、年份、会议、摘要、引用数、开放获取状态与链接、外部 ID 映射），
而且免费、无需申请密钥。它的开放获取数据本身来自 Unpaywall，
等于顺带拿到了"这篇有没有合法全文"的预判。

**它不做什么**：OpenAlex 是聚合器，元数据权威性弱于一手登记方，
所以它只负责"找到候选"，不负责最终定性。许可条款要问 Crossref。

两个实现要点：

- OpenAlex 不直接给 arXiv ID，但它的 location 链接里常含 ``arxiv.org/abs/...``。
  这里做的是**字段里的确定性提取**，不是猜测。
- 摘要以"倒排索引"形式返回（词 → 出现位置数组），需要重建。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    classify_openalex_type,
    coalesce,
    doc_types_to_openalex_filter,
    extract_arxiv_id_from_urls,
    resolve_oa_status,
)
from paper_evidence_qa.sources.payload import (
    as_bool,
    as_mapping,
    as_sequence,
    as_str,
)
from paper_evidence_qa.sources.query import doc_types_of

API_URL = "https://api.openalex.org/works"

SELECT_FIELDS = ",".join(
    (
        "id",
        "doi",
        "title",
        "display_name",
        "publication_year",
        "authorships",
        "primary_location",
        "best_oa_location",
        "locations",
        "type",
        "cited_by_count",
        "ids",
        "abstract_inverted_index",
        "open_access",
    )
)


class OpenAlexAdapter(SourceAdapter):
    """OpenAlex ``/works`` 检索适配器。"""

    name = SourceName.OPENALEX
    display_name = "OpenAlex"
    base_url = API_URL
    default_min_interval_seconds = 0.2
    """polite pool 允许 10 req/s，这里留出余量取 5 req/s。"""

    default_page_size = 25
    max_page_size = 200
    requires_mailto = True
    supports_open_access_filter = True
    """OpenAlex 支持 ``is_oa:true`` 服务端筛选，不需要本地二次过滤。"""

    # ---------------------------------------------------------------- 请求

    def build_request(self, query: SourceQuery) -> SourceRequest:
        params: dict[str, Any] = {
            "search": query.query_text,
            "per-page": self.limit_for(query),
            "page": 1,
            "select": SELECT_FIELDS,
            "sort": "relevance_score:desc",
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
            parts.append(f"from_publication_date:{int(year_from)}-01-01")
        if year_to:
            parts.append(f"to_publication_date:{int(year_to)}-12-31")

        values = doc_types_to_openalex_filter(doc_types_of(query))
        if values:
            parts.append("type:" + "|".join(values))

        if query.open_access_only:
            parts.append("is_oa:true")
        return parts

    # ---------------------------------------------------------------- 解析

    def parse_response(
        self, response: HttpResponse, query: SourceQuery
    ) -> ParsedPage:
        payload = parse_json(response, where="openalex.root")
        if not isinstance(payload, Mapping):
            raise ResponseShapeError(
                "OpenAlex 响应顶层不是对象",
                where="openalex.root",
                url=response.url,
            )
        results = payload.get("results")
        if not isinstance(results, list):
            raise ResponseShapeError(
                "OpenAlex 响应缺少 results 数组",
                where="openalex.results",
                url=response.url,
            )

        candidates: list[SourceCandidate] = []
        warnings = []
        for raw in results:
            if not isinstance(raw, Mapping):
                continue
            candidate, extra, _reason = self._build_one(raw)
            warnings.extend(extra)
            if candidate is not None:
                candidates.append(candidate)

        return ParsedPage(
            candidates=candidates, raw_count=len(results), warnings=warnings
        )

    def _build_one(self, raw: Mapping[str, Any]) -> tuple[Any, list[Any], str | None]:
        best_oa = as_mapping(raw.get("best_oa_location"))
        primary = as_mapping(raw.get("primary_location"))
        open_access = as_mapping(raw.get("open_access"))
        ids = as_mapping(raw.get("ids"))
        openalex_id = _short_openalex_id(as_str(raw.get("id")))

        oa = resolve_oa_status(
            is_open=as_bool(open_access.get("is_oa")),
            oa_url=as_str(
                coalesce(
                    best_oa.get("pdf_url"),
                    primary.get("pdf_url"),
                    open_access.get("oa_url"),
                )
            ),
            source="openalex",
        )

        authors = [
            as_str(as_mapping(item.get("author")).get("display_name"))
            for item in as_sequence(raw.get("authorships"))
            if isinstance(item, Mapping)
        ]

        return build_candidate(
            source_name=self.name,
            title=as_str(raw.get("title")) or as_str(raw.get("display_name")),
            oa=oa,
            now=self.now(),
            authors=authors,
            year=raw.get("publication_year"),
            venue=as_str(as_mapping(primary.get("source")).get("display_name")),
            doi=as_str(raw.get("doi")),
            arxiv_raw=extract_arxiv_id_from_urls(_location_urls(raw, primary, best_oa)),
            abstract=_abstract_from_inverted_index(raw.get("abstract_inverted_index")),
            doc_type=classify_openalex_type(as_str(raw.get("type"))),
            external_ids={
                "openalex": openalex_id,
                "mag": as_str(ids.get("mag")),
                "pmid": as_str(ids.get("pmid")),
                "pmcid": as_str(ids.get("pmcid")),
            },
            citation_count=raw.get("cited_by_count"),
            upstream_id=openalex_id,
            license_value=as_str(best_oa.get("license")),
        )


def _short_openalex_id(raw: str | None) -> str | None:
    """``https://openalex.org/W123`` → ``W123``。"""
    if not raw:
        return None
    return raw.rstrip("/").rsplit("/", 1)[-1] or None


def _location_urls(
    raw: Mapping[str, Any],
    primary: Mapping[str, Any],
    best_oa: Mapping[str, Any],
) -> list[str]:
    """收集所有可能含 arXiv 链接的位置字段。"""
    urls: list[str] = []
    locations: list[Mapping[str, Any]] = [primary, best_oa]
    for item in as_sequence(raw.get("locations")):
        if isinstance(item, Mapping):
            locations.append(item)
    for location in locations:
        for key in ("pdf_url", "landing_page_url"):
            value = as_str(location.get(key))
            if value:
                urls.append(value)
    return urls


def _abstract_from_inverted_index(raw: Any) -> str | None:
    """把 OpenAlex 的倒排索引还原成摘要正文。

    形如 ``{"This": [0], "is": [1, 5]}``，按位置排序拼接即可还原原句。
    """
    if not isinstance(raw, Mapping):
        return None
    positions: list[tuple[int, str]] = []
    for word, offsets in raw.items():
        if not isinstance(word, str) or not isinstance(offsets, Sequence):
            continue
        for offset in offsets:
            if isinstance(offset, int) and not isinstance(offset, bool):
                positions.append((offset, word))
    if not positions:
        return None
    positions.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positions)


__all__ = ["API_URL", "SELECT_FIELDS", "OpenAlexAdapter"]
