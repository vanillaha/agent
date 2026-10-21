"""Semantic Scholar 适配器（M2）—— 补充层，负责引用网络与跨源 ID 互映。

**它的独特价值**：一是引用/被引网络（支撑"多跳"问题——顺着引用往下爬，
这是本地库检索和普通关键词检索都做不到的），二是把 DOI / arXiv ID / PMID
统一在一个接口里返回，做跨源 ID 映射很方便。

**为什么放在补充层、不默认启用**：它的限流是四个源里最严的。
匿名访问共享公共配额，实测"连续第三次请求就可能被 429"；
即便申请了免费密钥，也只有 1 req/s（突发 10）。所以：

- 有密钥时限流间隔取 1 秒，无密钥时取 3 秒——**没有密钥就别指望它快**。
- 编排层应当只在真正需要扩展引用网络时才唤醒它，而不是每轮都查。

**许可要求**：Semantic Scholar 数据为 ODC-BY 1.0，对外产出必须署名；
摘要不得批量再分发。这一点由 M10 的记录与产物署名保证。
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
    classify_semantic_scholar_types,
    resolve_oa_status,
)
from paper_evidence_qa.sources.payload import (
    as_mapping,
    as_sequence,
    as_str,
    strings_in,
)

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

SELECT_FIELDS = ",".join(
    (
        "paperId",
        "title",
        "abstract",
        "year",
        "venue",
        "externalIds",
        "citationCount",
        "openAccessPdf",
        "publicationTypes",
        "authors",
    )
)

API_KEY_HEADER = "x-api-key"

MIN_INTERVAL_WITH_KEY = 1.0
MIN_INTERVAL_WITHOUT_KEY = 3.0


class SemanticScholarAdapter(SourceAdapter):
    """Semantic Scholar Graph API 适配器。"""

    name = SourceName.SEMANTIC_SCHOLAR
    display_name = "Semantic Scholar"
    base_url = API_URL
    default_min_interval_seconds = MIN_INTERVAL_WITHOUT_KEY
    default_page_size = 20
    max_page_size = 100
    requires_mailto = False
    supports_open_access_filter = False
    """该接口没有"只要开放获取"的筛选参数，只能在本地过滤并记告警。"""

    @property
    def min_interval_seconds(self) -> float:
        """有密钥 1 req/s；无密钥退到 3 秒，避免打死公共配额。"""
        if self.settings.api_key:
            return MIN_INTERVAL_WITH_KEY
        return MIN_INTERVAL_WITHOUT_KEY

    # ---------------------------------------------------------------- 请求

    def build_request(self, query: SourceQuery) -> SourceRequest:
        params: dict[str, Any] = {
            "query": query.query_text,
            "fields": SELECT_FIELDS,
            "limit": self.limit_for(query),
        }
        year_range = self._year_range(query)
        if year_range:
            params["year"] = year_range

        headers = self.polite_headers()
        if self.settings.api_key:
            headers[API_KEY_HEADER] = self.settings.api_key
        return SourceRequest(url=self.base_url, params=params, headers=headers)

    @staticmethod
    def _year_range(query: SourceQuery) -> str | None:
        """Semantic Scholar 用 ``2020-2025`` / ``2020-`` / ``-2025`` 表达年份区间。"""
        year_from = query.filters.get("year_from")
        year_to = query.filters.get("year_to")
        if not year_from and not year_to:
            return None
        start = str(int(year_from)) if year_from else ""
        end = str(int(year_to)) if year_to else ""
        return f"{start}-{end}"

    # ---------------------------------------------------------------- 解析

    def parse_response(
        self, response: HttpResponse, query: SourceQuery
    ) -> ParsedPage:
        payload = parse_json(response, where="semantic_scholar.root")
        if not isinstance(payload, Mapping):
            raise ResponseShapeError(
                "Semantic Scholar 响应顶层不是对象",
                where="semantic_scholar.root",
                url=response.url,
            )
        data = payload.get("data")
        if not isinstance(data, list):
            raise ResponseShapeError(
                "Semantic Scholar 响应缺少 data 数组",
                where="semantic_scholar.data",
                url=response.url,
            )

        candidates: list[SourceCandidate] = []
        warnings = []
        for raw in data:
            if not isinstance(raw, Mapping):
                continue
            candidate, extra, _reason = self._build_one(raw)
            warnings.extend(extra)
            if candidate is not None:
                candidates.append(candidate)

        return ParsedPage(
            candidates=candidates, raw_count=len(data), warnings=warnings
        )

    def _build_one(self, raw: Mapping[str, Any]) -> tuple[Any, list[Any], str | None]:
        external = as_mapping(raw.get("externalIds"))
        oa_pdf = as_mapping(raw.get("openAccessPdf"))
        oa_url = as_str(oa_pdf.get("url"))

        # 有 openAccessPdf 才敢说"开放"；字段缺失只代表"我们没有它的全文链接"，
        # 不代表这篇是封闭获取，所以传 None 而不是 False。
        oa = resolve_oa_status(
            is_open=True if oa_url else None,
            oa_url=oa_url,
            source="semantic_scholar",
        )

        paper_id = as_str(raw.get("paperId"))
        return build_candidate(
            source_name=self.name,
            title=as_str(raw.get("title")),
            oa=oa,
            now=self.now(),
            authors=[
                as_str(as_mapping(item).get("name"))
                for item in as_sequence(raw.get("authors"))
            ],
            year=raw.get("year"),
            venue=as_str(raw.get("venue")),
            doi=as_str(external.get("DOI")),
            arxiv_raw=as_str(external.get("ArXiv")),
            abstract=as_str(raw.get("abstract")),
            doc_type=classify_semantic_scholar_types(
                strings_in(raw.get("publicationTypes"))
            ),
            external_ids={
                "semantic_scholar": paper_id,
                "corpus_id": as_str(external.get("CorpusId")),
                "pmid": as_str(external.get("PubMed")),
                "pmcid": as_str(external.get("PubMedCentral")),
                "dblp": as_str(external.get("DBLP")),
            },
            citation_count=raw.get("citationCount"),
            upstream_id=paper_id,
            license_value=as_str(oa_pdf.get("license")),
        )


__all__ = [
    "API_KEY_HEADER",
    "API_URL",
    "MIN_INTERVAL_WITH_KEY",
    "MIN_INTERVAL_WITHOUT_KEY",
    "SELECT_FIELDS",
    "SemanticScholarAdapter",
]
