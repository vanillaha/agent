"""arXiv 适配器（M2）—— 补充层，负责全文与版本锚点。

**两个作用，第二个更重要：**

1. **全文最可靠的来源**。AI / NLP 方向的最新工作几乎都先上 arXiv，
   并且提供 PDF 直链——这是 M3 里"基本必然能下到全文"的一个源。
2. **arXiv ID 是 M4 判定版本关系的锚点**。"预印本与会议版是不是同一篇"
   最可靠的规则就是"arXiv ID ↔ DOI 映射"，而这个映射的源头就在这里。

**它的坑，必须写清楚：**

- arXiv 收录的**全部是预印本，未经同行评审**。所以 ``doc_type`` 一律写
  ``preprint``，绝不能被当成"已发表"的证据使用。
- 无密钥但要守礼仪：官方要求**调用间隔不短于 3 秒**。这个间隔由限流器强制，
  不靠调用方自觉。
- 返回的是 **Atom XML 而不是 JSON**，所以这个 adapter 是全项目唯一一个
  解析 XML 的地方。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any
from xml.etree.ElementTree import ParseError

from paper_evidence_qa.schemas import DocType, SourceName
from paper_evidence_qa.sources.base import (
    ParsedPage,
    SourceAdapter,
    SourceCandidate,
    SourceQuery,
    SourceRequest,
    build_candidate,
)
from paper_evidence_qa.sources.errors import ResponseShapeError
from paper_evidence_qa.sources.http import HttpResponse
from paper_evidence_qa.sources.normalize import (
    arxiv_version,
    collapse_whitespace,
    normalize_arxiv_id,
    resolve_oa_status,
)

API_URL = "https://export.arxiv.org/api/query"

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"

_ATOM = f"{{{ATOM_NS}}}"
_ARXIV = f"{{{ARXIV_NS}}}"

MIN_INTERVAL_SECONDS = 3.0
"""arXiv 官方礼仪要求的最小调用间隔。写死在适配器里，默认不允许调短。"""

_ADVANCED_QUERY_MARKERS = (" AND ", " OR ", " ANDNOT ")
_FIELD_PREFIXES = ("all:", "ti:", "au:", "abs:", "cat:", "jr:", "co:", "id:", "doi:")


class ArxivAdapter(SourceAdapter):
    """arXiv Atom API 适配器。"""

    name = SourceName.ARXIV
    display_name = "arXiv"
    base_url = API_URL
    default_min_interval_seconds = MIN_INTERVAL_SECONDS
    default_page_size = 25
    max_page_size = 200
    requires_mailto = False
    supports_open_access_filter = False
    """arXiv 没有"只要开放获取"的筛选参数——它本来就是全开放，无需筛选。"""

    # ---------------------------------------------------------------- 请求

    def build_request(self, query: SourceQuery) -> SourceRequest:
        params: dict[str, Any] = {
            "search_query": self.build_search_query(query),
            "start": 0,
            "max_results": self.limit_for(query),
        }
        headers = self.polite_headers()
        headers["Accept"] = "application/atom+xml"
        return SourceRequest(url=self.base_url, params=params, headers=headers)

    def build_search_query(self, query: SourceQuery) -> str:
        """把检索意图翻译成 arXiv 的查询语法。

        已经是 arXiv 高级语法（含 ``ti:`` 之类的字段前缀或布尔运算符）时原样放行——
        那说明是 M7 精心构造的检索式，不该再被包装一层。
        否则按词拆分并用 ``AND`` 连接：arXiv 不认"整句话当短语"，
        把一句话塞进引号会几乎检索不到结果。
        """
        text = query.query_text.strip()
        clauses: list[str] = []

        if self._is_advanced(text):
            clauses.append(text)
        else:
            terms = [token for token in text.split() if token]
            clauses.append(
                " AND ".join(f'all:"{_escape_term(token)}"' for token in terms)
            )

        date_clause = self._submitted_date_clause(query)
        if date_clause:
            clauses.append(date_clause)
        return " AND ".join(clauses)

    @staticmethod
    def _is_advanced(text: str) -> bool:
        lowered = text
        if any(marker in f" {lowered} " for marker in _ADVANCED_QUERY_MARKERS):
            return True
        return any(prefix in lowered for prefix in _FIELD_PREFIXES)

    @staticmethod
    def _submitted_date_clause(query: SourceQuery) -> str | None:
        """构造 ``submittedDate`` 区间。

        arXiv 要求区间两端都完整给出，所以只给一端时用极值补另一端——
        这不是猜测，而是把"不限制"显式写成 arXiv 能接受的边界值。
        """
        year_from = query.filters.get("year_from")
        year_to = query.filters.get("year_to")
        if not year_from and not year_to:
            return None
        start = f"{int(year_from)}01010000" if year_from else "180001010000"
        end = f"{int(year_to)}12312359" if year_to else "220012312359"
        return f"submittedDate:[{start} TO {end}]"

    # ---------------------------------------------------------------- 解析

    def parse_response(self, response: HttpResponse, query: SourceQuery) -> ParsedPage:
        try:
            root = ET.fromstring(response.text)
        except ParseError as exc:
            raise ResponseShapeError(
                f"arXiv 响应不是合法 XML：{exc}",
                where="arxiv.root",
                url=response.url,
                details={"preview": response.text[:200].replace("\n", " ")},
            ) from exc

        entries = root.findall(f"{_ATOM}entry")
        candidates: list[SourceCandidate] = []
        warnings = []
        for entry in entries:
            candidate, extra, _reason = self._build_one(entry)
            warnings.extend(extra)
            if candidate is not None:
                candidates.append(candidate)

        return ParsedPage(
            candidates=candidates, raw_count=len(entries), warnings=warnings
        )

    def _build_one(self, entry: ET.Element) -> tuple[Any, list[Any], str | None]:
        raw_id = collapse_whitespace(entry.findtext(f"{_ATOM}id"))
        arxiv_id = normalize_arxiv_id(raw_id)
        pdf_from_feed = self._pdf_url_from_feed(entry)
        pdf_url = pdf_from_feed or (
            f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None
        )

        if pdf_url is None:
            oa = resolve_oa_status(is_open=None, oa_url=None, source="arxiv")
        else:
            oa = resolve_oa_status(is_open=True, oa_url=pdf_url, source="arxiv")
            if pdf_from_feed is None:
                # 响应里没有 pdf 链接时，按 arXiv 官方 URL 规则由 ID 构造。
                # 这不是猜测，但要把"链接是构造来的"写进判定依据，便于人工复核。
                oa = oa.model_copy(
                    update={
                        "oa_evidence": (
                            "arxiv.api 未返回 pdf 链接，按 arXiv 官方规则"
                            f"由 ID {arxiv_id} 构造：{pdf_url}"
                        )
                    }
                )

        version = arxiv_version(raw_id)
        external_ids: dict[str, Any] = {
            "arxiv_category": self._category(entry),
            "arxiv_version": str(version) if version is not None else None,
        }

        # arXiv API 不提供引用数。这里留 None 而不是填 0——
        # 0 会被下游误读成"无人引用"，是两件完全不同的事。
        return build_candidate(
            source_name=self.name,
            title=entry.findtext(f"{_ATOM}title"),
            oa=oa,
            now=self.now(),
            authors=[
                node.findtext(f"{_ATOM}name")
                for node in entry.findall(f"{_ATOM}author")
            ],
            year=entry.findtext(f"{_ATOM}published"),
            venue=entry.findtext(f"{_ARXIV}journal_ref"),
            doi=entry.findtext(f"{_ARXIV}doi"),
            arxiv_raw=raw_id,
            abstract=entry.findtext(f"{_ATOM}summary"),
            doc_type=DocType.PREPRINT,
            external_ids=external_ids,
            citation_count=None,
            upstream_id=raw_id,
            license_value=None,
        )

    @staticmethod
    def _pdf_url_from_feed(entry: ET.Element) -> str | None:
        for link in entry.findall(f"{_ATOM}link"):
            title = (link.get("title") or "").lower()
            media = (link.get("type") or "").lower()
            if title == "pdf" or media == "application/pdf":
                href = link.get("href")
                if href:
                    return href
        return None

    @staticmethod
    def _category(entry: ET.Element) -> str | None:
        primary = entry.find(f"{_ARXIV}primary_category")
        if primary is not None:
            term = primary.get("term")
            if term:
                return term
        for node in entry.findall(f"{_ATOM}category"):
            term = node.get("term")
            if term:
                return term
        return None


def _escape_term(token: str) -> str:
    """转义引号，避免用户输入把查询语法撑破。"""
    return token.replace("\\", "\\\\").replace('"', '\\"')


__all__ = [
    "API_URL",
    "ARXIV_NS",
    "ATOM_NS",
    "MIN_INTERVAL_SECONDS",
    "ArxivAdapter",
]
