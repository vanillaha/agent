"""检索式构造与结构化筛选（M2）。

**这一层刻意不含大模型。** 检索式由参数拼装，规则确定、可复现、可断言。
让模型生成检索式是 M7 的事（它要在"证据不够"时想新角度），
而 M2 只负责把给定的意图翻译成每个源各自能懂的请求。

为什么按源分别构造：四个源的查询语法、筛选参数、字段名完全不同。
把"研究问题"硬塞成同一个字符串发给四家，只会四处碰壁；
正确做法是共用同一份意图，各源各自翻译。

筛选条件用固定的键名（见 ``FILTER_*`` 常量），它会被原样写进
``RetrievalRecord.filters``，所以键名一旦定下就是契约的一部分。
"""

from __future__ import annotations

from typing import Annotated, Any, Sequence

from pydantic import Field

from paper_evidence_qa.schemas import BaseSchema, DocType, PaperQARequest, SourceName
from paper_evidence_qa.sources.base import SourceQuery
from paper_evidence_qa.sources.normalize import collapse_whitespace

FILTER_YEAR_FROM = "year_from"
FILTER_YEAR_TO = "year_to"
FILTER_DOC_TYPES = "doc_types"
FILTER_OPEN_ACCESS_ONLY = "open_access_only"

MAX_KEYWORD_COUNT = 20


class SearchFilters(BaseSchema):
    """结构化筛选条件。写进 ``RetrievalRecord.filters`` 的正是这份内容。"""

    year_from: Annotated[int, Field(ge=1800, le=2200)] | None = None
    year_to: Annotated[int, Field(ge=1800, le=2200)] | None = None
    doc_types: list[DocType] = Field(default_factory=list)
    open_access_only: bool = False

    def to_record_dict(self) -> dict[str, Any]:
        """转成可落进 ``RetrievalRecord`` 的字典。文献类型用字符串值，便于比对。"""
        return {
            FILTER_YEAR_FROM: self.year_from,
            FILTER_YEAR_TO: self.year_to,
            FILTER_DOC_TYPES: [item.value for item in self.doc_types],
            FILTER_OPEN_ACCESS_ONLY: self.open_access_only,
        }

    @classmethod
    def from_record_dict(cls, raw: dict[str, Any]) -> "SearchFilters":
        """从字典还原。用于测试断言"构造出来的筛选条件与记录一致"。"""
        return cls(
            year_from=raw.get(FILTER_YEAR_FROM),
            year_to=raw.get(FILTER_YEAR_TO),
            doc_types=[
                DocType(value) for value in raw.get(FILTER_DOC_TYPES, []) if value
            ],
            open_access_only=bool(raw.get(FILTER_OPEN_ACCESS_ONLY, False)),
        )


def doc_types_of(query: SourceQuery) -> list[DocType]:
    """从 ``SourceQuery.filters`` 里还原文献类型。

    无法识别的取值直接跳过（而不是猜一个默认类型），
    因为"筛选条件被悄悄放宽"比"少筛一类"更危险。
    """
    result: list[DocType] = []
    for value in query.filters.get(FILTER_DOC_TYPES) or []:
        try:
            result.append(DocType(value))
        except ValueError:
            continue
    return result


class QueryBuilder:
    """把 ``PaperQARequest`` 翻译成各源可用的 ``SourceQuery``。"""

    def __init__(self, *, max_terms: int = MAX_KEYWORD_COUNT) -> None:
        self.max_terms = max_terms

    def base_query_text(self, request: PaperQARequest) -> str:
        """从请求里取出检索用词。

        有 ``keywords`` 就用关键词（用户显式给了意图）；
        没有才退回研究问题原文。这是**确定性规则**，不交给模型判断。
        """
        if request.keywords:
            terms = [
                value for value in (collapse_whitespace(k) for k in request.keywords) if value
            ][: self.max_terms]
            if terms:
                return " ".join(terms)
        text = collapse_whitespace(request.question)
        if not text:
            raise ValueError("请求既无关键词也无研究问题，无法构造检索式")
        return text

    def filters_for(
        self,
        request: PaperQARequest,
        *,
        open_access_only: bool = False,
    ) -> SearchFilters:
        return SearchFilters(
            year_from=request.year_from,
            year_to=request.year_to,
            doc_types=list(request.doc_types),
            open_access_only=open_access_only,
        )

    def build(
        self,
        source_name: SourceName,
        request: PaperQARequest,
        *,
        query_text: str | None = None,
        open_access_only: bool = False,
        limit: int = 20,
    ) -> SourceQuery:
        """构造发给某个源的检索请求。

        ``query_text`` 显式传入时优先使用它——M7 在后续轮次会给出新的检索角度，
        此时不应再回头用第一轮的默认检索式。
        """
        text = collapse_whitespace(query_text) or self.base_query_text(request)
        filters = self.filters_for(request, open_access_only=open_access_only)
        return SourceQuery(
            query_text=text,
            filters=filters.to_record_dict(),
            limit=limit,
            open_access_only=open_access_only,
        )

    def build_all(
        self,
        request: PaperQARequest,
        source_names: Sequence[SourceName],
        *,
        query_text: str | None = None,
        open_access_only: bool = False,
        limit: int = 20,
    ) -> dict[SourceName, SourceQuery]:
        return {
            name: self.build(
                name,
                request,
                query_text=query_text,
                open_access_only=open_access_only,
                limit=limit,
            )
            for name in source_names
        }


__all__ = [
    "FILTER_DOC_TYPES",
    "FILTER_OPEN_ACCESS_ONLY",
    "FILTER_YEAR_FROM",
    "FILTER_YEAR_TO",
    "MAX_KEYWORD_COUNT",
    "QueryBuilder",
    "SearchFilters",
    "doc_types_of",
]
