"""多源在线检索适配器（M2）。

对外统一从这里导入，不要直接依赖子模块路径：

    from paper_evidence_qa.sources import SearchService, SourceRegistry

模块划分：

| 模块 | 职责 |
| --- | --- |
| ``base`` | 适配器协议与对外结果模型；四个 adapter 共用的组装规则 |
| ``errors`` | 适配器异常与到 M1 ``ErrorCode`` 的映射 |
| ``http`` | HTTP 客户端：礼貌请求头、退避重试、缓存包装 |
| ``cache`` | 基于 SQLite 的响应缓存，支持离线重放 |
| ``ratelimit`` | 按源最小请求间隔的限流器 |
| ``normalize`` | DOI / arXiv ID / 标题 / 年份 / 作者 / OA 状态归一化 |
| ``payload`` | 上游 JSON 取值助手 |
| ``query`` | 检索式构造与结构化筛选 |
| ``openalex`` / ``crossref`` / ``arxiv`` / ``semantic_scholar`` | 四个来源适配器 |
| ``registry`` | 装配与跨源检索编排 |

关键行为约定（详见 ``spec/当前开发任务.md``）：

- 未授权联网时**不发起任何请求**，返回 ``online_disabled=True``，这是正常状态而非失败。
- 单源失败不影响其他源，失败信息落在该源的 ``RetrievalRecord.error`` 上。
- 检索失败与"检索成功但没有结果"在类型上可区分，不允许互相伪装。
- 候选统一为 M1 的 ``PaperRecord``，不重造字段；去重与版本关系属 M4。
"""

from paper_evidence_qa.sources.arxiv import ArxivAdapter
from paper_evidence_qa.sources.base import (
    AdapterSettings,
    ParsedPage,
    SourceAdapter,
    SourceCandidate,
    SourceFetchResult,
    SourceQuery,
    SourceRequest,
    build_candidate,
    make_failed_result,
)
from paper_evidence_qa.sources.cache import CachePolicy, CachedResponse, ResponseCache
from paper_evidence_qa.sources.crossref import CrossrefAdapter
from paper_evidence_qa.sources.errors import (
    OnlineDisabledError,
    RateLimitedError,
    ResponseShapeError,
    SourceError,
    UpstreamError,
)
from paper_evidence_qa.sources.http import (
    CachingHttpClient,
    HttpClient,
    HttpResponse,
    HttpxHttpClient,
    build_cache_key,
    parse_json,
)
from paper_evidence_qa.sources.normalize import (
    OAResolution,
    canonical_key,
    derive_paper_id,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
    parse_year,
    resolve_oa_status,
)
from paper_evidence_qa.sources.openalex import OpenAlexAdapter
from paper_evidence_qa.sources.query import QueryBuilder, SearchFilters, doc_types_of
from paper_evidence_qa.sources.ratelimit import RateLimiter
from paper_evidence_qa.sources.registry import (
    ONLINE_SOURCE_NAMES,
    SearchOutcome,
    SearchService,
    SourceRegistry,
    default_settings,
)
from paper_evidence_qa.sources.semantic_scholar import SemanticScholarAdapter

__all__ = [
    # base
    "AdapterSettings",
    "ParsedPage",
    "SourceAdapter",
    "SourceCandidate",
    "SourceFetchResult",
    "SourceQuery",
    "SourceRequest",
    "build_candidate",
    "make_failed_result",
    # errors
    "OnlineDisabledError",
    "RateLimitedError",
    "ResponseShapeError",
    "SourceError",
    "UpstreamError",
    # http
    "CachingHttpClient",
    "HttpClient",
    "HttpResponse",
    "HttpxHttpClient",
    "build_cache_key",
    "parse_json",
    # cache
    "CachePolicy",
    "CachedResponse",
    "ResponseCache",
    # ratelimit
    "RateLimiter",
    # normalize
    "OAResolution",
    "canonical_key",
    "derive_paper_id",
    "normalize_arxiv_id",
    "normalize_doi",
    "normalize_title",
    "parse_year",
    "resolve_oa_status",
    # query
    "QueryBuilder",
    "SearchFilters",
    "doc_types_of",
    # adapters
    "ArxivAdapter",
    "CrossrefAdapter",
    "OpenAlexAdapter",
    "SemanticScholarAdapter",
    # registry
    "ONLINE_SOURCE_NAMES",
    "SearchOutcome",
    "SearchService",
    "SourceRegistry",
    "default_settings",
]
