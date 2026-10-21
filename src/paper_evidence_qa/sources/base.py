"""适配器协议与 M2 新增模型。

这个模块定义"一个数据源适配器长什么样"，以及四个 adapter 共用的**组装规则**。

分三层，职责不重叠：

1. ``SourceQuery`` / ``SourceRequest`` / ``ParsedPage``：请求与解析的中间对象。
2. ``SourceCandidate`` / ``SourceFetchResult``：对外结果。**必须复用 M1 的
   ``PaperRecord``**，只额外携带 M1 没有的上游溯源信息（上游主键、许可证）。
3. ``SourceAdapter``：模板方法。子类只实现"怎么拼请求"和"怎么解析响应"，
   限流、失败封装、字段长度处理、开放获取筛选全部由基类统一做，
   保证四个源的行为一致——否则每个源都会长出自己的一套毛病。

一条关键约束：**"检索失败"与"检索成功但没有结果"必须在类型上可区分**。
前者 ``error`` 非空且 ``candidates`` 必为空；后者 ``error`` 为空。
由 ``SourceFetchResult`` 的校验器强制，不允许上游失败被伪装成空结果。
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, ClassVar, Mapping, Sequence

from pydantic import Field, StringConstraints, model_validator

from paper_evidence_qa.schemas import (
    BaseSchema,
    DocType,
    ErrorCode,
    ErrorInfo,
    MediumText,
    PaperRecord,
    ShortText,
    SourceName,
    WarningItem,
)
from paper_evidence_qa.sources.errors import (
    RateLimitedError,
    SourceError,
    UpstreamError,
)
from paper_evidence_qa.sources.http import HttpClient, HttpResponse
from paper_evidence_qa.sources.normalize import (
    OAResolution,
    coerce_int,
    collapse_whitespace,
    dedupe_authors,
    derive_paper_id,
    fit_text,
    normalize_arxiv_id,
    normalize_doi,
    parse_year,
    strip_markup,
)
from paper_evidence_qa.sources.ratelimit import RateLimiter

TITLE_LIMIT = 256
"""``ShortText`` 的长度上限。标题、会议名、许可证等短字段共用。"""

ABSTRACT_LIMIT = 2000
"""``MediumText`` 的长度上限。"""

URL_LIMIT = 2048

_CODE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
"""与 M1 ``CodeName`` 保持一致的校验规则，用于清洗外部 ID 的键。"""


class SourceQuery(BaseSchema):
    """送给单个数据源的检索请求。

    ``query_text`` 会原样落进 ``RetrievalRecord.query``，所以它必须是
    "这个源上真正跑过的检索式"，而不是笼统的研究问题——否则检索记录无法复现。
    """

    query_text: MediumText
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: Annotated[int, Field(ge=1, le=200)] = 20
    open_access_only: bool = False


class SourceRequest(BaseSchema):
    """一次 HTTP 请求的完整描述。便于在测试中断言"请求拼得对不对"。"""

    url: Annotated[str, StringConstraints(min_length=1, max_length=URL_LIMIT)]
    params: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)


class SourceCandidate(BaseSchema):
    """单篇候选文献 = M1 的 ``PaperRecord`` + 上游溯源信息。

    只新增 M1 里**确实没有**的字段：上游自己的主键、许可证、OA 判定依据。
    M1 已有的字段（标题、年份、DOI 等）一律复用，不在这里重造。
    """

    paper: PaperRecord
    upstream_id: ShortText | None = None
    """上游自身的主键，例如 OpenAlex 的 ``W2741809807``，便于人工比对。"""

    license: ShortText | None = None
    """许可证信息。Crossref 提供许可证链接，OpenAlex 可能只给代号（如 ``cc-by``）。"""

    oa_evidence: ShortText | None = None
    """开放获取状态的判定依据。必须能回答"凭什么判定它能下载"。"""


class ParsedPage(BaseSchema):
    """响应解析结果（adapter 内部使用）。"""

    candidates: list[SourceCandidate] = Field(default_factory=list)
    raw_count: Annotated[int, Field(ge=0)]
    """上游返回条数，归一化前。用于区分"上游没返回"与"我们全跳过了"。"""

    warnings: list[WarningItem] = Field(default_factory=list)


class SourceFetchResult(BaseSchema):
    """单源一次检索的结果。失败与空结果的区分由校验器强制。"""

    source_name: SourceName
    query: SourceQuery
    candidates: list[SourceCandidate] = Field(default_factory=list)
    raw_count: Annotated[int, Field(ge=0)] = 0
    request_url: Annotated[str, StringConstraints(max_length=URL_LIMIT)] | None = None
    from_cache: bool = False
    retry_count: Annotated[int, Field(ge=0)] = 0
    duration_ms: Annotated[int, Field(ge=0)] = 0
    skipped_count: Annotated[int, Field(ge=0)] = 0
    warnings: list[WarningItem] = Field(default_factory=list)
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _check_distinguishable(self) -> "SourceFetchResult":
        if self.error is None and self.raw_count < len(self.candidates):
            raise ValueError("raw_count 不得小于候选数量：解析结果不可能多于上游返回")
        if self.error is not None and self.candidates:
            raise ValueError("检索失败时不得携带候选结果：不允许把失败伪装成部分成功")
        return self

    @property
    def ok(self) -> bool:
        """是否检索成功（成功但无结果也算成功）。"""
        return self.error is None


class AdapterSettings(BaseSchema):
    """单源配置。

    ``api_key`` 只从环境变量或本对象读入，**不得写入日志、检索记录或测试文件**。
    """

    enabled: bool = True
    mailto: ShortText | None = None
    """进入 polite pool 的联系邮箱。缺失不会导致失败，但会产生告警。"""

    api_key: ShortText | None = None
    timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 20.0
    max_retries: Annotated[int, Field(ge=0, le=10)] = 2
    cache_ttl_seconds: Annotated[int, Field(ge=0, le=2592000)] = 86400
    page_size: Annotated[int, Field(ge=1, le=200)] | None = None


class SourceAdapter(ABC):
    """数据源适配器基类。

    子类只实现 ``build_request`` 与 ``parse_response``。基类负责：

    - 限流（按源礼仪设定最小请求间隔）
    - 失败封装（``SourceError`` → ``error`` 字段，不让异常穿透到上层）
    - 字段长度处理（超长截断并记告警，不静默改数据）
    - 开放获取筛选（源支持服务端筛选就用服务端，不支持就本地筛并记告警）
    """

    name: ClassVar[SourceName]
    display_name: ClassVar[str] = ""
    base_url: ClassVar[str] = ""
    default_min_interval_seconds: ClassVar[float] = 0.0
    default_page_size: ClassVar[int] = 20
    max_page_size: ClassVar[int] = 200
    requires_mailto: ClassVar[bool] = False
    """该源是否需要邮箱才能获得稳定服务（polite pool）。缺失时记告警。"""

    supports_open_access_filter: ClassVar[bool] = False
    """该源是否支持在服务端按开放获取筛选。不支持时本地筛选并记告警。"""

    def __init__(
        self,
        *,
        settings: AdapterSettings | None = None,
        http: HttpClient,
        limiter: RateLimiter | None = None,
        monotonic: Callable[[], float] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings or AdapterSettings()
        self._http = http
        self._monotonic = monotonic or time.monotonic
        self._now = now or (lambda: datetime.now(timezone.utc))
        if limiter is not None:
            self._limiter = limiter
        elif self.min_interval_seconds > 0:
            self._limiter = RateLimiter(self.min_interval_seconds)
        else:
            self._limiter = None

    # ------------------------------------------------------------------ 属性

    @property
    def min_interval_seconds(self) -> float:
        """同源两次请求的最小间隔。默认取类常量，子类可按配置覆盖。"""
        return self.default_min_interval_seconds

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def now(self) -> datetime:
        """当前时间。统一由外部注入，保证测试可复现。"""
        return self._now()

    def limit_for(self, query: SourceQuery) -> int:
        """把请求的 limit 夹到该源允许的区间内。"""
        configured = self.settings.page_size or self.default_page_size
        requested = min(query.limit, configured)
        return max(1, min(requested, self.max_page_size))

    def polite_headers(self) -> dict[str, str]:
        """礼貌请求头。带联系方式能让上游在出问题时联系到我们，而不是直接封禁。"""
        contact = self.settings.mailto
        if contact:
            agent = f"{self.display_name or self.name.value}/0.1 (+mailto:{contact})"
        else:
            agent = f"{self.display_name or self.name.value}/0.1"
        return {"User-Agent": agent, "Accept": "application/json"}

    # ------------------------------------------------------------------ 模板

    def fetch(self, query: SourceQuery) -> SourceFetchResult:
        """执行一次检索。

        只捕获 ``SourceError``（预期内的失败）；其他异常属于程序缺陷，
        必须向上抛出，由上层记为 ``internal_error``，不得伪装成上游故障。
        """
        started = self._monotonic()
        warnings: list[WarningItem] = []

        if not self.enabled:
            return self._failed_result(
                query,
                SourceError(
                    f"{self.name.value} 适配器已禁用，未发起请求",
                    details={"source": self.name.value, "reason": "adapter_disabled"},
                ),
                started,
                error_code=ErrorCode.ONLINE_DISABLED,
            )

        if self.requires_mailto and not self.settings.mailto:
            warnings.append(
                WarningItem(
                    code="mailto_missing",
                    message=(
                        f"{self.name.value} 未配置联系邮箱，将落在公共请求池，"
                        "响应可能变慢或被限流"
                    ),
                    details={"source": self.name.value},
                )
            )

        try:
            request = self.build_request(query)
            if self._limiter is not None:
                self._limiter.acquire()
            response = self._http.get(
                request.url, params=request.params, headers=request.headers
            )
            self._raise_for_status(response, request.url)
            page = self.parse_response(response, query)
        except SourceError as exc:
            return self._failed_result(query, exc, started, warnings=warnings)

        candidates = list(page.candidates)
        warnings.extend(page.warnings)

        if query.open_access_only and not self.supports_open_access_filter:
            kept = [c for c in candidates if c.paper.oa_status.value == "open"]
            if len(kept) != len(candidates):
                warnings.append(
                    WarningItem(
                        code="open_access_filter_local",
                        message=(
                            f"{self.name.value} 不支持服务端开放获取筛选，"
                            f"已本地过滤，丢弃 {len(candidates) - len(kept)} 条"
                        ),
                        details={
                            "source": self.name.value,
                            "dropped": len(candidates) - len(kept),
                        },
                    )
                )
            candidates = kept

        skipped = max(0, page.raw_count - len(page.candidates))
        if skipped:
            warnings.append(
                WarningItem(
                    code="records_skipped",
                    message=f"{self.name.value} 有 {skipped} 条记录因字段不合法被跳过",
                    details={"source": self.name.value, "skipped": skipped},
                )
            )

        return SourceFetchResult(
            source_name=self.name,
            query=query,
            candidates=candidates,
            raw_count=page.raw_count,
            request_url=response.url or request.url,
            from_cache=response.from_cache,
            retry_count=response.retry_count,
            duration_ms=self._elapsed_ms(started),
            skipped_count=skipped,
            warnings=warnings,
        )

    # ------------------------------------------------------------ 子类需实现

    @abstractmethod
    def build_request(self, query: SourceQuery) -> SourceRequest:
        """把 ``SourceQuery`` 拼成该源真实可用的 HTTP 请求。"""

    @abstractmethod
    def parse_response(self, response: HttpResponse, query: SourceQuery) -> ParsedPage:
        """解析响应，产出归一化候选。解析不了的记录跳过并记告警，不让整页失败。"""

    # ---------------------------------------------------------------- 内部

    def _elapsed_ms(self, started: float) -> int:
        return max(0, int(round((self._monotonic() - started) * 1000)))

    def _raise_for_status(self, response: HttpResponse, url: str) -> None:
        """把 HTTP 状态码翻译成结构化异常。

        HTTP 客户端已负责重试，走到这里说明重试已耗尽（或不该重试）。
        """
        status = response.status_code
        if 200 <= status < 300:
            return
        if status == 429:
            raise RateLimitedError(
                f"{self.name.value} 返回 429，重试 {response.retry_count} 次后仍被限流",
                attempts=response.retry_count,
                details={
                    "source": self.name.value,
                    "url": url,
                    "status_code": status,
                },
            )
        raise UpstreamError(
            f"{self.name.value} 返回 {status}",
            status_code=status,
            url=url,
            attempts=response.retry_count,
            details={"source": self.name.value},
        )

    def _failed_result(
        self,
        query: SourceQuery,
        error: SourceError,
        started: float,
        *,
        warnings: Sequence[WarningItem] | None = None,
        error_code: ErrorCode | None = None,
    ) -> SourceFetchResult:
        info = error.to_error_info()
        if error_code is not None and info.code is not error_code:
            info = ErrorInfo(code=error_code, message=info.message, details=info.details)
        return SourceFetchResult(
            source_name=self.name,
            query=query,
            candidates=[],
            raw_count=0,
            duration_ms=self._elapsed_ms(started),
            warnings=list(warnings or []),
            error=info,
        )


def make_failed_result(
    *,
    source_name: SourceName,
    query: SourceQuery,
    error: ErrorInfo,
    duration_ms: int = 0,
    warnings: Sequence[WarningItem] | None = None,
) -> SourceFetchResult:
    """由 ``ErrorInfo`` 直接构造失败结果。

    供上层在捕获到意外异常时使用（例如某个 adapter 里的程序缺陷）：
    必须显式给出错误码，且不允许附带任何候选——失败不能被包装成部分成功。
    """
    return SourceFetchResult(
        source_name=source_name,
        query=query,
        candidates=[],
        raw_count=0,
        duration_ms=duration_ms,
        warnings=list(warnings or []),
        error=error,
    )


def build_candidate(
    *,
    source_name: SourceName,
    title: str | None,
    oa: OAResolution,
    now: datetime,
    authors: Sequence[str | None] = (),
    year: Any = None,
    venue: str | None = None,
    doi: str | None = None,
    arxiv_raw: str | None = None,
    abstract: str | None = None,
    doc_type: DocType = DocType.OTHER,
    external_ids: Mapping[str, Any] | None = None,
    citation_count: Any = None,
    upstream_id: str | None = None,
    license_value: str | None = None,
) -> tuple[SourceCandidate | None, list[WarningItem], str | None]:
    """把各源解析出的零散字段组装成合法候选。

    返回 ``(候选, 告警, 跳过原因)``：

    - 组装成功：``(候选, 告警或空, None)``
    - 无法修复（缺标题、无任何同一性标识）：``(None, [], 原因)``，由 adapter 计入跳过
    - 可修复问题（摘要/标题超长）：截断并用告警说明，**不静默**。

    这样写的目的是：**字段长度这种"能救的"问题不要让整篇文献丢掉，
    而"救不了的"问题也不能悄悄塞进库里。**
    """
    warnings: list[WarningItem] = []

    clean_title = collapse_whitespace(strip_markup(title))
    if not clean_title:
        return None, warnings, "缺少标题"
    title_text, title_truncated = fit_text(clean_title, TITLE_LIMIT)
    if title_truncated:
        warnings.append(
            _truncation_warning("title", len(clean_title), TITLE_LIMIT, source_name)
        )

    abstract_text, abstract_truncated = fit_text(strip_markup(abstract), ABSTRACT_LIMIT)
    if abstract_truncated and abstract is not None:
        warnings.append(
            _truncation_warning(
                "abstract", len(abstract), ABSTRACT_LIMIT, source_name
            )
        )

    venue_text, venue_truncated = fit_text(
        collapse_whitespace(strip_markup(venue)), TITLE_LIMIT
    )
    if venue_truncated:
        warnings.append(
            _truncation_warning("venue", len(venue or ""), TITLE_LIMIT, source_name)
        )

    author_names = dedupe_authors(
        [
            fitted
            for fitted, _ in (
                fit_text(collapse_whitespace(name), TITLE_LIMIT)
                for name in authors
            )
            if fitted
        ]
    )

    doi_value = normalize_doi(doi)
    arxiv_value = normalize_arxiv_id(arxiv_raw)
    year_value = parse_year(year)

    try:
        paper_id = derive_paper_id(
            doi=doi_value,
            arxiv_id=arxiv_value,
            title=title_text,
            year=year_value,
        )
    except ValueError as exc:
        return None, warnings, str(exc)

    citation_value = _non_negative(citation_count)
    external = _clean_external_ids(external_ids, source_name)
    upstream = fit_text(collapse_whitespace(upstream_id), TITLE_LIMIT)[0]
    license_text = fit_text(collapse_whitespace(license_value), TITLE_LIMIT)[0]

    paper = PaperRecord(
        paper_id=paper_id,
        title=title_text,
        authors=author_names,
        year=year_value,
        venue=venue_text,
        doi=doi_value,
        arxiv_id=arxiv_value,
        abstract=abstract_text,
        doc_type=doc_type,
        sources=[source_name],
        external_ids=external,
        oa_status=oa.oa_status,
        oa_url=oa.oa_url,
        citation_count=citation_value,
        is_in_local_kb=False,
        version_group_id=None,
        first_seen_at=now,
        last_refreshed_at=now,
    )
    candidate = SourceCandidate(
        paper=paper,
        upstream_id=upstream,
        license=license_text,
        oa_evidence=oa.oa_evidence,
    )
    return candidate, warnings, None


def _truncation_warning(
    field: str, original_length: int, limit: int, source_name: SourceName
) -> WarningItem:
    return WarningItem(
        code="field_truncated",
        message=f"{source_name.value} 的 {field} 超过 {limit} 字符，已截断",
        details={
            "source": source_name.value,
            "field": field,
            "original_length": original_length,
            "limit": limit,
        },
    )


def _non_negative(raw: Any) -> int | None:
    value = coerce_int(raw)
    if value is None or value < 0:
        return None
    return value


def _clean_external_ids(
    raw: Mapping[str, Any] | None, source_name: SourceName
) -> dict[str, str]:
    """清洗外部 ID 映射：丢弃空值，并要求键符合 ``CodeName`` 规则。

    键不符合规则说明是我们代码写错了（例如 ``"OpenAlex ID"``），
    这里直接丢弃而不是改名——改名会掩盖 bug。
    """
    result: dict[str, str] = {}
    if not raw:
        return result
    for key, value in raw.items():
        if value is None:
            continue
        text = collapse_whitespace(str(value))
        if not text:
            continue
        if not _CODE_NAME_PATTERN.match(key):
            continue
        fitted, _ = fit_text(text, TITLE_LIMIT)
        if fitted:
            result[key] = fitted
    return result


__all__ = [
    "ABSTRACT_LIMIT",
    "TITLE_LIMIT",
    "URL_LIMIT",
    "AdapterSettings",
    "ParsedPage",
    "SourceAdapter",
    "SourceCandidate",
    "SourceFetchResult",
    "SourceQuery",
    "SourceRequest",
    "build_candidate",
    "make_failed_result",
]
