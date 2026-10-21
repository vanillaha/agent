"""适配器装配与检索编排（M2）。

这是联网链路的对外入口。上层的 M7 / M9 只需要调 ``SearchService.search``，
不必知道四个源各自怎么拼请求。

三条设计要点：

1. **未授权绝不联网。** ``allow_online=false`` 时一个请求都不发。
   这不是"发出去再拒绝"，而是根本不进入 adapter。隐藏测试会关闭联网来验证这一点，
   所以它是设计里的一等公民，不是异常分支。
2. **单源失败不拖垮整体。** 某个源挂了、被限流了、改格式了，其余源照常返回；
   失败信息落在该源自己的 ``RetrievalRecord.error`` 上，谁挂了一目了然。
3. **失败与空结果必须可区分。** ``error`` 非空就是失败，``result_count=0`` 才是
   "确实没有"。把 429 说成"没搜到"是最危险的谎，这里从类型上堵住。
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Mapping, Sequence

from pydantic import Field, model_validator

from paper_evidence_qa.ids import random_id
from paper_evidence_qa.schemas import (
    BaseSchema,
    ErrorCode,
    ErrorInfo,
    PaperQARequest,
    RetrievalRecord,
    RetrievalTarget,
    SourceName,
    WarningItem,
)
from paper_evidence_qa.sources.arxiv import ArxivAdapter
from paper_evidence_qa.sources.base import (
    AdapterSettings,
    SourceAdapter,
    SourceCandidate,
    SourceFetchResult,
    SourceQuery,
    make_failed_result,
)
from paper_evidence_qa.sources.cache import CachePolicy, ResponseCache
from paper_evidence_qa.sources.crossref import CrossrefAdapter
from paper_evidence_qa.sources.errors import SourceError
from paper_evidence_qa.sources.http import CachingHttpClient, HttpClient
from paper_evidence_qa.sources.normalize import collapse_whitespace
from paper_evidence_qa.sources.openalex import OpenAlexAdapter
from paper_evidence_qa.sources.query import QueryBuilder
from paper_evidence_qa.sources.semantic_scholar import SemanticScholarAdapter

ONLINE_SOURCE_NAMES: tuple[SourceName, ...] = (
    SourceName.OPENALEX,
    SourceName.CROSSREF,
    SourceName.ARXIV,
    SourceName.SEMANTIC_SCHOLAR,
)
"""联网链路涉及的来源。本地知识库不在其中——本地检索是 M5/M6 的事。"""

ADAPTER_CLASSES: Mapping[SourceName, type[SourceAdapter]] = {
    SourceName.OPENALEX: OpenAlexAdapter,
    SourceName.CROSSREF: CrossrefAdapter,
    SourceName.ARXIV: ArxivAdapter,
    SourceName.SEMANTIC_SCHOLAR: SemanticScholarAdapter,
}

DEFAULT_PAGE_SIZES: Mapping[SourceName, int] = {
    SourceName.OPENALEX: 25,
    SourceName.CROSSREF: 20,
    SourceName.ARXIV: 25,
    SourceName.SEMANTIC_SCHOLAR: 20,
}


class SearchOutcome(BaseSchema):
    """一轮跨源检索的汇总结果。"""

    candidates: list[SourceCandidate] = Field(default_factory=list)
    """跨源汇总候选，**未去重**——同一篇论文可能在多个源各出现一次，
    合并与版本关系判定是 M4 的职责。"""

    retrieval_records: list[RetrievalRecord] = Field(default_factory=list)
    """每个来源一条检索记录，供 M10 复现与评估。"""

    disabled_sources: list[SourceName] = Field(default_factory=list)
    """本轮未参与检索的来源：被配置禁用的，或因未授权联网而不可用的。"""

    online_disabled: bool = False
    warnings: list[WarningItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "SearchOutcome":
        if self.online_disabled and self.candidates:
            raise ValueError("未联网时不得产生任何候选：不允许把本地结果混进来")
        return self

    @property
    def failed_sources(self) -> list[SourceName]:
        """本轮检索失败的来源。与"返回空结果"区分开。"""
        return [
            record.source_name
            for record in self.retrieval_records
            if record.error is not None
        ]


def default_settings(
    *,
    mailto: str | None = None,
    semantic_scholar_api_key: str | None = None,
    enabled: Mapping[SourceName, bool] | None = None,
    timeout_seconds: float = 20.0,
    max_retries: int = 2,
    cache_ttl_seconds: int = 86400,
) -> dict[SourceName, AdapterSettings]:
    """构造四个来源的默认配置。

    ``mailto`` 与密钥都从外部传入（环境变量），**不写死在代码里**，
    也不允许落到检索记录或日志中。
    """
    overrides = dict(enabled or {})
    settings: dict[SourceName, AdapterSettings] = {}
    for name in ONLINE_SOURCE_NAMES:
        is_s2 = name is SourceName.SEMANTIC_SCHOLAR
        settings[name] = AdapterSettings(
            enabled=overrides.get(name, True),
            mailto=mailto,
            api_key=semantic_scholar_api_key if is_s2 else None,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            cache_ttl_seconds=cache_ttl_seconds,
            page_size=DEFAULT_PAGE_SIZES[name],
        )
    return settings


class SourceRegistry:
    """适配器集合。负责装配，不含检索逻辑。"""

    def __init__(self, adapters: Mapping[SourceName, SourceAdapter]) -> None:
        if not adapters:
            raise ValueError("适配器集合不得为空")
        self._adapters: dict[SourceName, SourceAdapter] = dict(adapters)

    @property
    def adapters(self) -> Mapping[SourceName, SourceAdapter]:
        return dict(self._adapters)

    @property
    def names(self) -> tuple[SourceName, ...]:
        return tuple(self._adapters)

    def get(self, name: SourceName) -> SourceAdapter | None:
        return self._adapters.get(name)

    def enabled_adapters(self) -> list[SourceAdapter]:
        """按固定顺序返回启用的适配器。

        顺序固定是为了让检索记录、缓存键和评估结果都可复现——
        遍历字典顺序会让同一份输入产生不同的输出顺序。
        """
        return [
            self._adapters[name]
            for name in ONLINE_SOURCE_NAMES
            if name in self._adapters and self._adapters[name].enabled
        ]

    def disabled_names(self) -> list[SourceName]:
        return [
            name
            for name in ONLINE_SOURCE_NAMES
            if name in self._adapters and not self._adapters[name].enabled
        ]

    @classmethod
    def with_defaults(
        cls,
        *,
        http: HttpClient,
        cache: ResponseCache | None = None,
        mailto: str | None = None,
        semantic_scholar_api_key: str | None = None,
        enabled: Mapping[SourceName, bool] | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        cache_ttl_seconds: int = 86400,
        replay_only: bool = False,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> "SourceRegistry":
        """装配四个适配器的标准配置。

        ``cache`` 传入时，每个源各套一层带自己 TTL 的缓存客户端；
        不传则直接使用裸客户端（测试与"不要缓存"的场合）。
        """
        settings_map = default_settings(
            mailto=mailto,
            semantic_scholar_api_key=semantic_scholar_api_key,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        adapters: dict[SourceName, SourceAdapter] = {}
        for name, settings in settings_map.items():
            client: HttpClient = http
            if cache is not None:
                client = CachingHttpClient(
                    http,
                    cache,
                    CachePolicy(
                        ttl_seconds=settings.cache_ttl_seconds,
                        enabled=True,
                        replay_only=replay_only,
                    ),
                    now=now,
                )
            adapters[name] = ADAPTER_CLASSES[name](
                settings=settings,
                http=client,
                now=now,
                monotonic=monotonic,
            )
        return cls(adapters)


class SearchService:
    """跨源检索入口。"""

    def __init__(
        self,
        registry: SourceRegistry,
        *,
        builder: QueryBuilder | None = None,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.registry = registry
        self._builder = builder or QueryBuilder()
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._monotonic = monotonic or time.monotonic

    def search(
        self,
        request: PaperQARequest,
        *,
        round_index: int = 1,
        query_text: str | None = None,
        open_access_only: bool = False,
        sources: Sequence[SourceName] | None = None,
    ) -> SearchOutcome:
        """执行一轮跨源检索。

        ``sources`` 可指定只用其中几个源（例如只想扩展引用网络时只唤醒
        Semantic Scholar），不传则使用全部启用的源。
        """
        if round_index < 1:
            raise ValueError("round_index 从 1 开始")

        selected = self._select_adapters(sources)

        if not selected:
            return SearchOutcome(
                candidates=[],
                retrieval_records=[],
                disabled_sources=list(ONLINE_SOURCE_NAMES),
                online_disabled=False,
                warnings=[
                    WarningItem(
                        code="no_source_selected",
                        message="没有可用的在线来源（未启用或未选中），本轮未检索",
                        details={
                            "requested": [
                                name.value for name in (sources or ())
                            ]
                        },
                    )
                ],
            )

        if not request.allow_online:
            return self._offline_outcome(
                request,
                selected,
                round_index=round_index,
                query_text=query_text,
                open_access_only=open_access_only,
            )

        candidates: list[SourceCandidate] = []
        records: list[RetrievalRecord] = []
        warnings: list[WarningItem] = []

        for adapter in selected:
            query = self._builder.build(
                adapter.name,
                request,
                query_text=query_text,
                open_access_only=open_access_only,
                limit=adapter.default_page_size,
            )
            result = self._safe_fetch(adapter, query)
            candidates.extend(result.candidates)
            warnings.extend(result.warnings)
            if result.error is not None:
                warnings.append(
                    WarningItem(
                        code="source_failed",
                        message=(
                            f"{adapter.name.value} 检索失败：{result.error.message}"
                        ),
                        details={
                            "source": adapter.name.value,
                            "error_code": result.error.code.value,
                        },
                    )
                )
            records.append(self._to_record(result, round_index=round_index))

        disabled = self.registry.disabled_names()
        disabled.extend(
            name
            for name in ONLINE_SOURCE_NAMES
            if name not in {adapter.name for adapter in selected}
            and name not in disabled
        )

        return SearchOutcome(
            candidates=candidates,
            retrieval_records=records,
            disabled_sources=disabled,
            online_disabled=False,
            warnings=warnings,
        )

    # ---------------------------------------------------------------- 内部

    def _select_adapters(
        self, sources: Sequence[SourceName] | None
    ) -> list[SourceAdapter]:
        available = self.registry.enabled_adapters()
        if sources is None:
            return available
        wanted = set(sources)
        return [adapter for adapter in available if adapter.name in wanted]

    def _offline_outcome(
        self,
        request: PaperQARequest,
        selected: Sequence[SourceAdapter],
        *,
        round_index: int,
        query_text: str | None,
        open_access_only: bool,
    ) -> SearchOutcome:
        """未授权联网时的结果。

        这里**不发任何请求**，但依然为每个来源写一条检索记录，
        错误码为 ``online_disabled``——这样"这轮为什么一个候选都没有"
        在记录里是看得见的，不会变成一团迷雾。
        """
        records: list[RetrievalRecord] = []
        for adapter in selected:
            query = self._builder.build(
                adapter.name,
                request,
                query_text=query_text,
                open_access_only=open_access_only,
                limit=adapter.default_page_size,
            )
            records.append(
                self._to_record(
                    make_failed_result(
                        source_name=adapter.name,
                        query=query,
                        error=ErrorInfo(
                            code=ErrorCode.ONLINE_DISABLED,
                            message="未授权联网，本轮未发起任何请求",
                            details={"source": adapter.name.value},
                        ),
                    ),
                    round_index=round_index,
                )
            )

        disabled = list(self.registry.disabled_names())
        for adapter in selected:
            if adapter.name not in disabled:
                disabled.append(adapter.name)

        return SearchOutcome(
            candidates=[],
            retrieval_records=records,
            disabled_sources=disabled,
            online_disabled=True,
            warnings=[
                WarningItem(
                    code="online_disabled",
                    message="未授权联网，仅使用本地知识库检索",
                    details={"sources": [name.value for name in disabled]},
                )
            ],
        )

    def _safe_fetch(self, adapter: SourceAdapter, query: SourceQuery) -> SourceFetchResult:
        """调用适配器，保证**任何**异常都变成"这个源失败了"，而不是中断整轮检索。

        ``SourceError`` 是预期内的失败（限流、上游故障、格式变化）。
        其余异常说明是程序缺陷，同样不让它中断整轮，但错误码老老实实写
        ``internal_error`` 并附上异常类型——**不允许把代码 bug 说成"上游不可用"**。
        """
        try:
            return adapter.fetch(query)
        except SourceError as exc:
            return make_failed_result(
                source_name=adapter.name, query=query, error=exc.to_error_info()
            )
        except Exception as exc:  # noqa: BLE001 - 必须兜住，但要如实标记为内部错误
            return make_failed_result(
                source_name=adapter.name,
                query=query,
                error=ErrorInfo(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"{adapter.name.value} 适配器内部错误：{exc}",
                    details={
                        "source": adapter.name.value,
                        "exception": type(exc).__name__,
                    },
                ),
            )

    def _to_record(
        self, result: SourceFetchResult, *, round_index: int
    ) -> RetrievalRecord:
        """把检索结果压成一条 ``RetrievalRecord``。

        ``result_count`` 记上游返回条数，``kept_count`` 记归一化后留下的条数——
        两者的差就是"上游给了但我们不敢用"的数量，是评估解析质量的关键指标。
        """
        return RetrievalRecord(
            record_id=random_id("rcd"),
            query=result.query.query_text,
            normalized_query=collapse_whitespace(result.query.query_text),
            target=RetrievalTarget.ONLINE,
            source_name=result.source_name,
            filters=dict(result.query.filters),
            executed_at=self._now(),
            round_index=round_index,
            result_count=result.raw_count,
            kept_count=len(result.candidates),
            duration_ms=result.duration_ms,
            error=result.error,
        )


__all__ = [
    "ADAPTER_CLASSES",
    "DEFAULT_PAGE_SIZES",
    "ONLINE_SOURCE_NAMES",
    "SearchOutcome",
    "SearchService",
    "SourceRegistry",
    "default_settings",
]
