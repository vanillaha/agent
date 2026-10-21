"""领域模型（M1）：文献、版本关系、分块、证据、断言、回答、充分性判断、检索与下载记录、知识库版本。

本模块定义的是**证据链**。从 PDF 到最终回答，所有事实都必须能沿着
``Chunk -> Evidence -> Claim -> Answer`` 这条链反查到原文位置。

关键约束（对应课程规范）：
- 本地证据与联网摘要不得混写：``Evidence.source_type`` 与 ``Evidence.source_name`` 必须如实标注。
- 无证据的断言不允许存在：``Claim.evidence_ids`` 不得为空。
- 全文证据必须能定位：``Evidence.source_type == fulltext`` 时必须带 ``chunk_id``。
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import Field, StringConstraints, field_validator, model_validator

from paper_evidence_qa.schemas.common import (
    BaseSchema,
    ChunkId,
    ClaimId,
    CodeName,
    DownloadId,
    ErrorInfo,
    EvidenceId,
    KBVersionId,
    LongText,
    MediumText,
    PaperId,
    RecordId,
    Sha256,
    ShortText,
    UtcDatetime,
    VersionGroupId,
)


class SourceName(str, Enum):
    """文献来源。每个数据源对应一个独立 adapter，名称固定不得混用。"""

    LOCAL_KB = "local_kb"
    OPENALEX = "openalex"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    ARXIV = "arxiv"
    CROSSREF = "crossref"
    USER_UPLOAD = "user_upload"


class RetrievalTarget(str, Enum):
    """检索目标：本地知识库还是联网。两条路径的结果不得混写。"""

    LOCAL_KB = "local_kb"
    ONLINE = "online"


class OAStatus(str, Enum):
    """开放获取状态。``unknown`` 不等价于 ``open``，未核查不得下载。"""

    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class SupportType(str, Enum):
    """证据等级。用于区分全文证据、摘要信息与模型推断。"""

    FULLTEXT = "fulltext"
    ABSTRACT = "abstract"
    METADATA = "metadata"
    INFERENCE = "inference"


class DocType(str, Enum):
    """文献类型，用于检索筛选与版本判断。"""

    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    PREPRINT = "preprint"
    REVIEW = "review"
    OTHER = "other"


class VersionType(str, Enum):
    """版本形态。同一论文的预印本、会议版与期刊版必须建立版本关系。"""

    PREPRINT = "preprint"
    CONFERENCE = "conference"
    JOURNAL = "journal"
    OTHER = "other"
    UNKNOWN = "unknown"


class JudgedBy(str, Enum):
    """判定来源。确定性规则优先，模型判断必须标记出来便于复核。"""

    RULE = "rule"
    LLM = "llm"
    HYBRID = "hybrid"


class DownloadStatus(str, Enum):
    """下载状态。跳过与失败必须区分，便于统计开放全文下载成功率。"""

    DOWNLOADED = "downloaded"
    SKIPPED_NOT_OPEN_ACCESS = "skipped_not_open_access"
    REJECTED_FILE_TYPE = "rejected_file_type"
    FAILED = "failed"


class Locator(BaseSchema):
    """证据在原文中的位置。

    至少包含页码或章节路径之一，否则证据无法回溯，视为无效证据。
    """

    page_from: Annotated[int, Field(ge=1)] | None = None
    page_to: Annotated[int, Field(ge=1)] | None = None
    section_path: list[ShortText] = Field(default_factory=list)
    table_id: ShortText | None = None
    figure_id: ShortText | None = None
    char_start: Annotated[int, Field(ge=0)] | None = None
    char_end: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def _check_locator(self) -> "Locator":
        if self.page_from is None and not self.section_path:
            raise ValueError("证据位置必须至少包含页码或章节路径之一")
        if self.page_to is not None and self.page_from is None:
            raise ValueError("提供 page_to 时必须同时提供 page_from")
        if (
            self.page_from is not None
            and self.page_to is not None
            and self.page_from > self.page_to
        ):
            raise ValueError("page_from 不得大于 page_to")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_start >= self.char_end
        ):
            raise ValueError("char_end 必须大于 char_start")
        return self

    def to_display(self) -> str:
        """生成人可读的位置描述，用于回答中的引用标注。"""
        parts: list[str] = []
        if self.page_from is not None:
            if self.page_to is not None and self.page_to != self.page_from:
                parts.append(f"pp.{self.page_from}-{self.page_to}")
            else:
                parts.append(f"p.{self.page_from}")
        if self.section_path:
            parts.append(" > ".join(self.section_path))
        if self.table_id:
            parts.append(f"table {self.table_id}")
        if self.figure_id:
            parts.append(f"figure {self.figure_id}")
        return ", ".join(parts)


class PaperRecord(BaseSchema):
    """一篇文献的元数据。不保存正文，正文以 ``Chunk`` 形式落库。"""

    paper_id: PaperId
    title: ShortText
    authors: list[ShortText] = Field(default_factory=list)
    year: Annotated[int, Field(ge=1800, le=2200)] | None = None
    venue: ShortText | None = None
    doi: ShortText | None = None
    arxiv_id: ShortText | None = None
    abstract: MediumText | None = None
    doc_type: DocType = DocType.OTHER
    sources: Annotated[list[SourceName], Field(min_length=1)]
    external_ids: dict[CodeName, ShortText] = Field(default_factory=dict)
    oa_status: OAStatus = OAStatus.UNKNOWN
    oa_url: Annotated[str, StringConstraints(min_length=1, max_length=2048)] | None = None
    citation_count: Annotated[int, Field(ge=0)] | None = None
    is_in_local_kb: bool = False
    version_group_id: VersionGroupId | None = None
    first_seen_at: UtcDatetime
    last_refreshed_at: UtcDatetime
    notes: MediumText | None = None

    @field_validator("doi")
    @classmethod
    def _normalize_doi(cls, value: str | None) -> str | None:
        """统一 DOI 写法：去掉 URL 前缀并转小写，避免同一论文被当成两条记录。"""
        if value is None:
            return None
        normalized = value.strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if normalized.lower().startswith(prefix):
                normalized = normalized[len(prefix) :]
        return normalized.lower() or None

    @field_validator("authors")
    @classmethod
    def _dedupe_authors(cls, value: list[str]) -> list[str]:
        """按出现顺序去重，保持作者顺序不变。"""
        seen: set[str] = set()
        result: list[str] = []
        for author in value:
            key = author.casefold()
            if key not in seen:
                seen.add(key)
                result.append(author)
        return result

    @model_validator(mode="after")
    def _check_paper(self) -> "PaperRecord":
        if self.last_refreshed_at < self.first_seen_at:
            raise ValueError("last_refreshed_at 不得早于 first_seen_at")
        if self.oa_status is OAStatus.OPEN and not self.oa_url:
            raise ValueError("oa_status=open 时必须提供 oa_url")
        if self.oa_status is OAStatus.CLOSED and self.oa_url:
            raise ValueError("oa_status=closed 时不得提供 oa_url")
        return self


class VersionRelation(BaseSchema):
    """版本关系：把预印本、会议版、期刊版归入同一个版本族。

    同一个版本族内至多一个 ``is_canonical=True``（该约束在服务层校验，
    因为需要跨记录比较）。
    """

    group_id: VersionGroupId
    paper_id: PaperId
    version_type: VersionType
    version_label: ShortText | None = None
    is_canonical: bool = False
    relation_evidence: ShortText
    """判定依据，例如 ``doi_equal``、``arxiv_to_doi``、``title_author_year_fuzzy``。"""

    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class Chunk(BaseSchema):
    """入库后的原文分块。定位信息随分块一起保存，是引用可追溯的基础。"""

    chunk_id: ChunkId
    paper_id: PaperId
    kb_version: KBVersionId
    chunk_index: Annotated[int, Field(ge=0)]
    text: LongText
    locator: Locator
    char_start: Annotated[int, Field(ge=0)]
    char_end: Annotated[int, Field(ge=1)]
    token_count: Annotated[int, Field(ge=0)] | None = None
    source_file_sha256: Sha256

    @model_validator(mode="after")
    def _check_chunk(self) -> "Chunk":
        if self.char_end <= self.char_start:
            raise ValueError("char_end 必须大于 char_start")
        return self


class Evidence(BaseSchema):
    """一条证据。回答中的每个事实都必须指向至少一条证据。"""

    evidence_id: EvidenceId
    source_type: SupportType
    source_name: SourceName
    paper_id: PaperId
    chunk_id: ChunkId | None = None
    quote: LongText
    """原文片段。必须逐字引用，不得改写或润色。"""

    locator: Locator
    is_local: bool
    """是否来自本地知识库全文。False 表示来自联网元数据/摘要。"""

    retrieval_score: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    retrieval_round: Annotated[int, Field(ge=0)]
    retrieved_at: UtcDatetime

    @model_validator(mode="after")
    def _check_evidence(self) -> "Evidence":
        if self.source_type is SupportType.FULLTEXT:
            if self.chunk_id is None:
                raise ValueError("全文证据必须提供 chunk_id，否则无法回溯到原文分块")
            if not self.is_local:
                raise ValueError("全文证据必须来自本地知识库（下载入库后再引用）")
        if self.source_type is SupportType.INFERENCE and self.is_local:
            raise ValueError("模型推断不得标记为本地证据")
        return self


class Claim(BaseSchema):
    """回答中的一条事实断言。无证据的断言不允许存在。"""

    claim_id: ClaimId
    text: MediumText
    evidence_ids: Annotated[list[EvidenceId], Field(min_length=1)]
    support_type: SupportType
    """该断言的支持等级。多证据取其中最低等级，由代码计算，不由模型自述。"""

    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None


class ConflictPair(BaseSchema):
    """相互冲突的证据对。冲突必须显式列出，不得悄悄选一个。"""

    evidence_ids: Annotated[list[EvidenceId], Field(min_length=2)]
    description: MediumText


class Answer(BaseSchema):
    """基于证据的回答。证据不足时应当拒答，而不是给出推测。"""

    question: MediumText
    summary: MediumText | None = None
    claims: list[Claim] = Field(default_factory=list)
    conflicts: list[ConflictPair] = Field(default_factory=list)
    unresolved_questions: list[MediumText] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: MediumText | None = None

    @model_validator(mode="after")
    def _check_answer(self) -> "Answer":
        if self.refused:
            if not self.refusal_reason:
                raise ValueError("拒答时必须说明 refusal_reason")
            if self.claims:
                raise ValueError("拒答时不得输出事实断言")
        else:
            if not self.summary:
                raise ValueError("非拒答回答必须提供 summary")
            if self.refusal_reason:
                raise ValueError("非拒答回答不得携带 refusal_reason")
        return self


class SufficiencyJudgment(BaseSchema):
    """证据充分性判断结果。决定"继续检索"还是"生成回答或拒答"。"""

    is_sufficient: bool
    coverage: Annotated[float, Field(ge=0.0, le=1.0)]
    judged_by: JudgedBy
    reasons: Annotated[list[MediumText], Field(min_length=1)]
    """判定理由。必须说明为什么认为充分或不充分，不允许只给一个布尔值。"""

    missing_aspects: list[MediumText] = Field(default_factory=list)
    next_queries: list[MediumText] = Field(default_factory=list)
    can_continue: bool = True
    """是否还有继续检索的余地（受轮次与预算上限约束）。"""

    round_index: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _check_judgment(self) -> "SufficiencyJudgment":
        if not self.is_sufficient and not self.missing_aspects:
            raise ValueError("判定证据不足时必须说明 missing_aspects")
        if self.can_continue and not self.is_sufficient and not self.next_queries:
            raise ValueError("可继续检索但未给出 next_queries，无法进入下一轮")
        return self


class RetrievalRecord(BaseSchema):
    """检索记录。保存检索式、来源、访问时间与结果数量，用于复现与评估。"""

    record_id: RecordId
    query: MediumText
    normalized_query: MediumText | None = None
    target: RetrievalTarget
    source_name: SourceName
    filters: dict[str, Any] = Field(default_factory=dict)
    executed_at: UtcDatetime
    round_index: Annotated[int, Field(ge=1)]
    result_count: Annotated[int, Field(ge=0)]
    kept_count: Annotated[int, Field(ge=0)] | None = None
    duration_ms: Annotated[int, Field(ge=0)] | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _check_record(self) -> "RetrievalRecord":
        if self.kept_count is not None and self.kept_count > self.result_count:
            raise ValueError("kept_count 不得大于 result_count")
        return self


class DownloadRecord(BaseSchema):
    """开放全文下载记录。记录合规检查结果，证明未绕过付费墙。"""

    download_id: DownloadId
    paper_id: PaperId
    url: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    status: DownloadStatus
    checked_at: UtcDatetime
    file_type: ShortText | None = None
    file_size_bytes: Annotated[int, Field(ge=0)] | None = None
    sha256: Sha256 | None = None
    failure_reason: MediumText | None = None

    @model_validator(mode="after")
    def _check_download(self) -> "DownloadRecord":
        if self.status is DownloadStatus.DOWNLOADED:
            if self.sha256 is None or self.file_size_bytes is None:
                raise ValueError("下载成功必须提供 sha256 与 file_size_bytes")
            if self.failure_reason:
                raise ValueError("下载成功不得携带 failure_reason")
        else:
            if not self.failure_reason:
                raise ValueError("未下载成功必须说明 failure_reason")
        return self


class KBVersion(BaseSchema):
    """本地知识库版本。检索结果与证据必须绑定版本，保证可复现。"""

    kb_version: KBVersionId
    built_at: UtcDatetime
    paper_count: Annotated[int, Field(ge=0)]
    chunk_count: Annotated[int, Field(ge=0)]
    parser_name: ShortText
    parser_version: ShortText
    chunk_strategy: ShortText
    chunk_size: Annotated[int, Field(ge=1)] | None = None
    chunk_overlap: Annotated[int, Field(ge=0)] | None = None
    embedding_model: ShortText | None = None
    notes: MediumText | None = None

    @model_validator(mode="after")
    def _check_kb(self) -> "KBVersion":
        if (
            self.chunk_size is not None
            and self.chunk_overlap is not None
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self
