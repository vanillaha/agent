"""公共模型（M1）：稳定标识、任务状态、错误、告警、事件、产物与数据集引用。

约定：
- 所有模型禁止未知字段（``extra="forbid"``），字段拼写错误必须报错，不得被静默忽略。
- 所有时间字段必须带时区，统一使用 UTC。
- 标识符使用带前缀的稳定格式，不在对外 JSON 中暴露数据库自增主键。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, TypeAlias

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints


class BaseSchema(BaseModel):
    """项目内所有模型的基类。"""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


def _require_tz(value: datetime) -> datetime:
    """时间字段必须带时区，避免本地时间与 UTC 混用导致评估与检索记录错乱。"""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("时间字段必须带时区，统一使用 UTC")
    return value


UtcDatetime = Annotated[datetime, AfterValidator(_require_tz)]


def _id_pattern(prefix: str) -> str:
    """生成 ``<prefix>_...`` 形式的标识正则，例如 ``ppr_0000000000000001``。

    统一前缀便于日志、轨迹和评估记录中快速识别对象类型。
    """
    return rf"^{prefix}_[A-Za-z0-9][A-Za-z0-9_-]{{2,63}}$"


# 标识类型别名必须用显式 TypeAlias 声明：静态检查器不执行函数调用，
# 通过函数返回值生成的"类型"在 PyCharm/mypy 眼里只是 Any，
# 用在 list[EvidenceId] 这类泛型参数里会被误判为"形参必须为类型"。
RequestId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("req"))]
TaskId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("tsk"))]
RunId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("run"))]
MessageId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("msg"))]
EventId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("evt"))]
ArtifactId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("art"))]
DatasetId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("dst"))]
PaperId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("ppr"))]
ChunkId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("chk"))]
EvidenceId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("evd"))]
ClaimId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("clm"))]
RecordId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("rcd"))]
DownloadId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("dl"))]
VersionGroupId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("vgr"))]
KBVersionId: TypeAlias = Annotated[str, StringConstraints(pattern=_id_pattern("kb"))]

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
"""文件内容摘要，用于版本绑定与产物校验。"""

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=256)]
"""短文本字段：名称、标签、单句说明。"""

MediumText = Annotated[str, StringConstraints(min_length=1, max_length=2000)]
"""中等文本字段：研究问题、检索式、断言文本。"""

LongText = Annotated[str, StringConstraints(min_length=1, max_length=4000)]
"""长文本字段：原文引用片段、回答正文。"""

CodeName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{2,63}$")]
"""错误码、告警码等机器可读的分类名。"""

EventType = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_.]{2,63}$")]
"""事件类型，允许点号分层，例如 ``retrieval.local.finished``。"""


class TaskStatus(str, Enum):
    """任务状态。取值与课程规范完全一致，不得另设同义状态。"""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """是否为终态（终态任务不得再推进）。"""
        return self in _TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        """是否为进行中的状态（可以继续推进或请求取消）。"""
        return not self.is_terminal


_TERMINAL_STATUSES = frozenset(
    {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)


class ErrorCode(str, Enum):
    """错误码。接口失败必须返回明确错误，不编造替代结果。"""

    INVALID_REQUEST = "invalid_request"
    FILE_NOT_FOUND = "file_not_found"
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    LOCAL_KB_UNAVAILABLE = "local_kb_unavailable"
    ONLINE_DISABLED = "online_disabled"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    RATE_LIMITED = "rate_limited"
    NOT_OPEN_ACCESS = "not_open_access"
    DOWNLOAD_FAILED = "download_failed"
    PARSE_FAILED = "parse_failed"
    INDEX_FAILED = "index_failed"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    BUDGET_EXCEEDED = "budget_exceeded"
    TOOL_CALL_LIMIT_EXCEEDED = "tool_call_limit_exceeded"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


class ErrorInfo(BaseSchema):
    """结构化错误。``code`` 供程序判断，``message`` 供人阅读，``details`` 供排查。"""

    code: ErrorCode
    message: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    details: dict[str, Any] = Field(default_factory=dict)


class WarningItem(BaseSchema):
    """非致命的告警。例如数据缺失、覆盖率不足，但任务仍可产出结果。"""

    code: CodeName
    message: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    details: dict[str, Any] = Field(default_factory=dict)


class Event(BaseSchema):
    """事件流中的一条记录，用于回放与失败定位。"""

    event_id: EventId
    at: UtcDatetime
    type: EventType
    message: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    data: dict[str, Any] = Field(default_factory=dict)


class DatasetRef(BaseSchema):
    """数据集引用。评估与检索记录必须能指回具体的数据集版本。"""

    dataset_id: DatasetId
    version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    row_count: Annotated[int, Field(ge=0)] | None = None
    uri: Annotated[str, StringConstraints(min_length=1, max_length=2048)] | None = None
    sha256: Sha256 | None = None


class Artifact(BaseSchema):
    """产物。例如文献清单、回答报告、知识库快照。"""

    artifact_id: ArtifactId
    artifact_type: CodeName
    name: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    uri: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    media_type: Annotated[str, StringConstraints(min_length=3, max_length=128)]
    size_bytes: Annotated[int, Field(ge=0)]
    sha256: Sha256 | None = None
    created_at: UtcDatetime
