"""任务模型（M1）：统一信封、任务请求、任务视图、状态响应、取消与人工输入消息。

``Envelope`` 是**唯一**的对外响应外壳：无论接口返回什么负载，都必须套在信封里，
这样错误、证据、数据集、产物、告警和事件的位置永远固定，前端与评估脚本不需要分情况处理。
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Generic, TypeVar

from pydantic import Field, model_validator

from paper_evidence_qa.schemas.common import (
    Artifact,
    BaseSchema,
    DatasetRef,
    ErrorInfo,
    Event,
    KBVersionId,
    MessageId,
    MediumText,
    RequestId,
    RunId,
    ShortText,
    TaskId,
    TaskStatus,
    UtcDatetime,
    WarningItem,
)
from paper_evidence_qa.schemas.domain import (
    Answer,
    DocType,
    Evidence,
    SufficiencyJudgment,
)

MAX_ROUNDS = 5
"""最大检索轮次上限。超过即停止并依据现有证据作答或拒答。"""

MAX_DOWNLOADS = 50
"""单次任务最大下载量上限。"""

MAX_TOOL_CALLS = 200
"""单次任务工具调用次数上限。"""

MAX_LOCAL_PDF_PATHS = 200
"""单次任务允许指定的本地 PDF 数量上限。"""


DataT = TypeVar("DataT")


class Envelope(BaseSchema, Generic[DataT]):
    """统一响应信封。公共字段的位置与名称固定，不得改名。"""

    request_id: RequestId
    task_id: TaskId
    run_id: RunId
    status: TaskStatus
    data: DataT | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    datasets: list[DatasetRef] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    warnings: list[WarningItem] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _check_error_consistency(self) -> "Envelope[DataT]":
        if self.status is TaskStatus.FAILED and self.error is None:
            raise ValueError("status=failed 时必须提供 error")
        if self.status is not TaskStatus.FAILED and self.error is not None:
            raise ValueError("非 failed 状态不得携带 error")
        return self


class PaperQARequest(BaseSchema):
    """创建问答任务的请求体。

    ``allow_online`` 默认为 False：未显式授权不得联网检索或下载。
    """

    request_id: RequestId
    requested_at: UtcDatetime
    question: MediumText
    keywords: list[ShortText] = Field(default_factory=list)
    domain: ShortText | None = None
    doc_types: list[DocType] = Field(default_factory=list)
    year_from: Annotated[int, Field(ge=1800, le=2200)] | None = None
    year_to: Annotated[int, Field(ge=1800, le=2200)] | None = None
    local_pdf_paths: Annotated[
        list[ShortText], Field(max_length=MAX_LOCAL_PDF_PATHS)
    ] = Field(default_factory=list)
    kb_version: KBVersionId | None = None
    allow_online: bool = False
    max_rounds: Annotated[int, Field(ge=1, le=MAX_ROUNDS)] = 3
    max_downloads: Annotated[int, Field(ge=0, le=MAX_DOWNLOADS)] = 5
    max_tool_calls: Annotated[int, Field(ge=1, le=MAX_TOOL_CALLS)] = 30
    notes: MediumText | None = None

    @model_validator(mode="after")
    def _check_request(self) -> "PaperQARequest":
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from 不得大于 year_to")
        if self.max_downloads > 0 and not self.allow_online:
            raise ValueError("允许下载时必须同时允许联网（allow_online=true）")
        return self


class TaskView(BaseSchema):
    """任务的可查询视图。用于 ``GET /api/v1/paper-qa-tasks/{task_id}``。"""

    task_id: TaskId
    run_id: RunId
    status: TaskStatus
    created_at: UtcDatetime
    updated_at: UtcDatetime
    finished_at: UtcDatetime | None = None
    request: PaperQARequest
    round_index: Annotated[int, Field(ge=0)] = 0
    tool_calls_used: Annotated[int, Field(ge=0)] = 0
    judgment: SufficiencyJudgment | None = None
    answer: Answer | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _check_view(self) -> "TaskView":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at 不得早于 created_at")
        if self.status.is_terminal:
            if self.finished_at is None:
                raise ValueError("终态任务必须提供 finished_at")
        elif self.finished_at is not None:
            raise ValueError("非终态任务不得提供 finished_at")
        if self.status is TaskStatus.SUCCEEDED and self.answer is None:
            raise ValueError("status=succeeded 时必须提供 answer（拒答也是一种回答）")
        if self.status is TaskStatus.FAILED and self.error is None:
            raise ValueError("status=failed 时必须提供 error")
        return self


class TaskStatusResponse(Envelope[TaskView]):
    """任务状态与结果查询响应。"""


class MessageKind(str, Enum):
    """人工输入类型。用于 ``waiting_for_input`` 状态下的确认与补充。"""

    CONFIRM = "confirm"
    REJECT = "reject"
    ANSWER = "answer"
    CANCEL = "cancel"


class UserInputMessage(BaseSchema):
    """人工输入消息。下载、写入等操作在获得确认前不得执行。"""

    message_id: MessageId
    kind: MessageKind
    content: MediumText
    at: UtcDatetime


class CancelRequest(BaseSchema):
    """取消请求。取消是请求而非立即生效，实际状态先转为 ``cancel_requested``。"""

    reason: MediumText | None = None
    requested_by: ShortText | None = None
