"""契约测试：任务模型与统一信封（M1）。

重点验证：
1. 信封的公共字段与错误一致性规则。
2. 请求参数的硬上限（轮次、下载量、工具调用数）不得被静默截断。
3. 任务视图的状态与产物必须自洽（终态、失败、成功各有必填项）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from factories import LATER, NOW, KB_VERSION, make_answer, make_judgment, make_request
from paper_evidence_qa.schemas import (
    MAX_DOWNLOADS,
    MAX_ROUNDS,
    CancelRequest,
    Envelope,
    ErrorCode,
    ErrorInfo,
    MessageKind,
    PaperQARequest,
    TaskStatus,
    TaskStatusResponse,
    TaskView,
    UserInputMessage,
)


def make_view(**overrides: object) -> TaskView:
    data: dict[str, object] = {
        "task_id": "tsk_0000000000000001",
        "run_id": "run_0000000000000001",
        "status": TaskStatus.RUNNING,
        "created_at": NOW,
        "updated_at": NOW,
        "request": make_request(),
        "round_index": 1,
        "tool_calls_used": 3,
    }
    data.update(overrides)
    return TaskView(**data)  # type: ignore[arg-type]


# ---------------------------------------------------------------- 正常场景


def test_envelope_with_typed_payload_is_valid() -> None:
    envelope: Envelope[dict[str, int]] = Envelope[dict[str, int]](
        request_id="req_0000000000000001",
        task_id="tsk_0000000000000001",
        run_id="run_0000000000000001",
        status=TaskStatus.RUNNING,
        data={"round_index": 1},
    )
    assert envelope.data == {"round_index": 1}
    assert envelope.error is None
    assert envelope.evidence == []


def test_status_response_carries_task_view() -> None:
    response = TaskStatusResponse(
        request_id="req_0000000000000001",
        task_id="tsk_0000000000000001",
        run_id="run_0000000000000001",
        status=TaskStatus.RUNNING,
        data=make_view(),
    )
    assert response.data is not None
    assert response.data.round_index == 1


def test_request_accepts_offline_default() -> None:
    """默认不联网、不下载：未显式授权就不得访问外部数据源。"""
    request = make_request()
    assert request.allow_online is False
    assert request.max_downloads == 0


def test_request_accepts_online_with_download_budget() -> None:
    request = make_request(allow_online=True, max_downloads=3)
    assert request.max_downloads == 3


def test_task_view_succeeded_requires_answer() -> None:
    view = make_view(
        status=TaskStatus.SUCCEEDED,
        finished_at=LATER,
        updated_at=LATER,
        answer=make_answer(),
        judgment=make_judgment(),
    )
    assert view.answer is not None
    assert view.status.is_terminal


def test_task_view_failed_carries_error() -> None:
    view = make_view(
        status=TaskStatus.FAILED,
        finished_at=LATER,
        updated_at=LATER,
        error=ErrorInfo(code=ErrorCode.PARSE_FAILED, message="样例解析失败"),
    )
    assert view.error is not None


def test_user_input_message_and_cancel_request_are_valid() -> None:
    message = UserInputMessage(
        message_id="msg_0000000000000001",
        kind=MessageKind.CONFIRM,
        content="确认下载这 3 篇开放论文",
        at=NOW,
    )
    cancel = CancelRequest(reason="研究问题已变更", requested_by="student-a")
    assert message.kind is MessageKind.CONFIRM
    assert cancel.reason


# ---------------------------------------------------------------- 边界场景


def test_request_rounds_at_upper_bound_is_allowed() -> None:
    request = make_request(max_rounds=MAX_ROUNDS)
    assert request.max_rounds == MAX_ROUNDS


def test_request_rejects_rounds_above_upper_bound() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_request(max_rounds=MAX_ROUNDS + 1)
    assert "max_rounds" in str(excinfo.value)


def test_request_rejects_zero_rounds() -> None:
    with pytest.raises(ValidationError):
        make_request(max_rounds=0)


def test_request_rejects_downloads_above_upper_bound() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_request(allow_online=True, max_downloads=MAX_DOWNLOADS + 1)
    assert "max_downloads" in str(excinfo.value)


def test_request_rejects_download_without_online_permission() -> None:
    """允许下载却不允许联网属于自相矛盾，必须报错而不是静默降级。"""
    with pytest.raises(ValidationError) as excinfo:
        make_request(allow_online=False, max_downloads=5)
    assert "allow_online" in str(excinfo.value)


def test_request_rejects_reversed_years() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_request(year_from=2025, year_to=2020)
    assert "year_from" in str(excinfo.value)


def test_request_rejects_unknown_doc_type() -> None:
    with pytest.raises(ValidationError):
        make_request(doc_types=["white_paper"])


def test_request_accepts_unknown_kb_version_identifier_format() -> None:
    """知识库版本号由 M5 分配，这里只校验格式，不校验是否存在。"""
    request = make_request(kb_version=KB_VERSION)
    assert request.kb_version == KB_VERSION


# ---------------------------------------------------------------- 失败场景


def test_envelope_failed_status_requires_error() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Envelope[dict[str, int]](
            request_id="req_0000000000000001",
            task_id="tsk_0000000000000001",
            run_id="run_0000000000000001",
            status=TaskStatus.FAILED,
            data={"round_index": 1},
        )
    assert "error" in str(excinfo.value)


def test_envelope_non_failed_status_forbids_error() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Envelope[dict[str, int]](
            request_id="req_0000000000000001",
            task_id="tsk_0000000000000001",
            run_id="run_0000000000000001",
            status=TaskStatus.RUNNING,
            error=ErrorInfo(code=ErrorCode.INTERNAL_ERROR, message="不应携带"),
        )
    assert "error" in str(excinfo.value)


def test_envelope_rejects_unknown_common_field_name() -> None:
    """公共字段名固定，拼错或改名必须报错。"""
    with pytest.raises(ValidationError) as excinfo:
        Envelope[dict[str, int]](
            request_id="req_0000000000000001",
            task_id="tsk_0000000000000001",
            run_id="run_0000000000000001",
            status=TaskStatus.RUNNING,
            evidence_list=[],
        )
    assert "evidence_list" in str(excinfo.value)


def test_task_view_terminal_without_finished_at_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_view(status=TaskStatus.CANCELLED, answer=None)
    assert "finished_at" in str(excinfo.value)


def test_task_view_active_with_finished_at_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_view(status=TaskStatus.RUNNING, finished_at=LATER, updated_at=LATER)


def test_task_view_succeeded_without_answer_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_view(status=TaskStatus.SUCCEEDED, finished_at=LATER, updated_at=LATER)
    assert "answer" in str(excinfo.value)


def test_task_view_updated_before_created_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_view(created_at=LATER, updated_at=NOW)


def test_task_view_failed_without_error_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_view(status=TaskStatus.FAILED, finished_at=LATER, updated_at=LATER)


def test_request_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        make_request(temperature=0.7)


# ---------------------------------------------------------------- 安全场景


def test_request_type_is_stable_for_api_contract() -> None:
    assert isinstance(make_request(), PaperQARequest)
    assert set(PaperQARequest.model_fields) >= {
        "request_id",
        "question",
        "allow_online",
        "max_rounds",
        "max_downloads",
        "max_tool_calls",
    }
