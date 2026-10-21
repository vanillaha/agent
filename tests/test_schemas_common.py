"""契约测试：公共模型（M1）。

覆盖正常、边界、失败与安全四类场景。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from paper_evidence_qa.schemas import (
    Artifact,
    DatasetRef,
    ErrorCode,
    ErrorInfo,
    Event,
    TaskStatus,
    WarningItem,
)

NOW = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
SHA = "b" * 64


# ---------------------------------------------------------------- 正常场景


def test_error_info_accepts_defined_code() -> None:
    error = ErrorInfo(
        code=ErrorCode.NOT_OPEN_ACCESS,
        message="该文献不提供开放全文，仅保存元数据与链接",
        details={"paper_id": "ppr_0000000000000001"},
    )
    assert error.code is ErrorCode.NOT_OPEN_ACCESS
    assert error.details["paper_id"] == "ppr_0000000000000001"


def test_event_accepts_utc_datetime() -> None:
    event = Event(
        event_id="evt_0000000000000001",
        at=NOW,
        type="retrieval.local.finished",
        message="本地检索完成",
        data={"result_count": 5},
    )
    assert event.at == NOW


def test_warning_keeps_machine_readable_code() -> None:
    warning = WarningItem(code="low_coverage", message="本地库覆盖率不足")
    assert warning.code == "low_coverage"


def test_artifact_and_dataset_ref_accept_valid_payload() -> None:
    artifact = Artifact(
        artifact_id="art_0000000000000001",
        artifact_type="paper_list_csv",
        name="文献清单.csv",
        uri="artifacts/paper_list.csv",
        media_type="text/csv",
        size_bytes=1024,
        sha256=SHA,
        created_at=NOW,
    )
    dataset = DatasetRef(
        dataset_id="dst_0000000000000001",
        version="v1",
        row_count=20,
        sha256=SHA,
    )
    assert artifact.size_bytes == 1024
    assert dataset.version == "v1"


# ---------------------------------------------------------------- 边界场景


def test_event_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Event(
            event_id="evt_0000000000000001",
            at=datetime(2026, 10, 11, 12, 0, 0),
            type="task.created",
            message="任务创建",
        )
    assert "时区" in str(excinfo.value)


def test_error_message_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        ErrorInfo(code=ErrorCode.INTERNAL_ERROR, message="   ")


def test_dataset_row_count_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        DatasetRef(dataset_id="dst_0000000000000001", version="v1", row_count=-1)


def test_dataset_sha256_must_be_full_digest() -> None:
    with pytest.raises(ValidationError):
        DatasetRef(dataset_id="dst_0000000000000001", version="v1", sha256="abc")


# ---------------------------------------------------------------- 失败场景


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ErrorInfo(
            code=ErrorCode.INTERNAL_ERROR,
            message="内部错误",
            unexpected_field="不应被静默忽略",
        )
    assert "unexpected_field" in str(excinfo.value)


def test_unknown_error_code_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ErrorInfo(code="something_else", message="未定义错误码")


def test_invalid_event_type_pattern_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Event(
            event_id="evt_0000000000000001",
            at=NOW,
            type="Retrieval Finished",
            message="事件类型必须是小写点分格式",
        )


def test_invalid_identifier_prefix_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Event(
            event_id="event-1",
            at=NOW,
            type="task.created",
            message="标识必须带固定前缀",
        )


# ---------------------------------------------------------------- 安全与规范一致性


def test_task_status_matches_course_specification() -> None:
    """状态取值必须与课程规范完全一致，不得另设同义状态。"""
    expected = {
        "queued",
        "running",
        "waiting_for_input",
        "cancel_requested",
        "succeeded",
        "failed",
        "cancelled",
    }
    assert {status.value for status in TaskStatus} == expected


def test_terminal_and_active_status_partition() -> None:
    terminal = {status for status in TaskStatus if status.is_terminal}
    active = {status for status in TaskStatus if status.is_active}
    assert terminal.isdisjoint(active)
    assert terminal | active == set(TaskStatus)
    assert terminal == {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }
