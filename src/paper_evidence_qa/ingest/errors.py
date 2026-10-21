"""ingest 层异常。

与 ``sources/errors.py`` 同一设计原则：异常只携带已经确定的事实，
到 ``ErrorInfo`` 的转换集中在服务层做。失败必须可区分、不吞异常。
"""

from __future__ import annotations

from paper_evidence_qa.schemas.common import ErrorCode


class IngestError(Exception):
    """入库链路（解析/分块/编排）的确定性失败。"""

    def __init__(self, code: ErrorCode, message: str, details: dict | None = None) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)
