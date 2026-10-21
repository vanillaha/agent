"""适配器层异常与到 M1 ``ErrorCode`` 的映射。

设计原则：

- 异常只携带**已经确定**的事实（错误码、上游响应码、请求地址），
  到 ``ErrorInfo`` 的转换集中在这里做，避免各 adapter 各写一套。
- 失败必须可区分：**"检索失败"与"检索成功但没有结果"绝不能长得一样**。
- 不吞异常。适配器只把 ``SourceError`` 转成结果对象里的 ``error``；
  真正的程序缺陷（``SourceError`` 之外的异常）必须向上抛出，由上层记为
  ``internal_error``，不得伪装成"上游不可用"。
"""

from __future__ import annotations

from typing import Any

from paper_evidence_qa.schemas import ErrorCode, ErrorInfo


class SourceError(RuntimeError):
    """适配器层异常基类。每个子类绑定一个 M1 错误码。"""

    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    def to_error_info(self) -> ErrorInfo:
        """转成对外的结构化错误。"""
        return ErrorInfo(
            code=self.error_code,
            message=self.message[:1000],
            details=self.details,
        )


class OnlineDisabledError(SourceError):
    """未授权联网时仍尝试发起请求。

    注意：正常情况下 ``allow_online=false`` 时根本不会走到 adapter，
    这个异常是**防御性**的——防止上层漏判而悄悄联网。
    """

    error_code = ErrorCode.ONLINE_DISABLED


class RateLimitedError(SourceError):
    """上游限流（HTTP 429）且重试已耗尽，或本地限流器判定不允许发出请求。"""

    error_code = ErrorCode.RATE_LIMITED

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        attempts: int = 0,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(details or {})
        merged["attempts"] = attempts
        if retry_after_seconds is not None:
            merged["retry_after_seconds"] = retry_after_seconds
        super().__init__(message, details=merged)
        self.retry_after_seconds = retry_after_seconds


class UpstreamError(SourceError):
    """上游返回错误状态或网络层失败（超时、连接中断）。"""

    error_code = ErrorCode.UPSTREAM_UNAVAILABLE

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        url: str | None = None,
        attempts: int = 0,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(details or {})
        merged["attempts"] = attempts
        if status_code is not None:
            merged["status_code"] = status_code
        if url is not None:
            merged["url"] = url
        super().__init__(message, details=merged)
        self.status_code = status_code
        self.url = url


class ResponseShapeError(UpstreamError):
    """响应能取到，但结构不符合预期。

    单独成类是为了在排查时能把"上游挂了"与"上游改格式了"分开。
    错误码仍归 ``upstream_unavailable``：对调用方而言两者都意味着
    "这个来源这次拿不到可用数据"。
    """

    def __init__(
        self,
        message: str,
        *,
        where: str,
        url: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(details or {})
        merged["where"] = where
        super().__init__(message, url=url, details=merged)
        self.where = where
