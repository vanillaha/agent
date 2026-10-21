"""解析协议（M5）。

适配器只依赖 ``Parser`` 协议，不依赖具体实现——测试注入假 Parser，
生产环境接入 Docling 适配器（待 Python 3.13 兼容性验证通过后实现）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Parser(Protocol):
    """解析器协议。

    返回 ``(pages, parser_name, parser_version)``，其中 ``pages`` 是
    按页序排列的文本块列表，每页由若干文本块组成（以换行连接）。
    解析失败时抛 ``IngestError``，绝不返回空结果冒充成功。
    """

    def parse(self, path: Path) -> tuple[list[list[str]], str, str]: ...
