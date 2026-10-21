"""pypdf 文本层解析器（M5 第一版真实现）。

定位：轻量、零模型、离线可用的文本层提取器，用于样例与常规电子版 PDF。
复杂版式（双栏、公式、表格密集）由 Docling 适配器接管（下一步接入），
两者都实现 ``Parser`` 协议，可按文件切换。
"""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from paper_evidence_qa.ingest.errors import IngestError
from paper_evidence_qa.schemas.common import ErrorCode

PARSER_NAME = "pypdf"


class PypdfParser:
    """实现 ``Parser`` 协议：每页产出一个文本块，页码即页序。"""

    def __init__(self) -> None:
        import pypdf

        self._version = pypdf.__version__

    def parse(self, path: Path) -> tuple[list[list[str]], str, str]:
        try:
            data = path.read_bytes()
            if not data.startswith(b"%PDF"):
                raise IngestError(
                    ErrorCode.UNSUPPORTED_FILE_TYPE,
                    "文件缺少 PDF 魔数（%PDF），不是有效的 PDF 文件",
                    {"file": path.name},
                )
            reader = PdfReader(io.BytesIO(data))
            pages: list[list[str]] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append([text])
        except IngestError:
            raise
        except PdfReadError as error:
            raise IngestError(
                ErrorCode.PARSE_FAILED, "PDF 结构损坏，无法解析", {"detail": str(error)}
            ) from error
        except Exception as error:  # noqa: BLE001 — 解析层兜底为明确错误，不静默
            raise IngestError(
                ErrorCode.PARSE_FAILED, "文本提取失败", {"detail": str(error)}
            ) from error
        return pages, PARSER_NAME, self._version
