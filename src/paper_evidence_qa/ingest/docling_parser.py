"""Docling 解析适配器（M5-6）。

定位：复杂版式（双栏、公式、表格密集）的解析器，实现 ``Parser`` 协议。
docling 依赖 torch 等重型组件，因此**惰性导入**：注入 converter（测试）
或显式使用 ``--parser docling`` 时才真正加载；未安装时给出明确错误。

页码策略：按每个文本项的 provenance（``item.prov[0].page_no``）归组；
无归属信息的项跳过；整页无文本（纯图页）输出空文本占位，保证页码
从 1 开始连续——这是 ``ParsedDocument`` 契约与页码可追溯的基础。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from paper_evidence_qa.ingest.errors import IngestError
from paper_evidence_qa.ingest.textnorm import normalize_text
from paper_evidence_qa.schemas.common import ErrorCode

PARSER_NAME = "docling"


class DoclingParser:
    """实现 ``Parser`` 协议。``converter`` 供测试注入鸭子类型假实现。"""

    def __init__(self, converter=None) -> None:  # noqa: ANN001
        self._converter = converter
        self._version = getattr(converter, "docling_version", None) if converter else None

    def _ensure_converter(self) -> None:
        if self._converter is not None:
            return
        try:
            import docling
            from docling.document_converter import DocumentConverter
        except ImportError as error:
            raise IngestError(
                ErrorCode.PARSE_FAILED,
                "docling 未安装，无法解析（可改用 --parser pypdf）",
                {"detail": str(error)},
            ) from error
        self._converter = DocumentConverter()
        self._version = docling.__version__

    def parse(self, path: Path) -> tuple[list[list[str]], str, str]:
        self._ensure_converter()
        try:
            result = self._converter.convert(str(path))
        except IngestError:
            raise
        except Exception as error:  # noqa: BLE001 — 上游异常统一映射为明确错误码
            raise IngestError(
                ErrorCode.PARSE_FAILED, "Docling 解析失败", {"detail": str(error)}
            ) from error

        items = getattr(getattr(result, "document", None), "texts", None)
        if items is None:
            raise IngestError(
                ErrorCode.PARSE_FAILED, "Docling 返回结果缺少 document.texts"
            )

        by_page: dict[int, list[str]] = defaultdict(list)
        for item in items:
            prov = getattr(item, "prov", None) or []
            page_no = getattr(prov[0], "page_no", None) if prov else None
            text = getattr(item, "text", "")
            if page_no is None or not text.strip():
                continue
            by_page[int(page_no)].append(text.strip())

        if not by_page:
            raise IngestError(
                ErrorCode.PARSE_FAILED, "Docling 未提取到任何带页码归属的文本"
            )

        total = max(by_page)
        pages: list[list[str]] = []
        for page_no in range(1, total + 1):
            lines = by_page.get(page_no, [])
            text = normalize_text("\n".join(lines))
            pages.append([text] if text else [])
        return pages, PARSER_NAME, self._version or "unknown"
