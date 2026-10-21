"""Docling 解析适配器测试（M5-6）。

核心约束：测试**不导入 docling**（避免拖入 torch 等重依赖），
通过注入鸭子类型的假 converter 覆盖分组与页码逻辑；
真实 docling 只在 `--parser docling` 的生产路径上惰性加载。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_evidence_qa.ingest.docling_parser import DoclingParser
from paper_evidence_qa.ingest.errors import IngestError
from paper_evidence_qa.schemas.common import ErrorCode


def _item(text: str, page_no: int | None):
    prov = None if page_no is None else SimpleNamespace(page_no=page_no)
    return SimpleNamespace(text=text, prov=[prov] if prov else [])


class FakeConverter:
    """鸭子类型 converter：行为对齐 docling 的 DocumentConverter。"""

    docling_version = "9.9.9-fake"

    def __init__(self, items: list, error: Exception | None = None) -> None:
        self._items = items
        self._error = error
        self.calls: list[str] = []

    def convert(self, source: str):
        self.calls.append(source)
        if self._error:
            raise self._error
        return SimpleNamespace(document=SimpleNamespace(texts=self._items, pages={}))


class TestDoclingParser:
    def test_items_grouped_by_page_in_order(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        items = [
            _item("page one line A", 1),
            _item("page one line B", 1),
            _item("page two line A", 2),
        ]
        parser = DoclingParser(converter=FakeConverter(items))
        pages, name, version = parser.parse(pdf)
        assert name == "docling"
        assert version == "9.9.9-fake"
        assert [p[0] for p in pages] == ["page one line A page one line B", "page two line A"]

    def test_items_without_provenance_are_skipped(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        items = [_item("real text", 1), _item("orphan text", None)]
        parser = DoclingParser(converter=FakeConverter(items))
        pages, _, _ = parser.parse(pdf)
        assert len(pages) == 1
        assert pages[0][0] == "real text"

    def test_missing_middle_page_keeps_page_numbering_continuous(
        self, tmp_path: Path
    ) -> None:
        # 第 2 页无文本（纯图页）：输出仍必须覆盖 1..3 页，页码连续
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        items = [_item("first", 1), _item("third", 3)]
        parser = DoclingParser(converter=FakeConverter(items))
        pages, _, _ = parser.parse(pdf)
        assert len(pages) == 3
        assert "\n".join(pages[1]) == "", "无文本页应产出空文本（页码占位不缺失）"
        assert pages[2][0] == "third"

    def test_conversion_error_maps_to_parse_failed(self, tmp_path: Path) -> None:
        pdf = tmp_path / "broken.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        parser = DoclingParser(
            converter=FakeConverter([], error=RuntimeError("layout boom"))
        )
        with pytest.raises(IngestError) as exc_info:
            parser.parse(pdf)
        assert exc_info.value.code is ErrorCode.PARSE_FAILED

    def test_empty_document_rejected(self, tmp_path: Path) -> None:
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        parser = DoclingParser(converter=FakeConverter([]))
        with pytest.raises(IngestError) as exc_info:
            parser.parse(pdf)
        assert exc_info.value.code is ErrorCode.PARSE_FAILED

    def test_converter_receives_path_as_string(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        converter = FakeConverter([_item("text", 1)])
        parser = DoclingParser(converter=converter)
        parser.parse(pdf)
        assert converter.calls == [str(pdf)], "必须以字符串路径调用 converter"
