"""ingest 契约模型测试（M5）。

覆盖正常、边界、失败、安全四类场景。全部离线。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from paper_evidence_qa.ingest.models import (
    IngestOptions,
    IngestRequest,
    ParsedDocument,
    ParsedPage,
)


def _page(page_no: int, text: str) -> ParsedPage:
    return ParsedPage(page_no=page_no, text=text, section_path=["第1章"])


class TestParsedDocument:
    def test_normal_multi_page(self) -> None:
        doc = ParsedDocument(
            pages=[_page(1, "第一页内容"), _page(2, "第二页内容")],
            parser_name="fake",
            parser_version="0.1",
            source_file_sha256="a" * 64,
        )
        assert doc.full_text == "第一页内容\n\n第二页内容"

    def test_empty_pages_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ParsedDocument(
                pages=[],
                parser_name="fake",
                parser_version="0.1",
                source_file_sha256="a" * 64,
            )

    def test_page_numbers_must_be_sequential_from_one(self) -> None:
        with pytest.raises(ValidationError):
            ParsedDocument(
                pages=[_page(1, "a"), _page(3, "b")],
                parser_name="fake",
                parser_version="0.1",
                source_file_sha256="a" * 64,
            )

    def test_sha256_must_be_64_hex(self) -> None:
        with pytest.raises(ValidationError):
            ParsedDocument(
                pages=[_page(1, "a")],
                parser_name="fake",
                parser_version="0.1",
                source_file_sha256="not-a-hash",
            )


class TestIngestOptions:
    def test_defaults_are_valid(self) -> None:
        options = IngestOptions()
        assert options.chunk_size == 1200
        assert options.chunk_overlap == 150

    def test_overlap_must_be_smaller_than_size(self) -> None:
        with pytest.raises(ValidationError):
            IngestOptions(chunk_size=200, chunk_overlap=200)

    def test_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            IngestOptions(chunk_size=0)


class TestIngestRequest:
    def test_requires_at_least_one_file(self) -> None:
        with pytest.raises(ValidationError):
            IngestRequest(files=[])

    def test_accepts_paths_with_default_options(self) -> None:
        request = IngestRequest(files=[Path("a.pdf")])
        assert request.options.chunk_size == 1200
