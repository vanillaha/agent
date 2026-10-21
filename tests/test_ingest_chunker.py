"""固定长度分块器测试（M5）。

核心验证三件事：
1. 覆盖性：chunk 拼起来能覆盖全文，`full_text[char_start:char_end] == text`。
2. 可追溯：`page_from/page_to` 能映射回真实页码。
3. 失败行为：空文档明确报错，绝不产出空 chunk 冒充成功。
"""

from __future__ import annotations

import pytest

from paper_evidence_qa.ingest.chunker import chunk_document
from paper_evidence_qa.ingest.errors import IngestError
from paper_evidence_qa.ingest.models import IngestOptions, ParsedDocument, ParsedPage
from paper_evidence_qa.schemas.common import ErrorCode


def _make_doc(page_texts: list[str]) -> ParsedDocument:
    pages = [
        ParsedPage(page_no=i + 1, text=text, section_path=["第1章", f"1.{i + 1}"])
        for i, text in enumerate(page_texts)
    ]
    return ParsedDocument(
        pages=pages,
        parser_name="fake",
        parser_version="0.1",
        source_file_sha256="b" * 64,
    )


def _long_text(marker: str, length: int) -> str:
    """生成一段带前缀标记、用空格连接的长文本（模拟自然段落）。"""
    words = [f"{marker}{i:04d}" for i in range(length // 8)]
    return " ".join(words)


class TestCoverage:
    def test_chunks_cover_full_text(self) -> None:
        doc = _make_doc([_long_text("p1", 500), _long_text("p2", 500)])
        chunks = chunk_document(doc, IngestOptions(chunk_size=200, chunk_overlap=50))
        assert chunks, "正常文档必须产出至少一个 chunk"
        assert chunks[0].char_start == 0
        assert chunks[-1].char_end == len(doc.full_text)
        for chunk in chunks:
            assert doc.full_text[chunk.char_start : chunk.char_end] == chunk.text

    def test_adjacent_chunks_overlap_when_configured(self) -> None:
        doc = _make_doc([_long_text("p", 800)])
        chunks = chunk_document(doc, IngestOptions(chunk_size=200, chunk_overlap=50))
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt.char_start < prev.char_end, "配置了 overlap 时相邻 chunk 必须重叠"

    def test_no_overlap_when_overlap_zero(self) -> None:
        doc = _make_doc([_long_text("p", 800)])
        chunks = chunk_document(doc, IngestOptions(chunk_size=200, chunk_overlap=0))
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt.char_start == prev.char_end

    def test_long_paragraph_hard_split(self) -> None:
        # 单个超长无空白段落：找不到自然边界时必须硬切，且稳步推进
        doc = _make_doc(["x" * 1000])
        chunks = chunk_document(doc, IngestOptions(chunk_size=300, chunk_overlap=60))
        assert len(chunks) >= 3
        assert chunks[-1].char_end == len(doc.full_text)


class TestPageMapping:
    def test_chunk_on_page_two_maps_to_page_two(self) -> None:
        marker = "SECONDPAGEMARKER"
        doc = _make_doc([_long_text("p", 400), f"{marker} " + _long_text("q", 400)])
        chunks = chunk_document(doc, IngestOptions(chunk_size=200, chunk_overlap=0))
        hit = [c for c in chunks if marker in c.text]
        assert hit, "标记文本必须出现在某个 chunk 中"
        assert all(c.page_from == 2 for c in hit)

    def test_cross_page_chunk_spans_page_range(self) -> None:
        # overlap>0 且页尾有自然边界时，允许出现合法的跨页 chunk
        doc = _make_doc([_long_text("p", 400), _long_text("q", 400)])
        chunks = chunk_document(doc, IngestOptions(chunk_size=200, chunk_overlap=50))
        for c in chunks:
            assert 1 <= c.page_from <= c.page_to <= len(doc.pages)

    def test_no_cross_page_chunks_without_overlap(self) -> None:
        # overlap=0 时切点落在页间边界，chunk 不应跨页（可追溯性更好）
        doc = _make_doc([_long_text("p", 400), _long_text("q", 400)])
        chunks = chunk_document(doc, IngestOptions(chunk_size=200, chunk_overlap=0))
        assert all(c.page_from == c.page_to for c in chunks)

    def test_no_duplicate_tails_around_page_boundary(self) -> None:
        # 回归测试：页边界附近不得产出一批结尾相同、仅差一两个字符的重复 chunk
        doc = _make_doc([_long_text("p", 400), _long_text("q", 400)])
        chunks = chunk_document(doc, IngestOptions(chunk_size=200, chunk_overlap=50))
        ends = [c.char_end for c in chunks]
        assert len(ends) == len(set(ends)), "chunk 结束偏移不得重复"


class TestFailure:
    def test_blank_document_raises_parse_failed(self) -> None:
        doc = _make_doc(["   ", ""])
        with pytest.raises(IngestError) as exc_info:
            chunk_document(doc, IngestOptions())
        assert exc_info.value.code is ErrorCode.PARSE_FAILED

    def test_single_char_document_yields_one_chunk(self) -> None:
        doc = _make_doc(["a"])
        chunks = chunk_document(doc, IngestOptions(chunk_size=50, chunk_overlap=10))
        assert len(chunks) == 1
        assert chunks[0].text == "a"
