"""pypdf 真解析器测试（M5）。

使用 fixtures/pdfs/sample_paper.pdf（由 make_sample_pdf.py 生成的
脱敏 3 页样例），全程离线验证"真实 PDF → 页码文本"路径。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paper_evidence_qa.ingest.errors import IngestError
from paper_evidence_qa.ingest.models import IngestOptions, IngestRequest
from paper_evidence_qa.ingest.pypdf_parser import PypdfParser
from paper_evidence_qa.ingest.service import IngestService
from paper_evidence_qa.schemas.common import ErrorCode

SAMPLE_PDF = Path(__file__).parent.parent / "fixtures" / "pdfs" / "sample_paper.pdf"


class TestPypdfParser:
    def test_sample_pdf_yields_three_pages(self) -> None:
        pages, name, version = PypdfParser().parse(SAMPLE_PDF)
        assert len(pages) == 3
        assert name == "pypdf"
        assert version
        for blocks in pages:
            assert any(block.strip() for block in blocks), "每页都应有可提取文本"

    def test_page_texts_are_ordered(self) -> None:
        pages, _, _ = PypdfParser().parse(SAMPLE_PDF)
        first_page = "\n".join(pages[0])
        second_page = "\n".join(pages[1])
        assert "Sample Paper" in first_page
        assert "2. Method" in second_page

    def test_non_pdf_magic_rejected(self, tmp_path: Path) -> None:
        fake = tmp_path / "fake.pdf"
        fake.write_bytes(b"not a pdf at all")
        with pytest.raises(IngestError) as exc_info:
            PypdfParser().parse(fake)
        assert exc_info.value.code is ErrorCode.UNSUPPORTED_FILE_TYPE


class TestEndToEndWithRealPdf:
    def test_ingest_real_sample_pdf(self) -> None:
        service = IngestService(parser=PypdfParser())
        result = service.process(
            IngestRequest(
                files=[SAMPLE_PDF],
                options=IngestOptions(chunk_size=600, chunk_overlap=100),
            )
        )
        assert result.failures == []
        paper = result.papers[0]
        assert paper.page_count == 3
        assert paper.chunk_count >= 2
        assert result.kb_version.parser_name == "pypdf"
        assert result.kb_version.chunk_size == 600

        full_check = True
        for chunk in result.chunks:
            assert chunk.locator.page_from is not None
            assert chunk.source_file_sha256 == paper.source_file_sha256
        assert full_check

    def test_ingest_is_reproducible(self) -> None:
        service = IngestService(parser=PypdfParser())
        request = IngestRequest(files=[SAMPLE_PDF], options=IngestOptions(chunk_size=600, chunk_overlap=100))
        first = service.process(request)
        second = service.process(request)
        assert {c.chunk_id for c in first.chunks} == {c.chunk_id for c in second.chunks}
