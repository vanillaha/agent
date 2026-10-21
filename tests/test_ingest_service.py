"""入库编排 IngestService 测试（M5）。

用假 Parser（协议注入）离线验证编排行为：
成功组装、失败隔离、确定性 chunk_id、KBVersion 参数回填。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paper_evidence_qa.ingest.errors import IngestError
from paper_evidence_qa.ingest.models import IngestOptions, IngestRequest
from paper_evidence_qa.ingest.parser import Parser
from paper_evidence_qa.ingest.service import IngestService
from paper_evidence_qa.schemas.common import ErrorCode


class FakeParser:
    """假 Parser：按文件名返回固定解析结果；fail_names 中的文件抛解析错误。"""

    def __init__(
        self,
        pages_by_name: dict[str, list[list[str]]],
        fail_names: set[str] | None = None,
    ) -> None:
        self._pages_by_name = pages_by_name
        self._fail_names = fail_names or set()
        self.calls: list[Path] = []

    def parse(self, path: Path) -> tuple[list[list[str]], str, str]:
        self.calls.append(path)
        if path.name in self._fail_names:
            raise IngestError(
                ErrorCode.PARSE_FAILED, "无可提取文本", {"file": path.name}
            )
        pages = self._pages_by_name.get(path.name, ["第1段内容 " * 20])
        return pages, "fake", "0.1"


class BrokenParser:
    """始终抛解析错误的假 Parser，用于失败路径。"""

    def parse(self, path: Path) -> tuple[list[list[str]], str, str]:
        raise IngestError(
            ErrorCode.PARSE_FAILED, "无可提取文本", {"file": path.name}
        )


def _request(names: list[str], tmp_path: Path) -> IngestRequest:
    return IngestRequest(
        files=[tmp_path / name for name in names],
        options=IngestOptions(chunk_size=200, chunk_overlap=50),
    )


def _touch(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4 fake")
    return path


class TestSuccess:
    def test_ingest_produces_chunks_and_kb_version(self, tmp_path: Path) -> None:
        _touch(tmp_path, "a.pdf")
        service = IngestService(parser=FakeParser({"a.pdf": [["第一页 " * 30], ["第二页 " * 30]]}))
        result = service.process(_request(["a.pdf"], tmp_path))

        assert len(result.papers) == 1
        assert result.failures == []
        paper = result.papers[0]
        assert paper.chunk_count > 0
        assert result.kb_version.paper_count == 1
        assert result.kb_version.chunk_count == paper.chunk_count
        assert result.kb_version.chunk_strategy == "fixed-v1"
        assert result.kb_version.chunk_size == 200
        assert result.kb_version.chunk_overlap == 50

    def test_chunk_ids_deterministic_across_reingest(self, tmp_path: Path) -> None:
        _touch(tmp_path, "a.pdf")
        parser = FakeParser({"a.pdf": [["内容 " * 40]]})
        first = IngestService(parser=parser).process(_request(["a.pdf"], tmp_path))
        second = IngestService(parser=parser).process(_request(["a.pdf"], tmp_path))

        ids_first = {c.chunk_id for c in first.chunks}
        ids_second = {c.chunk_id for c in second.chunks}
        assert ids_first == ids_second
        assert len(ids_first) == len(first.chunks)

    def test_chunks_carry_locator_and_sha(self, tmp_path: Path) -> None:
        _touch(tmp_path, "a.pdf")
        service = IngestService(parser=FakeParser({"a.pdf": [["正文 " * 40]]}))
        result = service.process(_request(["a.pdf"], tmp_path))
        for chunk in result.chunks:
            assert chunk.locator.page_from is not None
            assert chunk.source_file_sha256 == result.papers[0].source_file_sha256
            assert chunk.paper_id == result.papers[0].paper_id


class TestFailureIsolation:
    def test_missing_file_isolated_as_failure(self, tmp_path: Path) -> None:
        _touch(tmp_path, "a.pdf")
        service = IngestService(parser=FakeParser({}))
        result = service.process(_request(["a.pdf", "missing.pdf"], tmp_path))

        assert len(result.papers) == 1
        assert len(result.failures) == 1
        assert result.failures[0].code is ErrorCode.FILE_NOT_FOUND

    def test_non_pdf_extension_rejected(self, tmp_path: Path) -> None:
        _touch(tmp_path, "a.pdf")
        _touch(tmp_path, "notes.txt")
        service = IngestService(parser=FakeParser({}))
        result = service.process(_request(["a.pdf", "notes.txt"], tmp_path))
        assert len(result.papers) == 1
        assert result.failures[0].code is ErrorCode.UNSUPPORTED_FILE_TYPE

    def test_parser_error_isolated_as_failure(self, tmp_path: Path) -> None:
        _touch(tmp_path, "a.pdf")
        _touch(tmp_path, "scan.pdf")
        service = IngestService(parser=FakeParser({}, fail_names={"scan.pdf"}))
        result = service.process(_request(["a.pdf", "scan.pdf"], tmp_path))
        assert len(result.papers) == 1
        assert result.failures[0].code is ErrorCode.PARSE_FAILED

    def test_all_files_failed_raises(self, tmp_path: Path) -> None:
        service = IngestService(parser=FakeParser({}))
        with pytest.raises(IngestError) as exc_info:
            service.process(_request(["missing.pdf"], tmp_path))
        assert exc_info.value.code is ErrorCode.FILE_NOT_FOUND
