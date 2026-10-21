"""SQLite 知识库存储测试（M5-5）。

存储是"可复现"承诺的落点：入库结果必须能原样读回（chunk 逐字段相等）；
重复入库同一文件不得产生重复行；不同参数的知识库版本互不干扰。
全程使用临时库文件，不碰真实数据目录。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paper_evidence_qa.ingest.models import IngestOptions, IngestRequest
from paper_evidence_qa.ingest.pypdf_parser import PypdfParser
from paper_evidence_qa.ingest.service import IngestService
from paper_evidence_qa.storage.kb_store import KBStore

SAMPLE_PDF = Path(__file__).parent.parent / "fixtures" / "pdfs" / "sample_paper.pdf"


def _ingest(tmp_path: Path, chunk_size: int = 600):
    service = IngestService(parser=PypdfParser())
    request = IngestRequest(
        files=[SAMPLE_PDF],
        options=IngestOptions(chunk_size=chunk_size, chunk_overlap=100),
    )
    return service.process(request)


class TestSaveAndLoad:
    def test_save_then_load_chunks_roundtrip(self, tmp_path: Path) -> None:
        result = _ingest(tmp_path)
        store = KBStore(tmp_path / "kb.sqlite3")
        store.save_ingest(result)

        kb, chunks = store.load_kb(result.kb_version.kb_version)
        assert kb.kb_version == result.kb_version.kb_version
        assert kb.chunk_strategy == "fixed-v1"
        assert kb.chunk_size == 600
        assert len(chunks) == len(result.chunks)
        for saved, expected in zip(chunks, result.chunks, strict=True):
            assert saved.model_dump() == expected.model_dump()

    def test_duplicate_ingest_creates_no_duplicate_rows(self, tmp_path: Path) -> None:
        result_a = _ingest(tmp_path)
        result_b = _ingest(tmp_path)  # 同文件同参数再入库一次
        store = KBStore(tmp_path / "kb.sqlite3")
        store.save_ingest(result_a)
        store.save_ingest(result_b)

        _, chunks = store.load_kb(result_a.kb_version.kb_version)
        assert len(chunks) == len(result_a.chunks), "chunk 行数必须等于单次入库产出"
        assert len(store.list_kb_versions()) == 1
        papers = store.list_papers()
        assert len(papers) == 1, "同一 SHA256 的文件只允许一行 paper 记录"

    def test_different_chunk_size_yields_independent_kb_versions(
        self, tmp_path: Path
    ) -> None:
        small = _ingest(tmp_path, chunk_size=400)
        large = _ingest(tmp_path, chunk_size=1200)
        store = KBStore(tmp_path / "kb.sqlite3")
        store.save_ingest(small)
        store.save_ingest(large)

        assert len(store.list_kb_versions()) == 2
        _, chunks_small = store.load_kb(small.kb_version.kb_version)
        _, chunks_large = store.load_kb(large.kb_version.kb_version)
        small_ids = {c.chunk_id for c in chunks_small}
        large_ids = {c.chunk_id for c in chunks_large}
        assert small_ids and large_ids
        assert small_ids.isdisjoint(large_ids), (
            "不同分块参数必须产生不同 chunk_id（否则跨版本主键冲突）"
        )
        # paper_id 相同：同一篇论文跨版本共享身份
        assert {c.paper_id for c in chunks_small} == {c.paper_id for c in chunks_large}

    def test_paper_lookup_by_sha256(self, tmp_path: Path) -> None:
        result = _ingest(tmp_path)
        store = KBStore(tmp_path / "kb.sqlite3")
        store.save_ingest(result)

        paper = store.get_paper(result.papers[0].source_file_sha256)
        assert paper is not None
        assert paper["paper_id"] == result.papers[0].paper_id
        assert paper["page_count"] == result.papers[0].page_count
        assert store.get_paper("f" * 64) is None

    def test_load_missing_kb_raises_key_error(self, tmp_path: Path) -> None:
        store = KBStore(tmp_path / "kb.sqlite3")
        with pytest.raises(KeyError):
            store.load_kb("kb_does_not_exist")
