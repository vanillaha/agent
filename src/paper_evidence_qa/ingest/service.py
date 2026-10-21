"""入库编排（M5）。

职责：文件校验 → 调解析器 → 分块 → 组装 ``Chunk`` / ``KBVersion``。
确定性规则全部由代码执行；单文件失败被隔离为 ``PaperIngestFailure``，
全部失败才抛错——绝不产出"空成功"。
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from paper_evidence_qa.ingest.chunker import chunk_document
from paper_evidence_qa.ingest.errors import IngestError
from paper_evidence_qa.ingest.models import (
    IngestRequest,
    IngestResult,
    PaperIngestFailure,
    PaperIngestResult,
    ParsedDocument,
    ParsedPage,
)
from paper_evidence_qa.ids import deterministic_id
from paper_evidence_qa.schemas.common import ErrorCode
from paper_evidence_qa.schemas.domain import Chunk, KBVersion

CHUNK_STRATEGY = "fixed-v1"
"""本任务冻结的分块策略名。chunk size 对照实验引入新策略时再扩展。"""


class IngestService:
    """入库编排入口。``parser`` 按协议注入，测试用假实现，生产用 Docling。"""

    def __init__(self, parser) -> None:  # noqa: ANN001 — 协议注入，保持宽松
        self._parser = parser

    def process(self, request: IngestRequest) -> IngestResult:
        papers: list[PaperIngestResult] = []
        failures: list[PaperIngestFailure] = []
        chunks: list[Chunk] = []
        parser_name = ""
        parser_version = ""

        for source_path in request.files:
            outcome = self._ingest_one(source_path, request.options, None)
            if isinstance(outcome, PaperIngestFailure):
                failures.append(outcome)
                continue
            papers.append(outcome["summary"])
            chunks.extend(outcome["chunks"])
            parser_name = parser_name or outcome["parser_name"]
            parser_version = parser_version or outcome["parser_version"]

        if not papers:
            last = failures[-1] if failures else None
            raise IngestError(
                last.code if last else ErrorCode.INDEX_FAILED,
                "全部文件入库失败" if last else "无文件可入库",
                {"failure_count": len(failures)},
            )

        # kb_version 确定性派生：同文件集 + 同参数 + 同解析器 = 同版本号，
        # 重复入库天然幂等（与"检索结果绑定版本、可复现"的承诺一致）
        kb_id = deterministic_id(
            "kb",
            CHUNK_STRATEGY,
            str(request.options.chunk_size),
            str(request.options.chunk_overlap),
            parser_name,
            parser_version,
            *sorted(p.source_file_sha256 for p in papers),
        )
        for chunk in chunks:
            chunk.kb_version = kb_id
        kb = KBVersion(
            kb_version=kb_id,
            built_at=datetime.now(UTC),
            paper_count=len(papers),
            chunk_count=len(chunks),
            parser_name=parser_name or "unknown",
            parser_version=parser_version or "0",
            chunk_strategy=CHUNK_STRATEGY,
            chunk_size=request.options.chunk_size,
            chunk_overlap=request.options.chunk_overlap,
        )
        return IngestResult(papers=papers, failures=failures, chunks=chunks, kb_version=kb)

    def _ingest_one(self, source_path: Path, options, kb_id: str | None) -> dict | PaperIngestFailure:  # noqa: ANN001
        path = str(source_path)
        if not source_path.exists():
            return PaperIngestFailure(
                source_path=path, code=ErrorCode.FILE_NOT_FOUND, message="文件不存在"
            )
        if source_path.suffix.lower() != ".pdf":
            return PaperIngestFailure(
                source_path=path,
                code=ErrorCode.UNSUPPORTED_FILE_TYPE,
                message="只处理文本层 PDF",
            )

        data = source_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()

        try:
            page_blocks, parser_name, parser_version = self._parser.parse(source_path)
            parsed = ParsedDocument(
                pages=[
                    ParsedPage(page_no=i + 1, text="\n".join(blocks), section_path=[])
                    for i, blocks in enumerate(page_blocks)
                ],
                parser_name=parser_name,
                parser_version=parser_version,
                source_file_sha256=digest,
            )
            drafts = chunk_document(parsed, options)
        except IngestError as error:
            return PaperIngestFailure(
                source_path=path, code=error.code, message=error.message, details=error.details
            )

        paper_id = deterministic_id("ppr", digest)
        signature = f"{CHUNK_STRATEGY}:{options.chunk_size}:{options.chunk_overlap}"
        built = [
            Chunk(
                chunk_id=deterministic_id("chk", digest, signature, str(index)),
                paper_id=paper_id,
                kb_version="kb_placeholder",
                chunk_index=index,
                text=draft.text,
                locator={
                    "page_from": draft.page_from,
                    "page_to": draft.page_to,
                },
                char_start=draft.char_start,
                char_end=draft.char_end,
                source_file_sha256=digest,
            )
            for index, draft in enumerate(drafts)
        ]
        summary = PaperIngestResult(
            paper_id=paper_id,
            source_path=path,
            source_file_sha256=digest,
            page_count=len(parsed.pages),
            chunk_count=len(built),
        )
        return {
            "summary": summary,
            "chunks": built,
            "parser_name": parser_name,
            "parser_version": parser_version,
        }
