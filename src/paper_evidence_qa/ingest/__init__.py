"""M5：本地文献解析与分块索引。

对外入口：``IngestService``（编排）、``chunk_document``（纯函数分块器）、
``Parser``（解析协议）。数据形状见 ``models.py``。
"""

from paper_evidence_qa.ingest.chunker import ChunkDraft, chunk_document
from paper_evidence_qa.ingest.errors import IngestError
from paper_evidence_qa.ingest.models import (
    IngestOptions,
    IngestRequest,
    IngestResult,
    PaperIngestFailure,
    PaperIngestResult,
    ParsedDocument,
    ParsedPage,
)
from paper_evidence_qa.ingest.parser import Parser
from paper_evidence_qa.ingest.service import CHUNK_STRATEGY, IngestService

__all__ = [
    "CHUNK_STRATEGY",
    "ChunkDraft",
    "IngestError",
    "IngestOptions",
    "IngestRequest",
    "IngestResult",
    "IngestService",
    "PaperIngestFailure",
    "PaperIngestResult",
    "ParsedDocument",
    "ParsedPage",
    "Parser",
    "chunk_document",
]
