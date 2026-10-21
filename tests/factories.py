"""测试对象工厂。

只负责构造**合法**的内存对象，用于验证模型契约。
这里不包含任何业务规则，不硬编码任何答案或期望结果。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from paper_evidence_qa.schemas import (
    Answer,
    Chunk,
    Claim,
    DocType,
    Evidence,
    KBVersion,
    Locator,
    OAStatus,
    PaperQARequest,
    PaperRecord,
    SourceName,
    SufficiencyJudgment,
    SupportType,
)

NOW = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
LATER = datetime(2026, 10, 11, 12, 10, 0, tzinfo=timezone.utc)

SHA = "a" * 64
"""合法的占位摘要（64 位十六进制）。"""

PAPER_ID = "ppr_0000000000000001"
CHUNK_ID = "chk_0000000000000001"
EVIDENCE_ID = "evd_0000000000000001"
CLAIM_ID = "clm_0000000000000001"
KB_VERSION = "kb_20261011_001"


def make_locator(**overrides: Any) -> Locator:
    data: dict[str, Any] = {
        "page_from": 3,
        "page_to": 3,
        "section_path": ["3 Method", "3.2 Data"],
    }
    data.update(overrides)
    return Locator(**data)


def make_paper(**overrides: Any) -> PaperRecord:
    data: dict[str, Any] = {
        "paper_id": PAPER_ID,
        "title": "脱敏样例论文标题（契约测试用）",
        "authors": ["Zhang San", "Li Si"],
        "year": 2024,
        "venue": "Fixture Conference",
        "doi": "10.0000/fixture.2024.001",
        "arxiv_id": "2401.00001",
        "abstract": "脱敏样例摘要，仅用于契约测试。",
        "doc_type": DocType.CONFERENCE_PAPER,
        "sources": [SourceName.OPENALEX, SourceName.ARXIV],
        "external_ids": {"openalex": "W0000000001"},
        "oa_status": OAStatus.OPEN,
        "oa_url": "https://example.org/fixture.pdf",
        "citation_count": 12,
        "is_in_local_kb": True,
        "version_group_id": "vgr_0000000000000001",
        "first_seen_at": NOW,
        "last_refreshed_at": LATER,
    }
    data.update(overrides)
    return PaperRecord(**data)


def make_chunk(**overrides: Any) -> Chunk:
    data: dict[str, Any] = {
        "chunk_id": CHUNK_ID,
        "paper_id": PAPER_ID,
        "kb_version": KB_VERSION,
        "chunk_index": 0,
        "text": "脱敏样例正文片段，仅用于契约测试。",
        "locator": make_locator(),
        "char_start": 0,
        "char_end": 42,
        "token_count": 24,
        "source_file_sha256": SHA,
    }
    data.update(overrides)
    return Chunk(**data)


def make_evidence(**overrides: Any) -> Evidence:
    """全文证据（来自本地知识库）。"""
    data: dict[str, Any] = {
        "evidence_id": EVIDENCE_ID,
        "source_type": SupportType.FULLTEXT,
        "source_name": SourceName.LOCAL_KB,
        "paper_id": PAPER_ID,
        "chunk_id": CHUNK_ID,
        "quote": "脱敏样例引用片段，逐字摘录自样例正文。",
        "locator": make_locator(),
        "is_local": True,
        "retrieval_score": 0.87,
        "retrieval_round": 1,
        "retrieved_at": NOW,
    }
    data.update(overrides)
    return Evidence(**data)


def make_abstract_evidence(**overrides: Any) -> Evidence:
    """摘要证据（来自联网元数据，不是全文）。"""
    data: dict[str, Any] = {
        "evidence_id": "evd_0000000000000002",
        "source_type": SupportType.ABSTRACT,
        "source_name": SourceName.OPENALEX,
        "paper_id": PAPER_ID,
        "chunk_id": None,
        "quote": "脱敏样例摘要片段。",
        "locator": make_locator(page_from=None, page_to=None, section_path=["Abstract"]),
        "is_local": False,
        "retrieval_score": 0.51,
        "retrieval_round": 1,
        "retrieved_at": NOW,
    }
    data.update(overrides)
    return Evidence(**data)


def make_claim(**overrides: Any) -> Claim:
    data: dict[str, Any] = {
        "claim_id": CLAIM_ID,
        "text": "脱敏样例断言文本。",
        "evidence_ids": [EVIDENCE_ID],
        "support_type": SupportType.FULLTEXT,
        "confidence": 0.8,
    }
    data.update(overrides)
    return Claim(**data)


def make_answer(**overrides: Any) -> Answer:
    data: dict[str, Any] = {
        "question": "脱敏样例研究问题？",
        "summary": "基于样例证据的回答正文。",
        "claims": [make_claim()],
    }
    data.update(overrides)
    return Answer(**data)


def make_judgment(**overrides: Any) -> SufficiencyJudgment:
    data: dict[str, Any] = {
        "is_sufficient": True,
        "coverage": 0.9,
        "judged_by": "rule",
        "reasons": ["样例理由：本地库已覆盖问题的两个子方面。"],
        "round_index": 1,
    }
    data.update(overrides)
    return SufficiencyJudgment(**data)


def make_kb_version(**overrides: Any) -> KBVersion:
    data: dict[str, Any] = {
        "kb_version": KB_VERSION,
        "built_at": NOW,
        "paper_count": 3,
        "chunk_count": 42,
        "parser_name": "docling",
        "parser_version": "2.0.0",
        "chunk_strategy": "fixed_size",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "embedding_model": "BAAI/bge-m3",
    }
    data.update(overrides)
    return KBVersion(**data)


def make_request(**overrides: Any) -> PaperQARequest:
    data: dict[str, Any] = {
        "request_id": "req_0000000000000001",
        "requested_at": NOW,
        "question": "脱敏样例研究问题？",
        "keywords": ["retrieval augmented generation"],
        "domain": "natural_language_processing",
        "doc_types": [DocType.CONFERENCE_PAPER],
        "year_from": 2020,
        "year_to": 2025,
        "local_pdf_paths": [],
        "kb_version": KB_VERSION,
        "allow_online": False,
        "max_rounds": 3,
        "max_downloads": 0,
        "max_tool_calls": 20,
    }
    data.update(overrides)
    return PaperQARequest(**data)
