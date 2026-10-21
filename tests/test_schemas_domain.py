"""契约测试：领域模型与证据链（M1）。

重点验证三件事：
1. 证据链 ``Chunk -> Evidence -> Claim -> Answer`` 的约束是否成立。
2. 本地全文证据与联网摘要证据是否被严格区分。
3. 无证据的断言、无理由的拒答、无依据的继续检索是否被拒绝。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from factories import (
    LATER,
    NOW,
    PAPER_ID,
    SHA,
    make_abstract_evidence,
    make_answer,
    make_chunk,
    make_claim,
    make_evidence,
    make_judgment,
    make_kb_version,
    make_locator,
    make_paper,
)
from paper_evidence_qa.schemas import (
    Answer,
    Chunk,
    Claim,
    ConflictPair,
    DownloadRecord,
    DownloadStatus,
    KBVersion,
    OAStatus,
    RetrievalRecord,
    RetrievalTarget,
    SourceName,
    SupportType,
)


# ---------------------------------------------------------------- 正常场景


def test_evidence_chain_builds_end_to_end() -> None:
    """Chunk -> Evidence -> Claim -> Answer 整条证据链必须可构造且可回溯。"""
    chunk = make_chunk()
    evidence = make_evidence(chunk_id=chunk.chunk_id, locator=chunk.locator)
    claim = make_claim(evidence_ids=[evidence.evidence_id])
    answer = make_answer(claims=[claim])

    assert answer.refused is False
    assert answer.summary
    assert answer.claims[0].evidence_ids == [evidence.evidence_id]
    assert evidence.locator.to_display() == "p.3, 3 Method > 3.2 Data"


def test_abstract_evidence_is_not_fulltext() -> None:
    """联网摘要证据必须与全文证据区分，避免"本地证据与联网摘要混写"。"""
    evidence = make_abstract_evidence()
    assert evidence.source_type is SupportType.ABSTRACT
    assert evidence.is_local is False
    assert evidence.chunk_id is None


def test_paper_doi_is_normalized_and_authors_deduplicated() -> None:
    paper = make_paper(
        doi="https://doi.org/10.0000/Fixture.2024.001",
        authors=["Zhang San", "zhang san", "Li Si"],
    )
    assert paper.doi == "10.0000/fixture.2024.001"
    assert paper.authors == ["Zhang San", "Li Si"]


def test_closed_access_paper_keeps_metadata_only() -> None:
    """不提供开放全文时只保留元数据，不得编造下载地址。"""
    paper = make_paper(oa_status=OAStatus.CLOSED, oa_url=None)
    assert paper.oa_url is None


def test_download_record_success_requires_integrity_fields() -> None:
    record = DownloadRecord(
        download_id="dl_0000000000000001",
        paper_id=PAPER_ID,
        url="https://example.org/fixture.pdf",
        status=DownloadStatus.DOWNLOADED,
        checked_at=NOW,
        file_type="application/pdf",
        file_size_bytes=20480,
        sha256=SHA,
    )
    assert record.status is DownloadStatus.DOWNLOADED


def test_skipped_download_must_explain_reason() -> None:
    record = DownloadRecord(
        download_id="dl_0000000000000002",
        paper_id=PAPER_ID,
        url="https://example.org/closed.pdf",
        status=DownloadStatus.SKIPPED_NOT_OPEN_ACCESS,
        checked_at=NOW,
        failure_reason="检测到该文献不提供开放获取全文，未发起下载",
    )
    assert record.sha256 is None
    assert record.failure_reason


def test_retrieval_record_keeps_query_and_source() -> None:
    record = RetrievalRecord(
        record_id="rcd_0000000000000001",
        query="long context retrieval degradation",
        target=RetrievalTarget.ONLINE,
        source_name=SourceName.OPENALEX,
        executed_at=NOW,
        round_index=1,
        result_count=40,
        kept_count=8,
        duration_ms=820,
    )
    assert record.kept_count == 8


def test_refusal_answer_without_claims_is_valid() -> None:
    answer = Answer(
        question="样例问题？",
        refused=True,
        refusal_reason="本地库与联网检索均未找到支撑该结论的全文证据。",
    )
    assert answer.claims == []
    assert answer.summary is None


# ---------------------------------------------------------------- 边界场景


def test_locator_requires_page_or_section() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_locator(page_from=None, page_to=None, section_path=[])
    assert "页码或章节" in str(excinfo.value)


def test_locator_rejects_reversed_pages() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_locator(page_from=9, page_to=3)
    assert "page_from" in str(excinfo.value)


def test_locator_rejects_page_to_without_page_from() -> None:
    with pytest.raises(ValidationError):
        make_locator(page_from=None, page_to=5, section_path=["2 Related Work"])


def test_locator_rejects_invalid_char_range() -> None:
    with pytest.raises(ValidationError):
        make_locator(char_start=100, char_end=100)


def test_section_only_locator_is_allowed() -> None:
    locator = make_locator(page_from=None, page_to=None, section_path=["Abstract"])
    assert locator.to_display() == "Abstract"


def test_judgment_insufficient_requires_missing_aspects() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_judgment(
            is_sufficient=False,
            coverage=0.3,
            next_queries=["sample query"],
            missing_aspects=[],
            round_index=1,
        )
    assert "missing_aspects" in str(excinfo.value)


def test_judgment_insufficient_without_next_queries_blocks_continuation() -> None:
    """证据不足且仍有继续余地时，必须给出下一轮检索式，否则无法进入下一轮。"""
    with pytest.raises(ValidationError) as excinfo:
        make_judgment(
            is_sufficient=False,
            coverage=0.3,
            missing_aspects=["缺少对照实验结果"],
            next_queries=[],
            can_continue=True,
            round_index=2,
        )
    assert "next_queries" in str(excinfo.value)


def test_judgment_insufficient_but_no_room_left_is_valid() -> None:
    """达到轮次上限时可以不再给检索式，转为拒答。"""
    judgment = make_judgment(
        is_sufficient=False,
        coverage=0.4,
        missing_aspects=["缺少可获取的开放全文"],
        next_queries=[],
        can_continue=False,
        round_index=3,
    )
    assert judgment.can_continue is False


# ---------------------------------------------------------------- 失败场景


def test_paper_rejects_reversed_timestamps() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_paper(first_seen_at=LATER, last_refreshed_at=NOW)
    assert "last_refreshed_at" in str(excinfo.value)


def test_paper_open_status_requires_oa_url() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_paper(oa_status=OAStatus.OPEN, oa_url=None)
    assert "oa_url" in str(excinfo.value)


def test_paper_closed_status_forbids_oa_url() -> None:
    with pytest.raises(ValidationError):
        make_paper(oa_status=OAStatus.CLOSED, oa_url="https://example.org/x.pdf")


def test_paper_requires_at_least_one_source() -> None:
    with pytest.raises(ValidationError):
        make_paper(sources=[])


def test_chunk_rejects_invalid_text_span() -> None:
    with pytest.raises(ValidationError):
        make_chunk(char_start=50, char_end=50)


def test_fulltext_evidence_requires_chunk_id() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_evidence(chunk_id=None)
    assert "chunk_id" in str(excinfo.value)


def test_fulltext_evidence_must_be_local() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_evidence(is_local=False)
    assert "本地知识库" in str(excinfo.value)


def test_inference_cannot_be_marked_as_local_evidence() -> None:
    """模型推断不得伪装成本地全文证据。"""
    with pytest.raises(ValidationError):
        make_evidence(
            source_type=SupportType.INFERENCE,
            chunk_id=None,
            is_local=True,
        )


def test_claim_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_claim(evidence_ids=[])


def test_refusal_with_claims_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_answer(refused=True, refusal_reason="证据不足", claims=[make_claim()])
    assert "拒答" in str(excinfo.value)


def test_refusal_without_reason_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_answer(refused=True, refusal_reason=None, claims=[])


def test_non_refusal_requires_summary() -> None:
    with pytest.raises(ValidationError):
        make_answer(summary=None, claims=[])


def test_non_refusal_forbids_refusal_reason() -> None:
    with pytest.raises(ValidationError):
        make_answer(refusal_reason="不应出现")


def test_conflict_pair_requires_two_evidence_refs() -> None:
    with pytest.raises(ValidationError):
        ConflictPair(evidence_ids=["evd_0000000000000001"], description="仅一条不构成冲突")


def test_downloaded_record_without_integrity_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        DownloadRecord(
            download_id="dl_0000000000000003",
            paper_id=PAPER_ID,
            url="https://example.org/fixture.pdf",
            status=DownloadStatus.DOWNLOADED,
            checked_at=NOW,
            file_type="application/pdf",
        )
    assert "sha256" in str(excinfo.value)


def test_failed_download_without_reason_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DownloadRecord(
            download_id="dl_0000000000000004",
            paper_id=PAPER_ID,
            url="https://example.org/fixture.pdf",
            status=DownloadStatus.FAILED,
            checked_at=NOW,
        )


def test_kb_version_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_kb_version(chunk_size=512, chunk_overlap=512)
    assert "chunk_overlap" in str(excinfo.value)


def test_retrieval_record_kept_count_cannot_exceed_results() -> None:
    with pytest.raises(ValidationError):
        RetrievalRecord(
            record_id="rcd_0000000000000002",
            query="fixture query",
            target=RetrievalTarget.LOCAL_KB,
            source_name=SourceName.LOCAL_KB,
            executed_at=NOW,
            round_index=1,
            result_count=3,
            kept_count=5,
        )


def test_unknown_domain_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_evidence(unexpected="不应被静默忽略")


def test_models_are_immutable_by_contract_after_assignment() -> None:
    """赋值同样走校验：不允许把合法对象改成非法状态。"""
    chunk = make_chunk()
    with pytest.raises(ValidationError):
        chunk.char_end = 0


# ---------------------------------------------------------------- 安全场景


def test_chunk_and_evidence_types_are_distinct_classes() -> None:
    assert isinstance(make_chunk(), Chunk)
    assert isinstance(make_claim(), Claim)
    assert isinstance(make_kb_version(), KBVersion)
