"""归一化规则测试（M2）。

归一化是全项目最该被穷举测试的一层：它是纯函数、没有副作用，
而且一旦错了，错误会顺着证据链一路传到最终引用。

场景划分：正常 / 边界 / 失败 / 安全。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from paper_evidence_qa.schemas import DocType, OAStatus
from paper_evidence_qa.sources.normalize import (
    canonical_key,
    classify_crossref_type,
    classify_openalex_type,
    classify_semantic_scholar_types,
    coerce_int,
    collapse_whitespace,
    dedupe_authors,
    derive_paper_id,
    doc_types_to_crossref_filter,
    doc_types_to_openalex_filter,
    extract_arxiv_id_from_urls,
    fit_text,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
    parse_year,
    resolve_oa_status,
    strip_markup,
)

NOW = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- 正常场景


def test_normalize_doi_strips_resolver_prefix_and_lowercases() -> None:
    assert normalize_doi("https://doi.org/10.1234/AbC.DEF") == "10.1234/abc.def"
    assert normalize_doi("doi:10.1234/AbC") == "10.1234/abc"
    assert normalize_doi("http://dx.doi.org/10.5555/XYZ") == "10.5555/xyz"
    assert normalize_doi("10.7777/plain") == "10.7777/plain"


def test_normalize_arxiv_id_accepts_url_identifier_and_version_forms() -> None:
    assert normalize_arxiv_id("2401.00001") == "2401.00001"
    assert normalize_arxiv_id("arXiv:2401.00001v2") == "2401.00001"
    assert normalize_arxiv_id("http://arxiv.org/abs/2401.00001v2") == "2401.00001"
    assert normalize_arxiv_id("https://arxiv.org/pdf/2401.00001") == "2401.00001"
    assert normalize_arxiv_id("cs/0701001v1") == "cs/0701001"
    assert normalize_arxiv_id("math.GT/0309136") == "math.GT/0309136"


def test_normalize_title_collapses_whitespace_and_case() -> None:
    assert (
        normalize_title("  A   Study of\nRetrieval  ")
        == "a study of retrieval"
    )


def test_strip_markup_removes_jats_tags_and_decodes_entities() -> None:
    raw = "<jats:p>Retrieval &amp; ranking are <jats:italic>different</jats:italic>.</jats:p>"
    assert strip_markup(raw) == "Retrieval & ranking are different ."


def test_parse_year_handles_common_upstream_shapes() -> None:
    assert parse_year(2024) == 2024
    assert parse_year("2024") == 2024
    assert parse_year("2024-05-01") == 2024
    assert parse_year([[2024, 5, 1]]) == 2024
    assert parse_year({"date-parts": [[2019]]}) == 2019
    assert parse_year(datetime(2018, 3, 1, tzinfo=timezone.utc)) == 2018


def test_coerce_int_handles_strings_and_floats() -> None:
    assert coerce_int(7) == 7
    assert coerce_int("42") == 42
    assert coerce_int(3.9) == 3
    assert coerce_int("cited: 11 times") == 11


def test_dedupe_authors_is_case_insensitive_and_order_preserving() -> None:
    result = dedupe_authors(["Zhang San", "zhang san", "Li Si", None, "  "])
    assert result == ["Zhang San", "Li Si"]


def test_canonical_key_prefers_doi_then_arxiv_then_title() -> None:
    assert (
        canonical_key(doi="10.1/A", arxiv_id="2401.00001", title="X", year=2024)
        == "doi:10.1/a"
    )
    assert (
        canonical_key(arxiv_id="arXiv:2401.00001v2", title="X", year=2024)
        == "arxiv:2401.00001"
    )
    assert canonical_key(title="Some Title", year=2024) == "title:some title|2024"


def test_extract_arxiv_id_from_urls_finds_arxiv_link() -> None:
    urls = [
        "https://example.org/landing",
        "https://arxiv.org/abs/2401.00001v2",
    ]
    assert extract_arxiv_id_from_urls(urls) == "2401.00001"
    assert extract_arxiv_id_from_urls(["https://example.org/x"]) is None


def test_classify_doc_types_maps_known_values() -> None:
    assert classify_openalex_type("article") is DocType.JOURNAL_ARTICLE
    assert classify_openalex_type("preprint") is DocType.PREPRINT
    assert classify_openalex_type("proceedings-article") is DocType.CONFERENCE_PAPER
    assert classify_crossref_type("posted-content") is DocType.PREPRINT
    assert (
        classify_semantic_scholar_types(["Conference", "JournalArticle"])
        is DocType.CONFERENCE_PAPER
    )
    assert classify_semantic_scholar_types(["Review"]) is DocType.REVIEW


def test_doc_type_reverse_mapping_drops_unsupported_types() -> None:
    assert doc_types_to_openalex_filter(
        [DocType.JOURNAL_ARTICLE, DocType.OTHER]
    ) == ["article"]
    assert doc_types_to_crossref_filter(
        [DocType.CONFERENCE_PAPER, DocType.REVIEW]
    ) == ["proceedings-article", "journal-article"]


def test_resolve_oa_status_open_requires_url() -> None:
    resolution = resolve_oa_status(
        is_open=True, oa_url="https://example.org/a.pdf", source="openalex"
    )
    assert resolution.oa_status is OAStatus.OPEN
    assert resolution.oa_url == "https://example.org/a.pdf"
    assert "openalex" in resolution.oa_evidence


def test_resolve_oa_status_closed_has_no_url() -> None:
    resolution = resolve_oa_status(is_open=False, oa_url=None, source="openalex")
    assert resolution.oa_status is OAStatus.CLOSED
    assert resolution.oa_url is None


# ---------------------------------------------------------------- 边界场景


def test_fit_text_reports_truncation_instead_of_hiding_it() -> None:
    text, truncated = fit_text("abcdef", 3)
    assert text == "abc"
    assert truncated is True

    text, truncated = fit_text("abc", 3)
    assert text == "abc"
    assert truncated is False

    assert fit_text(None, 3) == (None, False)


def test_parse_year_rejects_out_of_range_and_garbage() -> None:
    assert parse_year(1700) is None
    assert parse_year(2300) is None
    assert parse_year("no year here") is None
    assert parse_year(None) is None
    assert parse_year(True) is None


def test_collapse_whitespace_returns_none_for_blank() -> None:
    assert collapse_whitespace("   \n\t ") is None
    assert collapse_whitespace(None) is None
    assert collapse_whitespace(" a  b ") == "a b"


def test_open_without_url_downgrades_to_unknown() -> None:
    resolution = resolve_oa_status(is_open=True, oa_url=None, source="openalex")
    assert resolution.oa_status is OAStatus.UNKNOWN
    assert resolution.oa_url is None
    assert "未核查" in resolution.oa_evidence


def test_url_without_open_flag_downgrades_to_unknown() -> None:
    resolution = resolve_oa_status(
        is_open=None, oa_url="https://example.org/maybe.pdf", source="openalex"
    )
    assert resolution.oa_status is OAStatus.UNKNOWN
    assert resolution.oa_url is None


def test_non_http_url_is_discarded() -> None:
    resolution = resolve_oa_status(
        is_open=True, oa_url="ftp://example.org/a.pdf", source="openalex"
    )
    assert resolution.oa_status is OAStatus.UNKNOWN
    assert resolution.oa_url is None


def test_normalize_doi_returns_none_for_blank_input() -> None:
    assert normalize_doi(None) is None
    assert normalize_doi("   ") is None


# ---------------------------------------------------------------- 失败场景


def test_canonical_key_rejects_records_without_any_identifier() -> None:
    with pytest.raises(ValueError) as excinfo:
        canonical_key(doi=None, arxiv_id=None, title="   ", year=None)
    assert "无法派生稳定标识" in str(excinfo.value)


def test_derive_paper_id_rejects_identifierless_record() -> None:
    with pytest.raises(ValueError):
        derive_paper_id(doi=None, arxiv_id=None, title="", year=2024)


def test_normalize_arxiv_id_returns_none_for_non_arxiv_text() -> None:
    assert normalize_arxiv_id("10.1234/not-arxiv") is None
    assert normalize_arxiv_id("hello world") is None


# ---------------------------------------------------------------- 安全场景


def test_same_paper_different_doi_writings_gets_same_paper_id() -> None:
    """同一篇论文的不同 DOI 写法必须派生同一个标识，否则库里会出现重复文献。"""
    variants = [
        "10.0000/fixture.2024.001",
        "10.0000/FIXTURE.2024.001",
        "https://doi.org/10.0000/fixture.2024.001",
        "doi:10.0000/fixture.2024.001",
    ]
    ids = {derive_paper_id(doi=value) for value in variants}
    assert len(ids) == 1


def test_same_paper_with_and_without_arxiv_version_gets_same_paper_id() -> None:
    ids = {
        derive_paper_id(arxiv_id=value)
        for value in ("2401.00001", "2401.00001v1", "arXiv:2401.00001v7")
    }
    assert len(ids) == 1


def test_different_papers_get_different_paper_ids() -> None:
    first = derive_paper_id(doi="10.0000/a")
    second = derive_paper_id(doi="10.0000/b")
    assert first != second


def test_paper_id_is_stable_across_calls() -> None:
    first = derive_paper_id(doi="10.0000/stable")
    second = derive_paper_id(doi="10.0000/stable")
    assert first == second
    assert first.startswith("ppr_")


def test_oa_resolution_never_violates_record_contract() -> None:
    """任意输入组合下，产出的结论都必须满足 M1 对 ``PaperRecord`` 的约束。

    约束是：``open`` 必须有链接；``closed`` 必须没有链接。
    这两条一旦被破坏，M1 的校验器会直接拒绝整条记录——
    与其等到那时才发现，不如在这里穷举所有组合先钉死。
    """
    cases = [
        (True, "https://example.org/a.pdf"),
        (True, None),
        (False, None),
        (False, "https://example.org/b.pdf"),
        (None, None),
        (None, "https://example.org/c.pdf"),
    ]
    for is_open, url in cases:
        resolution = resolve_oa_status(is_open=is_open, oa_url=url, source="fixture")
        if resolution.oa_status is OAStatus.OPEN:
            assert resolution.oa_url is not None, (is_open, url)
        if resolution.oa_status is OAStatus.CLOSED:
            assert resolution.oa_url is None, (is_open, url)


def test_conflicting_closed_with_url_is_downgraded_not_trusted() -> None:
    """上游自相矛盾时（说封闭却给了链接），保守按未核查处理，不凭猜测下载。"""
    resolution = resolve_oa_status(
        is_open=False, oa_url="https://example.org/conflict.pdf", source="fixture"
    )
    assert resolution.oa_status is OAStatus.UNKNOWN
    assert resolution.oa_url is None
    assert "自相矛盾" in resolution.oa_evidence


def test_oa_evidence_always_explains_the_judgement() -> None:
    resolution = resolve_oa_status(is_open=None, oa_url=None, source="fixture")
    assert resolution.oa_status is OAStatus.UNKNOWN
    assert resolution.oa_evidence
    assert "fixture" in resolution.oa_evidence
