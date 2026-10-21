"""四个来源适配器测试（M2）。

全部离线：用假 HTTP 客户端喂 `fixtures/sources/` 下的脱敏样本。

每个适配器都测四件事：**请求拼得对、响应解析得对、失败认得对、边界守得住**。
请求拼错是最难发现的一类 bug——它不会让测试变红，只会让线上一个结果都搜不到。

场景划分：正常 / 边界 / 失败 / 安全。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import pytest

from paper_evidence_qa.schemas import DocType, ErrorCode, OAStatus
from paper_evidence_qa.sources.arxiv import API_URL as ARXIV_URL
from paper_evidence_qa.sources.arxiv import ArxivAdapter
from paper_evidence_qa.sources.base import AdapterSettings, SourceQuery
from paper_evidence_qa.sources.crossref import API_URL as CROSSREF_URL
from paper_evidence_qa.sources.crossref import CrossrefAdapter
from paper_evidence_qa.sources.openalex import API_URL as OPENALEX_URL
from paper_evidence_qa.sources.openalex import OpenAlexAdapter
from paper_evidence_qa.sources.ratelimit import NullRateLimiter
from paper_evidence_qa.sources.semantic_scholar import API_URL as S2_URL
from paper_evidence_qa.sources.semantic_scholar import SemanticScholarAdapter

from fakes import (
    S2_URL as FAKE_S2_URL,
    TEST_API_KEY,
    TEST_MAILTO,
    FakeHttpClient,
    load_fixture_text,
    text_response,
)

NOW = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)

FIXTURES: Mapping[str, str] = {
    OPENALEX_URL: "openalex_works.json",
    CROSSREF_URL: "crossref_works.json",
    ARXIV_URL: "arxiv_feed.xml",
    S2_URL: "semantic_scholar_search.json",
}


def routes_with_fixture(url: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {url: load_fixture_text(FIXTURES[url])}
    payload.update(extra)
    return payload


def make_adapter(adapter_cls, routes, *, api_key=None, mailto=TEST_MAILTO):
    """构造适配器，使用不等待的限流器与固定时钟，保证测试快且可复现。"""
    http = FakeHttpClient(routes)
    adapter = adapter_cls(
        settings=AdapterSettings(mailto=mailto, api_key=api_key),
        http=http,
        limiter=NullRateLimiter(),
        monotonic=lambda: 0.0,
        now=lambda: NOW,
    )
    return adapter, http


def make_query(**overrides: Any) -> SourceQuery:
    data: dict[str, Any] = {
        "query_text": "long context retrieval",
        "filters": {},
        "limit": 10,
        "open_access_only": False,
    }
    data.update(overrides)
    return SourceQuery(**data)


# ================================================================ OpenAlex


def test_openalex_request_carries_query_paging_and_contact() -> None:
    adapter, http = make_adapter(
        OpenAlexAdapter, routes_with_fixture(OPENALEX_URL)
    )
    adapter.fetch(
        make_query(
            filters={"year_from": 2020, "year_to": 2025, "doc_types": ["review"]}
        )
    )
    params = http.last_params_for(OPENALEX_URL)
    assert params["search"] == "long context retrieval"
    assert params["per-page"] == 10
    assert params["mailto"] == TEST_MAILTO
    assert "from_publication_date:2020-01-01" in params["filter"]
    assert "to_publication_date:2025-12-31" in params["filter"]
    assert "type:review" in params["filter"]
    assert TEST_MAILTO in http.last_headers_for(OPENALEX_URL)["User-Agent"]


def test_openalex_request_adds_oa_filter_only_when_requested() -> None:
    adapter, http = make_adapter(
        OpenAlexAdapter, routes_with_fixture(OPENALEX_URL)
    )
    adapter.fetch(make_query())
    assert "filter" not in http.last_params_for(OPENALEX_URL)

    adapter.fetch(make_query(open_access_only=True))
    assert "is_oa:true" in http.last_params_for(OPENALEX_URL)["filter"]


def test_openalex_parses_fixture_and_skips_untitled_record() -> None:
    adapter, _ = make_adapter(OpenAlexAdapter, routes_with_fixture(OPENALEX_URL))
    result = adapter.fetch(make_query())

    assert result.ok is True
    assert result.raw_count == 4
    assert len(result.candidates) == 3
    assert result.skipped_count == 1
    assert any(w.code == "records_skipped" for w in result.warnings)


def test_openalex_normalizes_core_fields() -> None:
    adapter, _ = make_adapter(OpenAlexAdapter, routes_with_fixture(OPENALEX_URL))
    candidate = adapter.fetch(make_query()).candidates[0]
    paper = candidate.paper

    assert paper.doi == "10.0000/fixture.2024.001"
    assert paper.arxiv_id == "2401.00001"
    assert paper.title == "A Fixture Study of Retrieval Augmentation"
    assert paper.authors == ["Zhang San", "Li Si"]
    assert paper.venue == "Fixture Journal of Retrieval"
    assert paper.doc_type is DocType.JOURNAL_ARTICLE
    assert paper.citation_count == 17
    assert paper.abstract == (
        "This fixture abstract is synthetic and contains no real fixture content."
    )
    assert paper.sources == [paper.sources[0]]
    assert candidate.upstream_id == "W0000000001"
    assert candidate.license == "cc-by"
    assert paper.external_ids["mag"] == "9000000001"


def test_openalex_open_access_requires_downloadable_url() -> None:
    adapter, _ = make_adapter(OpenAlexAdapter, routes_with_fixture(OPENALEX_URL))
    candidates = adapter.fetch(make_query()).candidates

    downloadable = next(
        c for c in candidates if c.paper.doi == "10.0000/fixture.2024.001"
    )
    assert downloadable.paper.oa_status is OAStatus.OPEN
    assert downloadable.paper.oa_url == "https://example.org/fixture-001.pdf"

    flagged_but_no_link = next(
        c for c in candidates if c.paper.doi is None
    )
    assert flagged_but_no_link.paper.oa_status is OAStatus.UNKNOWN
    assert flagged_but_no_link.paper.oa_url is None
    assert "未核查" in (flagged_but_no_link.oa_evidence or "")

    closed = next(
        c for c in candidates if c.paper.doi == "10.0000/fixture.2021.004"
    )
    assert closed.paper.oa_status is OAStatus.CLOSED
    assert closed.paper.oa_url is None


def test_openalex_failure_paths_are_distinguishable() -> None:
    adapter, _ = make_adapter(
        OpenAlexAdapter,
        {OPENALEX_URL: text_response(OPENALEX_URL, "nope", status_code=500)},
    )
    result = adapter.fetch(make_query())
    assert result.ok is False
    assert result.candidates == []
    assert result.error is not None
    assert result.error.code is ErrorCode.UPSTREAM_UNAVAILABLE

    adapter, _ = make_adapter(
        OpenAlexAdapter,
        {OPENALEX_URL: text_response(OPENALEX_URL, "slow down", status_code=429)},
    )
    assert adapter.fetch(make_query()).error.code is ErrorCode.RATE_LIMITED


def test_openalex_malformed_payload_reports_location() -> None:
    adapter, _ = make_adapter(
        OpenAlexAdapter, {OPENALEX_URL: text_response(OPENALEX_URL, "<html>oops</html>")}
    )
    result = adapter.fetch(make_query())
    assert result.ok is False
    assert result.error is not None
    assert result.error.details["where"] == "openalex.root"


def test_openalex_missing_results_array_is_an_error_not_empty_success() -> None:
    """上游改格式时必须报错，而不是静默返回"零结果"。"""
    adapter, _ = make_adapter(
        OpenAlexAdapter, {OPENALEX_URL: '{"meta": {"count": 0}}'}
    )
    result = adapter.fetch(make_query())
    assert result.ok is False
    assert result.error.details["where"] == "openalex.results"


def test_openalex_success_with_zero_results_is_not_an_error() -> None:
    adapter, _ = make_adapter(
        OpenAlexAdapter, {OPENALEX_URL: '{"meta": {"count": 0}, "results": []}'}
    )
    result = adapter.fetch(make_query())
    assert result.ok is True
    assert result.raw_count == 0
    assert result.candidates == []


# ================================================================ Crossref


def test_crossref_request_uses_bibliographic_query_and_repeated_type_filter() -> None:
    adapter, http = make_adapter(
        CrossrefAdapter, routes_with_fixture(CROSSREF_URL)
    )
    adapter.fetch(
        make_query(
            filters={
                "year_from": 2020,
                "year_to": 2025,
                "doc_types": ["conference_paper", "review"],
            }
        )
    )
    params = http.last_params_for(CROSSREF_URL)
    assert params["query.bibliographic"] == "long context retrieval"
    assert params["rows"] == 10
    assert params["mailto"] == TEST_MAILTO
    assert "from-pub-date:2020-01-01" in params["filter"]
    assert "until-pub-date:2025-12-31" in params["filter"]
    assert "type:proceedings-article" in params["filter"]
    assert "type:journal-article" in params["filter"]


def test_crossref_parses_fixture_and_skips_untitled_record() -> None:
    adapter, _ = make_adapter(CrossrefAdapter, routes_with_fixture(CROSSREF_URL))
    result = adapter.fetch(make_query())

    assert result.raw_count == 2
    assert len(result.candidates) == 1
    assert result.skipped_count == 1


def test_crossref_normalizes_fields_and_keeps_license() -> None:
    adapter, _ = make_adapter(CrossrefAdapter, routes_with_fixture(CROSSREF_URL))
    candidate = adapter.fetch(make_query()).candidates[0]
    paper = candidate.paper

    assert paper.doi == "10.0000/fixture.2020.001"
    assert paper.title == "A Fixture Journal Article With Odd Spacing"
    assert paper.authors == ["Zhang San", "Fixture Consortium"]
    assert paper.year == 2020
    assert paper.venue == "Fixture Journal of Retrieval"
    assert paper.citation_count == 42
    assert paper.abstract == (
        "This fixture abstract is synthetic & contains no real content."
    )
    assert candidate.license == "https://creativecommons.org/licenses/by/4.0/"


def test_crossref_never_claims_open_access() -> None:
    """Crossref 不给可靠 OA 判定，一律 unknown，把判定权留给 OpenAlex。"""
    adapter, _ = make_adapter(CrossrefAdapter, routes_with_fixture(CROSSREF_URL))
    for candidate in adapter.fetch(make_query()).candidates:
        assert candidate.paper.oa_status is OAStatus.UNKNOWN
        assert candidate.paper.oa_url is None


def test_crossref_open_access_filter_is_applied_locally_with_warning() -> None:
    adapter, _ = make_adapter(CrossrefAdapter, routes_with_fixture(CROSSREF_URL))
    result = adapter.fetch(make_query(open_access_only=True))
    assert result.candidates == []
    assert any(w.code == "open_access_filter_local" for w in result.warnings)


def test_crossref_missing_items_is_an_error() -> None:
    adapter, _ = make_adapter(CrossrefAdapter, {CROSSREF_URL: '{"message": {}}'})
    result = adapter.fetch(make_query())
    assert result.ok is False
    assert result.error.details["where"] == "crossref.message.items"


def test_crossref_http_errors_are_reported() -> None:
    adapter, _ = make_adapter(
        CrossrefAdapter,
        {CROSSREF_URL: text_response(CROSSREF_URL, "bad request", status_code=400)},
    )
    result = adapter.fetch(make_query())
    assert result.error.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert result.error.details["status_code"] == 400


# ================================================================ arXiv


def test_arxiv_request_builds_boolean_query_and_date_clause() -> None:
    adapter, http = make_adapter(ArxivAdapter, routes_with_fixture(ARXIV_URL))
    adapter.fetch(make_query(filters={"year_from": 2020, "year_to": 2025}))
    params = http.last_params_for(ARXIV_URL)
    search = params["search_query"]
    assert 'all:"long"' in search
    assert 'all:"context"' in search
    assert " AND " in search
    assert "submittedDate:[202001010000 TO 202512312359]" in search
    assert params["max_results"] == 10
    assert http.last_headers_for(ARXIV_URL)["Accept"] == "application/atom+xml"


def test_arxiv_request_passes_through_advanced_syntax() -> None:
    """M7 精心构造的高级检索式不该被再包装一层。"""
    adapter, http = make_adapter(ArxivAdapter, routes_with_fixture(ARXIV_URL))
    adapter.fetch(make_query(query_text='ti:"retrieval" AND cat:cs.CL'))
    search = http.last_params_for(ARXIV_URL)["search_query"]
    assert search == 'ti:"retrieval" AND cat:cs.CL'


def test_arxiv_omits_date_clause_without_year_range() -> None:
    adapter, http = make_adapter(ArxivAdapter, routes_with_fixture(ARXIV_URL))
    adapter.fetch(make_query())
    assert "submittedDate" not in http.last_params_for(ARXIV_URL)["search_query"]


def test_arxiv_parses_atom_feed() -> None:
    adapter, _ = make_adapter(ArxivAdapter, routes_with_fixture(ARXIV_URL))
    result = adapter.fetch(make_query())

    assert result.raw_count == 2
    assert len(result.candidates) == 2

    modern = result.candidates[0]
    assert modern.paper.arxiv_id == "2401.00001"
    assert modern.paper.doi == "10.0000/fixture.2024.900"
    assert modern.paper.title == "A Fixture Preprint on Retrieval Augmentation"
    assert modern.paper.authors == ["Zhang San", "Li Si"]
    assert modern.paper.year == 2024
    assert modern.paper.venue == "Fixture Conference 2024"
    assert modern.paper.external_ids["arxiv_category"] == "cs.CL"
    assert modern.paper.external_ids["arxiv_version"] == "2"


def test_arxiv_records_are_always_preprints() -> None:
    """arXiv 全部是预印本，未经同行评审，不能被当作"已发表"的证据。"""
    adapter, _ = make_adapter(ArxivAdapter, routes_with_fixture(ARXIV_URL))
    for candidate in adapter.fetch(make_query()).candidates:
        assert candidate.paper.doc_type is DocType.PREPRINT


def test_arxiv_marks_open_access_and_keeps_pdf_link() -> None:
    adapter, _ = make_adapter(ArxivAdapter, routes_with_fixture(ARXIV_URL))
    candidates = adapter.fetch(make_query()).candidates

    assert candidates[0].paper.oa_status is OAStatus.OPEN
    assert candidates[0].paper.oa_url == "http://arxiv.org/pdf/2401.00001v2"

    old_style = candidates[1]
    assert old_style.paper.arxiv_id == "cs/0701001"
    assert old_style.paper.oa_status is OAStatus.OPEN
    assert old_style.paper.oa_url == "https://arxiv.org/pdf/cs/0701001"
    assert "构造" in (old_style.oa_evidence or "")


def test_arxiv_never_fabricates_citation_count() -> None:
    """arXiv 不提供引用数。留空，绝不填 0——0 会被误读成"无人引用"。"""
    adapter, _ = make_adapter(ArxivAdapter, routes_with_fixture(ARXIV_URL))
    for candidate in adapter.fetch(make_query()).candidates:
        assert candidate.paper.citation_count is None


def test_arxiv_invalid_xml_is_a_structured_error() -> None:
    adapter, _ = make_adapter(ArxivAdapter, {ARXIV_URL: "<feed><entry>"})
    result = adapter.fetch(make_query())
    assert result.ok is False
    assert result.error.details["where"] == "arxiv.root"


def test_arxiv_http_errors_are_reported() -> None:
    adapter, _ = make_adapter(
        ArxivAdapter, {ARXIV_URL: text_response(ARXIV_URL, "busy", status_code=503)}
    )
    result = adapter.fetch(make_query())
    assert result.error.code is ErrorCode.UPSTREAM_UNAVAILABLE


def test_arxiv_min_interval_is_configured() -> None:
    adapter, _ = make_adapter(ArxivAdapter, routes_with_fixture(ARXIV_URL))
    assert adapter.min_interval_seconds >= 3.0


# ================================================================ Semantic Scholar


def test_semantic_scholar_request_includes_year_range_and_fields() -> None:
    adapter, http = make_adapter(
        SemanticScholarAdapter, routes_with_fixture(S2_URL)
    )
    adapter.fetch(make_query(filters={"year_from": 2020, "year_to": 2025}))
    params = http.last_params_for(S2_URL)
    assert params["query"] == "long context retrieval"
    assert params["limit"] == 10
    assert params["year"] == "2020-2025"
    assert "openAccessPdf" in params["fields"]


def test_semantic_scholar_sends_api_key_only_when_configured() -> None:
    adapter, http = make_adapter(
        SemanticScholarAdapter, routes_with_fixture(S2_URL)
    )
    adapter.fetch(make_query())
    headers = http.last_headers_for(S2_URL)
    assert all(key.lower() != "x-api-key" for key in headers)

    adapter, http = make_adapter(
        SemanticScholarAdapter, routes_with_fixture(S2_URL), api_key=TEST_API_KEY
    )
    adapter.fetch(make_query())
    assert http.last_headers_for(S2_URL)["x-api-key"] == TEST_API_KEY


def test_semantic_scholar_interval_tightens_with_api_key() -> None:
    without_key = make_adapter(
        SemanticScholarAdapter, routes_with_fixture(S2_URL)
    )[0]
    with_key = make_adapter(
        SemanticScholarAdapter, routes_with_fixture(S2_URL), api_key=TEST_API_KEY
    )[0]
    assert with_key.min_interval_seconds < without_key.min_interval_seconds
    assert with_key.min_interval_seconds >= 1.0
    assert without_key.min_interval_seconds >= 3.0


def test_semantic_scholar_parses_fixture() -> None:
    adapter, _ = make_adapter(
        SemanticScholarAdapter, routes_with_fixture(S2_URL)
    )
    result = adapter.fetch(make_query())
    assert result.raw_count == 2
    assert len(result.candidates) == 2

    first = result.candidates[0]
    assert first.paper.doi == "10.0000/fixture.2024.777"
    assert first.paper.arxiv_id == "2401.00001"
    assert first.paper.year == 2024
    assert first.paper.citation_count == 9
    assert first.paper.doc_type is DocType.CONFERENCE_PAPER
    assert first.paper.external_ids["corpus_id"] == "900000001"
    assert first.upstream_id == "0" * 39 + "1"


def test_semantic_scholar_open_access_comes_with_license() -> None:
    adapter, _ = make_adapter(
        SemanticScholarAdapter, routes_with_fixture(S2_URL)
    )
    candidates = adapter.fetch(make_query()).candidates

    assert candidates[0].paper.oa_status is OAStatus.OPEN
    assert candidates[0].paper.oa_url == "https://example.org/fixture-s2-001.pdf"
    assert candidates[0].license == "CCBY"

    no_pdf = candidates[1]
    assert no_pdf.paper.oa_status is OAStatus.UNKNOWN
    assert no_pdf.paper.oa_url is None


def test_semantic_scholar_missing_data_is_an_error() -> None:
    adapter, _ = make_adapter(SemanticScholarAdapter, {S2_URL: '{"total": 0}'})
    result = adapter.fetch(make_query())
    assert result.ok is False
    assert result.error.details["where"] == "semantic_scholar.data"


def test_semantic_scholar_rate_limit_is_reported_as_rate_limited() -> None:
    adapter, _ = make_adapter(
        SemanticScholarAdapter,
        {S2_URL: text_response(S2_URL, "too many", status_code=429)},
    )
    result = adapter.fetch(make_query())
    assert result.error.code is ErrorCode.RATE_LIMITED
    assert result.candidates == []


# ================================================================ 共性约束


@pytest.mark.parametrize(
    ("adapter_cls", "url"),
    [
        (OpenAlexAdapter, OPENALEX_URL),
        (CrossrefAdapter, CROSSREF_URL),
        (ArxivAdapter, ARXIV_URL),
        (SemanticScholarAdapter, S2_URL),
    ],
)
def test_every_adapter_rejects_success_payload_with_wrong_shape(
    adapter_cls, url
) -> None:
    """四个源都必须把"格式变了"和"没有结果"分开，不能一律当成空结果。"""
    wrong_shape = (
        "not xml at all" if adapter_cls is ArxivAdapter else '{"unexpected": true}'
    )
    adapter, _ = make_adapter(adapter_cls, {url: wrong_shape})
    result = adapter.fetch(make_query())
    assert result.ok is False
    assert result.candidates == []
    assert result.error.code is ErrorCode.UPSTREAM_UNAVAILABLE


@pytest.mark.parametrize(
    ("adapter_cls", "url"),
    [
        (OpenAlexAdapter, OPENALEX_URL),
        (CrossrefAdapter, CROSSREF_URL),
        (ArxivAdapter, ARXIV_URL),
        (SemanticScholarAdapter, S2_URL),
    ],
)
def test_every_adapter_reports_disabled_state_without_network(
    adapter_cls, url
) -> None:
    """禁用某个源时必须短路，且不发出任何请求。"""
    http = FakeHttpClient({url: "{}"})
    adapter = adapter_cls(
        settings=AdapterSettings(enabled=False, mailto=TEST_MAILTO),
        http=http,
        limiter=NullRateLimiter(),
        monotonic=lambda: 0.0,
        now=lambda: NOW,
    )
    result = adapter.fetch(make_query())
    assert result.ok is False
    assert result.error.code is ErrorCode.ONLINE_DISABLED
    assert http.call_count == 0


@pytest.mark.parametrize(
    ("adapter_cls", "url"),
    [
        (OpenAlexAdapter, OPENALEX_URL),
        (CrossrefAdapter, CROSSREF_URL),
        (ArxivAdapter, ARXIV_URL),
        (SemanticScholarAdapter, S2_URL),
    ],
)
def test_every_adapter_warns_when_contact_email_missing(
    adapter_cls, url
) -> None:
    """缺联系邮箱不致命，但必须留下告警，不能悄悄跑在公共池里。"""
    routes = routes_with_fixture(url)
    adapter, _ = make_adapter(adapter_cls, routes, mailto=None)
    result = adapter.fetch(make_query())
    if adapter.requires_mailto:
        assert any(w.code == "mailto_missing" for w in result.warnings)
    else:
        assert not any(w.code == "mailto_missing" for w in result.warnings)


def test_all_fixture_urls_are_covered() -> None:
    """样本与适配器地址必须一一对应，避免改了 URL 却忘了改样本。"""
    assert set(FIXTURES) == {OPENALEX_URL, CROSSREF_URL, ARXIV_URL, FAKE_S2_URL}
