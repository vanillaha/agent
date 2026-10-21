"""跨源检索编排测试（M2）。

这一层最该被问的问题是**行为问题**，而不是字段问题：

- 没授权联网时，是不是真的一条请求都没发？
- 一个源挂了，其他源还活着吗？
- 失败会不会被伪装成"没搜到"？
- 检索记录里能不能看出"这轮到底发生了什么"？

场景划分：正常 / 边界 / 失败 / 安全。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from paper_evidence_qa.schemas import (
    ErrorCode,
    PaperQARequest,
    RetrievalTarget,
    SourceName,
)
from paper_evidence_qa.sources.registry import (
    ONLINE_SOURCE_NAMES,
    SearchService,
    SourceRegistry,
    default_settings,
)

from fakes import (
    ALL_SOURCE_URLS,
    OPENALEX_URL,
    S2_URL,
    TEST_API_KEY,
    TEST_MAILTO,
    FakeHttpClient,
    default_routes,
    empty_routes,
    load_fixture_text,
    text_response,
)

NOW = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)


def make_request(**overrides):
    data = {
        "request_id": "req_0000000000000001",
        "requested_at": NOW,
        "question": "长上下文下的检索增强是否退化？",
        "keywords": ["long context", "retrieval"],
        "year_from": 2020,
        "year_to": 2025,
        "allow_online": True,
        "max_downloads": 0,
    }
    data.update(overrides)
    return PaperQARequest(**data)


def make_service(routes=None, *, mailto=TEST_MAILTO, api_key=None, replay_only=False):
    http = FakeHttpClient(routes if routes is not None else default_routes())
    registry = SourceRegistry.with_defaults(
        http=http,
        mailto=mailto,
        semantic_scholar_api_key=api_key,
        replay_only=replay_only,
        now=lambda: NOW,
        monotonic=lambda: 0.0,
    )
    return SearchService(registry, now=lambda: NOW, monotonic=lambda: 0.0), http


# ---------------------------------------------------------------- 正常场景


def test_search_collects_candidates_from_every_source() -> None:
    service, http = make_service()
    outcome = service.search(make_request())

    sources_used = {candidate.paper.sources[0] for candidate in outcome.candidates}
    assert sources_used == set(ONLINE_SOURCE_NAMES)
    assert len(outcome.retrieval_records) == len(ONLINE_SOURCE_NAMES)
    assert outcome.online_disabled is False
    assert http.call_count == len(ONLINE_SOURCE_NAMES)


def test_search_records_every_source_for_reproducibility() -> None:
    service, _ = make_service()
    outcome = service.search(make_request())

    for record in outcome.retrieval_records:
        assert record.target is RetrievalTarget.ONLINE
        assert record.round_index == 1
        assert record.query == "long context retrieval"
        assert record.filters["year_from"] == 2020
        assert record.filters["year_to"] == 2025
        assert record.error is None
        assert record.kept_count is not None
        assert record.kept_count <= record.result_count


def test_search_keeps_failed_and_empty_results_distinguishable() -> None:
    """成功但零结果：``error`` 为空。失败：``error`` 非空。两者绝不混淆。"""
    service, _ = make_service(empty_routes())
    outcome = service.search(make_request())
    assert outcome.candidates == []
    assert outcome.failed_sources == []
    assert all(record.error is None for record in outcome.retrieval_records)


def test_search_can_be_limited_to_selected_sources() -> None:
    service, http = make_service()
    outcome = service.search(make_request(), sources=[SourceName.ARXIV])

    assert http.call_count == 1
    assert len(outcome.retrieval_records) == 1
    assert outcome.retrieval_records[0].source_name is SourceName.ARXIV
    assert set(ONLINE_SOURCE_NAMES) - set(outcome.disabled_sources) == {
        SourceName.ARXIV
    }


def test_round_index_is_propagated() -> None:
    service, _ = make_service()
    outcome = service.search(make_request(), round_index=3)
    assert all(record.round_index == 3 for record in outcome.retrieval_records)


def test_query_text_override_is_used() -> None:
    """M7 在后续轮次会给出新的检索角度，此时不应再用第一轮的检索式。"""
    service, http = make_service()
    service.search(make_request(), query_text="hybrid retrieval evaluation")
    assert http.last_params_for(OPENALEX_URL)["search"] == "hybrid retrieval evaluation"


# ---------------------------------------------------------------- 边界场景


def test_search_rejects_non_positive_round_index() -> None:
    service, _ = make_service()
    with pytest.raises(ValueError):
        service.search(make_request(), round_index=0)


def test_search_with_no_available_source_warns_instead_of_failing() -> None:
    http = FakeHttpClient(default_routes())
    registry = SourceRegistry.with_defaults(
        http=http,
        mailto=TEST_MAILTO,
        enabled={name: False for name in ONLINE_SOURCE_NAMES},
        now=lambda: NOW,
        monotonic=lambda: 0.0,
    )
    service = SearchService(registry, now=lambda: NOW, monotonic=lambda: 0.0)
    outcome = service.search(make_request())

    assert outcome.candidates == []
    assert outcome.online_disabled is False
    assert http.call_count == 0
    assert any(w.code == "no_source_selected" for w in outcome.warnings)


def test_disabled_source_is_reported_and_skipped() -> None:
    http = FakeHttpClient(default_routes())
    registry = SourceRegistry.with_defaults(
        http=http,
        mailto=TEST_MAILTO,
        enabled={SourceName.SEMANTIC_SCHOLAR: False},
        now=lambda: NOW,
        monotonic=lambda: 0.0,
    )
    service = SearchService(registry, now=lambda: NOW, monotonic=lambda: 0.0)
    outcome = service.search(make_request())

    assert SourceName.SEMANTIC_SCHOLAR in outcome.disabled_sources
    assert S2_URL not in http.urls()


def test_query_without_keywords_falls_back_to_question() -> None:
    service, http = make_service()
    service.search(make_request(keywords=[]))
    assert http.last_params_for(OPENALEX_URL)["search"] == "长上下文下的检索增强是否退化？"


def test_open_access_only_is_propagated_and_recorded() -> None:
    service, http = make_service()
    outcome = service.search(make_request(), open_access_only=True)

    assert "is_oa:true" in http.last_params_for(OPENALEX_URL)["filter"]
    for record in outcome.retrieval_records:
        assert record.filters["open_access_only"] is True


# ---------------------------------------------------------------- 失败场景


def test_one_failing_source_does_not_break_the_others() -> None:
    routes = default_routes()
    routes[S2_URL] = text_response(S2_URL, "server error", status_code=500)
    service, _ = make_service(routes)
    outcome = service.search(make_request())

    assert outcome.candidates, "其余来源必须照常返回候选"
    assert SourceName.SEMANTIC_SCHOLAR in outcome.failed_sources
    failed = next(
        record
        for record in outcome.retrieval_records
        if record.source_name is SourceName.SEMANTIC_SCHOLAR
    )
    assert failed.error is not None
    assert failed.error.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert any(w.code == "source_failed" for w in outcome.warnings)


def test_every_source_failing_still_returns_a_usable_outcome() -> None:
    routes = {url: text_response(url, "down", status_code=503) for url in ALL_SOURCE_URLS}
    service, _ = make_service(routes)
    outcome = service.search(make_request())

    assert outcome.candidates == []
    assert set(outcome.failed_sources) == set(ONLINE_SOURCE_NAMES)
    assert len(outcome.warnings) >= len(ONLINE_SOURCE_NAMES)


def test_unexpected_adapter_defect_is_labelled_internal_error() -> None:
    """代码缺陷不许说成"上游不可用" —— 那会把排查方向带偏。"""
    routes = default_routes()
    routes[OPENALEX_URL] = RuntimeError("模拟适配器内部缺陷")
    service, _ = make_service(routes)
    outcome = service.search(make_request())

    failed = next(
        record
        for record in outcome.retrieval_records
        if record.source_name is SourceName.OPENALEX
    )
    assert failed.error is not None
    assert failed.error.code is ErrorCode.INTERNAL_ERROR
    assert failed.error.details["exception"] == "RuntimeError"
    assert outcome.candidates, "其他来源仍应返回结果"


def test_rate_limited_source_is_reported_as_rate_limited() -> None:
    routes = default_routes()
    routes[S2_URL] = text_response(S2_URL, "429", status_code=429)
    service, _ = make_service(routes)
    outcome = service.search(make_request())

    failed = next(
        record
        for record in outcome.retrieval_records
        if record.source_name is SourceName.SEMANTIC_SCHOLAR
    )
    assert failed.error.code is ErrorCode.RATE_LIMITED


# ---------------------------------------------------------------- 安全场景


def test_offline_search_sends_no_request_at_all() -> None:
    """核心安全行为：未授权联网时零网络调用，且这是正常状态而非失败。"""
    service, http = make_service()
    outcome = service.search(make_request(allow_online=False))

    assert http.call_count == 0
    assert outcome.online_disabled is True
    assert outcome.candidates == []
    assert set(outcome.disabled_sources) == set(ONLINE_SOURCE_NAMES)
    assert any(w.code == "online_disabled" for w in outcome.warnings)


def test_offline_search_still_writes_audit_records() -> None:
    """"这轮为什么一个候选都没有"必须在记录里看得见，不能是一团迷雾。"""
    service, _ = make_service()
    outcome = service.search(make_request(allow_online=False))

    assert len(outcome.retrieval_records) == len(ONLINE_SOURCE_NAMES)
    for record in outcome.retrieval_records:
        assert record.error is not None
        assert record.error.code is ErrorCode.ONLINE_DISABLED
        assert record.result_count == 0
        assert record.kept_count == 0


def test_offline_outcome_never_carries_candidates() -> None:
    service, _ = make_service()
    outcome = service.search(make_request(allow_online=False))
    assert outcome.candidates == []


def test_api_key_never_leaks_into_records_or_warnings() -> None:
    """密钥只能出现在请求头里，不得进入检索记录、告警或候选。"""
    service, http = make_service(api_key=TEST_API_KEY)
    outcome = service.search(make_request())

    serialized = outcome.model_dump_json()
    assert TEST_API_KEY not in serialized
    assert http.last_headers_for(S2_URL)["x-api-key"] == TEST_API_KEY


def test_api_key_is_only_given_to_semantic_scholar() -> None:
    settings = default_settings(
        mailto=TEST_MAILTO, semantic_scholar_api_key=TEST_API_KEY
    )
    assert settings[SourceName.SEMANTIC_SCHOLAR].api_key == TEST_API_KEY
    for name in ONLINE_SOURCE_NAMES:
        if name is not SourceName.SEMANTIC_SCHOLAR:
            assert settings[name].api_key is None


def test_candidates_are_not_deduplicated_by_design() -> None:
    """跨源去重是 M4 的职责。M2 必须如实返回，不能偷偷合并——
    否则 M4 就没有输入可做了。"""
    routes = default_routes()
    routes[S2_URL] = load_fixture_text("semantic_scholar_search.json")
    service, _ = make_service(routes)
    outcome = service.search(make_request())

    arxiv_ids: dict[str, int] = {}
    for candidate in outcome.candidates:
        if candidate.paper.arxiv_id:
            arxiv_ids[candidate.paper.arxiv_id] = (
                arxiv_ids.get(candidate.paper.arxiv_id, 0) + 1
            )
    assert any(count > 1 for count in arxiv_ids.values()), (
        "样本里同一篇论文应在多个来源各出现一次，用于验证 M2 不做去重"
    )


# ---------------------------------------------------------------- 注册表


def test_registry_rejects_empty_adapter_set() -> None:
    with pytest.raises(ValueError):
        SourceRegistry({})


def test_registry_enabled_order_is_stable() -> None:
    service, _ = make_service()
    first = [adapter.name for adapter in service.registry.enabled_adapters()]
    second = [adapter.name for adapter in service.registry.enabled_adapters()]
    assert first == second == list(ONLINE_SOURCE_NAMES)


def test_local_kb_is_not_an_online_source() -> None:
    """本地知识库属 M5/M6，混进在线来源会让"关闭联网"的语义失效。"""
    assert SourceName.LOCAL_KB not in ONLINE_SOURCE_NAMES
    assert SourceName.USER_UPLOAD not in ONLINE_SOURCE_NAMES
