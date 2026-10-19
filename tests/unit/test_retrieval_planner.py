import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.rag import RAGConfig, RAGService
from app.services.retrieval_planner import CachedRetrievalPlanner


def plan_json(*queries: str, question: str = "Compare ALPHA and BETA") -> str:
    return json.dumps({"standalone_question": question, "facets": [
        {"query_original": query, "query_english": "", "explicit_qualifiers": []} for query in queries]})


class FakeLLM:
    def __init__(self, text: str):
        self.text, self.calls = text, []

    def complete(self, prompt: str):
        self.calls.append(prompt)
        return SimpleNamespace(text=self.text)


def config(tmp_path: Path, **values) -> RAGConfig:
    return RAGConfig(data_dir=tmp_path, qdrant_url="http://unused", qdrant_api_key=None,
        collection_name="test", baremetal_collection_name="test_bare", llm_model="configured-model",
        embedding_model="unused", embedding_dimensions=3, allow_external_data_transfer=True,
        product_scope="all", planner_enabled=True, **values)


def evidence(identity: str, score: float = .8) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=identity, text=f"Exact documentation for {identity}.",
        metadata={"source": identity+".pdf", "product": "test", "source_id": identity,
                  "source_hash": identity, "location_id": identity, "extraction_version": "corpus-v1"}), score=score)


def test_strict_plan_is_cached_by_exact_question_history_model_and_prompt(monkeypatch) -> None:
    llm = FakeLLM(plan_json("ALPHA", "BETA"))
    planner = CachedRetrievalPlanner(model="configured-model", llm=llm)
    history = [{"role": "user", "content": "Previous constraint"}]
    first = planner.plan("Compare ALPHA and BETA", history)
    second = planner.plan("Compare ALPHA and BETA", history)
    assert first.telemetry["status"] == "completed"
    assert second.telemetry["status"] == "cache_hit"
    assert len(llm.calls) == 1
    assert first.telemetry["plan_binding_sha256"] == second.telemetry["plan_binding_sha256"]
    assert first.telemetry["semantic_coverage"] is None
    planner.plan("Compare ALPHA and BETA", [{"role": "user", "content": "Changed constraint"}])
    assert len(llm.calls) == 2
    import app.services.retrieval_planner as module
    monkeypatch.setattr(module, "PROMPT_VERSION", "different-version")
    planner.plan("Compare ALPHA and BETA", history)
    assert len(llm.calls) == 3


def test_cache_binding_includes_full_last_history_message_beyond_model_input_bound() -> None:
    llm = FakeLLM(plan_json())
    planner = CachedRetrievalPlanner(model="configured-model", llm=llm)
    a = planner.plan("Compare ALPHA and BETA", [{"role": "user", "content": "x"*800+"A"}])
    b = planner.plan("Compare ALPHA and BETA", [{"role": "user", "content": "x"*800+"B"}])
    assert a.telemetry["history_truncated"] and b.telemetry["history_truncated"]
    assert a.telemetry["plan_binding_sha256"] != b.telemetry["plan_binding_sha256"]
    assert len(llm.calls) == 2


@pytest.mark.parametrize("raw", ["```json {} ```", plan_json("A", "B", "C", "D"),
    '{"standalone_question":"ALPHA","facets":[],"answer":"invented"}',
    '{"standalone_question":true,"facets":[]}'])
def test_invalid_schema_abstains_without_automatic_retry(raw: str) -> None:
    llm = FakeLLM(raw)
    outcome = CachedRetrievalPlanner(model="configured-model", llm=llm).plan("Compare ALPHA and BETA")
    assert outcome.plan is None
    assert outcome.telemetry["status"] == "failed"
    assert len(llm.calls) == 1
    assert "output_sha256" in outcome.telemetry


@pytest.mark.parametrize("facet", [
    {"query_original": "ALPHA v3", "query_english": "", "explicit_qualifiers": []},
    {"query_original": "ALPHA", "query_english": "", "explicit_qualifiers": ["invented Product"]},
])
def test_invented_numeric_or_declared_qualifier_is_rejected(facet: dict) -> None:
    llm = FakeLLM(json.dumps({"standalone_question": "ALPHA v2", "facets": [facet]}))
    outcome = CachedRetrievalPlanner(model="configured-model", llm=llm).plan("ALPHA v2")
    assert outcome.plan is None
    assert outcome.telemetry["status"] == "failed"
    assert len(llm.calls) == 1


def test_copied_explicit_qualifiers_are_retained_in_search_even_if_model_omits_them() -> None:
    raw = json.dumps({"standalone_question": "ALPHA in Product Z v2", "facets": [
        {"query_original": "ALPHA", "query_english": "ALPHA settings", "explicit_qualifiers": ["Product Z v2"]}]})
    outcome = CachedRetrievalPlanner(model="configured-model", llm=FakeLLM(raw)).plan("ALPHA in Product Z v2")
    assert outcome.plan is not None
    queries = outcome.plan.search_queries("ALPHA in Product Z v2")
    assert len(queries) == 1 and "Product Z v2" in queries[0]


def test_deadline_expired_never_invokes_model_and_error_body_is_not_diagnostic() -> None:
    llm = FakeLLM(plan_json())
    outcome = CachedRetrievalPlanner(model="configured-model", llm=llm).plan("ALPHA", deadline=time.perf_counter()-1)
    assert outcome.telemetry["status"] == "deadline_exhausted"
    assert not llm.calls

    def broken(_):
        raise RuntimeError("private provider body should not appear")

    outcome = CachedRetrievalPlanner(model="configured-model", llm=SimpleNamespace(complete=broken)).plan("ALPHA")
    assert outcome.plan is None and outcome.telemetry["error_type"] == "RuntimeError"
    assert "private provider" not in json.dumps(outcome.telemetry)


def test_existing_responses_adapter_is_used_with_bounded_output_and_timeout(monkeypatch) -> None:
    import app.services.runtime_model as runtime
    seen = {}

    class Adapter(FakeLLM):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(plan_json())

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(runtime, "ResponsesTextLLM", Adapter)
    outcome = CachedRetrievalPlanner(model="configured-model", timeout=5.).plan(
        "Compare ALPHA and BETA", deadline=time.perf_counter()+1.)
    assert outcome.plan is not None
    assert seen["max_output_tokens"] == 500
    assert seen["effort"] == "none" and 0 < seen["timeout"] <= 1.
    assert seen["closed"]


def test_planned_retrieval_keeps_original_and_at_most_three_facets_without_condense(tmp_path: Path) -> None:
    original = "Compare ALPHA BETA GAMMA"
    llm = FakeLLM(plan_json("ALPHA", "BETA", "GAMMA", question=original))
    planner = CachedRetrievalPlanner(model="configured-model", llm=llm)
    rag = RAGService(config(tmp_path, retrieval_concurrency=4), retrieval_planner=planner,
                     condense_hook=lambda *_: pytest.fail("Planner must replace condense"))
    calls = []

    def retrieve(query):
        calls.append(query)
        return [evidence(query)]

    rag._retriever = SimpleNamespace(retrieve=retrieve)
    result = rag.retrieve(original, history=[{"role": "user", "content": "Previous question"}],
        translated_question="Long faithful translated question about several independently documented settings")
    assert set(calls) == {original, "ALPHA", "BETA", "GAMMA"}
    assert len(calls) == 4 and len(llm.calls) == 1
    assert result["telemetry"]["retrieval_concurrency"] == 2
    assert result["telemetry"]["plan_mode"] == "planned"
    assert result["telemetry"]["retrieval_planning"]["translated_query_omitted"] == "dense_query_budget"
    assert len(result["telemetry"]["dense_queries"]) == 4
    assert {source["chunk_id"] for source in result["sources"]} == set(calls)
    assert all(row["selected_candidate_ids"] for row in result["telemetry"]["dense_queries"])


def test_planner_failure_uses_original_once_and_never_calls_condense(tmp_path: Path) -> None:
    llm = FakeLLM("not JSON")
    rag = RAGService(config(tmp_path), retrieval_planner=CachedRetrievalPlanner(model="configured-model", llm=llm),
                     condense_hook=lambda *_: pytest.fail("No second LLM after planner failure"))
    calls = []
    rag._retriever = SimpleNamespace(retrieve=lambda query: (calls.append(query) or [evidence("native")]))
    result = rag.retrieve("Original ALPHA question", history=[{"role": "user", "content": "Previous"}])
    assert calls == ["Original ALPHA question"]
    assert len(llm.calls) == 1
    assert result["telemetry"]["plan_mode"] == "planner_fallback"
    assert result["telemetry"]["retrieval_planning"]["status"] == "failed"


def test_planner_disabled_default_preserves_existing_route(tmp_path: Path) -> None:
    base = replace(config(tmp_path), planner_enabled=False)
    rag = RAGService(base, retrieval_planner=SimpleNamespace(plan=lambda *_a, **_kw: pytest.fail("Disabled planner")))
    rag._retriever = SimpleNamespace(retrieve=lambda _: [evidence("native")])
    result = rag.retrieve("Original question")
    assert result["telemetry"]["retrieval_planning"] == {"status": "disabled"}
    assert result["telemetry"]["dense_queries"] == []


def test_planner_settings_are_default_off_and_read_configured_values() -> None:
    assert not RAGConfig.from_settings(SimpleNamespace()).planner_enabled
    value = RAGConfig.from_settings(SimpleNamespace(rag_planner_enabled=True, rag_planner_model="candidate-model",
        rag_planner_effort="low", rag_planner_timeout=2.))
    assert value.planner_enabled and value.planner_model == "candidate-model"
    assert value.planner_effort == "low" and value.planner_timeout == 2.


def test_settings_aliases_keep_planner_disabled_until_explicit_optin(monkeypatch) -> None:
    from app.core.config import Settings
    values = {"_env_file": None, "openai_api_key": "sk-test-only",
              "internal_token": "test-internal-token-0000000000000001",
              "admin_token": "test-admin-token-000000000000000002"}
    monkeypatch.delenv("RAG_PLANNER_ENABLED", raising=False)
    assert Settings(**values).rag_planner_enabled is False
    monkeypatch.setenv("RAG_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_PLANNER_MODEL", "candidate-planner-model")
    monkeypatch.setenv("RAG_PLANNER_EFFORT", "none")
    monkeypatch.setenv("RAG_PLANNER_TIMEOUT", "2")
    configured = Settings(**values)
    assert configured.rag_planner_enabled
    assert configured.rag_planner_model == "candidate-planner-model"
    assert configured.rag_planner_timeout == 2.


def test_planner_cache_is_bounded_and_lru(monkeypatch) -> None:
    import app.services.retrieval_planner as module
    monkeypatch.setattr(module, "MAX_CACHE_ENTRIES", 2)
    llm = FakeLLM(plan_json())
    planner = CachedRetrievalPlanner(model="configured-model", llm=llm)
    for question in ["Compare ALPHA and BETA first", "Compare ALPHA and BETA second",
                     "Compare ALPHA and BETA first", "Compare ALPHA and BETA third", "Compare ALPHA and BETA second"]:
        planner.plan(question)
    assert len(llm.calls) == 4
    assert len(planner._cache) == 2


def test_original_query_cannot_invent_subject_words_without_declared_qualifier() -> None:
    raw = plan_json("InventedSubject", question="ALPHA")
    outcome = CachedRetrievalPlanner(model="configured-model", llm=FakeLLM(raw)).plan("ALPHA")
    assert outcome.plan is None and outcome.telemetry["status"] == "failed"
