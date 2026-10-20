from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest
from llama_index.core.llms import MockLLM
from llama_index.core.schema import NodeWithScore, TextNode

from app.services import lexical
from app.services.rag import RAGConfig, RAGService, RetrievalPlan


def count_tokenizations(monkeypatch):
    original = lexical._uncached_lexical_terms
    calls = Counter()
    lock = Lock()

    def counted(text):
        with lock:
            calls[text] += 1
        return original(text)

    monkeypatch.setattr(lexical, '_uncached_lexical_terms', counted)
    return calls


def test_memo_matches_uncached_tokens_and_keeps_mutable_results_independent(monkeypatch):
    texts = ['', 'Boundary boundaries stiffness', 'Грунты: жёсткость, жесткость и модели',
             'δ E50 3D β; MODEL_models 2021', 'running classes dependencies', 'soil ' * 300]
    expected = [lexical.lexical_terms(text) for text in texts]
    calls = count_tokenizations(monkeypatch)
    with lexical.memoized_terms():
        for text, tokens in zip(texts, expected, strict=True):
            first = lexical.lexical_terms(text)
            assert first == tokens
            first.append('caller mutation')
            assert lexical.lexical_terms(text) == tokens
    assert calls == Counter(dict.fromkeys(texts, 1))


def test_memo_nested_scopes_and_failure_restore_previous_scope(monkeypatch):
    calls = count_tokenizations(monkeypatch)
    text = 'boundary stiffness'
    with lexical.memoized_terms():
        lexical.lexical_terms(text)
        with pytest.raises(RuntimeError), lexical.memoized_terms():
            lexical.lexical_terms(text)
            raise RuntimeError('test failure')
        lexical.lexical_terms(text)
        assert calls[text] == 2
    lexical.lexical_terms(text)
    lexical.lexical_terms(text)
    assert calls[text] == 4  # Calls outside a selection are never retained.


def test_memo_isolated_between_concurrent_threads_and_later_requests(monkeypatch):
    calls = count_tokenizations(monkeypatch)
    barrier = Barrier(2)
    text = 'boundary stiffness'

    def request(_):
        with lexical.memoized_terms():
            barrier.wait(timeout=10)
            first = lexical.lexical_terms(text)
            barrier.wait(timeout=10)
            return first, lexical.lexical_terms(text)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(request, range(2)))
    assert results[0] == results[1]
    assert calls[text] == 2
    with lexical.memoized_terms():
        lexical.lexical_terms(text)
    assert calls[text] == 3


def selection_fixture(tmp_path):
    config = RAGConfig(data_dir=tmp_path, qdrant_url='http://unused.invalid', qdrant_api_key=None,
                       collection_name='memo-test', baremetal_collection_name='memo-baremetal',
                       llm_model='test', embedding_model='test', embedding_dimensions=3,
                       docstore_path=tmp_path / 'unused.json', lexical_retrieval_enabled=False)
    service = RAGService(config, client=SimpleNamespace(), llm=MockLLM())
    service._related_evidence = SimpleNamespace(resolve=lambda *_a, **_kw: [], source_relative_path=lambda _: None)
    texts = ['Boundary stiffness soil parameters. ' * 24,
             'Граничные условия и жесткость грунта. ' * 24,
             'Boundary mesh displacement. ' * 24]
    nodes = [NodeWithScore(node=TextNode(id_=f'node-{i}', text=text, metadata={
        'file_name': f'manual-{i}.pdf', 'product': 'gts_nx', 'version': 'unknown',
        'extraction_version': 'corpus-v1'}), score=.9 - i / 10) for i, text in enumerate(texts)]
    question = 'Boundary stiffness soil'
    kwargs = {'question': question, 'plan': RetrievalPlan(question, (question,), 'standalone'),
              'rankings': [nodes], 'lexical_candidates': nodes, 'lexical_query': question,
              'overview_question': False, 'preserve_full_context': True,
              'translated_question': 'Граничные условия и жесткость грунта'}
    return service, kwargs


def test_selection_matches_uncached_oracle_and_tokenizes_each_text_once(tmp_path, monkeypatch):
    service, kwargs = selection_fixture(tmp_path)
    calls = count_tokenizations(monkeypatch)
    expected = RAGService._select_context.__wrapped__(service, **deepcopy(kwargs))
    uncached_count = sum(calls.values())
    calls.clear()
    actual = service._select_context(**deepcopy(kwargs))
    assert actual == expected
    assert all(count == 1 for count in calls.values())
    assert sum(calls.values()) < uncached_count
    # A later selection starts fresh rather than depending on corpus/query TTLs.
    first_count = calls.copy()
    assert service._select_context(**deepcopy(kwargs)) == expected
    assert calls == first_count + first_count
