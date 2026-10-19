"""Offline provider/cache boundary regressions for the opt-in acceptance runner."""
from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from app.eval.acceptance_budget import (
    BudgetExceeded,
    BudgetLedger,
    build_embedding,
    fingerprint,
    metered_create,
)


def response(entries, *, incoming=10):
    return SimpleNamespace(data=[SimpleNamespace(index=index, embedding=vector) for index, vector in entries],
                           usage=SimpleNamespace(model_dump=lambda: {'input_tokens': incoming, 'output_tokens': 0}))


def install_fake_provider(monkeypatch, create):
    options = []
    def factory(**kwargs):
        options.append(kwargs)
        return SimpleNamespace(embeddings=SimpleNamespace(create=create))
    monkeypatch.setattr('openai.OpenAI', factory)
    return options


def test_metered_embeddings_deduplicate_realign_and_persist_cache(tmp_path, monkeypatch):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        assert kwargs['input'] == ['alpha', 'beta']
        return response([(1, [3., 4.]), (0, [1., 2.])])
    options = install_fake_provider(monkeypatch, create)
    ledger = BudgetLedger(tmp_path / 'budget.sqlite3', limit_usd=1)
    kwargs = {'ledger': ledger, 'model': 'text-embedding-3-small', 'dimensions': 2, 'api_key': 'fixture-not-a-secret',
              'api_base': 'https://api.openai.com/v1', 'cache_path': tmp_path / 'cache.sqlite3'}
    model = build_embedding(**kwargs)
    assert model._get_text_embeddings(['alpha', 'beta', 'alpha']) == [[1., 2.], [3., 4.], [1., 2.]]
    assert build_embedding(**kwargs)._get_text_embeddings(['beta', 'alpha']) == [[3., 4.], [1., 2.]]
    assert len(calls) == ledger.summary()['calls'] == 1
    assert all(option['max_retries'] == 0 for option in options)


@pytest.mark.parametrize('entries', [
    [(0, [1., 2.]), (0, [3., 4.])],
    [(0, [1., 2.])],
    [(0, [1., 2., 3.]), (1, [3., 4.])],
    [(0, [float('nan'), 2.]), (1, [3., 4.])],
    [(0, [float('inf'), 2.]), (1, [3., 4.])],
])
def test_malformed_embedding_response_never_pollutes_cache(tmp_path, monkeypatch, entries):
    calls = []
    install_fake_provider(monkeypatch, lambda **kwargs: calls.append(kwargs) or response(entries))
    ledger = BudgetLedger(tmp_path / 'budget.sqlite3', limit_usd=1)
    cache_path = tmp_path / 'cache.sqlite3'
    model = build_embedding(ledger=ledger, model='text-embedding-3-small', dimensions=2, api_key='fixture', cache_path=cache_path)
    with pytest.raises(ValueError):
        model._get_text_embeddings(['alpha', 'beta'])
    with sqlite3.connect(cache_path) as db:
        assert db.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0] == 0
    assert len(calls) == ledger.summary()['calls'] == 1


def test_provider_failure_has_one_attempt_and_preserves_reservation(tmp_path, monkeypatch):
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise TimeoutError('offline fixture')
    options = install_fake_provider(monkeypatch, fail)
    ledger = BudgetLedger(tmp_path / 'budget.sqlite3', limit_usd=1)
    model = build_embedding(ledger=ledger, model='text-embedding-3-small', dimensions=2, api_key='fixture')
    with pytest.raises(TimeoutError):
        model._get_text_embeddings(['alpha'])
    assert len(calls) == 1 and options[0]['max_retries'] == 0
    assert ledger.summary()['unknown_cost_calls'] == 1
    assert ledger.summary()['accounted_upper_usd'] > 0


@pytest.mark.parametrize('vector', [[1.], [float('nan'), 1.]])
def test_corrupt_cached_vectors_fail_closed_before_return(tmp_path, monkeypatch, vector):
    """Regression: cache hits need the same validation as provider responses."""
    install_fake_provider(monkeypatch, lambda **kwargs: pytest.fail('Corrupt cache must fail closed without provider request'))
    ledger = BudgetLedger(tmp_path / 'budget.sqlite3', limit_usd=1)
    cache_path = tmp_path / 'cache.sqlite3'
    model = build_embedding(ledger=ledger, model='text-embedding-3-small', dimensions=2, api_key='fixture', cache_path=cache_path)
    key = fingerprint(['text-embedding-3-small', 2, 'alpha'])
    with sqlite3.connect(cache_path) as db:
        db.execute('INSERT INTO embeddings VALUES (?,?,0)', (key, json.dumps(vector)))
    with pytest.raises(ValueError):
        model._get_text_embeddings(['alpha'])


def test_large_instructions_cannot_bypass_reservation_before_provider(tmp_path):
    """Regression: count all billed text fields, not only input/messages."""
    ledger = BudgetLedger(tmp_path / 'budget.sqlite3', limit_usd=.0005)
    calls = []
    def provider(**kwargs):
        calls.append(kwargs)
        return response([], incoming=10_000)
    with pytest.raises(BudgetExceeded):
        metered_create(provider, ledger, kind='review', model='gpt-6-luna', input='x',
                       instructions='word ' * 10_000, max_output_tokens=1)
    assert calls == []
    assert ledger.summary()['accounted_upper_usd'] <= ledger.summary()['limit_usd']


def test_server_side_history_requires_explicit_bound_before_provider(tmp_path):
    """Regression: unknown previous response history has no finite text bound."""
    ledger = BudgetLedger(tmp_path / 'budget.sqlite3', limit_usd=1)
    calls = []
    def provider(**kwargs):
        calls.append(kwargs)
        return response([])
    with pytest.raises(ValueError):
        metered_create(provider, ledger, kind='review', model='gpt-6-luna', input='x',
                       previous_response_id='offline-response-history', max_output_tokens=1)
    assert calls == []
