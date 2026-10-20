"""No-network contracts for the optional typed evaluator and repair routing."""
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from app.services.agent_persistent import build_agent
from app.services.answer_audit import CRITERIA, EvidenceAuditor, serialize_input
from app.services.turn_budget import turn_budget


def settings():
    return SimpleNamespace(default_model='configured-model', agent_audit_model='configured-audit-model',
        agent_audit_timeout=2, agent_reasoning_effort='medium', agent_service_tier='priority',
        openai_base_url='https://example.invalid/v1', openai_api_key=SecretStr('test-placeholder'))


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['decisions', 'responses'])
async def test_same_input_and_criteria_for_both_interfaces(mode):
    payload = serialize_input('question', 'draft', [], {})
    def handler(request):
        body = json.loads(request.content)
        assert body['input'] == payload and body['model'] == 'configured-audit-model'
        if mode == 'decisions':
            assert [q['name'] for q in body['questions']] == list(CRITERIA)
            assert all(q['type'] == 'predicate' and CRITERIA[q['name']] in q['instructions'] for q in body['questions'])
            raw = {'answers': [{'type': 'predicate', 'name': name, 'probability': 1.} for name in CRITERIA]}
        else:
            assert body['store'] is False and body['reasoning'] == {'effort': 'medium'}
            assert body['service_tier'] == 'priority'
            raw = {'status': 'completed', 'output': [{'type': 'message', 'content': [
                {'type': 'output_text', 'text': json.dumps(dict.fromkeys(CRITERIA, 1.))}]}]}
        return httpx.Response(200, json=raw)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await EvidenceAuditor(settings(), client).evaluate(payload, mode=mode)
    assert result['passed'] is True and result['status'] == 'completed'
    assert len(result['input_sha256']) == 64


@pytest.mark.parametrize('bad', [None, True, '1', float('nan'), -1, 2])
@pytest.mark.asyncio
async def test_invalid_probability_cannot_certify_quality(bad):
    raw = {'answers': [{'type': 'predicate', 'name': name, 'probability': bad} for name in CRITERIA]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=json.dumps(raw)))) as client:
        result = await EvidenceAuditor(settings(), client).evaluate('{}', mode='decisions')
    assert result['status'] == 'unavailable' and result['passed'] is False


@pytest.mark.parametrize('raw', [{'answers': []}, {'answers': [{'type': 'refusal', 'name': name} for name in CRITERIA]},
    {'answers': [{'type': 'predicate', 'name': 'requested_parts', 'probability': 1.}] * 4}])
@pytest.mark.asyncio
async def test_refusal_missing_and_duplicate_answers_are_not_passes(raw):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=raw))) as client:
        result = await EvidenceAuditor(settings(), client).evaluate('{}', mode='decisions')
    assert result['status'] == 'unavailable' and result['passed'] is False


@pytest.mark.asyncio
async def test_auth_failure_has_no_retry_or_sensitive_error_text():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(401, text='private provider body')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await EvidenceAuditor(settings(), client).evaluate('{}', mode='decisions')
    assert len(calls) == 1 and result['error_type'] == 'HTTPStatusError'
    assert 'private' not in json.dumps(result) and 'test-placeholder' not in json.dumps(result)


class RAG:
    supports_prepared_evidence = True
    config = SimpleNamespace(request_timeout=2)
    def retrieve(self, query, **kwargs):
        return {'status': 'ok', 'sources': [{'chunk_id': 'a', 'file_name': 'manual.txt', 'product': 'civil'}],
                'contexts': ['Parameter A uses unit X. Parameter B uses unit Y.']}


class Model:
    def __init__(self):
        self.calls, self.audits = [], []
    async def ainvoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return AIMessage(content='Parameter A. [1]' if len(self.calls) == 1 else 'Parameter A and B. [1]')
    async def audit_answer(self, payload, *, mode):
        self.audits.append(json.loads(payload))
        return {'status': 'completed', 'passed': len(self.audits) > 1,
                'failures': ['requested_parts'] if len(self.audits) == 1 else []}


@pytest.mark.asyncio
async def test_failed_check_repairs_with_search_tools_available_and_preserves_evidence(monkeypatch):
    monkeypatch.setattr('app.chat.service._o200k_base_encoding', lambda: None)
    model = Model()
    graph = build_agent(InMemorySaver(), model=model, rag_provider=RAG)
    state = await graph.ainvoke({'messages': [HumanMessage(content='Опиши параметры A и B.')]},
        {'configurable': {'thread_id': 'audit', 'answer_audit': 'decisions'}})
    assert len(model.calls) == len(model.audits) == 2
    assert all('tool_choice' not in kwargs for _, kwargs in model.calls)
    assert model.audits[0]['evidence'] == model.audits[1]['evidence']
    assert state['audit_results'][-1]['passed'] is True
    assert 'Parameter A and B' in state['messages'][-1].content


@pytest.mark.asyncio
async def test_disabled_path_never_calls_evaluator(monkeypatch):
    monkeypatch.setattr('app.chat.service._o200k_base_encoding', lambda: None)
    model = Model()
    graph = build_agent(InMemorySaver(), model=model, rag_provider=RAG)
    await graph.ainvoke({'messages': [HumanMessage(content='Опиши параметры A и B.')]},
        {'configurable': {'thread_id': 'off'}})
    assert len(model.calls) == 1 and model.audits == []


@pytest.mark.asyncio
async def test_optional_audit_skips_near_outer_deadline_without_using_provider():
    def forbidden(_):
        raise AssertionError('Optional audit must preserve the completed draft')
    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client, turn_budget(.1):
        result = await EvidenceAuditor(settings(), client).evaluate('{}', mode='decisions')
    assert result['status'] == 'skipped_budget' and result['passed'] is False


@pytest.mark.asyncio
async def test_explicit_second_rejection_is_not_reported_as_complete(monkeypatch):
    monkeypatch.setattr('app.chat.service._o200k_base_encoding', lambda: None)
    model = Model()
    async def reject(*args, **kwargs):
        return {'status': 'completed', 'passed': False, 'failures': ['grounding']}
    model.audit_answer = reject
    graph = build_agent(InMemorySaver(), model=model, rag_provider=RAG)
    state = await graph.ainvoke({'messages': [HumanMessage(content='Опиши параметры A и B.')]},
        {'configurable': {'thread_id': 'reject', 'answer_audit': 'decisions'}})
    assert len(model.calls) == 2 and state['audit_unresolved'] is True
    assert 'подтвердить не удалось' in state['messages'][-1].content


@pytest.mark.asyncio
async def test_last_prepared_turn_synthesizes_evidence_instead_of_another_search(monkeypatch):
    monkeypatch.setattr('app.chat.service._o200k_base_encoding', lambda: None)
    class SearchingModel:
        def __init__(self):
            self.options = []
        async def ainvoke(self, messages, **kwargs):
            self.options.append(kwargs)
            if kwargs.get('tool_choice') == 'none':
                return AIMessage(content='Parameter A uses unit X. Parameter B uses unit Y. [1]')
            return AIMessage(content='', tool_calls=[{'id': str(len(self.options)),
                'name': 'search_knowledge_base', 'args': {'query': 'Параметр A', 'english_query': 'Parameter A'}}])
    model = SearchingModel()
    graph = build_agent(InMemorySaver(), model=model, rag_provider=RAG)
    state = await graph.ainvoke({'messages': [HumanMessage(content='Опиши параметры A и B.')]},
        {'configurable': {'thread_id': 'final-evidence'}})
    assert len(model.options) == 6 and model.options[-1]['tool_choice'] == 'none'
    assert 'Parameter B uses unit Y' in state['messages'][-1].content
    assert state['sources'] and not state.get('answer_incomplete')
