"""Elapsed-time limits must preserve evidence and never replay an accepted turn."""
from __future__ import annotations

import asyncio
from contextvars import copy_context
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.chat.agent import ChatAgent
from app.chat.domain import Chat
from app.services.agent_persistent import _model_history
from app.services.parallel_retrieval import (
    independent_searches,
    search_while_translating,
)
from app.services.rag import RAGConfig, RAGService, build_llm_translate_hook
from app.services.retrieval_embedding import OpenAIEmbedding
from app.services.runtime_model import AgentModel
from app.services.turn_budget import AgentTurnTimeout, remaining_seconds, turn_budget
from bot.services.backend_client import BackendClient, BackendStreamError


@pytest.mark.asyncio
async def test_application_lifespan_owns_and_closes_model_connections(monkeypatch):
    from app.services.agent_persistent import agent_lifespan

    model = SimpleNamespace(aclose=AsyncMock())
    factory = Mock(return_value=model)
    monkeypatch.setenv('AGENT_CHECKPOINTER', 'memory')
    monkeypatch.setattr('app.services.agent_persistent._default_model', factory)
    graph_factory = Mock()
    monkeypatch.setattr('app.services.agent_persistent.build_agent', graph_factory)
    app = SimpleNamespace(state=SimpleNamespace())
    async with agent_lifespan(app):
        factory.assert_called_once_with()
        assert graph_factory.call_args.kwargs['model'] is model
        model.aclose.assert_not_awaited()
    model.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_nested_budgets_and_thread_workers_share_the_original_deadline():
    async with turn_budget(.2):
        await asyncio.sleep(.02)
        outer = remaining_seconds(10)
        async with turn_budget(10):
            assert 0 < remaining_seconds(10) <= outer
            outcomes = await asyncio.to_thread(independent_searches, ['a', 'b'],
                lambda _query: [remaining_seconds(10)])
            assert all(0 < row.items[0] <= outer for row in outcomes)
    assert remaining_seconds(10) == 10


@pytest.mark.asyncio
async def test_complete_turn_cancels_all_stages_and_does_not_start_another_generation():
    finished = []
    async def invoke(*_args):
        try:
            await asyncio.sleep(.02)
            await asyncio.sleep(1)
            finished.append('unexpected result')
        finally:
            finished.append('cancelled')
    settings = SimpleNamespace(agent_turn_timeout=.04, chat_model_context_tokens=10000,
        chat_safety_margin_tokens=100, chat_context_window=10)
    graph = SimpleNamespace(ainvoke=AsyncMock(side_effect=invoke))
    agent = ChatAgent(graph, None, settings=settings)
    with pytest.raises(AgentTurnTimeout):
        await agent._invoke({}, Chat(owner_external_id='isolated', interface='telegram'))
    assert finished == ['cancelled']
    graph.ainvoke.assert_awaited_once()
    assert remaining_seconds(10) == 10


@pytest.mark.asyncio
async def test_generation_timeout_includes_the_entire_sdk_retry_lifecycle():
    cancelled = []
    async def sdk_attempts(*_args, **_kwargs):
        try:
            while True:
                await asyncio.sleep(.01)  # SDK activity must not renew the clock.
        finally:
            cancelled.append(True)
    client = SimpleNamespace(ainvoke=AsyncMock(side_effect=sdk_attempts))
    settings = SimpleNamespace(default_model='gpt-6-luna', agent_generation_timeout=.03)
    model = AgentModel(settings, [], client=client)
    with pytest.raises(TimeoutError):
        await model.ainvoke([HumanMessage(content='Question')])
    assert cancelled == [True]
    client.ainvoke.assert_awaited_once()


def test_old_reasoning_is_removed_but_current_tool_loop_is_unchanged():
    old = AIMessage(content='Known fact [1].', id='old',
        additional_kwargs={'reasoning': {'encrypted_content': 'old-private-reasoning'}},
        response_metadata={'id': 'old-provider-continuation'})
    current = AIMessage(content='', additional_kwargs={'reasoning': {'encrypted_content': 'current'}},
        tool_calls=[{'id': 'search', 'name': 'search_knowledge_base', 'args': {'query': 'fact'}}])
    tool = ToolMessage(content='exact current evidence', tool_call_id='search')
    result = _model_history([HumanMessage(content='Previous question'), old,
                             HumanMessage(content='Current question'), current, tool])
    assert result[1].id == old.id and 'Known fact .' in result[1].content
    assert result[1].additional_kwargs == result[1].response_metadata == {}
    assert result[-2:] == [current, tool]
    assert old.additional_kwargs['reasoning']['encrypted_content'] == 'old-private-reasoning'


def test_original_search_starts_before_translation_and_retains_its_result():
    started = Event()
    def search():
        started.set()
        return ['unchanged Russian evidence']
    def translate():
        assert started.wait(1)
        return 'complete English question'
    original, translation = search_while_translating(search, translate)
    assert original == ['unchanged Russian evidence']
    assert translation == 'complete English question'


def test_failed_translation_does_not_discard_original_recall():
    def translate():
        raise TimeoutError('private provider response')
    assert search_while_translating(lambda: ['original evidence'], translate) == (['original evidence'], None)


@pytest.mark.parametrize('model', ['gpt-6-luna', 'gpt-5.4-mini'])
def test_translation_adapter_never_multiplies_its_timeout_with_sdk_retries(monkeypatch, model):
    factory = Mock()
    factory.return_value.complete.return_value = SimpleNamespace(text='Which elements exist?')
    target = ('app.services.runtime_model.ResponsesTextLLM' if model.startswith('gpt-6')
              else 'app.services.retrieval_translation.LlamaOpenAI')
    monkeypatch.setattr(target, factory)
    config = RAGConfig.from_settings(SimpleNamespace(default_model=model,
        rag_llm_model=model, rag_allow_external_data_transfer=True))
    hook = build_llm_translate_hook(config)
    assert hook('Какие элементы есть?') == 'Which elements exist?'
    assert factory.call_args.kwargs['max_retries'] == 0
    assert factory.call_args.kwargs['timeout'] <= 8


def test_query_embedding_adapter_disables_hidden_ten_attempt_retry_chain(monkeypatch):
    factory = Mock()
    monkeypatch.setattr('app.services.rag.OpenAIEmbedding', factory)
    service = RAGService(RAGConfig.from_settings(SimpleNamespace()))
    service._embedding_model()
    assert factory.call_args.kwargs['max_retries'] == 0


class Heartbeats(httpx.AsyncByteStream):
    closed = False
    async def __aiter__(self):
        while True:
            yield b': heartbeat\n\n'
            await asyncio.sleep(.003)
    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_heartbeat_cannot_keep_bot_waiting_forever_and_stream_is_closed(monkeypatch):
    monkeypatch.setattr('bot.services.backend_client.TURN_TIMEOUT_SECONDS', .03)
    stream = Heartbeats()
    attempts = []
    def handler(request):
        attempts.append(request)
        return httpx.Response(200, stream=stream)
    async with httpx.AsyncClient(base_url='http://test', transport=httpx.MockTransport(handler)) as http:
        client = BackendClient('http://test', client=http, internal_token='test-only')
        chat_id = uuid4()
        client._chat_owners[str(chat_id)] = 'isolated'
        with pytest.raises(BackendStreamError) as error:
            _ = [chunk async for chunk in client.send_message(chat_id, 'Question')]
    assert error.value.code == 'chat_turn_timeout'
    assert stream.closed and len(attempts) == 1


@pytest.mark.asyncio
async def test_accepted_stream_is_not_replayed_after_read_failure_before_first_token():
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b': heartbeat\n\n'
            raise httpx.ReadError('lost connection after acceptance')
    attempts = []
    def handler(request):
        attempts.append(request)
        return httpx.Response(200, stream=BrokenStream())
    async with httpx.AsyncClient(base_url='http://test', transport=httpx.MockTransport(handler)) as http:
        client = BackendClient('http://test', client=http, internal_token='test-only')
        chat_id = uuid4()
        client._chat_owners[str(chat_id)] = 'isolated'
        with pytest.raises(httpx.ReadError):
            _ = [chunk async for chunk in client.send_message(chat_id, 'Question')]
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_bot_deadline_never_cancels_consumer_delivering_a_ready_answer(monkeypatch):
    monkeypatch.setattr('bot.services.backend_client.TURN_TIMEOUT_SECONDS', .02)
    async def ready(*_args):
        yield 'complete answer'
    async with httpx.AsyncClient(base_url='http://test') as http:
        client = BackendClient('http://test', client=http, internal_token='test-only')
        monkeypatch.setattr(client, '_send_message', ready)
        delivered = []
        async for chunk in client.send_message(uuid4(), 'Question'):
            await asyncio.sleep(.05)
            delivered.append(chunk)
    assert delivered == ['complete answer']


@pytest.mark.asyncio
async def test_embedding_request_uses_residual_budget_and_never_starts_after_deadline():
    requests = []
    def handler(request):
        requests.append(request)
        assert 0 < request.extensions['timeout']['read'] <= 2
        return httpx.Response(200, json={'data': [{'embedding': [1., 2., 3.], 'index': 0}],
            'model': 'text-embedding-3-small', 'object': 'list',
            'usage': {'prompt_tokens': 1, 'total_tokens': 1}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        embedding = OpenAIEmbedding(model='text-embedding-3-small', api_key='test-only',
            timeout=30, max_retries=0, http_client=http)
        async with turn_budget(2):
            assert await asyncio.to_thread(embedding.get_query_embedding, 'public question') == [1., 2., 3.]
        assert embedding.timeout == 30  # A concurrent request retains its own budget.
        async with turn_budget(.01):
            expired_context = copy_context()
        await asyncio.sleep(.02)
        with pytest.raises(AgentTurnTimeout):
            expired_context.run(embedding.get_query_embedding, 'must not call provider')
        assert len(requests) == 1


@pytest.mark.asyncio
async def test_non_responses_translation_passes_residual_timeout(monkeypatch):
    factory = Mock()
    factory.return_value.complete.return_value = SimpleNamespace(text='English question')
    monkeypatch.setattr('app.services.retrieval_translation.LlamaOpenAI', factory)
    config = RAGConfig.from_settings(SimpleNamespace(rag_llm_model='gpt-5.4-mini',
        rag_allow_external_data_transfer=True))
    hook = build_llm_translate_hook(config)
    async with turn_budget(.2):
        assert await asyncio.to_thread(hook, 'Русский вопрос') == 'English question'
    timeout = factory.return_value.complete.call_args.kwargs['timeout']
    assert 0 < timeout.read <= .2
    assert timeout.connect <= timeout.read
