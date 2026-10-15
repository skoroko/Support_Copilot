"""Offline Telegram -> HTTP facade -> SQLite graph -> real RAG pipeline checks."""

from __future__ import annotations

import asyncio
import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from llama_index.core.schema import NodeWithScore, TextNode

from app.chat.agent import ChatAgent
from app.chat.deps import get_chat_service
from app.chat.domain import ChatMessage
from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.routes import router
from app.chat.service import ChatService
from app.core.auth import InMemoryRequestBudget, RequestBudgetLease
from app.core.config import Settings, get_settings
from app.services.agent_persistent import (
    CONTEXT_BUDGET_MESSAGE,
    KNOWLEDGE_UNAVAILABLE,
    build_agent,
    retrieval_worker_tracker,
)
from app.services.rag import NO_ANSWER, RAGConfig, RAGService
from bot.handlers._common import send_rendered_answer
from bot.handlers.approval import approve
from bot.keyboards.inline import approval_kb, encode_message_context
from bot.services.backend_client import BackendClient

INTERNAL = "test-internal-token-0000000000000001"


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.received = []

    async def ainvoke(self, messages, **kwargs):
        self.received.append(messages)
        return next(self.responses)


def search(query="Mesh Generate", call_id="search-1"):
    return AIMessage(
        content="Проверю мануал.",
        tool_calls=[
            {"name": "search_knowledge_base", "args": {"query": query}, "id": call_id}
        ],
    )


def send(call_id="send-1"):
    return AIMessage(
        content="Подготовлен черновик.",
        tool_calls=[
            {
                "name": "send_telegram_message",
                "args": {"chat_id": "4242", "text": "Инцидент передан инженеру."},
                "id": call_id,
            }
        ],
    )


def rag_service(tmp_path, *, product="gts_nx", score=0.9):
    service = RAGService(
        RAGConfig(
            data_dir=tmp_path,
            qdrant_url="http://unused",
            qdrant_api_key=None,
            collection_name="manuals",
            baremetal_collection_name="experiment",
            llm_model="offline",
            embedding_model="offline",
            embedding_dimensions=3,
            allow_external_data_transfer=True,
            condense_llm_enabled=False,
            lexical_retrieval_enabled=False,
            product_scope="strict",
        )
    )
    node = NodeWithScore(
        node=TextNode(
            id_="chunk-mesh",
            text="Mesh Generate: параметры генерации сетки.",
            metadata={
                "file_name": "GTS NX User Manual.pdf",
                "page": 7,
                "doc_id": "manual-gts",
                "product": product,
            },
        ),
        score=score,
    )
    questions = []

    def retrieve(question):
        questions.append(question)
        return [node]

    service._retriever = SimpleNamespace(retrieve=retrieve)
    service._language_model = lambda: pytest.fail(
        "retrieval must not generate an answer"
    )
    return service, questions


def settings():
    return Settings(
        _env_file=None,
        openai_api_key="offline",
        internal_token=INTERNAL,
        moderation_enabled=False,
        rag_enabled=False,
        chat_repository="json",
    )


async def collect(backend, chat, text):
    return "".join(
        str(delta)
        for delta in [item async for item in backend.send_message(chat, text)]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
async def test_compacted_answer_reaches_chat_transport_and_persistent_history(tmp_path, finish_reason):
    from app.services.answer_quality import INCOMPLETE_NOTICE
    rag, _ = rag_service(tmp_path)
    complete = "Mesh Generate задаёт параметры генерации сетки [1]."
    model = ScriptedModel([
        search(),
        AIMessage(content=complete + "\nКак это понимать практичесki\n- **", response_metadata={"finish_reason": finish_reason}),
    ])
    saver = InMemorySaver()
    graph = build_agent(saver, model=model, rag_provider=lambda: rag)
    repository = JsonChatRepository(tmp_path / "chat-data")
    service = ChatService(repository, object(), settings(), agent=ChatAgent(graph, saver))
    async with httpx.AsyncClient(base_url="http://backend", transport=httpx.ASGITransport(app=api_for(service))) as client:
        backend = BackendClient("http://backend", client=client, internal_token=INTERNAL)
        chat = await backend.get_or_create_chat("compact-user", "telegram")
        answer = await collect(backend, chat, "Как понимать параметры Mesh Generate?")
    notice = INCOMPLETE_NOTICE if finish_reason == "length" else (
        "Из ответа удалён фрагмент с ошибкой текста. "
        "Расширенный ответ можно запросить кнопкой «Подробнее»."
    )
    assert answer == complete + "\n\n" + notice
    assert backend.last_answer_info(chat) == {"mode": "brief", "incomplete": True, "expandable": True}
    assert backend.last_response_sources(chat)[0]["chunk_id"] == "chunk-mesh"
    history = await repository.list_messages(chat)
    assert history[-1].content == answer
    assert history[-1].rag_sources[0]["chunk_id"] == "chunk-mesh"
    assert len(model.received) == 2


def api_for(service):
    api = FastAPI()
    api.include_router(router)
    api.dependency_overrides[get_chat_service] = lambda: service
    api.dependency_overrides[get_settings] = settings
    return api


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_one_telegram_conversation_sources_followup_durable_preview_and_callback(
    tmp_path, approved
):
    rag, questions = rag_service(tmp_path)
    model = ScriptedModel(
        [
            search(),
            AIMessage(content="Команда Mesh Generate [1], [999]."),
            search("А параметры?", "search-2"),
            AIMessage(content="Параметры сетки [1]."),
            send(),
        ]
    )
    sender = AsyncMock(return_value="Сообщение передано Telegram-боту для чата 4242.")
    repository = JsonChatRepository(tmp_path / "chat-data")
    checkpoint_path = str(tmp_path / "checkpoint.sqlite")
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
        await saver.setup()
        graph = build_agent(
            saver, model=model, send_message=sender, rag_provider=lambda: rag
        )
        service = ChatService(
            repository, object(), settings(), agent=ChatAgent(graph, saver)
        )
        api = api_for(service)
        async with httpx.AsyncClient(
            base_url="http://backend", transport=httpx.ASGITransport(app=api)
        ) as client:
            backend = BackendClient(
                "http://backend", client=client, internal_token=INTERNAL
            )
            chat = await backend.get_or_create_chat("-100:11", "telegram")
            answer = await collect(backend, chat, "Что такое Mesh Generate?")
            assert "[1]" in answer and "999" not in answer
            sources = backend.last_response_sources(chat)
            assert sources[0]["chunk_id"] == "chunk-mesh"
            assert sources[0]["document_id"] == "manual-gts"
            assert sources[0]["page"] == 7
            assert (
                await backend.get_message_sources(
                    chat, backend.last_response_message_id(chat)
                )
                == sources
            )
            await collect(backend, chat, "А параметры?")
            assert any(
                "Mesh" in question and "параметр" in question for question in questions
            )
            await collect(
                backend, chat, "Отправь в чат 4242: Инцидент передан инженеру."
            )
            draft = backend.last_approval(chat)
            assert (
                draft["chat_id"] == "4242"
                and draft["text"] == "Инцидент передан инженеру."
            )
            assert "Точный текст" in backend.last_response_notice(chat)
            assert (
                max(
                    len(button.callback_data)
                    for row in approval_kb(chat, draft["approval_id"]).inline_keyboard
                    for button in row
                )
                <= 64
            )
            sender.assert_not_called()

    # Both bot transport and graph/saver restart; server resolves the same chat.
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
        await saver.setup()
        graph = build_agent(
            saver,
            model=ScriptedModel([]),
            send_message=sender,
            rag_provider=lambda: rag,
        )
        service = ChatService(
            JsonChatRepository(tmp_path / "chat-data"),
            object(),
            settings(),
            agent=ChatAgent(graph, saver),
        )
        async with httpx.AsyncClient(
            base_url="http://backend",
            transport=httpx.ASGITransport(app=api_for(service)),
        ) as client:
            backend = BackendClient(
                "http://backend", client=client, internal_token=INTERNAL
            )
            restored = await backend.get_or_create_chat("-100:11", "telegram")
            assert restored == chat
            assert await backend.pending_approval(chat) == draft
            assert len(await backend.list_messages(chat)) == 6
            with pytest.raises(httpx.HTTPStatusError) as forbidden:
                await backend.resume_approval(
                    chat, draft["approval_id"], True, owner_external_id="-100:12"
                )
            assert forbidden.value.response.status_code == 404
            with pytest.raises(httpx.HTTPStatusError) as stale:
                await backend.resume_approval(
                    chat, uuid4(), True, owner_external_id="-100:11"
                )
            assert stale.value.response.status_code == 409
            callback = SimpleNamespace(
                data=f"ap:{'y' if approved else 'n'}:{encode_message_context(chat, draft['approval_id'])}",
                from_user=SimpleNamespace(id=11),
                answer=AsyncMock(),
                message=SimpleNamespace(
                    chat=SimpleNamespace(id=-100, type="group"),
                    edit_reply_markup=AsyncMock(),
                    reply=AsyncMock(),
                ),
            )
            await approve(callback, backend)
            assert callback.message.reply.await_count == 1
            if approved:
                sender.assert_awaited_once()
            else:
                sender.assert_not_called()
            with pytest.raises(httpx.HTTPStatusError) as repeated:
                await backend.resume_approval(
                    chat, draft["approval_id"], True, owner_external_id="-100:11"
                )
            assert repeated.value.response.status_code == 409
            assert await backend.pending_approval(chat) is None
            await backend.clear_messages(chat)
            assert await backend.list_messages(chat) == []
            assert not (
                await graph.aget_state(
                    ChatAgent.config(await repository.get_chat(chat))
                )
            ).values


@pytest.mark.asyncio
async def test_clear_invalidates_pending_and_hydrates_preexisting_history(tmp_path):
    repo = JsonChatRepository(tmp_path / "data")
    chat = await repo.get_or_create_chat("11", "telegram")
    await repo.append_message(
        chat.id, ChatMessage(chat_id=chat.id, role="user", content="Mesh Generate")
    )
    saver = InMemorySaver()
    model = ScriptedModel([send(), AIMessage(content="После очистки.")])
    effect = AsyncMock()
    graph = build_agent(saver, model=model, send_message=effect)
    service = ChatService(repo, object(), settings(), agent=ChatAgent(graph, saver))
    _ = [item async for item in service.send_message(chat.id, "Отправь текст")]
    assert any(message.content == "Mesh Generate" for message in model.received[0])
    assert model.received[0][-1].content == "Отправь текст"
    draft = await service.pending_approval(chat.id)
    await service.clear_history(chat.id)
    from app.chat.agent import StaleApprovalError

    with pytest.raises(StaleApprovalError):
        await service.approve(chat.id, draft["approval_id"], True)
    _ = [item async for item in service.send_message(chat.id, "Новый вопрос")]
    assert not any(message.content == "Mesh Generate" for message in model.received[-1])
    effect.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["missing", "error", "empty"])
async def test_agent_reports_real_no_evidence_or_unavailability(tmp_path, mode):
    rag, _ = rag_service(tmp_path, score=0.01)
    if mode == "missing":
        rag = None
    if mode == "error":

        def fail(*_args, **_kwargs):
            raise RuntimeError("private provider details")

        rag.retrieve = fail
    graph = build_agent(
        InMemorySaver(),
        model=ScriptedModel([search(), AIMessage(content="Выдуманный ответ [1].")]),
        rag_provider=lambda: rag,
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Mesh Generate")]},
        {"configurable": {"thread_id": mode}},
    )
    assert result["messages"][-1].content == (
        NO_ANSWER if mode == "empty" else KNOWLEDGE_UNAVAILABLE
    )
    assert result["sources"] == []
    assert "private provider" not in result["messages"][-1].content


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_retrieval_worker_holds_principal_budget_after_timeout_or_disconnect(
    cancel,
):
    started, release = Event(), Event()

    def blocked(*_args, **_kwargs):
        started.set()
        release.wait(3)
        return {"status": "no_evidence", "sources": [], "contexts": []}

    rag = SimpleNamespace(
        config=SimpleNamespace(request_timeout=0.03), retrieve=blocked
    )
    budget = InMemoryRequestBudget(requests_per_minute=100, max_concurrency=1)
    await budget.acquire("owner")
    lease = RequestBudgetLease(budget, "owner")
    token = retrieval_worker_tracker.set(lease.track_worker)
    graph = build_agent(
        InMemorySaver(),
        model=ScriptedModel([search(), AIMessage(content="Недоступно.")]),
        rag_provider=lambda: rag,
    )
    try:
        task = asyncio.create_task(
            graph.ainvoke(
                {"messages": [HumanMessage(content="Вопрос")]},
                {"configurable": {"thread_id": "blocked"}},
            )
        )
        await asyncio.to_thread(started.wait, 2)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        await lease.release_request()
        assert budget._active["owner"] == 1
    finally:
        release.set()
        retrieval_worker_tracker.reset(token)
    for _ in range(100):
        if not budget._active.get("owner"):
            break
        await asyncio.sleep(0.01)
    assert not budget._active.get("owner")


def test_public_retrieval_preserves_gates_followup_ids_and_budget(tmp_path):
    service, questions = rag_service(tmp_path)
    result = service.retrieve(
        "А параметры?",
        history=[{"role": "user", "content": "Mesh Generate"}],
        max_context_chars=12,
    )
    assert result["status"] == "ok" and result["telemetry"]["plan_mode"] == "follow_up"
    assert 0 < sum(map(len, result["contexts"])) <= 12
    assert result["sources"][0]["chunk_id"] == "chunk-mesh"
    assert any("Mesh Generate" in query for query in questions)
    service, _ = rag_service(tmp_path, product="civil")
    assert service.retrieve("Mesh Generate")["status"] == "no_evidence"
    service, _ = rag_service(tmp_path, score=0.01)
    assert service.retrieve("Mesh Generate")["sources"] == []
    with pytest.raises(TimeoutError):
        service.retrieve("Mesh Generate", timeout_seconds=0.0000001)


@pytest.mark.asyncio
async def test_multiple_searches_use_stable_global_citation_numbers():
    def retrieve(query, **_kwargs):
        return {
            "status": "ok",
            "sources": [{"id": query, "chunk_id": query, "file_name": query + ".pdf"}],
            "contexts": [query],
        }

    model = ScriptedModel(
        [
            search("first"),
            search("second", "search-2"),
            AIMessage(content="Первый [1], второй [2], неизвестный [999]."),
        ]
    )
    graph = build_agent(
        InMemorySaver(),
        model=model,
        rag_provider=lambda: SimpleNamespace(retrieve=retrieve),
    )
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content="Сравни")]},
        {"configurable": {"thread_id": "sources"}},
    )
    assert [source["id"] for source in state["sources"]] == ["first", "second"]
    assert "999" not in state["messages"][-1].content
    observations = [
        json.loads(message.content)
        for message in state["messages"]
        if message.type == "tool"
    ]
    assert [item["citation_numbers"] for item in observations] == [[1], [2]]


@pytest.mark.asyncio
async def test_small_context_budget_refuses_without_calling_model():
    model = ScriptedModel([])
    graph = build_agent(InMemorySaver(), model=model)
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content="Материал " * 1000)]},
        {"configurable": {"thread_id": "small-budget", "context_budget": 10}},
    )
    assert not model.received
    assert state["messages"][-1].content == CONTEXT_BUDGET_MESSAGE


@pytest.mark.asyncio
async def test_history_window_keeps_whole_tool_groups():
    model = ScriptedModel(
        [
            search(),
            AIMessage(content="По мануалу [1]."),
            AIMessage(content="Следующий ответ."),
        ]
    )

    def retrieve(_query, **_kwargs):
        return {"status": "no_evidence", "sources": [], "contexts": []}

    graph = build_agent(
        InMemorySaver(),
        model=model,
        rag_provider=lambda: SimpleNamespace(retrieve=retrieve),
    )
    config = {
        "configurable": {
            "thread_id": "window",
            "history_window": 2,
            "context_budget": 5000,
        }
    }
    await graph.ainvoke({"messages": [HumanMessage(content="Первый вопрос")]}, config)
    await graph.ainvoke({"messages": [HumanMessage(content="Второй вопрос")]}, config)
    assert not any(message.type == "tool" for message in model.received[-1])
    assert not any(message.content == "Первый вопрос" for message in model.received[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_parallel_searches_and_delivery_keep_sources_and_next_turn_valid(
    approved,
):
    def retrieve(query, **_kwargs):
        return {
            "status": "ok",
            "sources": [{"id": query, "chunk_id": query, "file_name": query + ".pdf"}],
            "contexts": [query],
        }

    request = AIMessage(
        content="Сведения [1], [2]. Подготовлена отправка.",
        tool_calls=[
            *search("first").tool_calls,
            *search("second", "search-2").tool_calls,
            *send().tool_calls,
        ],
    )
    model = ScriptedModel([request, AIMessage(content="Диалог продолжается.")])
    sender = AsyncMock(return_value="Отправлено.")
    graph = build_agent(
        InMemorySaver(),
        model=model,
        send_message=sender,
        rag_provider=lambda: SimpleNamespace(retrieve=retrieve),
    )
    config = {"configurable": {"thread_id": "parallel-delivery"}}
    paused = await graph.ainvoke(
        {"messages": [HumanMessage(content="Найди и отправь")]}, config
    )
    assert [source["id"] for source in paused["sources"]] == ["first", "second"]
    observations = [
        json.loads(message.content)
        for message in paused["messages"]
        if message.type == "tool"
    ]
    assert [observation["citation_numbers"] for observation in observations] == [
        [1],
        [2],
    ]
    from langgraph.types import Command

    await graph.ainvoke(Command(resume=approved), config)
    await graph.ainvoke({"messages": [HumanMessage(content="Продолжим")]}, config)
    seen_calls = set()
    for message in model.received[-1]:
        if message.type == "ai":
            seen_calls.update(call["id"] for call in message.tool_calls)
        elif message.type == "tool":
            assert message.tool_call_id in seen_calls
    assert model.received[-1][-1].content == "Продолжим"
    assert sender.await_count == int(approved)


@pytest.mark.asyncio
async def test_caption_and_extracted_text_reach_agent_without_raw_image_checkpoint(
    tmp_path,
):
    repo = JsonChatRepository(tmp_path / "data")
    chat = await repo.get_or_create_chat("11", "telegram")
    saver = InMemorySaver()
    model = ScriptedModel([AIMessage(content="Нужно проверить материал.")])
    graph = build_agent(saver, model=model)
    service = ChatService(repo, object(), settings(), agent=ChatAgent(graph, saver))
    media = {
        "parts": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,PRIVATE"},
            },
            {"type": "text", "text": "Извлечённый текст Mesh Generate"},
        ]
    }
    _ = [
        item
        async for item in service.send_message(chat.id, "Подпись", media_refs=media)
    ]
    received = str(model.received[0])
    assert "Извлечённый текст" in received and "Подпись" in received
    assert "base64" not in str((await graph.aget_state(ChatAgent.config(chat))).values)


@pytest.mark.asyncio
async def test_long_preview_is_complete_escaped_and_buttons_on_last_part():
    from html import unescape

    message = SimpleNamespace(chat=SimpleNamespace(type="private"), answer=AsyncMock())
    exact = "Получатель: 4242\nТочный текст:\n" + "<>&😀 **буквально** `код`" * 200
    markup = approval_kb(uuid4(), uuid4())
    await send_rendered_answer(message, exact, reply_markup=markup)
    calls = message.answer.await_args_list
    assert len(calls) > 1
    assert "".join(unescape(call.args[0]) for call in calls) == exact
    assert all(
        len(unescape(call.args[0]).encode("utf-16-le")) // 2 <= 4096 for call in calls
    )
    assert all(call.kwargs["reply_markup"] is None for call in calls[:-1])
    assert calls[-1].kwargs["reply_markup"] is markup


@pytest.mark.asyncio
async def test_public_retrieve_api_evidence_contract_and_503(tmp_path):
    from app.routers.rag import router as rag_router

    rag, _ = rag_service(tmp_path)
    api = FastAPI()
    api.include_router(rag_router)
    api.dependency_overrides[get_settings] = settings
    api.state.rag_service = rag
    headers = {
        "X-Internal-Token": INTERNAL,
        "X-User-ID": "engineer",
        "X-User-Interface": "telegram",
    }
    async with httpx.AsyncClient(
        base_url="http://backend",
        transport=httpx.ASGITransport(app=api),
        headers=headers,
    ) as client:
        response = await client.post(
            "/rag/retrieve", json={"question": "Mesh Generate"}
        )
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "ok" and "answer" not in result
        assert result["sources"][0]["chunk_id"] == "chunk-mesh"
        api.state.rag_service = None
        assert (
            await client.post("/rag/retrieve", json={"question": "Mesh Generate"})
        ).status_code == 503
