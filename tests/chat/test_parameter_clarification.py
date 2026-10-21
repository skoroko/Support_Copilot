"""An unspecified first parameter request must not select a random RAG topic."""
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.chat.agent import ChatAgent
from app.chat.domain import Chat
from app.core.config import Settings
from app.services.agent_history import (
    PARAMETER_CLARIFICATION,
    needs_parameter_clarification,
)
from app.services.agent_persistent import build_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared", [True, False])
async def test_cold_parameter_question_is_persisted_without_model_or_retrieval(prepared):
    class NoModel:
        async def ainvoke(self, *args, **kwargs):
            pytest.fail("an object-free question must not generate technical settings")

    def no_rag():
        pytest.fail("an object-free question must not retrieve an arbitrary topic")

    saver = InMemorySaver()
    graph = build_agent(saver, model=NoModel(), rag_provider=no_rag)
    chat = Chat(id=uuid4(), owner_external_id="clarification-test", interface="telegram")
    settings = Settings(_env_file=None, openai_api_key="offline", agent_prepared_evidence_enabled=prepared)
    adapter = ChatAgent(graph, saver, settings=settings)
    result = await adapter.turn(chat, "Какие параметры нужно задать?")
    assert result["answer"] == PARAMETER_CLARIFICATION
    assert result["sources"] == []
    assert not result["answer_info"]["expandable"]
    recreated = ChatAgent(graph, saver)
    snapshot = await recreated.graph.aget_state(recreated.config(chat))
    assert snapshot.values["messages"][-1].content == PARAMETER_CLARIFICATION
    assert snapshot.values["knowledge_status"] == "unused"


@pytest.mark.parametrize("question", ["Какие параметры?", "Какие настройки выбрать?", "Что нужно настроить?", "What parameters should I set?"])
def test_other_unqualified_parameter_questions_ask_for_context(question):
    assert needs_parameter_clarification([HumanMessage(content=question)])


@pytest.mark.parametrize("messages", [
    [HumanMessage(content="Какие параметры нужно задать для сетки AUTO-SOLID?")],
    [HumanMessage(content="Как задать K0 в GTS NX?")],
    [HumanMessage(content="Настраиваю AUTO-SOLID"), AIMessage(content="Уточним сетку."),
     HumanMessage(content="Какие параметры нужно задать?")],
    [HumanMessage(content=[{"type": "text", "text": "Какие параметры нужно задать?"},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64,offline"}}])],
])
def test_specific_history_and_media_questions_keep_the_normal_agent_route(messages):
    assert not needs_parameter_clarification(messages)
