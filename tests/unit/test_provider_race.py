"""Slow-provider hedges preserve full responses, cancellation and accounting."""
import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app.services.provider_race import backup_allowance, first_complete
from app.services.runtime_model import AgentModel, normalize_response
from app.services.turn_budget import turn_budget


def answer(text="Complete.", *, status="completed", finish=None):
    metadata = {"status": status}
    if finish:
        metadata["finish_reason"] = finish
    return AIMessage(content=text, response_metadata=metadata,
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


@pytest.mark.asyncio
async def test_fast_primary_never_starts_backup():
    calls = []

    async def invoke():
        calls.append(1)
        return answer()

    result = await first_complete(invoke, delay=.02, backup_slots=asyncio.Semaphore(1))
    assert calls == [1] and "generation_race" not in result.response_metadata


@pytest.mark.asyncio
async def test_backup_wins_only_when_complete_and_primary_stream_is_closed():
    calls, closed = [], asyncio.Event()

    async def invoke():
        calls.append(1)
        if len(calls) == 1:
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        return answer("Complete backup.")

    slots = asyncio.Semaphore(1)
    result = await first_complete(invoke, delay=.01, backup_slots=slots)
    assert result.content == "Complete backup." and closed.is_set() and not slots.locked()
    assert result.usage_metadata["total_tokens"] == 15
    race = result.response_metadata["generation_race"]
    assert race["attempts"] == 2 and race["winner"] == 1
    assert len(race["known_usage"]) == 1 and race["unknown_usage_attempts"] == 1
    assert race["known_usage"][0]["attempt"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status,finish,invalid", [
    ("incomplete", None, False), ("completed", "length", False),
    ("completed", "content_filter", False), ("failed", None, False),
    ("completed", None, True), ("in_progress", None, False),
    ("queued", None, False),
])
async def test_fast_partial_or_invalid_backup_cannot_beat_complete_primary(status, finish, invalid):
    calls, backup_finished = [], asyncio.Event()

    async def invoke():
        calls.append(1)
        if len(calls) == 1:
            await backup_finished.wait()
            return answer("Full primary.")
        message = answer("Partial.", status=status, finish=finish)
        if invalid:
            message.invalid_tool_calls = [{"name": "search_knowledge_base", "args": "{", "id": "bad", "error": "invalid"}]
        backup_finished.set()
        return normalize_response(message)

    result = await first_complete(invoke, delay=.01, backup_slots=asyncio.Semaphore(1))
    assert result.content == "Full primary."
    assert len(result.response_metadata["generation_race"]["known_usage"]) == 2


@pytest.mark.asyncio
async def test_completed_tool_call_is_eligible_without_visible_text():
    calls = []

    async def invoke():
        calls.append(1)
        if len(calls) == 1:
            await asyncio.Event().wait()
        return AIMessage(content="", response_metadata={"status": "completed"},
            tool_calls=[{"name": "search_knowledge_base", "args": {"query": "fact"}, "id": "one"}])

    result = await first_complete(invoke, delay=.01, backup_slots=asyncio.Semaphore(1))
    assert len(result.tool_calls) == 1


@pytest.mark.asyncio
async def test_early_error_does_not_add_retry():
    calls = []

    async def invoke():
        calls.append(1)
        raise ValueError("invalid request")

    with pytest.raises(ValueError, match="invalid request"):
        await first_complete(invoke, delay=.02, backup_slots=asyncio.Semaphore(1))
    assert calls == [1]


@pytest.mark.asyncio
async def test_terminal_request_error_after_hedge_cancels_other_stream():
    calls, closed = [], asyncio.Event()

    class Forbidden(Exception):
        status_code = 403

    async def invoke():
        calls.append(1)
        if len(calls) == 1:
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        raise Forbidden()

    with pytest.raises(Forbidden):
        await first_complete(invoke, delay=.01, backup_slots=asyncio.Semaphore(1))
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_backup", [False, True])
async def test_outer_cancellation_drains_every_started_stream(after_backup):
    started, closed = [], []
    ready = asyncio.Event()
    slots = asyncio.Semaphore(1)

    async def invoke():
        number = len(started)
        started.append(number)
        if len(started) == (2 if after_backup else 1):
            ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.append(number)

    task = asyncio.create_task(first_complete(invoke, delay=.01 if after_backup else 10, backup_slots=slots))
    await asyncio.wait_for(ready.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(closed) == started and not slots.locked()


@pytest.mark.asyncio
async def test_only_one_backup_per_whole_turn_and_nested_budgets_share_allowance():
    calls = []

    async def invoke():
        calls.append(1)
        if len(calls) == 1:
            await asyncio.Event().wait()
        if len(calls) == 3:
            await asyncio.sleep(.03)
        return answer()

    async with turn_budget(2):
        await first_complete(invoke, delay=.01, backup_slots=asyncio.Semaphore(1))
        assert backup_allowance.get().used
        async with turn_budget(1):
            await first_complete(invoke, delay=.01, backup_slots=asyncio.Semaphore(1))
        assert len(calls) == 3
    assert backup_allowance.get() is None


@pytest.mark.asyncio
async def test_two_backup_profile_shares_nested_limit_and_third_slow_call_stays_single():
    calls = []

    async def invoke():
        calls.append(1)
        if len(calls) in {1, 3}:
            await asyncio.Event().wait()
        if len(calls) == 5:
            await asyncio.sleep(.03)
        return answer()

    slots = asyncio.Semaphore(1)
    async with turn_budget(2, max_backups=2):
        await first_complete(invoke, delay=.01, backup_slots=slots)
        async with turn_budget(1):
            await first_complete(invoke, delay=.01, backup_slots=slots)
            assert backup_allowance.get().used == 2
            await first_complete(invoke, delay=.01, backup_slots=slots)
        assert len(calls) == 5 and backup_allowance.get().used == 2
    assert backup_allowance.get() is None and not slots.locked()


@pytest.mark.asyncio
async def test_busy_backup_slot_does_not_queue_unrelated_turn_or_start_more_calls():
    slots, calls = asyncio.Semaphore(1), []
    await slots.acquire()

    async def invoke():
        calls.append(1)
        await asyncio.sleep(.03)
        return answer()

    await first_complete(invoke, delay=.01, backup_slots=slots)
    assert calls == [1] and slots.locked()
    slots.release()


@pytest.mark.asyncio
async def test_agent_model_duplicates_identical_parameters_and_preserves_expansion_single_call():
    requests = []

    async def invoke(messages, **kwargs):
        requests.append((messages, kwargs))
        if len(requests) == 1:
            await asyncio.Event().wait()
        return answer()

    config = SimpleNamespace(default_model="gpt-6-luna", agent_llm_api="responses",
        agent_reasoning_effort="medium", agent_expanded_reasoning_effort="medium",
        request_timeout=60, agent_generation_timeout=60, agent_generation_backup_after=.01)
    model = AgentModel(config, [], client=SimpleNamespace(ainvoke=invoke))
    messages = ["identical input"]
    await model.ainvoke(messages, allow_backup=True)
    assert requests[0] == requests[1]
    assert requests[0][1]["reasoning"] == {"effort": "medium"}
    await model.ainvoke(messages, expanded=True)
    assert len(requests) == 3


@pytest.mark.asyncio
async def test_langgraph_message_stream_exposes_only_selected_node_result():
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk, HumanMessage
    from langchain_core.outputs import ChatGenerationChunk
    from langgraph.graph import END, START, MessagesState, StateGraph

    calls, closed = [], asyncio.Event()

    class StreamingProvider(BaseChatModel):
        @property
        def _llm_type(self):
            return "test-streaming-provider"

        def _generate(self, *args, **kwargs):
            raise AssertionError("Expected async streaming")

        async def _astream(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                try:
                    yield ChatGenerationChunk(message=AIMessageChunk(content="LOSER PRIVATE CHUNK"))
                    await asyncio.Event().wait()
                finally:
                    closed.set()
            else:
                yield ChatGenerationChunk(message=AIMessageChunk(content="Complete winner.",
                    response_metadata={"status": "completed"}))

    config = SimpleNamespace(default_model="gpt-6-luna", agent_llm_api="responses",
        agent_reasoning_effort="medium", request_timeout=60,
        agent_generation_timeout=60, agent_generation_backup_after=.01)
    model = AgentModel(config, [], client=StreamingProvider())

    async def node(state):
        message = await model.ainvoke(state["messages"], allow_backup=True,
            config={"tags": ["existing-tag"], "metadata": {"test": "retained"}})
        return {"messages": [message]}

    builder = StateGraph(MessagesState)
    builder.add_node("answer", node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    events = [event async for event in builder.compile().astream(
        {"messages": [HumanMessage(content="question")]}, stream_mode="messages")]
    assert calls == [1, 1] and closed.is_set()
    assert [message.content for message, _metadata in events] == ["Complete winner."]
