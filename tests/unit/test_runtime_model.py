from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from app.services.runtime_model import (
    AgentModel,
    ResponsesTextLLM,
    normalize_response,
    visible_text,
)


def settings(**changes):
    return SimpleNamespace(
        **(
            {
                "default_model": "gpt-6-luna",
                "agent_llm_api": "auto",
                "agent_reasoning_effort": "low",
                "agent_expanded_reasoning_effort": None,
                "agent_reasoning_token_reserve": 8192,
                "request_timeout": 30,
                "agent_generation_timeout": 120,
                "openai_api_key": SecretStr("test-only"),
                "openai_base_url": None,
            }
            | changes
        )
    )


def test_blank_environment_reasoning_inherits_instead_of_breaking_startup():
    from app.core.config import Settings

    config = Settings(
        _env_file=None,
        openai_api_key=SecretStr("test-only"),
        agent_reasoning_effort="",
        agent_expanded_reasoning_effort=" ",
    )
    assert config.agent_reasoning_effort is None
    assert config.agent_expanded_reasoning_effort is None


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh"])
async def test_reasoning_modes_and_total_budget(effort):
    client = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(content="Ответ."))
    )
    model = AgentModel(settings(agent_reasoning_effort=effort), [], client=client)
    await model.ainvoke([], max_tokens=500)
    kwargs = client.ainvoke.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": effort}
    assert kwargs["max_tokens"] == (500 if effort == "none" else 8692)
    assert "temperature" not in kwargs


@pytest.mark.asyncio
async def test_expansion_can_use_different_effort_and_disable_tools():
    client = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(content="Ответ."))
    )
    model = AgentModel(
        settings(agent_expanded_reasoning_effort="xhigh"), [], client=client
    )
    await model.ainvoke([], max_tokens=1500, expanded=True, tool_choice="none")
    assert client.ainvoke.call_args.kwargs == {
        "max_tokens": 9692,
        "reasoning": {"effort": "xhigh"},
        "tool_choice": "none",
    }


def test_constructor_explicitly_selects_responses_and_disables_storage(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("app.services.runtime_model.ChatOpenAI", factory)
    AgentModel(settings(), ["tool"])
    kwargs = factory.call_args.kwargs
    assert kwargs["use_responses_api"] is True
    assert kwargs["output_version"] == "v0"
    assert kwargs["temperature"] is None
    assert kwargs["store"] is False
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    factory.return_value.bind_tools.assert_called_once_with(["tool"])


@pytest.mark.asyncio
async def test_mini_baseline_keeps_existing_chat_contract():
    client = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(content="Ответ."))
    )
    model = AgentModel(
        settings(default_model="gpt-5.4-mini", agent_reasoning_effort=None),
        [],
        client=client,
    )
    await model.ainvoke([], max_tokens=500)
    assert model.responses is False
    assert client.ainvoke.call_args.kwargs == {"max_tokens": 500}


def test_reasoning_with_tools_is_rejected_on_unsupported_chat_endpoint():
    with pytest.raises(ValueError, match="requires Responses"):
        AgentModel(settings(agent_llm_api="chat_completions"), [], client=Mock())


def test_only_visible_text_is_extracted_and_usage_preserved():
    usage = {
        "input_tokens": 10,
        "output_tokens": 30,
        "total_tokens": 40,
        "output_token_details": {"reasoning": 20},
    }
    msg = AIMessage(
        content=[
            {"type": "reasoning", "summary": [{"text": "private"}]},
            {"type": "text", "text": "Ответ."},
        ],
        response_metadata={"status": "completed"},
        usage_metadata=usage,
    )
    result = normalize_response(msg)
    assert result.content == "Ответ."
    assert result.usage_metadata == usage
    assert result.response_metadata["finish_reason"] == "stop"
    assert visible_text([{"type": "function_call", "text": "hidden"}]) == ""


def test_incomplete_maps_to_existing_length_contract():
    msg = AIMessage(
        content="Часть.",
        response_metadata={
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    )
    assert normalize_response(msg).response_metadata["finish_reason"] == "length"


def test_rag_luna_completion_does_not_depend_on_llama_registry():
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=Mock(
                return_value=SimpleNamespace(
                    status="completed", output_text="Самостоятельный вопрос?"
                )
            )
        )
    )
    llm = ResponsesTextLLM(
        model="gpt-6-luna", api_key=None, base_url=None, timeout=10, client=client
    )
    assert llm.complete("Уточни").text == "Самостоятельный вопрос?"
    assert client.responses.create.call_args.kwargs["reasoning"] == {"effort": "none"}


def test_rag_incomplete_is_not_treated_as_a_complete_answer():
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=Mock(
                return_value=SimpleNamespace(
                    status="incomplete", output_text="Неоконченный"
                )
            )
        )
    )
    llm = ResponsesTextLLM(
        model="gpt-6-luna", api_key=None, base_url=None, timeout=10, client=client
    )
    with pytest.raises(RuntimeError, match="did not complete"):
        llm.complete("Вопрос")


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_rag_stream_only_emits_visible_deltas_and_checks_final_status(status):
    from unittest.mock import MagicMock

    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.__iter__.return_value = iter(
        [
            SimpleNamespace(type="response.reasoning_text.delta", delta="private"),
            SimpleNamespace(type="response.output_text.delta", delta="Ответ."),
        ]
    )
    stream.get_final_response.return_value = SimpleNamespace(status=status)
    client = SimpleNamespace(
        responses=SimpleNamespace(stream=Mock(return_value=stream))
    )
    llm = ResponsesTextLLM(
        model="gpt-6-luna", api_key=None, base_url=None, timeout=10, client=client
    )
    output = llm.stream_complete("Вопрос")
    assert next(output).delta == "Ответ."
    if status == "completed":
        assert list(output) == []
    else:
        with pytest.raises(RuntimeError, match="did not complete"):
            next(output)


@pytest.mark.parametrize("configured", [None, 0, 3])
def test_agent_model_retries_transient_provider_errors(configured, monkeypatch):
    factory = Mock()
    monkeypatch.setattr("app.services.runtime_model.ChatOpenAI", factory)
    overrides = {} if configured is None else {"llm_max_retries": configured}
    AgentModel(settings(**overrides), ["tool"])
    assert factory.call_args.kwargs["max_retries"] == (2 if configured is None else configured)
    factory.return_value.bind_tools.assert_called_once_with(["tool"])


@pytest.mark.parametrize("configured", [None, 5])
def test_rag_llm_retries_transient_provider_errors(configured, monkeypatch):
    factory = Mock()
    monkeypatch.setattr("app.services.runtime_model.OpenAI", factory)
    kwargs = {} if configured is None else {"max_retries": configured}
    ResponsesTextLLM(model="gpt-6-luna", api_key="k", base_url=None, timeout=10, **kwargs)
    assert factory.call_args.kwargs["max_retries"] == (2 if configured is None else configured)


def test_llm_max_retries_defaults_to_two_when_unset():
    from app.core.config import Settings
    from pydantic import SecretStr

    config = Settings(_env_file=None, openai_api_key=SecretStr("test-only"))
    assert config.llm_max_retries == 2
