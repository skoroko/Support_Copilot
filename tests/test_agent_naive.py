from __future__ import annotations

from types import SimpleNamespace

import app.services.agent_naive as agent
import app.services.agent_naive_tools as tools


def _response(*, content: str | None = None, tool_name: str | None = None, arguments: str = "{}"):
    calls = []
    if tool_name:
        calls = [SimpleNamespace(id="call-1", function=SimpleNamespace(name=tool_name, arguments=arguments))]
    message = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


class _Completions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return next(self.responses)


def _install_client(monkeypatch, responses):
    completions = _Completions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    settings = SimpleNamespace(
        openai_api_key="sk-test",
        openai_base_url="https://example.test/v1",
        llm_timeout_sec=1,
        openai_model="gpt-5.4-mini",
    )
    monkeypatch.setattr(agent, "load_settings", lambda: settings)
    monkeypatch.setattr(agent, "OpenAI", lambda **kwargs: client)
    return completions


def test_agent_executes_tool_and_returns_trace(monkeypatch):
    calls = _install_client(
        monkeypatch,
        [_response(tool_name="get_current_time", arguments='{"timezone":"Europe/Moscow"}'),
         _response(content="Готово")],
    )

    result = agent.run_agent("Сколько времени в Москве?")

    assert result["answer"] == "Готово"
    assert result["steps"] == 2
    assert result["trace"][0]["tool_name"] == "get_current_time"
    assert result["trace"][0]["llm_input_tokens"] == 11
    assert calls.requests[0]["parallel_tool_calls"] is False
    assert calls.requests[1]["messages"][-1]["role"] == "tool"
    assert calls.requests[1]["messages"][-1]["tool_call_id"] == "call-1"


def test_unknown_tool_is_model_visible_and_loop_continues(monkeypatch):
    calls = _install_client(
        monkeypatch,
        [_response(tool_name="get_user_balance", arguments='{"user_id":"7"}'),
         _response(content="Такого инструмента нет")],
    )

    result = agent.run_agent("Проверь баланс пользователя 7")

    assert result["answer"] == "Такого инструмента нет"
    assert "неизвестный tool get_user_balance" in result["trace"][0]["tool_result"]
    assert "неизвестный tool get_user_balance" in calls.requests[1]["messages"][-1]["content"]


def test_max_steps_stops_loop(monkeypatch):
    _install_client(monkeypatch, [_response(tool_name="get_current_time")])

    result = agent.run_agent("Продолжай бесконечно", max_steps=1)

    assert result["steps"] == 1
    assert result["error"] == "Достигнут лимит max_steps=1"


def test_telegram_tool_is_only_a_stub(capsys):
    result = tools.send_telegram_message("12345", "Тест")

    assert result == "Сообщение отправлено в 12345"
    assert capsys.readouterr().out.strip() == "[TELEGRAM → 12345] Тест"


def test_tool_descriptions_have_two_sentences():
    for tool in tools.TOOLS:
        assert tool["function"]["description"].count(".") >= 2


def test_knowledge_base_rejects_weak_match():
    result = tools.search_knowledge_base("ремонт квантового телепорта QX-9000")

    assert result.startswith("В базе знаний нет надежного совпадения")
