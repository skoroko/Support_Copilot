from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from support_copilot import cli
from support_copilot.core.tool_assistant import JSONLToolLogger, SupportToolAssistant
from support_copilot.infrastructure.cache import LLMCache

USER_MARKER = "user-private-4292@example.invalid"
ANSWER_MARKER = "answer-private-4292@example.invalid"
ERROR_MARKER = "provider-private-4292@example.invalid"


class _TextClient:
    def __init__(self) -> None:
        self.events = [
            SimpleNamespace(
                attempt=1,
                code="provider_error",
                provider="synthetic-provider",
                message=ERROR_MARKER,
                delay_sec=None,
            )
        ]

    def chat(self, _messages, **_kwargs):
        return SimpleNamespace(
            answer=ANSWER_MARKER,
            provider=ERROR_MARKER,
            model="synthetic-model",
            usage=None,
            unavailable=False,
        )


def _assistant(tmp_path: Path, *, real: bool = False) -> cli.SupportAssistant:
    return cli.SupportAssistant(
        llm_client=_TextClient(),
        cache=LLMCache(),
        mode_name="test",
        real=real,
        stream=False,
        vision_provider=None,
        model_cache_id="synthetic-model",
        temperature=0,
        max_tokens=100,
        history_limit=10,
        logger=cli.FileSessionLogger(tmp_path / "session.txt"),
    )


def test_cli_text_log_keeps_metrics_without_request_answer_or_provider_text(tmp_path: Path) -> None:
    assistant = _assistant(tmp_path)

    displayed = assistant.answer(USER_MARKER)
    assistant.clear_history()
    repeated = assistant.answer(USER_MARKER)
    log = assistant.logger.path.read_text(encoding="utf-8")

    assert ANSWER_MARKER in displayed
    assert ANSWER_MARKER in repeated
    assert USER_MARKER not in log
    assert ANSWER_MARKER not in log
    assert ERROR_MARKER not in log
    assert "USER chars=" in log
    assert "ASSISTANT source=provider chars=" in log
    assert "PROVIDER attempt=1" in log
    assert "TOKENS kind=text" in log
    assert "CACHE hit" in log


def test_cli_image_and_command_errors_keep_path_question_and_exception_out_of_log(
    tmp_path: Path, monkeypatch,
) -> None:
    assistant = _assistant(tmp_path, real=True)
    image_path = tmp_path / f"{USER_MARKER}.png"
    monkeypatch.setattr(cli, "resolve_user_image_path", lambda _raw: image_path)
    monkeypatch.setattr(
        cli,
        "image_info",
        lambda _path: SimpleNamespace(mime_type="image/png", size_bytes=12, width=2, height=2),
    )
    monkeypatch.setattr(
        cli, "analyze_image_real_with_fallback", lambda *_args: (_ for _ in ()).throw(RuntimeError(ERROR_MARKER)),
    )
    monkeypatch.setattr(cli, "analyze_image_offline", lambda *_args: ANSWER_MARKER)

    displayed = assistant.answer_image(str(image_path), USER_MARKER)
    assert ANSWER_MARKER in displayed
    assert ERROR_MARKER in displayed

    monkeypatch.setattr(
        cli,
        "analyze_image_real_with_fallback",
        lambda *_args: SimpleNamespace(
            answer=ANSWER_MARKER,
            provider=ERROR_MARKER,
            model=ERROR_MARKER,
            usage=None,
            fallback_events=[ERROR_MARKER],
        ),
    )
    assistant.answer_image(str(image_path), USER_MARKER)

    monkeypatch.setattr(cli, "resolve_user_image_path", lambda _raw: (_ for _ in ()).throw(ValueError(ERROR_MARKER)))
    _, command_output = cli.handle_command(f"/image {USER_MARKER}", assistant)
    assert ERROR_MARKER in command_output

    log = assistant.logger.path.read_text(encoding="utf-8")
    for marker in (USER_MARKER, ANSWER_MARKER, ERROR_MARKER, str(image_path)):
        assert marker not in log
    assert "USER_IMAGE path_chars=" in log
    assert "VISION_ERROR type=RuntimeError" in log
    assert "VISION_FALLBACK chars=" in log
    assert "COMMAND_IMAGE_ERROR type=ValueError" in log


class _ToolClient:
    def chat_with_tools(self, messages, **kwargs):
        assert messages[-1]["content"] == USER_MARKER
        sink = kwargs["event_sink"]
        sink("tool_call", {"tool_call_id": "synthetic-call", "name": "search_knowledge_base", "arguments": {"query": USER_MARKER}})
        sink("tool_result", {"tool_call_id": "synthetic-call", "name": "search_knowledge_base", "result": {"answer": ERROR_MARKER}})
        return SimpleNamespace(
            answer=ANSWER_MARKER,
            provider="synthetic-provider",
            model="synthetic-model",
            direct_answer=False,
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, cached_tokens=0, reasoning_tokens=0, total_tokens=5),
            cost_usd=0.01,
        )


def test_tool_mode_jsonl_log_omits_input_tool_payload_and_answer(tmp_path: Path) -> None:
    logger = JSONLToolLogger(tmp_path / "tools.jsonl")
    assistant = SupportToolAssistant(_ToolClient(), logger)

    result = assistant.answer(USER_MARKER)
    log = logger.path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in log.splitlines()]

    assert result.answer == ANSWER_MARKER
    for marker in (USER_MARKER, ANSWER_MARKER, ERROR_MARKER):
        assert marker not in log
    assert {record["event"] for record in records} == {"input", "tool_call", "tool_result", "final_answer", "usage"}
    assert records[0]["text_chars"] == len(USER_MARKER)
    assert "arguments_digest" in records[1]
    assert "result_digest" in records[2]
    assert records[-1]["total_tokens"] == 5


def test_tool_logger_does_not_trust_event_names_or_numeric_payloads(tmp_path: Path) -> None:
    logger = JSONLToolLogger(tmp_path / "tools.jsonl")
    logger.write_event(USER_MARKER, {USER_MARKER: 4292, "result": 4292})

    record = json.loads(logger.path.read_text(encoding="utf-8"))
    assert record["event"] == "other"
    assert USER_MARKER not in record
    assert "field_0_digest" in record
    assert "result_digest" in record
    assert "result" not in record
