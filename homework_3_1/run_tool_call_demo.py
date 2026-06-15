from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "results" / "tool_call_demo.jsonl"
DEFAULT_SUMMARY_PATH = Path(__file__).resolve().parent / "results" / "demo_summary.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from support_copilot.config import load_settings  # noqa: E402
from support_copilot.core.tool_assistant import JSONLToolLogger, SupportToolAssistant  # noqa: E402
from support_copilot.infrastructure.llm import ProviderConfig  # noqa: E402
from support_copilot.infrastructure.tool_calling import ToolCallingClient  # noqa: E402


CASES = [
    {
        "name": "Точно нужен tool",
        "input": "Найди в базе знаний, что запросить у клиента, если лицензия не активируется второй день.",
    },
    {
        "name": "Точно не нужен tool",
        "input": "Сформулируй короткое вежливое сообщение, что инженер принял обращение в работу.",
    },
    {
        "name": "Пограничный случай",
        "input": "Клиент пишет: после обновления расчет падает. Что ответить сначала?",
    },
]


@dataclass(frozen=True)
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class FakeFunctionCall:
    name: str
    arguments: str


@dataclass(frozen=True)
class FakeToolCall:
    id: str
    type: str
    function: FakeFunctionCall


@dataclass(frozen=True)
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass(frozen=True)
class FakeChoice:
    message: FakeMessage


@dataclass(frozen=True)
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage


class DemoToolCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **params: Any) -> FakeResponse:
        self.calls += 1
        messages = params.get("messages", [])
        user_input = next(
            (str(message.get("content", "")) for message in reversed(messages) if message.get("role") == "user"),
            "",
        )
        tool_messages = [message for message in messages if message.get("role") == "tool"]

        if tool_messages:
            tool_result = json.loads(str(tool_messages[-1].get("content", "{}")))
            answer = self._final_answer_from_tool(tool_result)
            return self._text_response(messages, answer)

        if params.get("tools") and params.get("tool_choice") != "none" and self._should_call_tool(user_input):
            arguments = {
                "query": user_input,
                "product": self._choose_product(user_input),
            }
            tool_call = FakeToolCall(
                id=f"call_demo_{self.calls}",
                type="function",
                function=FakeFunctionCall(
                    name="search_knowledge_base",
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
            return self._tool_response(messages, tool_call)

        return self._text_response(
            messages,
            "Инженер принял обращение в работу и вернется с уточнениями после первичной проверки.",
        )

    @staticmethod
    def _should_call_tool(user_input: str) -> bool:
        lowered = user_input.casefold()
        markers = (
            "базе знаний",
            "найди",
            "лиценз",
            "активац",
            "установ",
            "документац",
            "инструкц",
            "расчет",
            "обновлен",
            "ошибк",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _choose_product(user_input: str) -> str:
        lowered = user_input.casefold()
        if "лиценз" in lowered or "активац" in lowered:
            return "licensing"
        if "установ" in lowered:
            return "installation"
        if "расчет" in lowered or "обновлен" in lowered:
            return "calculation"
        if "документац" in lowered or "инструкц" in lowered:
            return "documentation"
        return "support_copilot"

    @staticmethod
    def _final_answer_from_tool(tool_result: dict[str, Any]) -> str:
        results = tool_result.get("results") or []
        if not results:
            return "В базе знаний нет надежного совпадения. Запрос нужно передать инженеру поддержки для проверки."
        top = results[0]
        return (
            f"По базе знаний {top['id']}: {top['answer']} "
            "Если данных не хватает, не обещайте решение до проверки инженером."
        )

    @staticmethod
    def _usage(messages: list[dict[str, Any]], answer: str) -> FakeUsage:
        prompt_tokens = max(1, sum(len(str(message.get("content", "")).split()) for message in messages) * 2)
        completion_tokens = max(12, len(answer.split()) * 2)
        return FakeUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _text_response(self, messages: list[dict[str, Any]], answer: str) -> FakeResponse:
        return FakeResponse(
            choices=[FakeChoice(FakeMessage(content=answer))],
            usage=self._usage(messages, answer),
        )

    def _tool_response(self, messages: list[dict[str, Any]], tool_call: FakeToolCall) -> FakeResponse:
        return FakeResponse(
            choices=[FakeChoice(FakeMessage(content=None, tool_calls=[tool_call]))],
            usage=self._usage(messages, tool_call.function.arguments),
        )


class DemoToolChat:
    def __init__(self) -> None:
        self.completions = DemoToolCompletions()


class DemoToolOpenAIClient:
    def __init__(self) -> None:
        self.chat = DemoToolChat()


def build_demo_client() -> ToolCallingClient:
    provider = ProviderConfig(
        name="demo-openai",
        api_key="sk-demo",
        base_url="https://api.openai.com/v1",
        model="demo-function-calling-model",
        input_price_per_1m=0.75,
        output_price_per_1m=4.50,
    )
    return ToolCallingClient(
        providers=[provider],
        client_factory=lambda _config: DemoToolOpenAIClient(),
        max_retries=0,
        sleep_fn=lambda _delay: None,
        random_fn=lambda: 0.0,
    )


def build_real_client() -> ToolCallingClient:
    return ToolCallingClient.from_env(str(DEFAULT_ENV_PATH))


def write_summary(path: Path, rows: list[tuple[dict[str, str], Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Demo summary",
        "",
        "| Case | Tool called | Arguments | Final answer | Total tokens |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case, result in rows:
        if result.tool_calls:
            first_call = result.tool_calls[0]
            tool_called = first_call.name
            arguments = json.dumps(first_call.arguments, ensure_ascii=False)
        else:
            tool_called = "no"
            arguments = "-"
        answer = result.answer.replace("|", "\\|")
        lines.append(
            f"| {case['name']} | {tool_called} | `{arguments}` | {answer} | {result.usage.total_tokens} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    settings = load_settings(DEFAULT_ENV_PATH)
    parser = argparse.ArgumentParser(description="Homework 3.1 Function Calling demo.")
    parser.add_argument("--real", action="store_true", help="Use real provider chain from .env.")
    parser.add_argument("--log-path", default=settings.tool_call_log_path or str(DEFAULT_LOG_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--temperature", type=float, default=settings.tool_call_temperature)
    parser.add_argument("--max-tokens", type=int, default=settings.tool_call_max_tokens)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = JSONLToolLogger(args.log_path, reset=True)
    client = build_real_client() if args.real else build_demo_client()
    assistant = SupportToolAssistant(
        client,
        logger,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    rows = []
    print("Homework 3.1 Function Calling demo")
    print("=" * 80)
    for case in CASES:
        print(f"\n[{case['name']}]\nUser: {case['input']}")
        result = assistant.answer(case["input"])
        tool_text = result.tool_calls[0].name if result.tool_calls else "no tool"
        print(f"Tool: {tool_text}")
        print(f"Answer: {result.answer}")
        print(f"Tokens: {result.usage.total_tokens}")
        rows.append((case, result))

    write_summary(Path(args.summary_path), rows)
    print(f"\nLog: {Path(args.log_path).resolve()}")
    print(f"Summary: {Path(args.summary_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
