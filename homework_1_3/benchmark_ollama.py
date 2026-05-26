from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODELS = ["qwen3:1.7b", "smollm2:1.7b", "gemma3:1b"]


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    prompt: str
    expected_keywords: tuple[str, ...]


TASKS = [
    BenchmarkTask(
        name="classification",
        prompt=(
            "Классифицируй обращение в одну категорию: faq, technical, "
            "complaint или escalation. Ответь одной строкой с категорией и "
            "коротким объяснением. Обращение: Пользователь пишет, что после "
            "обновления не открывается проект, появляется ошибка 500, работа "
            "стоит уже второй час."
        ),
        expected_keywords=("technical", "ошибка", "500"),
    ),
    BenchmarkTask(
        name="summary",
        prompt=(
            "Сделай краткое резюме обращения для инженера поддержки в 2 "
            "предложения. Обращение: Клиент не может активировать лицензию "
            "на новом ноутбуке. Старый ноутбук сломан, доступа к нему нет. "
            "Нужно понять, можно ли перенести лицензию и какие данные запросить."
        ),
        expected_keywords=("лиценз", "перен", "данн"),
    ),
    BenchmarkTask(
        name="escalation",
        prompt=(
            "Определи, нужна ли эскалация специалисту второй линии. Ответь: "
            "да или нет, затем одно предложение причины. Обращение: Пользователь "
            "третий раз сообщает о потере данных после импорта, просит срочно "
            "подключить старшего инженера и сохранить журнал ошибки."
        ),
        expected_keywords=("да", "эскалац", "старш"),
    ),
]


CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["faq", "technical", "complaint", "escalation"],
        },
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
        },
        "summary": {"type": "string"},
        "needs_human": {"type": "boolean"},
        "suggested_action": {"type": "string"},
    },
    "required": [
        "category",
        "priority",
        "summary",
        "needs_human",
        "suggested_action",
    ],
}


STRUCTURED_PROMPT = (
    "/no_think\n"
    "Проанализируй обращение техподдержки и верни только JSON без markdown.\n"
    "Схема JSON:\n"
    "{\n"
    '  "category": "faq | technical | complaint | escalation",\n'
    '  "priority": "low | medium | high | critical",\n'
    '  "summary": "краткое резюме на русском",\n'
    '  "needs_human": true,\n'
    '  "suggested_action": "следующий шаг для инженера поддержки"\n'
    "}\n"
    "Правила классификации: ошибка 500 - technical; повторная нерешенная "
    "проблема и просьба подключить специалиста - escalation или high priority.\n"
    "Обращение: Пользователь сообщает, что после установки обновления MIDAS "
    "проект не открывается, появляется ошибка 500. Он уже обращался вчера, "
    "проблема повторяется, просит подключить специалиста."
)


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(base_url: str, path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_ollama_running(base_url: str) -> None:
    try:
        get_json(base_url, "/api/tags")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            "Ollama API недоступен. Установите Ollama, запустите приложение "
            "или выполните `ollama serve`, затем повторите запуск.\n"
            f"URL: {base_url}\nОшибка: {exc}"
        ) from exc


def installed_models(base_url: str) -> set[str]:
    payload = get_json(base_url, "/api/tags")
    names = {item.get("name", "") for item in payload.get("models", [])}
    families = {name.split(":")[0] for name in names if name}
    return names | families


def ensure_models_available(base_url: str, models: list[str]) -> None:
    available = installed_models(base_url)
    missing = [model for model in models if model not in available]
    if missing:
        commands = "\n".join(f"ollama pull {model}" for model in missing)
        raise SystemExit(
            "Не все модели скачаны в Ollama. Выполните команды:\n"
            f"{commands}"
        )


def chat(base_url: str, model: str, prompt: str) -> tuple[str, float]:
    started_at = time.perf_counter()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты ассистент техподдержки. Отвечай на русском языке, "
                    "кратко и по делу."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 220},
    }
    response = post_json(base_url, "/api/chat", payload)
    elapsed = time.perf_counter() - started_at
    return response["message"]["content"].strip(), elapsed


def score_answer(answer: str, expected_keywords: tuple[str, ...]) -> float:
    normalized = answer.lower()
    matched = sum(1 for keyword in expected_keywords if keyword in normalized)
    return round(matched / len(expected_keywords), 2)


def run_benchmark(base_url: str, models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for task in TASKS:
            answer, latency = chat(base_url, model, task.prompt)
            chars_per_sec = len(answer) / latency if latency > 0 else 0
            rows.append(
                {
                    "model": model,
                    "task": task.name,
                    "latency_sec": round(latency, 2),
                    "approx_chars_per_sec": round(chars_per_sec, 1),
                    "quality_score": score_answer(answer, task.expected_keywords),
                    "answer_preview": answer.replace("\n", " ")[:160],
                }
            )
    return rows


def run_structured_output(base_url: str, model: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты классификатор обращений техподдержки. "
                    "Верни только валидный JSON. Не добавляй рассуждения."
                ),
            },
            {"role": "user", "content": STRUCTURED_PROMPT},
        ],
        "stream": False,
        # Маленькие thinking-модели Qwen надежнее проходят JSON mode, чем
        # строгую JSON Schema: схема остается в prompt-контракте выше.
        "format": "json",
        "options": {"temperature": 0, "num_predict": 360},
    }
    response = post_json(base_url, "/api/chat", payload)
    content = response["message"]["content"]
    parsed = json.loads(content)
    validate_structured_output(parsed)
    return parsed


def validate_structured_output(payload: dict[str, Any]) -> None:
    allowed_categories = set(CLASSIFICATION_SCHEMA["properties"]["category"]["enum"])
    allowed_priorities = set(CLASSIFICATION_SCHEMA["properties"]["priority"]["enum"])
    required = set(CLASSIFICATION_SCHEMA["required"])
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Structured output missing fields: {sorted(missing)}")
    if payload["category"] not in allowed_categories:
        raise ValueError(f"Unexpected category: {payload['category']}")
    if payload["priority"] not in allowed_priorities:
        raise ValueError(f"Unexpected priority: {payload['priority']}")
    if not isinstance(payload["needs_human"], bool):
        raise ValueError("Field needs_human must be boolean")


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "model",
        "task",
        "latency_sec",
        "approx_chars_per_sec",
        "quality_score",
        "answer_preview",
    ]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark локальных моделей Ollama для Support Copilot."
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama API URL. По умолчанию: {DEFAULT_OLLAMA_URL}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Список моделей для benchmark.",
    )
    parser.add_argument(
        "--structured-model",
        default="qwen3:1.7b",
        help="Модель для structured output.",
    )
    parser.add_argument(
        "--structured-only",
        action="store_true",
        help="Запустить только structured output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_ollama_running(args.ollama_url)

    required_models = [args.structured_model] if args.structured_only else args.models
    if args.structured_model not in required_models:
        required_models.append(args.structured_model)
    ensure_models_available(args.ollama_url, required_models)

    if not args.structured_only:
        rows = run_benchmark(args.ollama_url, args.models)
        print("\nBenchmark results\n")
        print_table(rows)

    structured = run_structured_output(args.ollama_url, args.structured_model)
    print("\nStructured output\n")
    print(json.dumps(structured, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
