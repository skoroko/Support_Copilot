from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from support_copilot.config import load_settings
from support_copilot.prompts.loader import load_prompt_file

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional for offline demo
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - offline demo can run without SDK
    OpenAI = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - image metadata is optional
    Image = None


CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parents[0]
PROJECT_ROOT = PACKAGE_ROOT.parent
EXAMPLES_DIR = PROJECT_ROOT / "homework_2_6" / "examples"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_LOG_PATH = PACKAGE_ROOT / "results" / "vision_demo_log.txt"

SYSTEM_PROMPT = load_prompt_file("vision_system_v1.txt").strip()

DEMO_CASES = [
    {
        "label": "screenshot_error",
        "image": EXAMPLES_DIR / "ошибка расчета.jpg",
        "question": "Опиши ошибку на скриншоте и подскажи, что запросить у пользователя.",
    },
    {
        "label": "engineering_result",
        "image": EXAMPLES_DIR / "УГВ.png",
        "question": "Опиши расчетную картину и возможный контекст обращения.",
    },
    {
        "label": "function_chart",
        "image": EXAMPLES_DIR / "func.png",
        "question": "Что видно на графике функции и какие данные нужно проверить?",
    },
]


@dataclass(frozen=True)
class VisionConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout_sec: float
    max_tokens: int
    max_tokens_param: str


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    mime_type: str
    size_bytes: int
    width: int | None
    height: int | None


@dataclass(frozen=True)
class VisionUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class VisionAnalysisResult:
    answer: str
    provider: str
    model: str
    usage: VisionUsage = field(default_factory=VisionUsage)
    fallback_events: tuple[str, ...] = ()


class VisionEmptyResponseError(RuntimeError):
    """Raised when a vision provider returns a response without visible text."""


def value_from(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            text = value_from(item, "text") or value_from(item, "content")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip()


def extract_vision_answer(response: Any) -> str:
    choices = value_from(response, "choices", [])
    if not choices:
        return ""

    first_choice = choices[0]
    message = value_from(first_choice, "message", {})
    answer = text_from_content(value_from(message, "content"))
    if answer:
        return answer

    refusal = text_from_content(value_from(message, "refusal"))
    if refusal:
        return f"Модель отказалась обработать изображение: {refusal}"

    return ""


def extract_vision_usage(response: Any) -> VisionUsage:
    raw_usage = value_from(response, "usage")
    if raw_usage is None:
        return VisionUsage()

    prompt_tokens = int(value_from(raw_usage, "prompt_tokens", value_from(raw_usage, "input_tokens", 0)) or 0)
    completion_tokens = int(
        value_from(raw_usage, "completion_tokens", value_from(raw_usage, "output_tokens", 0)) or 0
    )
    total_tokens = int(value_from(raw_usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    prompt_details = value_from(raw_usage, "prompt_tokens_details") or value_from(raw_usage, "input_tokens_details") or {}
    completion_details = (
        value_from(raw_usage, "completion_tokens_details")
        or value_from(raw_usage, "output_tokens_details")
        or {}
    )
    cached_tokens = int(value_from(prompt_details, "cached_tokens", 0) or 0)
    reasoning_tokens = int(value_from(completion_details, "reasoning_tokens", 0) or 0)

    return VisionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def vision_finish_reason(response: Any) -> str:
    choices = value_from(response, "choices", [])
    if not choices:
        return "no_choices"
    return str(value_from(choices[0], "finish_reason", "unknown") or "unknown")


def text_from_delta_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            text = value_from(item, "text") or value_from(item, "content")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)


def extract_vision_delta(chunk: Any) -> str:
    choices = value_from(chunk, "choices", [])
    if not choices:
        return ""
    first_choice = choices[0]
    delta = value_from(first_choice, "delta", {})
    return text_from_delta_content(value_from(delta, "content"))


def vision_chunk_finish_reason(chunk: Any) -> str:
    choices = value_from(chunk, "choices", [])
    if not choices:
        return "no_choices"
    return str(value_from(choices[0], "finish_reason", "unknown") or "unknown")


def extract_vision_stream_error(chunk: Any) -> str | None:
    error = value_from(chunk, "error")
    if not error:
        return None
    message = value_from(error, "message")
    code = value_from(error, "code")
    if message and code:
        return f"{code}: {message}"
    if message:
        return str(message)
    return str(error)


def sanitize_error(message: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "sk-***", message)


def normalize_openai_base_url(base_url: str, *, provider: str) -> str:
    normalized = base_url.rstrip("/")
    if provider == "ollama" and not normalized.endswith("/v1"):
        return f"{normalized}/v1"
    return normalized


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def env_float(name: str, default: float) -> float:
    value = env_str(name)
    return default if not value else float(value)


def env_int(name: str, default: int) -> int:
    value = env_str(name)
    return default if not value else int(value)


def first_csv_item(value: str) -> str:
    return next((item.strip() for item in value.split(",") if item.strip()), "")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    candidates = (
        [path]
        if path.is_absolute()
        else [
            Path.cwd() / path,
            PROJECT_ROOT / path,
            CURRENT_DIR / path,
            EXAMPLES_DIR / path.name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def image_info(path: Path) -> ImageInfo:
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    width: int | None = None
    height: int | None = None
    if Image is not None:
        with Image.open(path) as image:
            width, height = image.size

    return ImageInfo(
        path=path,
        mime_type=mime_type,
        size_bytes=path.stat().st_size,
        width=width,
        height=height,
    )


def image_to_data_url(path: Path) -> str:
    info = image_info(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{info.mime_type};base64,{encoded}"


def load_vision_config(
    provider_override: str | None = None,
    *,
    allow_generic_model: bool = True,
) -> VisionConfig:
    settings = load_settings(DEFAULT_ENV_PATH)

    provider = (provider_override or settings.vision_provider).lower()
    timeout_sec = settings.vision_timeout_sec
    max_tokens = settings.vision_max_tokens
    generic_model = settings.vision_model if allow_generic_model else ""

    if provider == "openai":
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url
        model = settings.openai_vision_model or generic_model or settings.openai_model
        max_tokens_param = "max_completion_tokens"
    elif provider == "openrouter":
        api_key = settings.openrouter_api_key
        base_url = settings.openrouter_base_url
        model = settings.openrouter_vision_model or generic_model or settings.openrouter_model
        max_tokens_param = "max_tokens"
    elif provider == "ollama":
        api_key = settings.ollama_api_key
        base_url = settings.ollama_openai_compatible_base_url
        model = (
            settings.ollama_vision_model
            or generic_model
            or settings.selected_ollama_model
        )
        max_tokens_param = "max_tokens"
    else:
        raise ValueError("VISION_PROVIDER must be 'openai', 'openrouter' or 'ollama'.")

    if not api_key:
        raise ValueError(f"API key for vision provider '{provider}' is not configured.")
    if not model:
        raise ValueError(
            "Vision model is not configured. Set VISION_MODEL or provider-specific *_VISION_MODEL."
        )

    return VisionConfig(
        provider=provider,
        api_key=api_key,
        base_url=normalize_openai_base_url(base_url, provider=provider),
        model=model,
        timeout_sec=timeout_sec,
        max_tokens=max_tokens,
        max_tokens_param=max_tokens_param,
    )


def vision_provider_order(provider_override: str | None = None) -> list[str]:
    if load_dotenv is not None:
        load_dotenv(DEFAULT_ENV_PATH, override=False)

    primary = (provider_override or env_str("VISION_PROVIDER", "openai")).lower()
    order: list[str] = []
    for provider in [primary, "openai", "openrouter", "ollama"]:
        if provider in {"openai", "openrouter", "ollama"} and provider not in order:
            order.append(provider)
    return order


def load_vision_config_chain(provider_override: str | None = None) -> list[VisionConfig]:
    if load_dotenv is not None:
        load_dotenv(DEFAULT_ENV_PATH, override=False)

    primary = (provider_override or env_str("VISION_PROVIDER", "openai")).lower()
    configs: list[VisionConfig] = []
    for provider in vision_provider_order(provider_override):
        try:
            configs.append(
                load_vision_config(
                    provider,
                    allow_generic_model=(provider == primary),
                )
            )
        except ValueError:
            continue
    return configs


def build_vision_messages(question: str, data_url: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def build_vision_request_params(
    config: VisionConfig,
    path: Path,
    question: str,
    *,
    stream: bool,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": config.model,
        "messages": build_vision_messages(question, image_to_data_url(path)),
        "temperature": 0.2,
        config.max_tokens_param: config.max_tokens,
    }
    if stream:
        params["stream"] = True
        if config.provider in {"openai", "openrouter"}:
            params["stream_options"] = {"include_usage": True}
    return params


def build_openai_client(config: VisionConfig) -> Any:
    if OpenAI is None:
        raise RuntimeError("Install OpenAI SDK first: python -m pip install openai python-dotenv")

    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_sec,
        max_retries=0,
    )


def analyze_image_with_config(config: VisionConfig, path: Path, question: str) -> tuple[str, VisionUsage]:
    client = build_openai_client(config)
    params = build_vision_request_params(config, path, question, stream=False)
    response = client.chat.completions.create(**params)
    answer = extract_vision_answer(response)
    if answer:
        return answer, extract_vision_usage(response)

    finish_reason = vision_finish_reason(response)
    raise VisionEmptyResponseError(
        "Vision provider returned an empty answer "
        f"(provider={config.provider}, model={config.model}, finish_reason={finish_reason})."
    )


def analyze_image_stream_with_config(
    config: VisionConfig,
    path: Path,
    question: str,
    *,
    on_token: Any,
) -> tuple[str, VisionUsage]:
    client = build_openai_client(config)
    params = build_vision_request_params(config, path, question, stream=True)
    chunks: list[str] = []
    emitted_chars = 0
    finish_reason = "unknown"
    usage = VisionUsage()

    try:
        stream = client.chat.completions.create(**params)
        for chunk in stream:
            stream_error = extract_vision_stream_error(chunk)
            if stream_error:
                raise RuntimeError(f"stream error: {stream_error}")

            finish_reason = vision_chunk_finish_reason(chunk)
            chunk_usage = extract_vision_usage(chunk)
            if chunk_usage.total_tokens:
                usage = chunk_usage

            content = extract_vision_delta(chunk)
            if content:
                chunks.append(content)
                emitted_chars += len(content)
                on_token(content)
    except Exception as exc:
        if emitted_chars > 0:
            note = (
                "\n\n[Потоковый Vision-ответ прерван провайдером; "
                "проверьте лог и при необходимости повторите запрос.]"
            )
            on_token(note)
            return "".join(chunks).strip() + note, usage
        raise exc

    answer = "".join(chunks).strip()
    if answer:
        return answer, usage

    raise VisionEmptyResponseError(
        "Vision provider returned an empty streamed answer "
        f"(provider={config.provider}, model={config.model}, finish_reason={finish_reason})."
    )


def analyze_image_real_with_fallback(
    path: Path,
    question: str,
    provider: str | None = None,
) -> VisionAnalysisResult:
    configs = load_vision_config_chain(provider)
    if not configs:
        raise ValueError(
            "No Vision providers are configured. Set OPENROUTER_API_KEY/OPENROUTER_VISION_MODEL "
            "or OPENAI_API_KEY/OPENAI_VISION_MODEL."
        )

    events: list[str] = []
    for config in configs:
        try:
            answer, usage = analyze_image_with_config(config, path, question)
            return VisionAnalysisResult(
                answer=answer,
                provider=config.provider,
                model=config.model,
                usage=usage,
                fallback_events=tuple(events),
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary should normalize all errors
            events.append(
                f"{config.provider}/{config.model}: {type(exc).__name__}: {sanitize_error(str(exc))}"
            )
            continue

    raise RuntimeError("All Vision providers failed. " + " | ".join(events))


def analyze_image_real_stream_with_fallback(
    path: Path,
    question: str,
    *,
    on_token: Any,
    provider: str | None = None,
) -> VisionAnalysisResult:
    configs = load_vision_config_chain(provider)
    if not configs:
        raise ValueError(
            "No Vision providers are configured. Set OPENROUTER_API_KEY/OPENROUTER_VISION_MODEL "
            "or OPENAI_API_KEY/OPENAI_VISION_MODEL."
        )

    events: list[str] = []
    for config in configs:
        try:
            answer, usage = analyze_image_stream_with_config(
                config,
                path,
                question,
                on_token=on_token,
            )
            return VisionAnalysisResult(
                answer=answer,
                provider=config.provider,
                model=config.model,
                usage=usage,
                fallback_events=tuple(events),
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary should normalize all errors
            events.append(
                f"{config.provider}/{config.model}: {type(exc).__name__}: {sanitize_error(str(exc))}"
            )
            continue

    raise RuntimeError("All Vision providers failed. " + " | ".join(events))


def analyze_image_real(path: Path, question: str, provider: str | None = None) -> str:
    return analyze_image_real_with_fallback(path, question, provider).answer


def analyze_image_offline(path: Path, question: str) -> str:
    info = image_info(path)
    name = path.name.casefold()
    dimensions = (
        "unknown size"
        if info.width is None or info.height is None
        else f"{info.width}x{info.height}px"
    )

    if "ошибка" in name or "error" in name:
        visible = "скриншот окна Output с ошибкой ERROR [1004] и сообщениями Internal Error"
        problem = "похоже на некорректно заданный элемент/ID или поврежденные входные данные модели"
        request = "текст ошибки, версию продукта, файл модели, лог расчета и шаги воспроизведения"
        escalation = "да, если ошибка воспроизводится на минимальной модели или блокирует расчет"
    elif "угв" in name:
        visible = "интерфейс результатов фильтрации/консолидации с расчетной сеткой и линией на профиле"
        problem = "пользователь, вероятно, уточняет интерпретацию результатов или отображение кривой депрессии/УГВ"
        request = "исходную модель, выбранный результат, этап нагрузки, единицы измерения и ожидаемую картину"
        escalation = "нет, сначала собрать контекст расчета"
    elif "func" in name:
        visible = "диалог создания функции с таблицей значений и графиком зависимости"
        problem = "возможна ошибка в задании функции материала/грунта или в интерполяции значений"
        request = "назначение функции, диапазон значений, единицы измерения и ожидаемую зависимость"
        escalation = "нет, если нет падения расчета или блокировки работы"
    else:
        visible = "изображение пользовательского обращения к техподдержке"
        problem = "по одному изображению недостаточно данных для уверенного вывода"
        request = "описание задачи, версию продукта, ожидаемый и фактический результат"
        escalation = "по ситуации после уточнения влияния на работу"

    return "\n".join(
        [
            f"1. Что видно: {visible}. Файл: {path.name}, {info.mime_type}, {dimensions}, {info.size_bytes} байт.",
            f"2. Возможная проблема: {problem}.",
            f"3. Что запросить у пользователя: {request}.",
            f"4. Нужно ли эскалировать: {escalation}.",
            f"Вопрос инженера: {question}",
        ]
    )


class DemoLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, text: str = "") -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(text.rstrip() + "\n")


def run_single(args: argparse.Namespace) -> int:
    if not args.image:
        raise SystemExit("--image is required unless --demo is used.")

    path = resolve_path(args.image)
    question = args.question
    answer = (
        analyze_image_real(path, question, args.provider)
        if args.real
        else analyze_image_offline(path, question)
    )
    print(answer)
    return 0


def run_demo(args: argparse.Namespace) -> int:
    logger = DemoLogger(Path(args.log_path))
    logger.write(f"Vision demo | {utc_now()}")
    logger.write(f"mode={'real' if args.real else 'offline'}")
    logger.write()

    print("Support Copilot Vision demo")
    print("=" * 80)
    for index, case in enumerate(DEMO_CASES, start=1):
        path = case["image"]
        question = case["question"]
        print(f"\n[{index}] {case['label']}: {path}")
        print(f"Question: {question}")
        answer = (
            analyze_image_real(path, question, args.provider)
            if args.real
            else analyze_image_offline(path, question)
        )
        print(answer)

        logger.write(f"[{index}] {case['label']}")
        logger.write(f"image={path}")
        logger.write(f"question={question}")
        logger.write(answer)
        logger.write()

    print(f"\nLog: {Path(args.log_path).resolve()}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homework 2.6A Vision API demo.")
    parser.add_argument("--image", help="Path to a PNG/JPEG image.")
    parser.add_argument(
        "--question",
        default="Опиши изображение как вложение к обращению в техподдержку MIDAS.",
        help="Question for the vision model.",
    )
    parser.add_argument("--demo", action="store_true", help="Analyze 3 prepared demo images.")
    parser.add_argument("--real", action="store_true", help="Call real Vision API instead of offline demo.")
    parser.add_argument(
        "--provider",
        choices=("openai", "openrouter"),
        help="Override VISION_PROVIDER from .env for real mode.",
    )
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG_PATH),
        help="Path for demo log.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.demo:
        return run_demo(args)
    return run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
