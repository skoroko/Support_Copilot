"""CLI entry point for homework 2.6: image analysis through Vision API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from support_copilot.config import env_float, env_int, load_env_file
from support_copilot.multimodal import (
    ConfigurationError,
    VisionAPIError,
    VisionConfig,
    analyze_image,
    append_analysis_log,
    format_result_for_console,
)


DEFAULT_QUESTION = (
    "Проанализируй изображение как вложение к обращению в техподдержку MIDAS."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a support image with an OpenAI-compatible Vision API."
    )
    parser.add_argument("image_path", help="Путь к PNG/JPEG/WebP/GIF изображению.")
    parser.add_argument(
        "-q",
        "--question",
        default=DEFAULT_QUESTION,
        help="Вопрос к модели по изображению.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Файл для добавления человекочитаемого лога работы.",
    )
    parser.add_argument(
        "--env-file",
        default=str(PROJECT_ROOT / ".env"),
        help="Локальный .env файл. Секреты не выводятся в лог.",
    )
    parser.add_argument("--model", default=None, help="Переопределить VISION_MODEL.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base_url.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--detail", choices=["auto", "low", "high"], default=None)
    parser.add_argument("--timeout-sec", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        load_env_file(args.env_file)

        model = args.model or os.getenv("VISION_MODEL") or ""
        base_url = (
            args.base_url
            or os.getenv("VISION_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        api_key = os.getenv("VISION_API_KEY") or os.getenv("OPENAI_API_KEY")
        temperature = (
            args.temperature
            if args.temperature is not None
            else env_float("VISION_TEMPERATURE", 0.2)
        )
        max_tokens = (
            args.max_tokens if args.max_tokens is not None else env_int("VISION_MAX_TOKENS", 500)
        )
        detail = args.detail or os.getenv("VISION_DETAIL") or "auto"
        timeout_sec = (
            args.timeout_sec
            if args.timeout_sec is not None
            else env_float("VISION_TIMEOUT_SEC", 120.0)
        )

        config = VisionConfig(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            detail=detail,
            timeout_sec=timeout_sec,
        )
        result = analyze_image(
            args.image_path,
            args.question,
            config,
        )
        print(format_result_for_console(result))

        if args.log_file:
            append_analysis_log(args.log_file, result)
            print(f"\nLog appended to: {args.log_file}")

    except (ConfigurationError, FileNotFoundError, ValueError) as exc:
        print(f"Ошибка конфигурации или входных данных: {exc}", file=sys.stderr)
        return 2
    except VisionAPIError as exc:
        print(f"Ошибка Vision API: {exc}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
