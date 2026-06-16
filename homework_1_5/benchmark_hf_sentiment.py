from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI
from sklearn.metrics import accuracy_score
from transformers import pipeline


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[0]
CASES_PATH = ROOT / "sentiment_cases.json"
RESULTS_DIR = ROOT / "results"

HF_MODELS = [
    "cointegrated/rubert-tiny-sentiment-balanced",
    "blanchefort/rubert-base-cased-sentiment",
    "tabularisai/multilingual-sentiment-analysis",
]

OPENROUTER_MODEL = "openai/gpt-oss-120b:free"


@dataclass
class Prediction:
    case_id: str
    expected: str
    predicted: str
    latency_sec: float
    raw_label: str
    score: float
    text: str


def load_cases() -> list[dict[str, str]]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized in {"negative", "neg", "label_0", "0", "1 star", "1", "2 stars", "2"}:
        return "negative"
    if normalized in {"neutral", "neu", "label_1", "1", "3 stars", "3"}:
        return "neutral"
    if normalized in {"positive", "pos", "label_2", "2", "4 stars", "4", "5 stars", "5"}:
        return "positive"
    if "negative" in normalized or "негатив" in normalized:
        return "negative"
    if "neutral" in normalized or "нейтр" in normalized:
        return "neutral"
    if "positive" in normalized or "позит" in normalized:
        return "positive"
    return normalized


def get_license(model_id: str) -> str:
    info = HfApi().model_info(model_id)
    card_data = getattr(info, "cardData", None)
    if isinstance(card_data, dict):
        return str(card_data.get("license") or "not specified")
    license_value = getattr(card_data, "license", None)
    return str(license_value or "not specified")


def run_hf_model(model_id: str, cases: list[dict[str, str]]) -> dict[str, Any]:
    classifier = pipeline("text-classification", model=model_id, truncation=True)
    predictions: list[Prediction] = []
    for case in cases:
        started = time.perf_counter()
        result = classifier(case["text"])[0]
        latency = time.perf_counter() - started
        raw_label = str(result["label"])
        predictions.append(
            Prediction(
                case_id=case["id"],
                expected=case["expected"],
                predicted=normalize_label(raw_label),
                latency_sec=round(latency, 4),
                raw_label=raw_label,
                score=round(float(result.get("score", 0.0)), 4),
                text=case["text"],
            )
        )

    expected = [item.expected for item in predictions]
    predicted = [item.predicted for item in predictions]
    return {
        "provider": "HuggingFace Pipeline",
        "model": model_id,
        "license": get_license(model_id),
        "accuracy": round(float(accuracy_score(expected, predicted)), 3),
        "avg_latency_sec": round(sum(item.latency_sec for item in predictions) / len(predictions), 4),
        "predictions": [asdict(item) for item in predictions],
    }


def parse_cloud_label(content: str) -> str:
    try:
        payload = json.loads(content)
        return normalize_label(str(payload.get("sentiment", "")))
    except json.JSONDecodeError:
        return normalize_label(content)


def run_openrouter(cases: list[dict[str, str]], model: str = OPENROUTER_MODEL) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return {
            "provider": "OpenRouter",
            "model": model,
            "status": "skipped",
            "reason": "OPENROUTER_API_KEY is empty",
        }

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    predictions: list[Prediction] = []
    for case in cases:
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify Russian support feedback sentiment. "
                        "Return only JSON: {\"sentiment\":\"positive|neutral|negative\"}."
                    ),
                },
                {"role": "user", "content": case["text"]},
            ],
            temperature=0,
            max_tokens=40,
        )
        latency = time.perf_counter() - started
        content = response.choices[0].message.content or ""
        predictions.append(
            Prediction(
                case_id=case["id"],
                expected=case["expected"],
                predicted=parse_cloud_label(content),
                latency_sec=round(latency, 4),
                raw_label=content.strip(),
                score=0.0,
                text=case["text"],
            )
        )

    expected = [item.expected for item in predictions]
    predicted = [item.predicted for item in predictions]
    return {
        "provider": "OpenRouter",
        "model": model,
        "accuracy": round(float(accuracy_score(expected, predicted)), 3),
        "avg_latency_sec": round(sum(item.latency_sec for item in predictions) / len(predictions), 4),
        "predictions": [asdict(item) for item in predictions],
    }


def make_markdown(results: list[dict[str, Any]]) -> str:
    rows = [
        "| Провайдер | Модель | Лицензия | Accuracy | Средняя latency |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for result in results:
        if result.get("status") == "skipped":
            rows.append(f"| {result['provider']} | `{result['model']}` | n/a | skipped | skipped |")
            continue
        rows.append(
            "| {provider} | `{model}` | {license} | {accuracy:.3f} | {latency:.4f} сек |".format(
                provider=result["provider"],
                model=result["model"],
                license=result.get("license", "n/a"),
                accuracy=result["accuracy"],
                latency=result["avg_latency_sec"],
            )
        )

    completed = [result for result in results if result.get("status") != "skipped"]
    best_overall = max(
        completed,
        key=lambda item: (item["accuracy"], -item["avg_latency_sec"]),
    )
    best_hf = max(
        (result for result in results if result.get("status") != "skipped"),
        key=lambda item: (item["accuracy"], -item["avg_latency_sec"]),
    )
    hf_completed = [result for result in completed if result["provider"] == "HuggingFace Pipeline"]
    if hf_completed:
        best_hf = max(hf_completed, key=lambda item: (item["accuracy"], -item["avg_latency_sec"]))

    return "\n".join(
        [
            "# Benchmark sentiment analysis",
            "",
            "Задача: классификация тональности русских support-отзывов.",
            "",
            *rows,
            "",
            f"Лучший общий baseline по accuracy/latency: `{best_overall['model']}`.",
            f"Лучшая HuggingFace-модель для Gradio-демо: `{best_hf['model']}`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HuggingFace sentiment models and OpenRouter baseline.")
    parser.add_argument("--models", nargs="+", default=HF_MODELS)
    parser.add_argument("--skip-cloud", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    cases = load_cases()
    results = [run_hf_model(model_id, cases) for model_id in args.models]
    if not args.skip_cloud:
        results.append(run_openrouter(cases))

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "benchmark_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "benchmark_report.md").write_text(make_markdown(results), encoding="utf-8")
    print(make_markdown(results))


if __name__ == "__main__":
    main()
