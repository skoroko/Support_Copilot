"""Validate saved blinded reviews and export comparison metrics without network calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_reviews(comparison, reviews, inputs):
    expected = {
        (name, r["case_id"]): r
        for name, v in comparison["variants"].items()
        for r in v["results"]
    }
    scored, calls, provenance = {}, [], {}
    for review_path, input_path in zip(reviews, inputs, strict=True):
        review, packet = read(review_path), read(input_path)
        if review["signature"]["input_sha256"] != digest(input_path):
            raise ValueError("Review input hash mismatch")
        source_rows = {
            (name, r["case_id"]): r
            for name, v in packet["variants"].items()
            for r in v["results"]
        }
        for row in review["scores"]:
            key = row["variant"], row["case_id"]
            if key in scored or key not in expected:
                raise ValueError("Duplicate or unknown review row")
            if any(
                source_rows[key].get(k) != expected[key].get(k)
                for k in ("answer", "sources", "evidence", "answer_info")
            ):
                raise ValueError(
                    "Reviewed answer/evidence differs from final comparison"
                )
            scored[key] = row
        calls.extend(review["calls"])
        provenance[review_path.name] = {
            "review_sha256": digest(review_path),
            "input_sha256": digest(input_path),
            "signature": review["signature"],
        }
    if scored.keys() != expected.keys():
        raise ValueError("Semantic review incomplete")
    return list(scored.values()), calls, provenance


def quality(rows):
    grounds = [r["grounding"] for r in rows if r["grounding"] is not None]
    return {
        "n": len(rows),
        "correctness_completeness": statistics.mean(
            r["correctness_completeness"] for r in rows
        ),
        "grounding": statistics.mean(grounds) if grounds else None,
        "grounding_n": len(grounds),
        "citation_errors": sum(r["citation_error"] for r in rows),
        "judge_critical_flags": sum(r["critical_unsupported"] for r in rows),
    }


def metrics(comparison, scores):
    output = {}
    for name, variant in comparison["variants"].items():
        rows = [r for r in scores if r["variant"] == name]
        output[name] = {
            "runtime": variant["summary"],
            "quality": quality(rows),
            "brief": quality([r for r in rows if r["mode"] == "brief"]),
            "expanded": quality([r for r in rows if r["mode"] == "expanded"]),
            "iteration_limit_answers": sum(
                "Превышен лимит из 6 итераций" in r.get("answer", "")
                for r in variant["results"]
            ),
        }
    return output


def paired_intervals(comparison, scores):
    scores = {
        (r["variant"], r["case_id"]): r["correctness_completeness"] for r in scores
    }
    definitions = {c["id"]: c for c in comparison["cases"]}
    result = {}
    for candidate, baseline in (
        ("luna-low", "mini"),
        ("luna-medium", "mini"),
        ("luna-high", "mini"),
        ("luna-xhigh", "mini"),
    ):
        clusters = {}
        for row in comparison["variants"][baseline]["results"]:
            ident = row["case_id"]
            conversation = definitions[ident.removesuffix(":expanded")]["conversation"]
            clusters.setdefault(conversation, []).append(
                scores[candidate, ident] - scores[baseline, ident]
            )
        rng, groups = random.Random("luna-ci-v1:" + candidate), list(clusters.values())
        samples = []
        for _ in range(10000):
            values = [v for group in rng.choices(groups, k=len(groups)) for v in group]
            samples.append(statistics.mean(values))
        samples.sort()
        result[candidate + "-" + baseline] = {
            "clusters": len(groups),
            "mean_delta": statistics.mean(v for g in groups for v in g),
            "cluster_bootstrap_95_ci": [samples[249], samples[9749]],
        }
    return result


def table(metrics, title):
    lines = [
        title,
        "",
        "| Вариант | Полнота/верность | Grounding (n) | Медиана / P95, с | Генерация, USD |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "mini",
        "luna-none",
        "luna-low",
        "luna-medium",
        "luna-high",
        "luna-xhigh",
    ):
        r, q = metrics[name]["runtime"], metrics[name]["quality"]
        lines.append(
            f"| {name} | {q['correctness_completeness']:.3f} | {q['grounding']:.3f} ({q['grounding_n']}) | {r['median_ms'] / 1000:.2f} / {r['p95_ms'] / 1000:.2f} | ${r['generation_cost_usd']:.5f} |"
        )
    return "\n".join(lines)


def run(args):
    fixed, live = read(args.fixed), read(args.live)
    scores, judge_calls, provenance = merge_reviews(
        fixed, args.reviews, args.review_inputs
    )
    live_scores, live_calls, live_provenance = merge_reviews(
        live, args.live_reviews, args.live_review_inputs
    )
    summary = {
        "fixed_sha256": digest(args.fixed),
        "live_sha256": digest(args.live),
        "fixed": metrics(fixed, scores),
        "live": metrics(live, live_scores),
        "paired_correctness": paired_intervals(fixed, scores),
        "review_provenance": provenance,
        "live_review_provenance": live_provenance,
        "judge_cost_usd": sum(c.get("cost_usd", 0) for c in judge_calls + live_calls),
        "judge_unknown_reservations_usd": sum(
            c.get("reservation_usd", 0)
            for c in judge_calls + live_calls
            if "cost_usd" not in c
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "reviews.json").write_text(
        json.dumps(
            {
                "fixed": scores,
                "live": live_scores,
                "provenance": provenance,
                "live_provenance": live_provenance,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    text = "# GPT-6 Luna: eval Support Copilot, 15.10.2026\n\n"
    text += "384 ответа: 43 сценария + 5 expanded на фиксированных фрагментах (6 вариантов, 288 ответов); 11 сценариев + 5 expanded с живым Qdrant-поиском (96 ответов). Эталоны использованы только оценщиком. Telegram-отправка выключена.\n\n"
    text += (
        table(summary["fixed"], "## Одинаковые фрагменты: 48 ответов на вариант")
        + "\n\n"
    )
    text += table(summary["live"], "## Живой поиск: 16 ответов на вариант") + "\n\n"
    text += "## Как читать результаты\n\nВерность/полнота и grounding — средние оценки слепого GPT-6.1 Sol low по шкале 0 / 0.5 / 1, **не доля правильных ответов и не независимая человеческая экспертиза**. Grounding оценивается по конкретным процитированным фрагментам; чистые отказы/уточнения имеют null и исключаются из его знаменателя. Низкая полнота учитывает необоснованные отказы. Метки моделей скрыты, порядок перемешан детерминированно. Исходные заметки оценщика сохранены; его critical flags требуют проверки инженером.\n\n"
    text += "Фрагменты первой части fixed-оценки одинаковы между вариантами (проверено по SHA-256); для второй части и live каждая цитата дополнительно привязана к хешу конкретного текста. История follow-up строится из ответов самого кандидата. Fixed retrieval возвращает ту же выдачу при повторном поиске: это не проверка качества поисковых запросов. Один xhigh fixed-сценарий достиг лимита итераций; это считается потерей полезности, а не сетевой ошибкой.\n\n"
    text += "Лимиты mini/Luna none — 500/1500; при reasoning общий max_output_tokens — 8692/9692. Отдельного жёсткого лимита видимого ответа API не даёт. Краткость задаётся одинаковой инструкцией 3–5 пунктов. Стоимость рассчитана по usage, включая cache read/write; output уже включает reasoning, второй раз он не прибавляется. Live-стоимость включает генерацию агента, исключает query embeddings и вспомогательный condense. Задержка — до полного ответа, **не TTFT**; concurrency=3, единичный прогон, сеть и cache влияют на время. Luna none fixed повторена после ошибки Windows при сохранении артефакта; исходная попытка сохранена отдельно.\n\n"
    text += "Набор небольшой и ранее использовался при настройке промптов. Доверительные интервалы парных различий полноты (bootstrap по разговорам, 10 000 выборок) сохранены в summary.json; они не подтверждают устойчивое превосходство Luna над mini. Для production-замены нужен отдельный набор новых обращений и повторения.\n\n"
    text += "## Тарифы и воспроизведение\n\n[OpenAI pricing](https://developers.openai.com/api/docs/pricing), проверено 15.10.2026: mini input $0.75 / cached $0.075 / output $4.50; Luna input $0.10 / cached $0.01 / cache write $0.125 / output $0.50 за 1 млн токенов. Использован Standard и короткий контекст. [Контракт Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), [reasoning](https://developers.openai.com/api/docs/guides/reasoning#controlling-costs).\n\n"
    text += "Скрипты: scripts/compare_luna.py (fixed/live), scripts/judge_luna.py (слепая оценка), scripts/report_luna.py (проверка fingerprint и экспорт). Сетевые прогоны требуют отдельного разрешения на передачу документации и бюджет. Все ответы: var/luna_migration; CSV — answers.csv.\n"
    (args.output / "report.md").write_text(text, encoding="utf-8")
    with (args.output / "answers.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "stage",
                "variant",
                "case_id",
                "question",
                "answer",
                "correctness_completeness",
                "grounding",
                "latency_ms",
                "notes",
            ],
        )
        writer.writeheader()
        for stage, comparison, ratings in (
            ("fixed", fixed, scores),
            ("live", live, live_scores),
        ):
            definitions = {c["id"]: c for c in comparison["cases"]}
            lookup = {(r["variant"], r["case_id"]): r for r in ratings}
            for name, variant in comparison["variants"].items():
                for row in variant["results"]:
                    score = lookup[name, row["case_id"]]
                    writer.writerow(
                        {
                            "stage": stage,
                            "variant": name,
                            "case_id": row["case_id"],
                            "question": definitions[
                                row["case_id"].removesuffix(":expanded")
                            ]["question"],
                            "answer": row["answer"],
                            "correctness_completeness": score[
                                "correctness_completeness"
                            ],
                            "grounding": score["grounding"],
                            "latency_ms": row["latency_ms"],
                            "notes": score["notes"],
                        }
                    )
    print(
        json.dumps(
            {
                "fixed": {n: v["quality"] for n, v in summary["fixed"].items()},
                "live": {n: v["quality"] for n, v in summary["live"].items()},
                "judge_cost_usd": summary["judge_cost_usd"],
                "judge_unknown_reservations_usd": summary[
                    "judge_unknown_reservations_usd"
                ],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("fixed", "live", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("reviews", "review-inputs", "live-reviews", "live-review-inputs"):
        parser.add_argument("--" + flag, type=Path, nargs="+", required=True)
    run(parser.parse_args())
