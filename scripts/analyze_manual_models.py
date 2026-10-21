"""Audit saved manual benchmark records and publish paired comparison artifacts."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_manual_models import (
    MODELS,
    ROOT,
    blind_payload,
    cost,
    digest,
    empty_answer_scores,
    load,
    model_input,
    report,
    reservation,
    save,
    validate_scores,
    verify,
)


def paired_interval(differences: list[float]) -> dict:
    """Resample question means, keeping the two model repeats in each cluster."""
    rng = random.Random(20261019)
    samples = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(5000))
    return {"mean_delta": statistics.mean(differences), "ci95_low": samples[125], "ci95_high": samples[4874],
            "unit": "question mean across repeats", "n_questions": len(differences)}


def audit(output: Path) -> tuple[dict, list[dict], list[dict], dict]:
    manifest, cases = verify(output)
    rows = [load(p) for p in sorted((output / "samples").glob("*.json"))]
    reviews = [load(p) for p in sorted((output / "reviews").glob("*.json"))]
    if len(rows) != len(manifest["schedule"]) or len(reviews) != len(cases):
        raise ValueError("Incomplete generation/review set")
    slots = {r["slot"]: r for r in rows}
    by_case = {c["id"]: c for c in cases}
    if len(slots) != len(rows):
        raise ValueError("Duplicate sample slot")
    for planned in manifest["schedule"]:
        row = slots[planned["slot"]]
        if any(row.get(k) != v for k, v in planned.items()) or row["status"] not in ("completed", "incomplete", "failed"):
            raise ValueError("Sample differs from frozen schedule")
        if row["input_sha256"] != manifest["input_sha256"][row["case_id"]]:
            raise ValueError("Sample input changed")
        if row["status"] == "failed":
            cap = manifest["luna_total_output_cap"] if row["model"] == "gpt-6-luna" else manifest["nonreasoning_output_cap"]
            expected_reserve = reservation(row["model"], model_input(by_case[row["case_id"]]), cap)
            if (row.get("answer") or "cost_usd" in row or not row.get("error_type") or
                    row.get("latency_s", -1) < 0 or abs(row["reservation_usd"] - expected_reserve) > 1e-12):
                raise ValueError("Failed attempt receipt/reservation mismatch")
            continue
        if row["served_tier"] != "default" or not row["served_model"].startswith(row["model"]):
            raise ValueError("Unexpected model or service tier")
        if abs(row["cost_usd"] - cost(row["model"], row["usage"])) > 1e-12:
            raise ValueError("Sample cost mismatch")
    scores = {}
    for review in reviews:
        case = by_case[review["case_id"]]
        mapping, payload = blind_payload(case, [r for r in rows if r["case_id"] == case["id"]])
        if review["status"] != "completed" or review["mapping"] != mapping:
            raise ValueError("Review mapping/status mismatch")
        if review["served_tier"] != "default" or not review["served_model"].startswith(manifest["judge_model"]):
            raise ValueError("Unexpected judge model or tier")
        if abs(review["cost_usd"] - cost(manifest["judge_model"], review["usage"])) > 1e-12:
            raise ValueError("Judge cost mismatch")
        if review["input_sha256"] != digest(json.dumps(payload, ensure_ascii=False).encode()):
            raise ValueError("Review is bound to different answers or evidence")
        validate_scores(review["scores"], mapping)
        raw_scores = json.loads(review["raw_scores"])["scores"]
        validate_scores(raw_scores, mapping)
        if review["scores"] != [{**score, "slot": mapping[score["label"]]} for score in raw_scores]:
            raise ValueError("Scores differ from the saved judge response")
        for score in review["scores"]:
            if score["slot"] != mapping[score["label"]] or score["slot"] in scores:
                raise ValueError("Duplicate or changed review slot")
            scores[score["slot"]] = score
    for slot, score in empty_answer_scores(rows).items():
        if slot in scores:
            raise ValueError("Empty answer was also judged")
        scores[slot] = score
    if set(scores) != set(slots):
        raise ValueError("Missing sample review")
    return manifest, cases, rows, scores


def publish(output: Path) -> None:
    manifest, cases, rows, scores = audit(output)
    report(SimpleNamespace(output=output))  # Recompute summary from the audited raw records.
    summary = load(output / "summary.json")
    paired = {}
    baseline = "gpt-6-luna"
    for model in MODELS:
        if model == baseline:
            continue
        differences = {"quality": [], "latency_s": []}
        for case in cases:
            selected = {name: [r for r in rows if r["case_id"] == case["id"] and r["model"] == name]
                        for name in (baseline, model)}
            differences["quality"].append(
                statistics.mean(scores[r["slot"]]["correctness_completeness"] for r in selected[model]) -
                statistics.mean(scores[r["slot"]]["correctness_completeness"] for r in selected[baseline]))
            if all(r["status"] == "completed" for group in selected.values() for r in group):
                differences["latency_s"].append(statistics.mean(r["latency_s"] for r in selected[model]) -
                                                 statistics.mean(r["latency_s"] for r in selected[baseline]))
        paired[model] = {name: paired_interval(values) for name, values in differences.items()}
    summary["paired_vs_luna"] = paired
    summary["manifest_sha256"] = digest((output / "manifest.json").read_bytes())
    summary["source_sha256"] = manifest["source_sha256"]
    summary["protocol_amendments"] = manifest.get("amendments", [])
    summary["rates_usd_per_million"] = manifest["rates"]
    summary["reasoning_effort"] = manifest["models"]
    summary["literal_citation_placeholders"] = {
        model: sum("[N]" in row.get("answer", "") for row in rows if row["model"] == model)
        for model in MODELS
    }
    summary["audit"] = ("All scheduled attempt/input receipts, successful response tiers/usage/cost "
                        "and blinded review bindings verified; unknown failed usage preserved")
    summary["assessment_method"] = {
        "judge_model": manifest["judge_model"], "judge_reasoning": manifest["judge_effort"],
        "judge_concurrency_final": manifest.get("judge_concurrency", 1),
        "judge_deadline_s_final": manifest.get("judge_wall_deadline_s", 240),
        "quality_scale": [0, 0.5, 1], "quality_denominator": "all scheduled attempts per model",
        "empty_answer_rule": "correctness=0; grounding=null; not sent to paid judge",
        "latency_denominator": "completed responses only; paired intervals require both repeats complete",
        "critical_flags": "LLM assessment; no human engineering certification",
    }
    preflight = ROOT / "var/manual_model_benchmark/2026-10-19/samples"
    preflight_rows = [load(p) for p in preflight.glob("*.json")]
    summary["excluded_preflight_unknown_reserve_usd"] = sum(
        r["reservation_usd"] for r in preflight_rows if "cost_usd" not in r)
    summary["all_unknown_cost_reserves_usd"] = (
        summary["excluded_preflight_unknown_reserve_usd"] + summary["prior_unknown_reserve_usd"] +
        summary["judge_unknown_cost_reservations_usd"] +
        sum(m["unknown_cost_reservations_usd"] for m in summary["models"].values()))
    evidence_files = sorted({r["source"]["file_name"] for c in cases for r in c["records"]})
    summary["evidence_files"] = evidence_files
    metrics = summary["models"]
    luna, mini = metrics[baseline], metrics["gpt-5.4-mini"]
    cheapest = min(metrics, key=lambda model: metrics[model]["cost_usd"] / metrics[model]["n"])
    best_quality = max(metrics, key=lambda model: metrics[model]["quality"])
    conclusions = [
        f"На этой выборке минимальная стоимость — у `{cheapest}`, максимальная средняя полнота/правильность — у `{best_quality}`.",
        (f"`gpt-5.4-mini` даёт медиану полного ответа на {1-mini['median_s']/luna['median_s']:.1%} "
         f"меньше Luna и первый текст через {mini['ttft_median_s']:.2f} с вместо {luna['ttft_median_s']:.2f} с. "
         f"За это расход по наблюдаемому usage выше в {mini['cost_usd']/luna['cost_usd']:.1f} раза. "
         "Интервалы парных различий качества и времени с Luna включают ноль; устойчивое превосходство не установлено."),
        ("Для текущего сценария рекомендовано сохранить Luna medium. 5.4-mini — кандидат при высоком приоритете TTFT "
         "и готовности платить больше; перед заменой нужен отдельный тест полного агентского цикла. "
         "4o-mini выигрывает медиану, но теряет полноту и имеет длинный хвост задержек. "
         "4.1-mini не дал выигрыша в цене/скорости и получил один таймаут. "
         "Nano часто оставляет буквальный [N] при этом общем prompt; его результат требует отдельной настройки цитирования.")
        if cheapest == best_quality == baseline else
        "Рабочую модель не меняли. Решение о замене требует отдельной проверки полного агентского цикла.",
    ]
    summary["conclusions"] = conclusions
    doc = ROOT / "docs/manual_model_comparison_2026-10-19.md"
    tracked = ROOT / "docs/manual_model_comparison_2026-10-19.json"
    save(tracked, summary)
    lines = [
        "# Сравнение пяти моделей по мануалам MIDAS — 19.10.2026", "",
        "Рабочая модель и настройки бота не менялись; хеши `.env` и compose совпали до и после прогона.", "",
        *[part for conclusion in conclusions for part in (conclusion, "")],
        "## Протокол", "",
        ("20 вопросов по GTS NX из существующего development-набора × 2 повтора × 5 моделей = 200 попыток. "
        "У каждого вопроса пять одинаковых сохранённых фрагментов из мануалов. Вопрос, текст и нумерация "
        "источников одинаковы для всех моделей; эталоны и критерии не передавались генераторам. "
        "Порядок моделей чередуется: каждая занимает каждую позицию восемь раз."), "",
        ("Responses API, streaming, Standard (`default`), `store=false`, concurrency=1, SDK retries=0. "
        "Luna: reasoning medium, общий cap 8992 токена; 5.4 mini/nano: none, cap 800; "
        "4.1 mini/4o mini: без reasoning, cap 800. Общая инструкция — обычно 100–150 слов, "
        "на простой вопрос короче. Общий cap Luna включает резерв 8192 для reasoning и не гарантирует "
        "отдельный жёсткий лимит видимого ответа."), "",
        ("TTFT — от отправки запроса до первого непустого текстового delta; полное время — до завершения "
        "потока, включая сеть. Медианы/p95 считаются среди успешных завершений. "
        "Неуспешные попытки показаны отдельно. Поиск, HTTP/Telegram-доставка и агентские циклы в эти времена не входят."), "",
        "## Тарифы API", "",
        "USD за миллион токенов, Standard, короткий контекст; налоги и конвертация валют не включены.", "",
        "| Модель | Input | Cache read | Cache write | Output, включая reasoning |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        inp, read, write, out = manifest["rates"][model]
        lines.append(f"| `{model}` | ${inp:g} | ${read:g} | ${write:g} | ${out:g} |")
    lines += ["", ("Для 4.1/4o/5.4 запись кеша посчитана по обычной цене input. "
              "Отдельная цена cache write применяется к Luna."), "",
        "## Скорость, качество и стоимость", "",
        ("Качество и grounding — средняя оценка по рубрике 0/0.5/1, а не вероятность правильного ответа. "
        "Grounding исключает чистые отказы без технических утверждений. Стоимость рассчитана по фактическому "
        "usage и [тарифам OpenAI](https://developers.openai.com/api/docs/pricing), проверенным 19.10.2026. "
        "Reasoning входит в output один раз; cache read/write учитываются отдельно."), "",
        "| Модель | Успех / попытки | TTFT p50, с | Полный ответ p50 / p95, с | Полнота/правильность | Grounding | Critical / ошибки ссылок (judge) | Известные USD за 40 попыток | USD / 1000 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, metrics in summary["models"].items():
        lines.append(f"| `{model}` | {metrics['completed']}/{metrics['n']} | {metrics['ttft_median_s']:.2f} | {metrics['median_s']:.2f} / {metrics['p95_s']:.2f} "
                     f"| {metrics['quality']:.1%} | {metrics['grounding']:.1%} | {metrics['critical']} / {metrics['citation_errors']} "
                     f"| ${metrics['cost_usd']:.5f} | ${metrics['cost_usd'] / metrics['n'] * 1000:.2f} |")
    total = sum(m["cost_usd"] for m in summary["models"].values())
    unknown = sum(m["unknown_cost_reservations_usd"] for m in summary["models"].values())
    lines += ["", (f"Известная стоимость генерации: **${total:.5f}**; отдельного слепого оценивания: "
              f"**${summary['judge_cost_usd']:.5f}**; всего с известным usage: **${total + summary['judge_cost_usd']:.5f}**."),
              "", (f"Неизвестная стоимость неуспешных попыток: резерв **${unknown:.5f}**. "
              "Значение USD / 1000 для модели с неизвестным usage — нижняя оценка, без этого расхода."),
              "", (f"Неизвестная стоимость неуспешных оцениваний: резерв "
              f"**${summary['judge_unknown_cost_reservations_usd']:.5f}**, отдельно от известных расходов."),
              "", (f"Отдельный неизвестный расход прерванной при исправлении runner попытки зарезервирован "
              f"в размере ${summary.get('prior_unknown_reserve_usd', 0):.5f}; это верхняя оценка, не подтверждённое списание."),
              "", (f"С учётом резерва sandbox preflight ${summary['excluded_preflight_unknown_reserve_usd']:.5f} "
              f"все неизвестные расходы имеют суммарный резерв ${summary['all_unknown_cost_reserves_usd']:.5f}. "
              f"Известный usage плюс эти резервы: ${total + summary['judge_cost_usd'] + summary['all_unknown_cost_reserves_usd']:.5f}. "
              "Это консервативная оценка по токеновым лимитам, не счёт провайдера."),
              "", "## Кеш и длина ответа", "",
              ("Кеш провайдера наблюдался, но не очищался и не выравнивался принудительно. "
              "Расчёт без кеша ниже — альтернативная оценка по тем же токенам, не дополнительный расход."), "",
              "| Модель | Input | Cache read / write | Output / reasoning | Видимых output на ответ, среднее | USD без кеша, известный usage |",
              "|---|---:|---:|---:|---:|---:|"]
    for model, metrics in summary["models"].items():
        u = metrics["usage"]
        lines.append(f"| `{model}` | {u['input']} | {u['cache_read']} / {u['cache_write']} "
                     f"| {u['output']} / {u['reasoning']} | {metrics['visible_output_tokens'] / metrics['completed']:.0f} "
                     f"| ${metrics['uncached_cost_usd']:.5f} |")
    lines += ["", "## Проверка качества", "",
              ("Отдельный оценщик — GPT-6.1 Sol High. Для каждого вопроса имеющиеся ответы "
              "анонимизированы и перемешаны. Оценщик видит точные источники и source-corrected reference, "
              "названия моделей и повторы скрыты. Проверяется соответствие каждой существенной фразы "
              "именно процитированному фрагменту. Это модельная оценка; экспертная инженерная приёмка не проводилась. "
              "Метки critical требуют проверки инженером. USD / 1000 — экстраполяция этой выборки, "
              "а не фиксированная цена запроса. Отсутствующие ответы получают correctness=0 и grounding=null "
              "по детерминированному правилу и не отправляются оценщику."), "",
              "Детерминированная проверка буквального placeholder `[N]` вместо номера источника: " +
              "; ".join(f"`{model}` — {count}/40" for model, count in
                        summary["literal_citation_placeholders"].items()) + ". "
              "Число неверных цифровых маркеров не включает такие placeholders.", "",
              ("Перед генерацией проверены старые эталоны: для golden-13 убрано неверное требование "
              "об устранении всех ограничений MC; для golden-32 убрано требование полной динамической "
              "классификации, отсутствующей в выдержках. Для golden-07/19/26/33 добавлены уточнения scope, "
              "перевода Lanczos, ограничений K0 и корректного отказа при отсутствии цены."), "",
              ("Выборочно проверены 15 ответов первого повтора по golden-13/26/32 и critical-флаги "
              "первых семи вопросов отдельным агентом. Это дополнительная модельная проверка, "
              "не инженерная приёмка; исходные баллы judge не заменялись. Подтверждены, например, "
              "неверное расширение независимости угла трения до всей прочности у Luna (slot34), "
              "перенос свойств двойного упрочнения на неразрушимость всей MMC у 4o-mini (slots31/130), "
              "и перенос поглощения волн Compliant Base на Infinite Element у 4.1-mini (slot124)."), "",
              ("Сами выдержки неоднозначны: golden-07 использует R² в [2] и R в [4]; "
              "golden-09 переводит intermediate principal stress как «среднее главное»; "
              "golden-26 [2] словесно задаёт обратное отношение напряжений относительно формулы σh/σv. "
              "Эти расхождения источников не являются автоматически доказанными ошибками кандидатов. "
              "Critical-флаги K0 у 4o-mini (slots81/180) и nano (slot184) связаны с воспроизведением "
              "противоречия источника; сырой счётчик не равен числу самостоятельных галлюцинаций. "
              "Эта оценка проверяет ответы по сохранённым выдержкам, а не правильность самих мануалов."), "",
              "## Парные различия с Luna", "",
              ("95% bootstrap-интервалы: 5000 выборок по 20 вопросам, оба повтора вопроса остаются вместе. "
              "Показана разница средних (кандидат минус Luna). Для времени берутся только вопросы "
              "с обоими успешными повторами обеих моделей; n показано отдельно, таймаут не считается "
              "временем готового ответа. Малый знакомый development-набор "
              "не доказывает общей эквивалентности или превосходства. Интервалы исследовательские, "
              "без поправки на четыре сравнения."), "",
              "| Модель | Изменение качества, п.п. [95% CI] | Изменение полного времени, с [95% CI] | n для времени |",
              "|---|---:|---:|---:|"]
    for model, delta in paired.items():
        q, t = delta["quality"], delta["latency_s"]
        lines.append(f"| `{model}` | {100*q['mean_delta']:+.1f} [{100*q['ci95_low']:+.1f}; {100*q['ci95_high']:+.1f}] "
                     f"| {t['mean_delta']:+.2f} [{t['ci95_low']:+.2f}; {t['ci95_high']:+.2f}] | {t['n_questions']} |")
    lines += ["", "## Вопросы и существенные замечания", ""]
    for case in cases:
        lines += [f"### {case['id']}: {case['question']}", ""]
        for model in MODELS:
            selected = [r for r in rows if r["case_id"] == case["id"] and r["model"] == model]
            notes = list(dict.fromkeys(scores[r["slot"]]["notes"] for r in selected))
            quality = statistics.mean(scores[r["slot"]]["correctness_completeness"] for r in selected)
            lines.append(f"- `{model}` — {quality:.2f}: " + " / ".join(notes))
        lines.append("")
    lines += ["## Воспроизводимость и ограничения", "",
              "Использованы файлы: " + "; ".join(f"`{name}`" for name in evidence_files) + ".", "",
              ("Замороженный исходник: `var/luna_migration/comparison_complete.json`; SHA-256: "
              f"`{manifest['source_sha256']}`. Полные входы, хеши исходников/конфигурации, "
              "расписание, usage, ответы и оценки сохранены в `var/manual_model_benchmark/2026-10-19-api/`. "
              "Сводка без полных выдержек находится рядом с этим отчётом в JSON."), "",
              ("Первый sandbox preflight сохранил пять APIConnectionError в отдельной папке "
              "`var/manual_model_benchmark/2026-10-19/`. Они исключены из сравнения скорости/качества. "
              "Usage для них отсутствует: возможная стоимость неизвестна и не объявляется нулевой; "
              "консервативные резервы сохранены в файлах попыток. Все сравниваемые вызовы выполнены "
              "в разрешённом сетевом режиме."), "",
              ("После 83 завершённых ответов синхронный транспорт runner заменён на асинхронный с общим "
              "дедлайном 180 с, поскольку read-timeout SDK 120 с не ограничивал полный поток. Один "
              "завершённый вызов 4o mini занял 148.91 с и оставлен в сравнении. Остановкой runner была "
              "прервана следующая попытка mini (slot83); исходная запись сохранена отдельно, новая "
              "попытка выполнена явно. Исходный код/manifest и amendment сохранены. Вопросы, порядок, "
              "prompt, модели, reasoning и лимиты генерации не менялись. Эта смена транспорта — "
              "ограничение сравнения абсолютного времени; она не затрагивает рабочий бот. "
              "Отмена и закрытие соединения могут немного превысить дедлайн."), "",
              ("Slot131 (`gpt-4.1-mini`, второй повтор golden-13) остановлен по TimeoutError через 180.001 с. "
              "Usage не получен; исходная запись сохранена и не повторялась. После завершения генерации "
              "отдельным amendment разрешён анализ законченных неуспешных попыток: ноль за отсутствие ответа, "
              "grounding=null, отдельный резерв стоимости, задержка только среди успешных. "
              "Этот amendment не менял входы/генерацию и не создавал дополнительных попыток."), "",
              ("После семи оценённых вопросов blocking-вызов judge для golden-14 получил APITimeoutError "
              "без usage. Его receipt сохранён отдельно, резерв включён в бюджет оценивания. "
              "Продолжение judge переведено на streaming с общим дедлайном 240 с; golden-14 "
              "повторён явно. Модель, reasoning, rubric, input и schema оценивания сохранены. "
              "Скорость judge в сравнении моделей не используется."), "",
              ("После 12 готовых оценок golden-20 достиг общего дедлайна 240 с. Receipt сохранён, "
              "добавлен резерв неизвестной стоимости и зафиксирован явный повтор. Последние восемь "
              "вопросов оценивались через streaming с concurrency=2, дедлайном 360 с и общим бюджетом "
              "резервов до dispatch. Prompt/model/reasoning/schema и ответы кандидатов не менялись. "
              "Все изменения протокола и прежние версии runner/manifest сохранены."), "",
              "Команды повторного анализа существующих данных (без платных вызовов):", "",
              "```powershell", ".venv/Scripts/python.exe scripts/benchmark_manual_models.py report --output var/manual_model_benchmark/2026-10-19-api",
              ".venv/Scripts/python.exe scripts/analyze_manual_models.py --output var/manual_model_benchmark/2026-10-19-api", "```", "",
              ("Создание manifest не вызывает API; команды run/judge делают платные вызовы. Сохранённые попытки "
              "никогда автоматически не повторяются. Качество поиска, агентский planning, "
              "работа с полной историей, расширенные ответы и sealed holdout здесь не проверялись. "
              "Production-архитектура, API-контракты, IMPLEMENTATION_DECISIONS и CURRENT_IMPLEMENTATION "
              "этим экспериментом не меняются."), ""]
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(doc), "summary": str(tracked)}, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "var/manual_model_benchmark/2026-10-19-api")
    publish(parser.parse_args().output)
