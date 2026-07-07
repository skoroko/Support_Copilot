# Домашнее задание 3.7: тестирование и оценка качества ИИ-приложений

В этом инкременте к FastAPI LLM Service добавлен отдельный testing/evaluation
слой:

- `eval/golden_dataset.json` - versioned golden dataset из 22 кейсов Support Copilot;
- `tests/unit/` - быстрые unit-тесты без API-ключей и сети;
- `eval/run_evaluation.py` - ручной LLM-as-judge прогон по golden dataset;
- `eval/check_thresholds.py` и `eval/thresholds.yaml` - проверка порогов качества;
- `eval/runs/2026-07-07.json` - offline smoke artifact для проверки формата и thresholds.

## Быстрые unit-тесты

```powershell
python -m pytest tests/unit/ -m "not llm"
```

Покрытие:

- prompt builder: порядок ролей и безопасная обработка фигурных скобок;
- JSON parser: markdown-fence, обычный JSON, malformed JSON;
- tool calls parser;
- Pydantic validation: пустое и слишком длинное сообщение;
- PII-safe repr;
- расчёт стоимости из usage;
- cache hit/miss без реального LLM;
- retry на 429;
- FastAPI dependency override вместо реального LLM-сервиса.

## Реальный evaluation-прогон

Сначала поднять FastAPI-сервис:

```powershell
uvicorn app.main:app --port 8000
```

Затем запустить evaluation:

```powershell
python eval/run_evaluation.py --golden eval/golden_dataset.json --judge gpt-5.2 --out eval/runs/2026-07-07.json
```

Runner отправляет production-вызовы с `temperature=0`. Judge вызывается с
`temperature=0` и `response_format={"type": "json_object"}`. Prompt judge требует
G-Eval порядок `reasoning` перед `scores`.

## Offline smoke

Без ключей и сети можно проверить формат артефакта, агрегаты и thresholds:

```powershell
python eval/run_evaluation.py --offline-smoke --out eval/runs/2026-07-07.json
python eval/check_thresholds.py eval/runs/2026-07-07.json
```

Offline smoke не заменяет реальный LLM-as-judge прогон.
