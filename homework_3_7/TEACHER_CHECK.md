# Краткая сводка для проверки ДЗ 3.7

## Что добавлено

- `eval/golden_dataset.json` - 22 кейса, `version: 1`, 6 категорий, 8 hard-кейсов.
- `tests/unit/` - 18 быстрых unit-тестов без сети и без API-ключей.
- `eval/run_evaluation.py` - CLI для ручного evaluation-прогона через `/chat` и judge-модель.
- `eval/check_thresholds.py` - release gate по агрегатам.
- `eval/thresholds.yaml` - пороги `correctness_avg >= 4.0`, `min_correctness >= 2.0`.
- `eval/runs/2026-07-07.json` - offline smoke artifact для проверки JSON-формата и threshold checker.

## Команды проверки

```powershell
python -m pytest tests/unit/ -m "not llm"
python eval/run_evaluation.py --offline-smoke --out eval/runs/2026-07-07.json
python eval/check_thresholds.py eval/runs/2026-07-07.json
python -m compileall app eval tests/unit homework_3_7
```

## Реальный прогон с judge

```powershell
uvicorn app.main:app --port 8000
python eval/run_evaluation.py --golden eval/golden_dataset.json --judge gpt-5.2 --out eval/runs/2026-07-07.json
```

Реальный прогон требует запущенный FastAPI-сервис и настроенный `OPENAI_API_KEY`.
Offline smoke нужен только для локальной проверки проводки без затрат.
