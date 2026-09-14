# Проверка домашнего задания 5.6

1. Выполнить `scripts/verify_eval_setup.py` в `.venv-eval`: должны быть RAGAS
   0.4.x, пять метрик и строка `eval setup: OK`.
2. Проверить, что `tests/eval/golden_dataset.json` содержит не менее 30 записей
   с `user_input`, `reference`, `reference_contexts`, а сырой CSV сохранён
   отдельно.
3. Запустить либо открыть три timestamped CSV в `tests/eval/results`: baseline,
   `top_k_5`, `chunk_1024`.
4. Сверить агрегаты, пороги и 3–5 худших случаев в `docs/rag_evaluation.md`.
5. Выполнить `docker compose config --quiet`, затем открыть
   <http://localhost:6006> и убедиться, что в проекте `support-copilot-rag`
   есть не менее 20 RAG-трасс. То же условие и наличие дочерних
   retriever/embedding/LLM spans проверяются командой
   `.venv-eval\Scripts\python.exe scripts\verify_phoenix_traces.py --minimum 20`.
6. Запустить тесты:

```powershell
$env:PHOENIX_WORKING_DIR="$PWD\var\phoenix-test"
.venv-eval\Scripts\python.exe -m pytest -q
python -m ruff check .
```

Перед любым live-прогоном требуется явный флаг
`--allow-external-data-transfer`: без него скрипты завершаются до внешнего
вызова.
