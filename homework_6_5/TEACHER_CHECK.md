# Проверка домашнего задания 6.5

1. Проверить, что `experiments/shared.py` содержит единственный `search_knowledge_base`: он возвращает два-три mock-документа с `content`, `source`, `score`, используется и researcher, и baseline, а production RAG/Qdrant не импортируется.
2. Убедиться, что `EXPERIMENT_CASES` состоит ровно из пяти одинаковых входов: три `corpus`, один `multi_step`, один `out_of_base`.
3. В `multi_agent_langgraph.py` проверить два `langchain.agents.create_agent`: `researcher` с общим tool и `writer` с `tools=[]`; researcher не пишет финальный ответ, writer требует цитаты `[1]`, `[2]`.
4. Проследить graph path `START -> supervisor -> researcher -> supervisor -> writer -> END`. Это manual `StateGraph` с `Command`, central supervisor и бюджетом трёх handoff; выбран вместо `langgraph-supervisor` согласно актуальной рекомендации README пакета.
5. Проверить `InMemorySaver`, стабильный `thread_id="exp-langgraph"` и `app.stream(..., stream_mode="updates")`. Mermaid в `docs/architecture-multi-agent.md` должен быть выводом `draw_mermaid()`.
6. Проверить `single_agent_baseline.py`: там один `create_agent`, тот же `search_knowledge_base` и объединённый prompt. `handoff_count` baseline всегда 0.
7. Открыть `experiments/results.json`: там 10 фактических `completed` записей `case × implementation` с числовыми `total_tokens`, `llm_calls`, `latency_ms`, `handoff_count`, `quality_score` и финальным `answer` с `[1]`; верхний статус — `completed_live_run`, judge — `completed_one_batched_call`.
8. Запустить офлайн-проверку:

   ```powershell
   $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
   .venv-eval\Scripts\python.exe -B -m pytest -q -p pytest_asyncio.plugin tests\test_multi_agent_experiment.py
   ```

   Фейковая модель проверяет tool-call turns, сумму всех `usage_metadata`, три handoff, 5 × 2 contract и один batched LLM judge parser без сети.

9. Сверить `docs/multi-agent-report.md`: таблица отражает фактические средние 1100.2 / 1470.2 tokens, p50 2248.48 / 3987.97 ms, quality 5.0 / 3.0; в выводе отмечены два incorrect out-of-base fallback и окончательное решение оставить single-agent.
10. Разрешённый live запуск уже выполнен; не повторять его без нового разрешения. Один judge вызов хранится отдельно и не добавляется в agent metrics; диагностика конфигурации по-прежнему разрешена только через `scripts/check_env_safety.py`.
