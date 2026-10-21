# Домашнее задание 6.3 — LangGraph ReAct agent

Фактический сравнительный прогон от 21 октября 2026 года сохранён в
[отдельном отчёте](BENCHMARK_2026-10-21.md) и
[результатах 45 исполнений](results/benchmark_2026-10-21.json): 44 завершения,
один timeout native ReAct; custom и prebuilt — по 15/15 завершений.
Он использует одну текущую Luna и
учебную локальную KB; завершение исполнения и качество ответа оцениваются
раздельно. Публичные результаты исключают messages/checkpoints и opaque
reasoning payload.

В инкременте ручной ReAct loop из 6.2 сохранён без изменений и дополнен двумя
LangGraph-вариантами:

- `app/services/agent_graph.py` — явный `StateGraph` с тремя узлами и чистым
  роутером;
- тот же модуль экспортирует `prebuilt_graph`, созданный через
  `langchain.agents.create_agent` с теми же prompt/model/tools;
- `scripts/visualize_graph.py` создаёт Mermaid-исходники в `docs/`;
- `scripts/bench_agents.py` сравнивает исходный 6.2 loop, custom graph и
  prebuilt agent на неизменённых пяти кейсах 6.2.

## Офлайн-проверка

```powershell
.venv-eval\Scripts\python.exe -m pytest -q tests/test_agent_graph.py
.venv-eval\Scripts\python.exe scripts/visualize_graph.py
.venv-eval\Scripts\python.exe scripts/bench_agents.py
```

Последняя команда намеренно завершится до сетевого вызова и напомнит про
`--allow-external-api`.

## Live benchmark (только после отдельного разрешения)

```powershell
.venv-eval\Scripts\python.exe scripts/bench_agents.py --allow-external-api
```

Команда выполняет минимум 45 агентных запусков; провайдерских запросов может
быть больше, если агент использует несколько шагов. Она сохраняет только
метаданные benchmark в `docs/agent-graph-benchmark.json` и не выводит
содержимое `.env` или API-ключи. О наличии конфигурации можно узнать только
статусной командой:

```powershell
.venv-eval\Scripts\python.exe scripts/check_env_safety.py
```

Подробности архитектуры, ограничения и честное состояние benchmark находятся
в [`docs/agent-graph-report.md`](../docs/agent-graph-report.md).
