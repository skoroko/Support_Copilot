# Домашнее задание 6.5 — Multi-agent исследование

Работа изолирована в `experiments/`, поэтому она не изменяет production RAG или будущий `app/agents/` до обоснованного решения.

- `experiments/shared.py` — единый mock `search_knowledge_base`, пять входов, честный сбор usage и batched judge parser;
- `experiments/single_agent_baseline.py` — один `create_agent` с тем же tool;
- `experiments/multi_agent_langgraph.py` — manual supervisor, researcher, writer, `InMemorySaver`, update streaming и bound из трёх handoff;
- `experiments/results.json` — 10 фактических записей разрешённого live-прогона, `completed_live_run`;
- `docs/multi-agent-report.md` и `docs/architecture-multi-agent.md` — вывод и схема;
- [`docs/multi-agent-benchmark-answers.md`](../docs/multi-agent-benchmark-answers.md) — компактные ответы и сравнение по пяти кейсам;
- [`DEFENSE_GUIDE.md`](DEFENSE_GUIDE.md) — сценарий защиты, offline-проверки и ответы на вопросы.

## Офлайн-проверка

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.venv-eval\Scripts\python.exe -B -m pytest -q -p pytest_asyncio.plugin tests\test_multi_agent_experiment.py
.venv-eval\Scripts\python.exe -B -m py_compile experiments\shared.py experiments\single_agent_baseline.py experiments\multi_agent_langgraph.py
.venv-eval\Scripts\python.exe -B -m experiments.multi_agent_langgraph --write-architecture
```

Тесты используют `FakeMessagesListChatModel` с no-op `bind_tools`; они не обращаются к OpenAI, Qdrant, Redis или сети. Последняя команда строит `ChatOpenAI`, но не вызывает его; она только заново сохраняет Mermaid из `draw_mermaid()`.

## Выполненный live benchmark

Перед запуском не показывайте `.env`; разрешён только безопасный статус:

```powershell
.venv-eval\Scripts\python.exe -B scripts\check_env_safety.py
```

После отдельного разрешения был выполнен следующий запуск (повторно его не запускайте без нового разрешения):

```powershell
.venv-eval\Scripts\python.exe -B -m experiments.multi_agent_langgraph --allow-external-api
```

Все 10 agent runs завершились успешно, и один judge вызов сохранён отдельно от agent metrics. Фактическое сравнение: single-agent — 1100.2 tokens, 2.0 calls, p50 2248.48 ms, quality 5.0; multi-agent — 1470.2 tokens, 3.0 calls, p50 3987.97 ms, quality 3.0 и 3 handoff. Подробная таблица, интерпретация и окончательное решение находятся в [`docs/multi-agent-report.md`](../docs/multi-agent-report.md). Модель по-прежнему берётся из существующего `OPENAI_MODEL` settings; при отсутствии используется безопасный fallback `gpt-5.4-mini`.

Отдельный baseline можно запустить тем же opt-in способом, не перезаписывая comparison file:

```powershell
.venv-eval\Scripts\python.exe -B -m experiments.single_agent_baseline --allow-external-api
```
