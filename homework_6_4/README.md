# Домашнее задание 6.4 - persistent LangGraph + HIL

ДЗ Б6.4 добавляет персистентность и human-in-the-loop к отдельному графу,
не меняя `app/services/agent_graph.py` из Б6.3.

- `app/services/agent_persistent.py` - `build_agent(checkpointer)` и lifespan
  с `memory`, `sqlite` и `postgres` backend;
- `app/routers/agent.py` - защищённый `POST /agent/stream` с SSE,
  приватным namespace и `Command(resume=...)`;
- `scripts/time_travel_demo.py` - полностью офлайн SQLite in-memory demo;
- `docs/agent-persistent-report.md` - дизайн, лог и известные ограничения.

## Офлайн-проверка

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.venv-eval\Scripts\python.exe -B -m pytest -q -p pytest_asyncio.plugin tests\test_agent_persistent.py
$env:PYTHONUTF8 = "1"
.venv-eval\Scripts\python.exe -B scripts\time_travel_demo.py
```

`AsyncMock` заменяет реальный вызов существующего внутреннего bot `/notify`
endpoint. Тесты и demo не обращаются к OpenAI, Telegram или Postgres. Граф
также сохраняет лимит в шесть model turns из Б6.3 и парит каждый tool call с
`ToolMessage` даже на границе лимита.

## Конфигурация

Локальный безопасный default:

```dotenv
AGENT_CHECKPOINTER=sqlite
AGENT_SQLITE_PATH=./var/agent_checkpoints.sqlite
```

В compose используется существующая база `support_copilot`:

```dotenv
AGENT_CHECKPOINTER=postgres
LANGGRAPH_STRICT_MSGPACK=true
```

Production smoke проверен на этом compose: сервис `app` стал healthy, а
`AsyncPostgresSaver.setup()` создал четыре checkpoint-таблицы в существующей
базе `support_copilot`.

Не создавайте новый `thread_id` для каждого HTTP-вызова. Используйте стабильное
значение, например `tg-123456-license-escalation`, затем передавайте его же при
`{"resume": true, "approval_id": "<точный 32-символьный ID из preview>"}`
с теми же `X-Internal-Token` и `X-User-ID`. Сервер хеширует
identity и thread id; роль `full` дополнительно требует `X-Admin-Token`.

Опасная доставка использует детерминированный idempotency key. Bot резервирует
его через Redis `SET NX`; compose включает AOF, поэтому повторный resume не
дублирует Telegram-сообщение.

Текущая интеграция Telegram не требует ручных HTTP-команд или `/agent`:
обычные сообщения идут в persistent graph через `/chats`, а точный recipient
и текст подтверждаются кнопками. История, sources/feedback и pending draft
сохраняются после перезапуска; `/clear` отменяет pending. Production search
использует мануалы RAG, без синтетической базы. См. `docs/unified_agent.md`.
