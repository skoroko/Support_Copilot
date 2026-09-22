# Проверка домашнего задания 6.4

1. Проверить `pyproject.toml` и `uv.lock`: пакеты `langgraph-checkpoint-sqlite` и
   `langgraph-checkpoint-postgres` закреплены в диапазоне 3.x, есть
   `aiosqlite` и `psycopg[binary,pool]` v3.
2. В `app/services/agent_persistent.py` проверить `build_agent(checkpointer)`,
   `agent_lifespan()` и default `AGENT_CHECKPOINTER=sqlite`; старый
   `agent_graph.py` не менялся.
3. Проследить topology опасного действия:
   `prepare_telegram_message -> confirm_and_execute_telegram_message`. До
   `interrupt()` только draft; side effect вызывается только после
   `Command(resume=True)`.
4. Запустить офлайн-тесты и убедиться, что они используют реальный
   `AsyncSqliteSaver(":memory:")`, проверяют snapshot.next, approve, reject и
   `AsyncMock.assert_not_called()`.
5. Запустить `scripts/time_travel_demo.py`: вывод должен показать interrupt,
   историю checkpoint, past state и две ветки на отдельных thread id.
6. Проверить `/agent/stream`: SSE содержит `updates`, `messages` и
   `__interrupt__`; повторный запрос с тем же `thread_id` и `resume=true`
   продолжает сохранённый граф только с теми же `X-Internal-Token` и
   `X-User-ID`. System-role во входе отклоняется, а `full` требует admin token.
7. В `alembic/env.py` проверить `include_name` для строго четырёх checkpoint
   таблиц. Их схему создаёт saver.setup(), а не Alembic.
8. Сверить `docs/agent-persistent-report.md`: в нём восемь разделов, честный
   статус Postgres smoke test, idempotency boundary и permission policy.
9. Проверить, что parallel tool calls и последний вызов на шестом model turn
   получают парные `ToolMessage`, а ошибка графа завершается безопасным SSE
   `agent_stream_failed`.
10. В `bot/web.py` проверить Redis `SET NX` по idempotency key; повторный ключ
    не вызывает `bot.send_message` второй раз.
