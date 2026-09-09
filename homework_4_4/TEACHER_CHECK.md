# Проверка ДЗ 4.4

1. `python -m pytest` - unit/API/contract-тесты.
2. Запрещённый input возвращает HTTP 403 с `moderation_blocked` до LLM-вызова.
3. `X-Admin-Token` обязателен для всех `/chats/admin/*` endpoints.
4. `IsAdmin` установлен через `admin_router.message.filter(...)`.
5. После ответа Telegram отображаются 👍/👎; повторный голос не создаёт дубль.
6. `/broadcast <text>` создаёт pending-запись, background worker отправляет её.
7. `docker compose up -d --build` поднимает app + bot + postgres и применяет Alembic.
8. PostgreSQL использует named volume `pg-data`.

OpenAI Moderation API является опциональным вторым слоем. Model ID читается из
`MODERATION_MODEL`; значение по умолчанию - `omni-moderation-latest`.
