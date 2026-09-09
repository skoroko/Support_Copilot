# ДЗ 4.4 - production-обвязка

## Реализовано

- `app/moderation/ModerationService` выполняет локальную keyword/regex-проверку
  из `moderation_keywords.yaml`, затем опционально вызывает OpenAI Moderation API.
- Вход блокируется до LLM-вызова с HTTP `403` и кодом `moderation_blocked`.
- Ответ LLM накапливается в backend, проверяется целиком и только после этого
  передаётся пользователю; запрещённый ответ заменяется безопасным сообщением.
- Инциденты логируются через structlog: только `sha256[:16]`, PII-маскированный
  preview, категории, направление и слой блокировки.
- `GET /chats/admin/stats`, `GET /chats/admin/users` и
  `POST /chats/admin/broadcast` защищены заголовком `X-Admin-Token`.
- PostgreSQL хранит feedback с `UNIQUE (owner_external_id, message_id)`, события
  модерации, latency и очередь рассылок. JSON backend предоставляет локальный fallback.
- Бот показывает 👍/👎 после каждого ответа, сохраняет голос в backend и убирает
  клавиатуру после callback.
- Admin-команды `/stats`, `/users`, `/broadcast <text>` проходят router-level
  фильтр `IsAdmin` по `BOT_ADMIN_IDS`.
- Фоновый worker бота забирает pending-рассылки, отправляет их Telegram-пользователям
  и переводит запись в `sent` или `failed`.
- `docker compose up` запускает app, bot и PostgreSQL, автоматически выполняет
  `alembic upgrade head`; данные БД находятся в volume `pg-data`.

## Настройка

Добавьте в `.env`:

```dotenv
ADMIN_TOKEN=отдельный-длинный-случайный-секрет
BOT_ADMIN_IDS=ваш_telegram_user_id
MODERATION_ENABLED=true
MODERATION_OPENAI_ENABLED=false
MODERATION_MODEL=omni-moderation-latest
MODERATION_THRESHOLDS={}
BROADCAST_POLL_SECONDS=5
BOT_HOST_PORT=19000
APP_HOST_PORT=18000
```

`ADMIN_TOKEN` должен отличаться от `INTERNAL_TOKEN`. Чтобы включить облачный слой,
задайте `MODERATION_OPENAI_ENABLED=true`. Локальные regex-правила работают всегда,
когда `MODERATION_ENABLED=true`.

## Запуск

```powershell
docker compose up -d --build
docker compose ps
```

Проверка admin API:

```powershell
curl.exe http://localhost:18000/chats/admin/stats `
  -H "X-Admin-Token: $env:ADMIN_TOKEN"
```

Проверка запрещённого ввода:

```powershell
curl.exe -X POST http://localhost:18000/chats/CHAT_ID/messages `
  -F "content=убей себя"
```

Ожидается HTTP `403` и `error.code = moderation_blocked`.

## Тесты

```powershell
python -m pytest
python -m compileall app bot alembic tests homework_4_4
docker compose config
```

PostgreSQL contract-тесты запускаются дополнительно при наличии
`CHAT_TEST_DATABASE_URL`. Остальные проверки работают без сети и реальных токенов.
