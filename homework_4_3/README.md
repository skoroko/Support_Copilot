# ДЗ 4.3 - мультимодальность и streaming

## Реализовано

- `POST /chats/{chat_id}/messages` принимает единый `multipart/form-data` контракт:
  обязательный `content` и опциональный `media`.
- `app/chat/media.py` преобразует изображения в `image_url` data URI, аудио - в
  текст через OpenAI Audio Transcriptions, PDF/DOCX - в извлечённый текст.
- Изображение передаётся прямо в основной `chat.completions.create`; отдельного
  vision-вызова нет. Для OGG/MP3/M4A не используется FFmpeg.
- `media_refs.part` сохраняется в JSON/PostgreSQL истории и восстанавливается в
  следующих LLM-вызовах.
- `BackendClient.send_message(...)` обслуживает и текст, и медиа, разбирая
  типизированные SSE-события `token`/`done`.
- Telegram handlers принимают фото до 2 МБ, голос, аудио и PDF/DOCX до 10 МБ.
- Ответ отображается через `send_message_draft`, затем фиксируется обычным
  `send_message`; до первого токена показывается `typing`.
- Бот поднимает защищённый `POST /notify`, backend содержит notifier и demo endpoint
  `POST /chats/{chat_id}/system-message`.
- HTTP-клиент один на процесс бота; connect errors повторяются только до начала
  стрима, а пользователь получает понятные сообщения для connect/timeout/429/5xx.

## Настройка

Скопируйте `.env.example` в `.env` и замените как минимум:

```dotenv
OPENAI_API_KEY=...
BOT_TOKEN=...
INTERNAL_TOKEN=длинный-случайный-секрет
TRANSCRIPTION_MODEL=whisper-1
```

Model ID берутся только из окружения. Контракт image content-part и Audio
Transcriptions проверен по актуальной официальной документации OpenAI.

## Запуск

Локально в двух терминалах:

```powershell
uvicorn app.main:app --reload --port 8000
python -m bot
```

Или весь стек:

```powershell
docker compose up -d --build
alembic upgrade head
```

Пример текстового multipart-запроса:

```powershell
curl.exe -N -X POST http://localhost:8000/chats/CHAT_ID/messages `
  -F "content=Как проверить лицензию?"
```

Пример с изображением:

```powershell
curl.exe -N -X POST http://localhost:8000/chats/CHAT_ID/messages `
  -F "content=Что показано на скриншоте?" `
  -F "media=@screen.png;type=image/png"
```

Проверка обратного канала:

```powershell
curl.exe -X POST http://localhost:9000/notify `
  -H "X-Internal-Token: $env:INTERNAL_TOKEN" `
  -H "Content-Type: application/json" `
  -d '{"chat_id":123456789,"text":"Фоновая задача завершена"}'
```

## Проверка

```powershell
python -m pytest
python -m compileall app bot alembic tests homework_4_3
```

PostgreSQL contract-тесты запускаются при заданном `CHAT_TEST_DATABASE_URL`.
Остальные тесты не требуют сети, Telegram token или реального OpenAI API key.
