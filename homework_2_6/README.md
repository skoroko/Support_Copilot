# Homework 2.6A - Vision In Support Copilot CLI

Цель: добавить мультимодальный сценарий в накопительный Support Copilot, а не
делать отдельный одноразовый прогон тестовых картинок.

## Основной способ запуска

Vision-сценарий теперь доступен в общем CLI из ДЗ 2.5:

```powershell
python -m support_copilot --real
```

Внутри CLI:

```text
/image ".\homework_2_6\examples\ошибка расчета.jpg" | Опиши ошибку и подскажи, что запросить у пользователя.
```

Команда `/image`:

- принимает путь к PNG/JPEG-файлу;
- кодирует изображение в base64 data URL;
- отправляет его в OpenAI-compatible Vision API в real mode;
- при запуске CLI с `--stream` выводит real Vision-ответ по токенам;
- при пустом ответе или ошибке пробует следующий настроенный Vision-провайдер;
- в offline mode дает локальный шаблонный ответ без сетевого вызова;
- добавляет текстовый результат анализа в историю CLI;
- отражается в `/stats` как `image requests`.

## Настройки `.env`

Fallback-цепочка Vision в CLI:

```text
OpenAI -> OpenRouter -> Ollama
```

Для OpenAI как основного провайдера:

```dotenv
VISION_PROVIDER=openai
VISION_MODEL=gpt-5.4-mini
OPENAI_VISION_MODEL=gpt-5.4-mini
VISION_TIMEOUT_SEC=60
VISION_MAX_TOKENS=1200
```

Для OpenRouter fallback:

```dotenv
OPENROUTER_VISION_MODEL=nex-agi/nex-n2-pro:free
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Для Ollama fallback:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_VISION_MODEL=gemma3:4b
```

Модель должна поддерживать вход `text+image -> text`. `OPENROUTER_MODEL` и
`OPENAI_MODEL` относятся к обычному текстовому CLI, а `VISION_MODEL`,
`OPENROUTER_VISION_MODEL` и `OPENAI_VISION_MODEL` - к анализу изображений.

## Вспомогательный скрипт

`vision_assistant.py` сохранен как wrapper на `support_copilot.core.vision` и
одиночный smoke-test:

```powershell
python .\homework_2_6\vision_assistant.py --real --image ".\homework_2_6\examples\func.png" --question "Что видно на графике?"
```

Для сдачи накопительного проекта предпочтителен запуск через
`python -m support_copilot` и команду `/image`.
