# ДЗ 2.6: мультимодальный Vision API

Вариант A: анализ изображений.

Что реализовано:

- прием пути к изображению от пользователя;
- кодирование изображения в base64 data URL;
- отправка в OpenAI-compatible Vision API через `POST /chat/completions`;
- текстовый ответ на русском языке для инженера поддержки;
- лог работы программы;
- примеры входных изображений в `homework_2_6/examples/`.

## Переменные окружения

Секреты хранятся в `.env`, не в коде:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
VISION_MODEL=<актуальная vision-capable модель из официальной документации>
VISION_TEMPERATURE=0.2
VISION_MAX_TOKENS=500
VISION_DETAIL=auto
```

`VISION_MODEL` должен быть OpenAI-compatible model id, который поддерживает
изображения. CLI не подставляет текстовую `OPENAI_MODEL`, чтобы случайно не
отправить изображение в модель без Vision-поддержки.

## Запуск через Vision API

```powershell
python homework_2_6/vision_image_analysis.py homework_2_6/examples/conv.png --question "Что видно на изображении?" --log-file homework_2_6/demo_log.txt
python homework_2_6/vision_image_analysis.py homework_2_6/examples/func.png --question "Какая проблема показана?" --log-file homework_2_6/demo_log.txt
```

CLI не выводит API-ключ и не пишет base64 изображения в лог.

Тот же Vision-сценарий подключен к общему CLI из ДЗ 2.5:

```text
/image homework_2_6/examples/conv.png Что видно на скриншоте?
```

Если CLI уже запущен, не вставляйте строку `python -m support_copilot` в `you>`.
Сначала выйдите командой `/quit`, затем запускайте новую команду в PowerShell.
