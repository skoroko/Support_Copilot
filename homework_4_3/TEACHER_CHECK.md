# Проверка ДЗ 4.3

1. `python -m pytest tests/app/chat tests/chat tests/bot -q`
2. Убедиться, что `/chats/{id}/messages` принимает form-data и не существует
   `/messages/with-media`.
3. Отправить боту фото: backend получает multipart, а основной chat-вызов -
   `image_url` part.
4. Отправить голосовое OGG: OpenAI Audio Transcriptions получает исходный файл без
   FFmpeg.
5. Проверить повторный вопрос в том же чате: `media_refs.part` остаётся в истории.
6. Проверить `/notify` с неверным и верным `X-Internal-Token` (401 и 200).
7. На длинном ответе проверить рост Telegram draft и финальное сообщение.
