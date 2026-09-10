# Проверка ДЗ 5.5

1. `scripts/ingest.py` читает PDF, DOCX, HTML, Markdown, TXT и CHM.
2. `data/` содержит 70 реальных файлов; точный состав и размер приведены в
   `docs/data_inventory.md`.
3. `app/services/ingestion.py` добавляет source, timestamps, author, category и
   version; технические поля исключены из embedding.
4. Индексация использует `IngestionPipeline`, `SimpleDocumentStore`,
   `DocstoreStrategy.UPSERTS`, `SentenceSplitter(512, 64)` и Qdrant.
5. Повторный полный offline ingest: `0 changed, 1992 unchanged, 0 failed`.
6. Query выполняет retrieval top-10, optional BGE top-5 и score guard до LLM.
7. Ответ содержит `[1]`, `[2]`; sources имеют `id`, `file_name`, `page`,
   `score`, `snippet`, `confident`.
8. Multi-turn test проверяет «Каков срок возврата?» → «А для электроники?».
9. Низкий score возвращает честный отказ без вызова LLM; project threshold
   `0.45` обоснован в `docs/rag.md`.
10. `POST /rag/query` синхронный; `/chats/{id}/messages` отправляет JSON-safe
    токены и финальный `event: sources`.
11. `POST /documents/upload` сохраняет allowlisted файл и отвечает 202 перед
    background UPSERT.
12. Telegram редактирует одно сообщение с debounce 0.8 секунды; 👍/👎 сохраняют
    feedback вместе с показанными sources.
13. `compose.yaml` поднимает app, Qdrant, Redis и bot одной командой, сохраняя
    также существующие Postgres/Phoenix и персистентные данные.
14. `docs/rag.md` содержит две Mermaid-схемы, параметры, threshold и endpoints.
15. Живой OpenAI ingestion запускался с явным opt-in; региональный ответ
    провайдера `403 unsupported_country_region_territory` корректно переводит
    только RAG в degraded-режим, не останавливая FastAPI и Telegram-бота.
