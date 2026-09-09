# Проверка ДЗ 5.3

1. В `requirements.txt` закреплены три пакета LlamaIndex и CHM extractor; версии продублированы в `docs/rag.md`.
2. В `data/rag-block-03` ровно 10 документов: RU/EN PDF, CHM и намеренно нерелевантный Civil 2020.
3. `app/services/chm_reader.py` реально читает Windows-1251 CHM на Windows и создаёт документы по HTML-страницам.
4. `app/services/rag.py`: `SimpleDirectoryReader`, `SentenceSplitter(512/64)`, `QdrantVectorStore`, `StorageContext`, `VectorStoreIndex`, top-3 query engine.
5. `app/services/rag_baremetal.py`: явные loaders/chunks/embeddings, отдельная flat-payload коллекция, `query_points`, system+context+question через Chat Completions.
6. Обе `answer()` возвращают одинаковый контракт: `answer`, `top_score`, `sources[text, source, score]`.
7. `POST /rag/query` отделён от `/chat`, а индекс создаётся один раз в lifespan при `RAG_ENABLED=true`.
8. Сравнение, четырёхфразовый вывод для диплома и пять согласованных вопросов находятся в `docs/rag.md` и `homework_5_3/questions.json`.
9. `scripts/evaluate_rag.py` не допускает случайной отправки руководств без `--allow-external-data-transfer`.
10. Реальный прогон обеих реализаций сохранён в `results/rag_evaluation.json`, а таблица релевантности заполнена в `docs/rag.md`.
11. `RAG_REBUILD_ON_START=false` подключает готовые коллекции без повторного чтения, embedding и upsert; это покрыто регрессионными тестами.
12. Проверки: 12 RAG unit tests и полный набор `132 passed, 8 skipped`; Ruff по файлам инкремента и `docker compose config --quiet` пройдены.
