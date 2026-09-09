# ДЗ 5.1 - эмбеддинги и семантический поиск

## Что реализовано

- `tests/eval/mini_benchmark.json` содержит 8 русскоязычных пар
  `query/relevant/irrelevant` из предметной области Support Copilot. Фрагменты
  основаны на локальной базе знаний и документации текущего проекта.
- `app/services/embeddings.py` предоставляет требуемый публичный интерфейс
  `embed_texts(texts: list[str]) -> list[list[float]]`.
- OpenAI-вызовы выполняются батчами по 100 текстов. Допустимый диапазон настройки
  - 32-2048.
- Каждый вектор L2-нормализуется и проверяется на ожидаемую размерность.
- Retry применяется только к 429, timeout, connection errors и HTTP 5xx.
  Ошибки 400/401/403 не повторяются.
- SQLite-кеш переживает перезапуск процесса. SHA-256 ключ включает provider,
  model, dimensions и исходный текст, поэтому смена модели или размерности не
  переиспользует несовместимый вектор.
- До запроса проверяется лимит 8192 токена на один input через `tiktoken`.
- `homework_5_1/benchmark.py` считает cosine score релевантного и нерелевантного
  документа, accuracy, mean margin, latency и число вызовов провайдера.

## Выбор модели

Рабочая модель: `text-embedding-3-large` с `dimensions=1024`.

Решение проверено по официальной документации OpenAI 2026-09-09:

| Критерий | `text-embedding-3-small` | `text-embedding-3-large` (выбор) |
|---|---:|---:|
| Описание | компактная embedding-модель | наиболее качественная модель, заявлена для English и non-English |
| Общий MTEB из OpenAI guide | 62.3% | 64.6% |
| Размерность по умолчанию | 1536 | 3072, сокращена API-параметром до 1024 |
| Цена, USD / 1M input tokens | 0.02 | 0.13 |
| Max input | 8192 tokens | 8192 tokens |

Почему выбран `large`: корпус и запросы преимущественно русскоязычные, поэтому
качество non-English retrieval важнее минимальной разницы в одноразовой стоимости
маленького корпуса. Размерность 1024 уменьшает хранение и вычисления vector DB; API
поддерживает сокращение через `dimensions`. Это не E5-модель, поэтому префиксы
`query:`/`passage:` и отдельные `embed_query`/`embed_documents` ей не требуются.

Источники:

- [OpenAI: text-embedding-3-large](https://developers.openai.com/api/docs/models/text-embedding-3-large)
- [OpenAI: text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small)
- [OpenAI: Vector embeddings guide](https://developers.openai.com/api/docs/guides/embeddings)
- [OpenAI: Batch API](https://developers.openai.com/api/docs/guides/batch)

## Стоимость индексации 50+ документов

Расчётная база: 60 документов по 20 страниц, 800 токенов на страницу, одна
страница равна одному chunk. Всего 1200 chunks и 960 000 input tokens.

| Модель | Синхронный API | Batch API (-50%) | Float32-векторы без metadata |
|---|---:|---:|---:|
| `text-embedding-3-small`, 1536d | $0.0192 | $0.0096 | 7.0312 MiB |
| `text-embedding-3-large`, 1024d | $0.1248 | $0.0624 | 4.6875 MiB |

Формула: `tokens / 1_000_000 * price`. Batch API поддерживает `/v1/embeddings`
и даёт скидку 50%, но текущий `embed_texts` использует синхронный endpoint, потому
что это online-сервис; Batch имеет смысл для отдельного офлайн-процесса индексации.
Расчёт воспроизводится командой:

```powershell
python homework_5_1/cost_estimate.py
```

Цены зафиксированы на дату проверки. Перед production-индексацией их нужно снова
сверить с официальной страницей OpenAI.

## Настройка

Скопируйте параметры из `.env.example`. Значения по умолчанию:

```dotenv
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=1024
EMBEDDING_BATCH_SIZE=100
EMBEDDING_CACHE_PATH=./var/embeddings.sqlite3
EMBEDDING_TIMEOUT_SECONDS=30
EMBEDDING_MAX_ATTEMPTS=3
EMBEDDING_RETRY_MIN_SECONDS=0.25
EMBEDDING_RETRY_MAX_SECONDS=4
EMBEDDING_MAX_INPUT_TOKENS=8192
```

Пример использования:

```python
from app.services.embeddings import embed_texts

vectors = embed_texts(["Не активируется лицензия", "Проверьте версию продукта"])
```

Если `.env` изменён в уже работающем процессе, вызовите
`reset_embedding_service()` перед следующим module-level вызовом. При обычном
рестарте новые настройки подхватятся автоматически.

## Проверка

Полностью офлайн:

```powershell
python -m pytest tests/unit/test_embeddings.py -q -p no:cacheprovider
python homework_5_1/benchmark.py --offline-smoke
python homework_5_1/benchmark.py --offline-smoke
```

Второй CLI-запуск 2026-09-09 получил 24/24 cache hits и сделал 0 provider-like
вызовов. Краткий лог находится в `results/cache_smoke.json`. Локальная hash-модель
нужна только для smoke-теста pipeline/cache; её accuracy нельзя использовать для
выбора semantic embedding-модели.

Реальный сравнительный прогон:

```powershell
python homework_5_1/benchmark.py `
  --candidate text-embedding-3-small:1536 `
  --candidate text-embedding-3-large:1024
```

Команда отправляет 24 учебных фрагмента во внешний OpenAI API. Запускайте её
только когда такой экспорт явно разрешён. В текущей автоматизированной сессии
реальный прогон не выполнялся: внешняя отправка локальных фрагментов не была
отдельно подтверждена.
