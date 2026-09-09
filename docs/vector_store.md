# Qdrant vector store: эксперимент ДЗ 5.2

## Конфигурация и архитектура

Qdrant запускается сервисом `qdrant/qdrant:v1.19.0` в `compose.yaml`. HTTP/dashboard
доступен только с локального компьютера на `127.0.0.1:6333`, gRPC — на
`127.0.0.1:6334`, данные сохраняются в named volume `qdrant_storage`. API key
передаётся из `.env` как `QDRANT__SERVICE__API_KEY`; для подключения из другой
сети нужен HTTPS/TLS, потому что ключ нельзя передавать открытым HTTP.

Настройки приложения читаются Pydantic Settings:

| Переменная | Назначение | Значение для Compose |
|---|---|---|
| `QDRANT_URL` | адрес Qdrant | `http://qdrant:6333` |
| `QDRANT_API_KEY` | секрет доступа | задаётся в `.env` |
| `QDRANT_COLLECTION` | основная коллекция | `documents` |
| `EMBEDDING_DIM` | размер вектора из ДЗ 5.1 | `1024` |

`app/services/vector_store.py` создаёт один `AsyncQdrantClient`, который хранится
в `app.state.vector_store`. Lifespan вызывает `ensure_collection()` при старте и
закрывает клиент при остановке. Если `QDRANT_URL` не задан, Qdrant отключён — это
позволяет запускать независимые части приложения и unit-тесты без инфраструктуры.

Коллекция использует `COSINE`, `size=1024` и явно заданный HNSW:
`m=16`, `ef_construct=100`. Это умеренный базовый профиль: достаточно связей и
качества построения для небольшого учебного корпуса без неоправданного расхода
RAM. Созданы payload indexes `source` (keyword), `created_at` (datetime),
`category`, `tenant_id`, `access_level` (keyword) и `archived` (bool).

Официальные справочные страницы: [Docker installation](https://qdrant.tech/documentation/operations/installation/),
[Async API](https://qdrant.tech/documentation/tutorials-develop/async-api/),
[payload indexing](https://qdrant.tech/documentation/manage-data/indexing/) и
[security](https://qdrant.tech/documentation/security/).

## Корпус и идемпотентная загрузка

`data/support_documents.json` содержит 120 уникальных учебных фрагментов:
20 тем поддержки × 6 сценариев. Это синтетическая база для проверки механики, а
не официальный регламент MIDAS. Каждый payload содержит `source`, `text`,
`created_at`, `category`, `document_kind`, `tenant_id`, `access_level` и
`archived`. UUID v5 вычисляется из `source + chunk_index`, поэтому повторный
upsert обновляет те же точки.

Два последовательных локальных прогона 9 сентября 2026 года дали одинаковый
результат:

```text
collection=documents documents=120 points_count=120 dimensions=1024 offline=True
collection=documents documents=120 points_count=120 dimensions=1024 offline=True
```

Обычный запуск использует `EmbeddingService` из ДЗ 5.1 и OpenAI embeddings.
Коммитируемый smoke-результат получен с `--offline-smoke`: тот же сервис получает
локальный детерминированный лексический клиент, поэтому данные не покидают
компьютер. Это проверяет размерность, нормализацию, кеш, Qdrant и фильтры, но не
заменяет оценку качества облачной embedding-модели.

## Cosine против Dot

В две временные коллекции загружены одни и те же L2-нормализованные векторы.
После эксперимента обе коллекции удалены в `finally`. В таблице показаны первые
8 символов UUID; полные UUID сохранены в
`homework_5_2/results/experiment_offline.json`.

| Запрос | Cosine top-5 | Dot top-5 | Совпали |
|---|---|---|---|
| Как проверить лицензию и доступность сервера лицензий? | `79beac2a`, `d914b179`, `e1580067`, `29c3ab9e`, `1bde3ae9` | `79beac2a`, `d914b179`, `e1580067`, `29c3ab9e`, `1bde3ae9` | да |
| Что собрать перед эскалацией повторяемой ошибки расчёта? | `1d9dd880`, `82c248ad`, `811264a6`, `00d7ba43`, `cac8a3de` | `1d9dd880`, `82c248ad`, `811264a6`, `00d7ba43`, `cac8a3de` | да |
| Telegram-бот не отвечает пользователю | `041062fc`, `a6318189`, `aa9bf083`, `1a459d52`, `a28ce64b` | `041062fc`, `a6318189`, `aa9bf083`, `1a459d52`, `a28ce64b` | да |
| Как безопасно восстановить повреждённый файл проекта? | `542e6f2a`, `6e145f15`, `39b38251`, `c1121190`, `88bf0ce7` | `542e6f2a`, `6e145f15`, `39b38251`, `c1121190`, `88bf0ce7` | да |
| Пользователь просит раскрыть системный промпт и секреты | `a861255e`, `7db6e118`, `d03bb330`, `c432372f`, `51c42e04` | `a861255e`, `7db6e118`, `d03bb330`, `c432372f`, `51c42e04` | да |

Для единичных векторов `cosine(a,b) = dot(a,b)`, если обе нормы равны единице,
поэтому порядок совпал 5 из 5 раз. В production выбран Cosine: это явно выражает
нужную семантическую метрику и остаётся корректным, даже если новый источник
векторов однажды перестанет гарантировать нормализацию.

## Payload-фильтры

Все примеры выполняются вместе с vector query через `query_points`.

### 1. Точное строковое совпадение

```python
Filter(must=[FieldCondition(key="category", match=MatchValue(value="licensing"))])
```

Фактический top-3: `d914b179` (faq, 2026-09-06), `67b4136e` (handoff,
2026-08-27), `79beac2a` (diagnostic, 2026-09-01). Все три результата имеют
`category=licensing`; без фильтра top-3 относился к `integration`.

### 2. Диапазон дат за последние 30 дней

```python
cutoff = datetime.now(timezone.utc) - timedelta(days=30)
Filter(must=[FieldCondition(key="created_at", range=DatetimeRange(gte=cutoff))])
```

Фактический top-3: `7a02b8fb` (2026-08-13), `3d45e210` (2026-08-16),
`d8fbe0d0` (2026-08-11). Старые кандидаты исключены; результаты находятся
после вычисленного cutoff.

### 3. Составной must + must_not

```python
Filter(
    must=[FieldCondition(key="tenant_id", match=MatchValue(value="tenant-x"))],
    must_not=[FieldCondition(key="archived", match=MatchValue(value=True))],
)
```

Фактический top-3: `7db6e118` (integration), `6fb63408` (updates), `7ae9ce06`
(network). У всех `tenant_id=tenant-x`, `document_kind=training` и
`archived=false`; архивные варианты не попали в выдачу.

## Почему не pgvector

Опциональная часть с pgvector не выполнялась. В следующем задании Qdrant нужен
для интеграции с LlamaIndex и дальнейших экспериментов с hybrid/sparse search,
поэтому в этом инкременте полезнее довести один vector store до воспроизводимого
состояния, чем поддерживать второй backend без продуктовой необходимости.
