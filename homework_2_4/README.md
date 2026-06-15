# Homework 2.4 - LLM cache

Цель: реализовать кеширование ответов LLM для Support Copilot.

## Что реализовано

- `LLMCache` — in-memory кеш с TTL по умолчанию 1 час.
- Ключ строится как SHA-256 от canonical JSON payload.
- Payload ключа учитывает `model`, `messages`, `temperature`,
  `prompt_version` и `service_facts_version`.
- Просроченные записи не возвращаются и удаляются при чтении.
- `stats()` возвращает `hits`, `misses`, `writes`, `entries`, `expired` и
  `hit_rate_percent`.
- `RedisLLMCache` — Redis backend с тем же интерфейсом, JSON-сериализацией и TTL
  через `SETEX`.
- `demo_cache.py` показывает miss, write, hit, miss при другой temperature,
  miss при другой prompt version и expiry по TTL.

## Запуск in-memory demo

Из родительской папки проекта:

```powershell
python .\HelpDesk_Assistant\homework_2_4\demo_cache.py
```

Из папки `HelpDesk_Assistant`:

```powershell
python .\homework_2_4\demo_cache.py
```

## Запуск Redis demo

Redis backend опционален. Нужен установленный пакет:

```powershell
python -m pip install redis
```

Запуск:

```powershell
python .\homework_2_4\demo_cache.py --redis-url redis://localhost:6379/0
```

## Результаты

Лог проверенного in-memory запуска находится в `results/cache_demo_log.txt`.
