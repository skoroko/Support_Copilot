# ДЗ 2.4: кеширование

Основной класс: `support_copilot/infrastructure/cache.py::LLMCache`.

Покрыто:

- ключ SHA-256 от `model`, `temperature`, `messages`;
- TTL, по умолчанию 1 час;
- `hits`, `misses`, `hit_rate_percent`;
- метод `stats()`;
- optional `RedisLLMCache` с тем же интерфейсом.

Демо:

```powershell
python homework_2_4/cache_demo.py --log-file homework_2_4/cache_demo_log.txt
```
