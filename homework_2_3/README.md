# ДЗ 2.3: надёжность

Основной класс: `support_copilot/infrastructure/llm.py::RobustLLMClient`.

Покрыто:

- retry для 429, 5xx и timeout;
- exponential backoff `1s -> 2s -> 4s -> 8s -> 16s` + jitter;
- до 5 попыток;
- fallback-цепочка OpenAI -> OpenRouter -> сообщение о недоступности;
- логирование timestamp, provider, попытки, кода ошибки;
- usage tracking и расчет стоимости.

```powershell
python homework_2_3/robust_client_demo.py --log-file homework_2_3/retry_fallback_log.txt
```

Скрипт использует реальные провайдеры из `.env`. Если `OPENROUTER_API_KEY` и
`OPENROUTER_MODEL` пустые, fallback-провайдер будет пропущен.
