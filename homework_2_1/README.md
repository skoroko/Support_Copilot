# ДЗ 2.1: API провайдеров

Скрипт: `api_providers_demo.py`.

Покрыто:

- OpenAI SDK;
- ключ из `.env`;
- system/user messages;
- temperature experiment: `0`, `0.7`, `1.5`;
- streaming с TTFT и total time;
- переключение провайдера через `LLM_PROVIDER`, `base_url`, `model`.

```powershell
python homework_2_1/api_providers_demo.py --provider openai --log-file homework_2_1/demo_log.txt
```

OpenRouter используется только если заполнены `OPENROUTER_API_KEY` и
`OPENROUTER_MODEL`:

```powershell
python homework_2_1/api_providers_demo.py --provider openrouter
```
