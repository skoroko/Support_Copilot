# Homework 1.4 - model comparison and cost

Цель: провести полный цикл выбора модели для Support Copilot: eval-набор 20+ задач, benchmark 3+ моделей, расчет стоимости и рекомендация.

## Модели

Основной локальный кандидат: `qwen3:1.7b`.

Для выполнения требования ДЗ "2 локальные + 1 облачная" по умолчанию настроены:

- `qwen3:1.7b` - основной локальный кандидат для русского support-сценария;
- `gemma3:4b` - второй локальный кандидат для сравнения;
- `gpt-5.4-mini` через OpenAI API - облачный кандидат, модель задается через `OPENAI_MODEL`.

OpenAI model id и цены вынесены в `.env`, чтобы не хардкодить их в бизнес-логике. На 2026-05-25 официальная документация OpenAI рекомендует меньшие варианты вроде `gpt-5.4-mini`/`gpt-5.4-nano` для задач, где важны latency и cost; публичная цена для `gpt-5.4-mini`: input $0.75 / 1M tokens, cached input $0.075 / 1M tokens, output $4.50 / 1M tokens.

Источники проверки:

- https://developers.openai.com/api/docs/models
- https://openai.com/api/pricing/

## Подготовка окружения

Файлы уже подготовлены:

- `.env.example` - шаблон переменных окружения;
- `.env` - локальный файл для ключей, сейчас без секретов.

Перед облачным benchmark вставьте ключ:

```powershell
notepad .\.env
```

Минимально нужно заполнить:

```text
OPENAI_API_KEY=sk-...
```

Локальные модели:

```powershell
ollama pull qwen3:1.7b
ollama pull gemma3:4b
```

## Eval-набор

`eval_set.json` содержит 21 задачу:

- FAQ по установке, лицензированию, документации и тарифам;
- технические ошибки, логи, воспроизведение;
- жалобы и негативные обращения;
- явная просьба позвать специалиста;
- вопросы вне роли Support Copilot;
- prompt injection;
- повторный вопрос для проверки будущего кеша;
- кейсы, где ассистент должен признать нехватку данных.

Каждая задача содержит prompt, ожидаемые ключевые слова и запрещенные ключевые слова. Accuracy считается автоматически как доля найденных expected keywords с штрафом за forbidden keywords.

## Запуск benchmark

Только локальные модели:

```powershell
cd "C:\00.MIDAS\2026\DEV\PROJECTS\19. HelpDesk_Assistant\HelpDesk_Assistant"
python .\homework_1_4\benchmark_models.py
```

Локальные модели + OpenAI API:

```powershell
python .\homework_1_4\benchmark_models.py --include-cloud
```

Результаты сохраняются в:

- `homework_1_4/results/benchmark_results.json`
- `homework_1_4/results/benchmark_summary.json`

Метрики:

- `accuracy` - автоматическая оценка по проверочным функциям;
- `ttft_sec` - time to first token;
- `throughput_tok_sec` - скорость генерации;
- `total_sec` - полное время ответа.

## Расчет стоимости

```powershell
python .\homework_1_4\cost_calculator.py
```

Результат сохраняется в `homework_1_4/results/cost_report.json`.

Базовые assumptions в `.env`:

- 300 запросов в день;
- 900 input tokens на запрос;
- 350 output tokens на запрос;
- cache hit rate 35%;
- 30 дней в месяце.

Локальная модель в расчете API-токенов имеет $0 cloud-cost, но для итогового выбора нужно отдельно учитывать скорость, качество и стоимость владения ноутбуком/сервером. Для учебного сравнения cloud-cost считается только для OpenAI API.

## Рекомендация для отчета

Предварительная рекомендация до фактического прогона: выбрать `qwen3:1.7b` как основной локальный baseline для учебного CLI и `gpt-5.4-mini` как облачный кандидат для сложных обращений и проверки качества. Финальное решение нужно принять после benchmark на этой машине: если `qwen3:1.7b` дает достаточную accuracy и приемлемый TTFT, его стоит оставить для локального режима; если качество на prompt injection, эскалации и признании нехватки данных ниже облачной модели, облако использовать как strong fallback.
