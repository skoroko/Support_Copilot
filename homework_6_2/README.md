# Домашнее задание 6.2 — ReAct-агент с критиком

Инкремент добавляет отдельный ReAct-агент поверх накопительного Support
Copilot. Он не меняет наивный baseline ДЗ 6.1: бизнес-обработчики переиспользуются
как явный allowlist, а новые строгие Chat Completions schemas находятся в
`app/services/agent_react_tools.py`.

## Реализовано

- `run_react_agent(task, max_iterations=10, timeout_per_iteration_sec=10)`;
  допустимые диапазоны — 8–20 итераций и 5–15 секунд соответственно;
- native Chat Completions с `tools`, `tool_choice="auto"` и
  `parallel_tool_calls=False`; несколько tool calls в одном ответе безопасно
  отклоняются;
- строгие JSON Schema (`strict: true`, `additionalProperties: false`) и
  локальная runtime-проверка JSON, обязательных и лишних аргументов;
- отдельный critic после каждого tool observation: `OK` либо `REVISE: причина`,
  максимум две правки за запуск;
- единый дедлайн на main LLM, tool и critic. Медленный local tool выполняется
  в завершаемом дочернем процессе, поэтому timeout не оставляет его работать в
  фоне;
- учёт всех main и critic tokens в `usage`, структурированные логи каждого
  шага через существующий `structlog` adapter;
- при `--premium-after-revision` следующий main-вызов после замечания критика
  переключается с `gpt-5.4-mini` на `gpt-5.4`; по умолчанию этой эскалации нет.

`OPENAI_MODEL` продолжает задавать основную модель; если переменная пуста,
используется `gpt-5.4-mini`. Поддержка Chat Completions/function calling для
выбранной модели сверялась по официальным страницам OpenAI до реализации.

## Офлайн-проверка

```powershell
$env:PHOENIX_WORKING_DIR="$PWD\var\phoenix-test"
.venv-eval\Scripts\python.exe -m pytest -q tests/test_agent_react.py
python -m app.services.agent_react --help
```

## Один внешний прогон

Команда ниже вызывает провайдера OpenAI. Локальный Telegram tool остаётся
заглушкой и не создаёт сетевого сообщения.

```powershell
python -m app.services.agent_react "Сколько времени в Москве?" --trace
```

## Сравнение с baseline

Ровно пять предметных кейсов (два простых, два составных, один
провокационный без tools) запускаются только после явного разрешения внешнего
API:

```powershell
python scripts/run_agent_comparison.py --allow-external-api
```

Для повторного запуска одного кейса без лишних API-вызовов:

```powershell
python scripts/run_agent_comparison.py --allow-external-api --case-id 04_compositional_confirmed_stub
```

JSON-результаты live-прогона записаны в `docs/agent-comparison-results/`, а
фактические итерации, tokens, revisions и ручная оценка корректности перенесены
в [`docs/agent-react-report.md`](../docs/agent-react-report.md).

## Безопасная диагностика `.env`

Содержимое `.env` нельзя выводить или искать терминальными командами. Проверка
наличия настроек выполняется только статусным скриптом, который никогда не
печатает значения:

```powershell
python scripts/check_env_safety.py
```
