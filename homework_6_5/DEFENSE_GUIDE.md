# HW 6.5 — сценарий защиты

## Тезис решения

В этой постановке multi-agent не оправдал добавленную сложность. На пяти
одинаковых support-кейсах single-agent получил качество 5 против 3 у
multi-agent, использовал в среднем 1100.2 против 1470.2 токенов, а p50 latency
составила 2248.48 против 3987.97 мс. Поэтому текущий production agent-layer
остаётся single-agent; multi-agent код изолирован в `experiments/` и не
переносится в `app/agents/`.

## План показа на 5–7 минут

| Время | Что показать | Что сказать |
|---|---|---|
| 0:00–0:40 | `homework_6_5/README.md` и цель работы | Сравниваются один агент и ручной multi-agent workflow на пяти одинаковых вопросах поддержки MIDAS. Цель — принять решение по данным, а не добавить агентов ради архитектуры. |
| 0:40–1:25 | `docs/architecture-multi-agent.md` | Путь графа: `START -> supervisor -> researcher -> supervisor -> writer -> END`. Researcher работает с KB tool, writer формирует ответ без tools, supervisor детерминированно передаёт управление. |
| 1:25–2:15 | `experiments/shared.py` и контракт в `experiments/results.json` | Обе реализации используют один mock `search_knowledge_base`, один набор вопросов и одну модель. Mock-корпус делает результат воспроизводимым и не смешивает учебное сравнение с production RAG. |
| 2:15–3:10 | `docs/multi-agent-benchmark-answers.md` | Показать aggregate: single 1100.2 tokens / 2 calls / p50 2248.48 ms / quality 5; multi 1470.2 / 3 / 3987.97 ms / quality 3 / 3 handoffs. |
| 3:10–4:00 | В таблице кейсы 01, 04 и 05 | В 01 и 04 multi ошибочно выдал out-of-base fallback, хотя single нашёл релевантные документы. Причину не придумываем: judge reasons не сохранены. В 05 оба корректно эскалировали неизвестный вопрос. |
| 4:00–4:50 | Фрагменты `multi_agent_langgraph.py` | `InMemorySaver` хранит checkpoint состояния в памяти процесса, `app.stream(..., stream_mode="updates")` показывает обновления узлов, а три handoff заданы явным budget. Скрытого LLM routing call у manual supervisor нет. |
| 4:50–5:35 | Метрики и judge в `results.json` | `total_tokens` суммирует usage всех AIMessage, включая turns с tool call; `llm_calls` считает эти сообщения; latency измеряется от входа до final state. Один batched LLM judge оценивает ответы по шкале 1–5 и не входит в agent metrics. |
| 5:35–6:20 | Offline-проверки ниже | Показать, что тесты, компиляция и Mermaid-архитектура воспроизводятся без внешнего API. Benchmark повторно не запускать. |
| 6:20–7:00 | Итог и вопросы | Вывод: для общего короткого контекста и последовательной зависимости single-agent надёжнее и дешевле. Multi-agent имеет смысл проверять отдельно на задачах с независимыми источниками или параллельным research. |

## Безопасные offline-команды

Команды ниже не используют `--allow-external-api` и не выполняют внешний
benchmark. Не выводите `.env`; для проверки наличия конфигурации используйте
только безопасный статусный скрипт.

```powershell
.\.venv-eval\Scripts\python.exe -B scripts\check_env_safety.py

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.\.venv-eval\Scripts\python.exe -B -m pytest -q -p pytest_asyncio.plugin tests\test_multi_agent_experiment.py
.\.venv-eval\Scripts\python.exe -B -m py_compile experiments\shared.py experiments\single_agent_baseline.py experiments\multi_agent_langgraph.py
.\.venv-eval\Scripts\python.exe -B -m experiments.multi_agent_langgraph --write-architecture
```

Тесты используют `FakeMessagesListChatModel` и не обращаются к OpenAI, Qdrant,
Redis или сети. `--write-architecture` только заново строит Mermaid-граф; он не
вызывает модель. Для показа фактических результатов достаточно открыть уже
сохранённые `experiments/results.json` и документы benchmark/report.

## Как объяснить честность сравнения

- **Same cases:** обе реализации получили ровно пять одинаковых вопросов:
  три corpus-кейса, один многошаговый и один out-of-base.
- **Same tool:** обе использовали один детерминированный
  `search_knowledge_base`; у writer multi-agent tools нет, но researcher
  получает тот же tool, что и baseline.
- **Same model:** запуск использовал одну и ту же модель из конфигурации
  (`gpt-5.4-mini` записана в результатах).
- **Same contract:** ответы должны быть короткими, на русском, с проверяемыми
  ссылками и честной эскалацией при отсутствии данных.
- **Separate judge:** один batched LLM judge оценил все ответы; его usage
  сохранён отдельно и не улучшает/не ухудшает token и call метрики агентов.

## Ключевые детали реализации

**Manual supervisor.** Использован явный `StateGraph`/`Command`, а не скрытый
router. Цепочка и budget из трёх передач видны в коде и в streaming updates:
`supervisor -> researcher -> supervisor -> writer`. Это даёт наблюдаемое
управление, но не создаёт отдельный model call для маршрутизации.

**InMemorySaver.** Это in-process checkpointer LangGraph: он сохраняет state
графа для текущего запуска и `thread_id`, но не является production persistent
history или Redis-хранилищем.

**Streaming.** `app.stream(..., stream_mode="updates")` выдаёт обновления
узлов по мере прохождения графа. В консоль выводятся имена узлов, а не полные
промпты или ответы, поэтому прогресс виден без лишнего вывода данных.

**Metrics.** Токены и model calls считаются по каждому `AIMessage`, включая
модельные turns с tool call. Latency — end-to-end время до final state;
`handoff_count` — отдельная метрика переходов supervisor/worker. p50 посчитан
по пяти запускам каждой реализации.

**LLM judge.** Judge получил батч ответов и вернул score 1–5. Это полезнее
простого сравнения длины, но остаётся оценкой модели: в JSON сохранены scores,
а причины оценок не сохранялись, поэтому по ним нельзя доказывать конкретную
причину провала.

**Anthropic как ориентир.** В указанном в отчёте исследовании Anthropic
multi-agent Research превзошёл single-agent на 90.2% во внутреннем breadth-first
research eval, но расходовал примерно в 15 раз больше токенов. Это не прямой
benchmark для нашего сценария: здесь один общий KB tool, короткий общий
контекст и последовательный workflow; наш multiplier — `1470.2 / 1100.2 =
1.336x`. Ориентир полезен как аргумент проверять multi-agent только там, где
есть независимые источники, специализация или параллелизм.

**Почему не создан `app/agents/`.** Фактический eval не показал выигрыша, а
multi-agent ухудшил качество на двух corpus-кейсах. Поэтому production-код не
изменён: сначала выбран минимальный single-agent слой, а multi-agent оставлен
как изолированный эксперимент для нового, отдельно сформулированного eval.

## Ожидаемые вопросы преподавателя

### 1. Зачем вообще сравнивать multi-agent, если single-agent проще?

Чтобы проверить гипотезу измерением. В задачах с независимым research несколько
специалистов могут дать выигрыш, но в этом общем коротком контексте данные
показали обратное.

### 2. Сравнение действительно честное?

Да: пять одинаковых вопросов, один mock tool, одна модель и один output
contract. Отличается только orchestration: baseline совмещает роли, multi
разделяет researcher и writer.

### 3. Почему использован mock tool, а не production RAG?

Чтобы не смешивать качество orchestration с сетью, Qdrant и изменениями
retrieval. Для учебного вопроса нужен воспроизводимый общий источник; новый
RAG-eval должен быть отдельным экспериментом.

### 4. Почему supervisor ручной, а не отдельный LLM-агент?

Manual supervisor делает переходы и handoff budget детерминированными и видимыми.
Иначе routing model call смешался бы с измерением и добавил скрытую стоимость.

### 5. Что означают три handoff и три LLM calls?

Три handoff — переходы `supervisor -> researcher`, `researcher -> supervisor`
и `supervisor -> writer`. Три LLM calls — model turns multi-agent; это разные
метрики. Supervisor не является скрытым четвёртым LLM call.

### 6. Почему multi провалил кейсы 01 и 04?

Наблюдается ошибочный out-of-base fallback: multi написал, что подтверждённых
сведений нет, хотя single нашёл релевантные документы. В данных нет сохранённой
причины маршрутизации, поэтому точное объяснение утверждать нельзя.

### 7. Можно ли доверять оценке LLM judge?

Это единая сопоставимая оценка по шкале 1–5, а не абсолютная истина. Поэтому
она рассматривается вместе с текстами ответов, sources, tokens и latency; judge
usage вынесен за пределы agent metrics.

### 8. Почему результат Anthropic не означает, что нужно выбрать multi-agent?

Их результат относится к другому breadth-first research eval и другой стоимости.
Наши пять tight-coupled support-кейсов не имеют независимых веток research;
решение принимается по собственному benchmark, а не по чужому проценту.
