# Комплект документов для сдачи диплома

Проверка и редакция — 21.10.2026. Пользовательская приёмка Telegram пройдена.
Функциональная версия — PR53 / `c484959`; последующая редакция документов
не меняет принятый код, модель, индекс или результаты измерений.
[Происхождение версии](publication_2026-10-21.md).

В репозитории каталог называется `docs/`. Ниже учтены и материалы вне него.
Документы описывают нашу реализацию; ссылки на чужие проекты и сравнительные
истории чужих систем не используются как обоснование архитектуры.

## Основное чтение для проверяющего

| Порядок | Документ | На какой вопрос отвечает |
|---:|---|---|
| 1 | [README](../README.md) | Для кого система, как пользоваться и запускать |
| 2 | [Финальное техническое описание](final_technical_documentation.md) | Что входит в принятую версию и почему выбраны её параметры |
| 3 | [Технологии](technology_stack.md) | Что именно делает каждая библиотека и хранилище |
| 4 | [Архитектурные приёмы](architecture_patterns.md) | Как устроены обработка запроса, поиск, состояние, контроль действий и сбои |
| 5 | [Граф агента](final_agent_graph.md), [SVG](final-agent-graph.svg), [исходник Mermaid](final-agent-graph.mmd) | Какие узлы реально исполняются и где требуется согласие |
| 6 | [Инвентаризация корпуса](data_inventory.md) | Какие документы индексируются и чего в базе нет |
| 7 | [Приёмка](final_acceptance.md) | Что проверено, какие результаты получены и какие ограничения остались |
| 8 | [Матрица всех 34 ДЗ](homework_coverage.md) | Где доказательство каждого учебного результата и какие пробелы сохранены |

[INDEX](INDEX.md) служит краткой навигацией. Указанные материалы читаются из
репозитория без внешнего workspace `tasks/`, учебного `TEMP/`, локального `var/`
и пользовательских переписок. Отдельные исторические raw-данные в `var/`
не объявляются частью переносимого комплекта.

## Подробные документы действующей системы

| Материал | Назначение |
|---|---|
| [architecture.md](architecture.md) | Текущий контур и отдельно помеченный исторический CLI-план |
| [unified_agent.md](unified_agent.md) | Chat adapter, checkpoint, ownership, approvals, очистка и расширение |
| [rag.md](rag.md) | Ingestion, поиск, metadata, эксплуатационные параметры |
| [documentation_evidence.md](documentation_evidence.md) | Структурный evidence и хронология исследований; незакрытые research gates указаны явно |
| [chat.md](chat.md) | Хранилище чатов, контекст и HTTP/SSE-контракт |
| [telegram_bot.md](telegram_bot.md) | Команды, renderer, источники, feedback и кнопки согласия |
| [observability/README.md](observability/README.md), [скриншот](observability/phoenix_trace.png) | Логи и опциональный tracing; скриншот относится к учебному запуску |
| [dependency-risk-acceptance.md](dependency-risk-acceptance.md) | Зафиксированные риски зависимостей; это не новый скан уязвимостей |

## Результаты и служебные документы

| Материал | Как использовать |
|---|---|
| [evidence/diploma_2026-10-21](evidence/diploma_2026-10-21/) | Замороженные ответы, rubric, источники, индекс, runtime, tests, restart и решение пользователя; читать по ссылкам протокола |
| [evidence manifest](evidence/diploma_2026-10-21/evidence_manifest.json) | Хэши и размеры 20 первоначальных файлов; последующая пользовательская приёмка сохранена отдельно |
| [проверка документов](evidence/documentation_check_2026-10-21.json) | Локальные пути, переносимость, матрица ДЗ, неизменность доказательств и графа; не оценка модели |
| [полный реестр проверки](evidence/diploma_document_audit_2026-10-21.json) | Учтённые файлы `docs/` и внешние учебные документы, охват проверки ссылок и её границы |
| [publication_2026-10-21.md](publication_2026-10-21.md) | Коммиты и происхождение принятой версии |
| [diploma_completion_plan_2026-10-21.md](diploma_completion_plan_2026-10-21.md) | Исходный план и последующие решения пользователя; старые пункты не являются новым заданием на все опыты |
| [telegram_delivery_acceptance.md](telegram_delivery_acceptance.md), [luna_speed_options.md](luna_speed_options.md) | Сценарии и варианты исследований; требования этих планов не выдаются за выполненные измерения |

Исходные JSON приёмки сохраняют состояние на момент замера: например,
`pending` до ручной проверки и ещё незакоммиченное дерево. Последующая
публикация и решение пользователя описаны отдельно; переписывать прошлые
измерения для придания им нового статуса нельзя.

Обе проверки повторяются локально без API-вызовов:

```powershell
.venv/Scripts/python.exe scripts/verify_diploma_documentation.py
```

Скрипт обновляет только два JSON-отчёта проверки документов. Хэши замороженных
данных сверяются с manifest; собственные хэши самих проверочных отчётов
не используются. Переходы по внешним URL и Markdown anchors не проверяются.

## Исторические учебные приложения внутри docs

Эти материалы нужны для накопительного характера проекта и объяснения выбора.
Их параметры, модели и метрики относятся к датированному опыту, а не к
новой проверке финального runtime.

| Группа | Файлы |
|---|---|
| CLI и ранняя архитектура | [architecture-historical-cli.pdf](architecture-historical-cli.pdf) |
| Первые агенты | [agent-react-report.md](agent-react-report.md), [traces README](agent-naive-traces/README.md), JSON в [agent-naive-traces](agent-naive-traces/), [agent-persistent-report.md](agent-persistent-report.md) |
| Графы | [agent-graph-report.md](agent-graph-report.md), [custom](agent-graph-custom.mmd), [prebuilt](agent-graph-prebuilt.mmd), JSON в [agent-comparison-results](agent-comparison-results/) |
| Single / multi-agent | [multi-agent-report.md](multi-agent-report.md), [architecture-multi-agent.md](architecture-multi-agent.md), [ответы](multi-agent-benchmark-answers.md) |
| Vector store и RAG | [vector_store.md](vector_store.md), [chunking_experiment.md](chunking_experiment.md), [rag_evaluation.md](rag_evaluation.md), [rag-context-audit.md](rag-context-audit.md) |
| Исторический FSM бота | [homework_4_2_history.md](homework_4_2_history.md) |
| Ранние Telegram-оценки | [baseline 14.10](telegram_agent_evaluation_baseline_2026-10-14.md), [citation fix](telegram_agent_evaluation_citation_fix_2026-10-14.md), [semantic fix](telegram_agent_evaluation_semantic_fix_2026-10-14.md), [15.10](telegram_agent_evaluation.md) |
| Выбор Luna | [runtime evaluation](luna_runtime_evaluation.md), [runtime rollout](luna_runtime_rollout.md), [RAG rollout](rag_luna_rollout.md), [Fast A/B](luna_fast_ab.md), [optimized A/B](optimized_model_ab.md) |
| Сравнение облачных моделей | [manual_model_comparison_2026-10-19.md](manual_model_comparison_2026-10-19.md), [JSON](manual_model_comparison_2026-10-19.json) |
| Контекст и подготовка evidence | [context_latency_optimization.md](context_latency_optimization.md), [overview_prefetch.md](overview_prefetch.md), [verified_overview.md](verified_overview.md), [live](verified_overview_live.md) |
| Ускорение 19–20.10 | [latency fix](telegram_latency_fix_2026-10-19.md), [JSON](telegram_latency_fix_2026-10-19.json), [preparation](telegram_preparation_2026-10-20.md), [JSON](telegram_preparation_2026-10-20.json), [warmup](evidence_warmup_2026-10-20.md) |
| Учёт корпуса 16.10 | [documentation_corpus_audit_2026-10-16.json](documentation_corpus_audit_2026-10-16.json) |
| Ранний security baseline | [до](security/garak_baseline_2026-07-07.md), [после](security/garak_after_2026-07-07.md), HTML-результаты в [security/reports](security/reports/) |
| Неиспользуемый вариант proxy | [litellm/config.yaml](litellm/config.yaml); не входит в принятый runtime |

Ссылки на собственные коммиты/PR фиксируют происхождение этого проекта.
Сохранённые официальные сведения о тарифах и security advisories в старых
отчётах остаются датированными справочными материалами. В новых архитектурных
главах внешних ссылок нет. Чужие реализации и их численные результаты из
архитектурного рассказа исключены.

## Материалы вне docs

| Расположение | Что нужно сохранить |
|---|---|
| `homework_*/README.md`, `TEACHER_CHECK.md` | Учебные инструкции и критерии проверки; вход через [матрицу](homework_coverage.md), а не отдельное чтение всех папок подряд |
| `homework_*/results/` | Сохранённые результаты конкретных заданий с указанием live/offline/mock |
| `homework_5_3/historical/`, `homework_5_4/historical/` | Восстановленные отчёты прежних версий без отката приложения |
| [ДЗ 5.2](../homework_5_2/REAL_INDEX_2026-10-21.md) | Проверка 100 реальных векторов в изолированных учебных коллекциях |
| [ДЗ 5.6](../homework_5_6/CITATION_JUDGE_2026-10-21.md) | Отдельная проверка цитат 34 сохранённых ответов, включая false positive |
| [ДЗ 6.3](../homework_6_3/BENCHMARK_2026-10-21.md) | 45 попыток учебных графов: 44 завершены и один timeout; это не оценка production RAG |
| `homework_4_2/historical/` | Исторический FSM и manifest экспорта из Git |
| [eval/README.md](../eval/README.md), [corpus suite](../tests/eval/corpus_suite/README.md), [review](../tests/eval/corpus_suite/review/README.md) | Наборы и правила eval; наличие runner не означает выполнение всех его метрик |
| `tests/eval/results/`, `eval/runs/`, [experiments/results.json](../experiments/results.json) | Машинные приложения и учебные сравнения по ссылкам соответствующих отчётов |
| [support_copilot/README.md](../support_copilot/README.md) | Ранний CLI; самостоятельный путь, не текущий Telegram-layer |
| [pyproject.toml](../pyproject.toml), [uv.lock](../uv.lock), [Compose](../compose.yaml), [Dockerfile](../Dockerfile) | Воспроизводимое окружение и запуск; конфигурационные имена описаны в README |
| [data](../data/) | Разрешённые мануалы и происхождение корпуса; состав — в инвентаризации |

`.env`, API-ключи, приватные checkpoints, пользовательские Telegram-истории,
игнорируемые локальные raw и дампы БД в комплект не включаются. `TEMP/` остаётся
неизменным учебным контекстом за пределами Git-проекта.

## Что остаётся до передачи диплома

Документационный этап завершает описание принятой реализации. Новые модели,
корпус и архитектурные улучшения для него не подключаются. Пропущенные по
решению пользователя опыты ДЗ 1–4 остаются видимыми в матрице; S0-005 открыта.

[Презентация из 11 слайдов](presentation/README.md) подготовлена:
[PPTX](presentation/Support_Copilot_Diploma_2026-10-21_v3.pptx),
[PDF](presentation/Support_Copilot_Diploma_2026-10-21_v3.pdf) и
[заметки на 8 минут](presentation/speaker_notes.md). Описаны задача,
архитектура, технологии, причины решений, измерения и ограничения принятой версии.

Ещё требуются демонстрация 5–7 минут с голосом и финальная упаковка с
проверкой запуска из чистой копии. Они пока не объявлены готовыми.
