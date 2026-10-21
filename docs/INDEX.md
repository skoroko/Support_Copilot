# Техническая документация Support Copilot

Срез для сдачи — **21 октября 2026 года**. Пользовательская проверка Telegram
принята. Функциональная версия опубликована в `main` через PR53 / `c484959`;
source commit — `78daea7`, первоначальная база приёмки — PR51 / `df96941`.
[История публикации](publication_2026-10-21.md); точные идентификаторы
проверенного кода/образа/индекса находятся в протоколе. PR52 не включён.

## Порядок чтения

| Шаг | Материал | Содержание |
|---:|---|---|
| 1 | [README](../README.md) | Сценарий инженера, подготовка конфигурации, одна команда запуска, проверки |
| 2 | [Техническое описание](final_technical_documentation.md) | Компоненты, принятый профиль и семь решений: выбор → альтернативы → причина |
| 3 | [Покрытие всех 34 ДЗ](homework_coverage.md) | Обязательное/дополнительное, доказательства, даты, режимы и оставленные пробелы |
| 4 | [Финальный граф](final_agent_graph.md), [SVG](final-agent-graph.svg), [Mermaid](final-agent-graph.mmd) | Семь исполняемых узлов, retrieval-first, уточнение, approval и Подробнее |
| 5 | [Протокол приёмки](final_acceptance.md) | Tests, качество, источники, restart, ограничения и пользовательское решение |

Материалы выше самодостаточны внутри проекта: внешний workspace `tasks/`,
исходные задания `TEMP/` и private checkpoints для их чтения не требуются.
[Презентация из 11 слайдов](presentation/README.md) готова: PPTX, PDF и
заметки докладчика примерно на 8 минут. Демонстрация 5–7 минут и чистый
комплект остаются следующими задачами.

[Проверка документации S0-010](evidence/documentation_check_2026-10-21.json):
локальные пути во всех Markdown-документах `docs/` и README, 34 строки ДЗ, хэши доказательств,
байты исторического FSM и совпадение Mermaid/SVG. Внешние URL и доступность
будущего комплекта этой проверкой не подтверждаются.

[Расширенный реестр](evidence/diploma_document_audit_2026-10-21.json) включает
файлы `docs/`, учебные Markdown вне него и проверку их локальных ссылок.
Повтор проверки без сети: `.venv/Scripts/python.exe scripts/verify_diploma_documentation.py`.

## Полный комплект и объяснение реализации

- [Путеводитель сдачи](diploma_submission_guide.md): обязательное чтение,
  приложения внутри и вне `docs/`, история и ещё не готовые материалы.
- [Технологии](technology_stack.md): роль каждой библиотеки, runtime,
  опциональные возможности и учебные инструменты.
- [Архитектурные приёмы](architecture_patterns.md): путь запроса, provenance,
  поиск, состояние, подтверждения, ограничения и компромиссы.

## Текущая система

| Тема | Документ | Граница |
|---|---|---|
| Telegram | [telegram_bot.md](telegram_bot.md) | Команды, меню, renderer, feedback и согласие |
| Архитектура | [architecture.md](architecture.md), [unified_agent.md](unified_agent.md) | Текущий контур и контракты; нижние датированные разделы сохраняют историю |
| RAG | [rag.md](rag.md) | Ingestion, retrieval, metadata/provenance, эксплуатация |
| Корпус | [data_inventory.md](data_inventory.md) | 10 канонических источников, 60 исключённых зеркал, страницы/topics и пробелы |
| Общий контур evidence | [documentation_evidence.md](documentation_evidence.md) | Развёрнутые компоненты PR51 и отдельно незакрытые research gates S0-005 |
| Chat storage | [chat.md](chat.md) | Repository и API; graph adapter описан в unified_agent |
| Checkpoint / interrupt | [agent-persistent-report.md](agent-persistent-report.md) | Saver и учебный снимок HIL; текущий restart подтверждён в приёмке |
| Observability | [observability/README.md](observability/README.md) | Логи/метрики/tracing, optional Phoenix |
| Ограничения зависимостей | [dependency-risk-acceptance.md](dependency-risk-acceptance.md) | Известные риски и срок пересмотра; не новый полный audit |

## Проверяемые результаты 21.10

| Артефакт | Что подтверждает |
|---|---|
| [Ответы](evidence/diploma_2026-10-21/evaluated_answers.json), [сводка](evidence/diploma_2026-10-21/acceptance_summary.json), [rubric](evidence/diploma_2026-10-21/semantic_review.json) | 43 сценария + 5 expanded до узкой коррекции уточнения; 27 полных / 7 частичных из 34 |
| [Sources audit](evidence/diploma_2026-10-21/source_audit.json), [ручная сверка](evidence/diploma_2026-10-21/manual_source_review.json) | 154 использования источников; отдельно от смысловой полноты ответов |
| [Tests](evidence/diploma_2026-10-21/test_summary.json) | 1522 PASS / 12 skip; отдельно 25 PostgreSQL PASS |
| [Runtime](evidence/diploma_2026-10-21/post_restart_runtime_manifest.json), [code manifest](evidence/diploma_2026-10-21/final_code_manifest.json), [source match](evidence/diploma_2026-10-21/post_restart_source_identity.json) | Проверенный образ, настройки и совпадение исходников app/bot |
| [Индекс](evidence/diploma_2026-10-21/post_restart_index_audit.json), [restart](evidence/diploma_2026-10-21/restart_index_comparison.json), [история](evidence/diploma_2026-10-21/history_after.json) | Согласованность источников/docstore/Qdrant и сохранение 40 учебных чатов |

В исходном code manifest `pending` относится к моменту freeze до ручного
пользовательского прогона. Последующее решение зафиксировано в протоколе;
измерения не перезаписывались. [Пользовательская приёмка](evidence/diploma_2026-10-21/user_acceptance.json)
зафиксирована отдельно.

## Исторические эксперименты и обоснование выбора

| Тема | Датированные материалы | Как читать |
|---|---|---|
| Runtime Luna/reasoning | [eval 15.10](luna_runtime_evaluation.md), [rollout бота](luna_runtime_rollout.md), [rollout RAG](rag_luna_rollout.md) | 384 ответа; выбор medium, не новый score принятой версии |
| Подготовка и Fast | [19.10 latency](telegram_latency_fix_2026-10-19.md), [20.10 preparation](telegram_preparation_2026-10-20.md), [warmup](evidence_warmup_2026-10-20.md) | Принятые инфраструктурные улучшения и неудачные варианты |
| Быстрые облачные модели | [manual_model_comparison_2026-10-19.md](manual_model_comparison_2026-10-19.md), [JSON](manual_model_comparison_2026-10-19.json) | 20 вопросов × 2 повтора × 5 моделей, fixed evidence; runtime не менялся |
| Предыдущие Telegram-оценки | [15.10](telegram_agent_evaluation.md), [semantic 14.10](telegram_agent_evaluation_semantic_fix_2026-10-14.md), [citation baseline](telegram_agent_evaluation_citation_fix_2026-10-14.md), [первый baseline](telegram_agent_evaluation_baseline_2026-10-14.md) | Сохранены ошибки/неуспехи и прежние версии; актуальный протокол выше |
| Early cloud/local | [две cloud-модели](../homework_1_1/results/comparison_report.md), [local/cloud](../homework_1_4/results/benchmark_summary.json) | Учебные результаты и их незакрытые условия — в матрице ДЗ |
| Embeddings | [ДЗ 5.1](../homework_5_1/README.md) | Выбор large/1024; датированные цены, live comparison не заявлен |
| Qdrant | [vector_store.md](vector_store.md), [реальные фрагменты 5.2](../homework_5_2/REAL_INDEX_2026-10-21.md) | Учебные коллекции отделены от production |
| RAG 5.3 | [исторический отчёт](../homework_5_3/historical/2026-09-09_rag_6514f10.md) | Восстановленный снимок Git, приложение не откатано |
| Chunking | [ранний опыт](chunking_experiment.md), [исторический BGE](../homework_5_4/historical/2026-09-09_chunking_fb594ad.md), [RAGAS 12.09](rag_evaluation.md) | От offline 512/64 к large/1024; не текущая оценка агента |
| Citation judge | [5.6, 21.10](../homework_5_6/CITATION_JUDGE_2026-10-21.md) | 34 сохранённых ответа; false positive не скрыт |
| Учебные графы | [6.3 report](agent-graph-report.md), [45 попыток](../homework_6_3/BENCHMARK_2026-10-21.md), [custom](agent-graph-custom.mmd), [prebuilt](agent-graph-prebuilt.mmd) | 44 complete / 1 timeout, учебная KB и Telegram stub |
| Single / multi-agent | [отчёт](multi-agent-report.md), [схема эксперимента](architecture-multi-agent.md) | Пять mock-KB случаев × два варианта; обоснование single-agent |
| Ранняя архитектура | [CLI PDF](architecture-historical-cli.pdf) | Исторический учебный эскиз; не схема финальной системы |

Некоторые исторические отчёты ссылаются на private raw в ignored `var/` или
старые локальные UI. Такие ссылки не объявляются частью переносимого комплекта;
доказательства, необходимые проверяющему, указаны в матрице и текущем протоколе.
