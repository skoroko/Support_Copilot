# Покрытие домашних заданий Support Copilot

Сверка для сдачи: 21.10.2026. Всего **34 ДЗ**: модуль 1 — 5, модуль 2 — 6,
модуль 3 — 8, модуль 4 — 4, модуль 5 — 6, модуль 6 — 5.
Требования повторно сверены с исходными заданиями курса; ниже их краткое
изложение и ссылки на доказательства внутри проекта. Оригинальные задания
и примеры преподавателя для чтения этой таблицы не требуются.

Пользователь принял текущую Telegram-версию и разрешил оформление диплома.
При этом он сознательно исключил дозакрытие недостающих опытов ДЗ1–4.
Эти пробелы сохранены; наличие строки или кода не означает полного выполнения
всех условий. Доказательства ДЗ5–6 дополнены согласованными опытами без новых
моделей/зависимостей. Текущая техническая приёмка: [протокол](final_acceptance.md).

**Режимы:** live — реально выполненный внешний/контейнерный опыт;
offline — тесты/заглушки без внешнего LLM; исторический — результат прежней
версии; подготовлено — есть код, но требуемый опыт не подтверждён.
Дата «не установлена» относится к самому опыту, а не к дате этой сверки.
Дополнения отделены от обязательной части, где это существенно.

| ДЗ | Обязательная часть и дополнения | Доказательства | Дата и режим | Фактический статус / предел |
|---|---|---|---|---|
| 1.1 | Две модели, пять support-запросов, качество/скорость/полнота | [Сравнение](../homework_1_1/results/comparison_report.md) | Дата не установлена; исторический live | Результат есть; не оценка текущего агента. |
| 1.2 | Нагрузка, токены, cloud/local TCO, break-even | [Экономика](../homework_1_2/README.md) | Цены 16.06.2026; исторический расчёт | Расчёт есть; старые тарифы не текущий billing. |
| 1.3 | Локальные модели, benchmark, Modelfile, structured classification | [Benchmark](../homework_1_3/results/benchmark_summary.md), [Modelfile](../homework_1_3/Modelfile), [указатель](../homework_1_3/TEACHER_CHECK.md) | 16.06.2026; исторический локальный live | Три модели/JSON и неуспехи сохранены. Требуемый снимок результатов не подтверждён; пробел не дозакрывается. |
| 1.4 | 20+ задач; **две локальные + одна облачная**; TTFT/throughput/accuracy/cost | [Набор](../homework_1_4/eval_set.json), [сводка](../homework_1_4/results/benchmark_summary.json) | Дата не установлена; исторический live / подготовлено | Есть 21×2, а не 21×3. Второй локальный прогон/полный итог не подтверждены; октябрьское cloud-сравнение их не заменяет. |
| 1.5 | HF Hub, модели прикладной задачи против cloud, доступное демо | [Benchmark](../homework_1_5/results/benchmark_report.md), [Gradio](../homework_1_5/gradio_demo.py) | Дата не установлена; исторический live / локальное демо | Три HF-модели и cloud baseline есть. Внешняя доступная демо-ссылка не подтверждена. |
| 2.1 | SDK, system/user, env, temperature-опыт, streaming, смена endpoint | [Описание](../homework_2_1/README.md), [runner](../homework_2_1/provider_api_demo.py) | Дата не установлена; подготовлено | Код есть; temperature-наблюдения предварительные, сохранённого реального опыта нет. |
| 2.2 | РРФО, few-shot, injection/tests, токены, сравнение с/без few-shot; Jinja2 — дополнение | [Описание](../homework_2_2/README.md), [offline-лог](../homework_2_2/results/test_run_log.txt) | Дата не установлена; offline | Prompt/контракт/tests есть; настоящий LLM-опыт с/без few-shot не подтверждён. |
| 2.3 | Retry временных ошибок, fallback, logging, usage/cost | [Лог](../homework_2_3/results/retry_fallback_log.txt), [описание](../homework_2_3/README.md) | 15.06.2026; исторический offline | Fault simulation подтверждён; это не реальные сбои cloud. |
| 2.4 | SHA-256 key, TTL, hit/miss/stats; Redis — дополнение | [Лог](../homework_2_4/results/cache_demo_log.txt), [описание](../homework_2_4/README.md) | Дата не установлена; исторический offline | Базовый кеш и демонстрация есть. Redis реализован отдельно. |
| 2.5 | Общий CLI, clear/stats/quit, история, кеш, классификация, эскалация | [Сессия](../homework_2_5/results/session_demo_log.txt), [CLI](../support_copilot/cli.py) | 15.06.2026; исторический offline/demo | Инкремент есть; provider errors в логе смоделированы. |
| 2.6 | Один мультимодальный путь; выбран vision с тремя типами изображений, оба пути не обязательны | [Описание](../homework_2_6/README.md), [лог](../homework_2_6/results/vision_demo_log.txt), [код](../support_copilot/core/vision.py) | Лог 15.06.2026; исторический пример | Vision есть, сохранён один пример. Три типа изображений не подтверждены. |
| 3.1 | Полный tool loop; нужен tool / не нужен / пограничный случай | [Сводка](../homework_3_1/results/demo_summary.md), [traces](../homework_3_1/results/tool_call_demo.jsonl) | Дата не установлена; исторический live | Три сценария есть. Учебная KB не является регламентом MIDAS. |
| 3.2 | Архитектура/ADR, критические latency/cost участки, короткий LiteLLM-опыт | [Архитектура](architecture.md), [LiteLLM config](litellm/config.yaml) | Дата не установлена; проектирование / подготовлено | Архитектура есть; наблюдения LiteLLM-опыта и часть пометок не подтверждены. Production proxy не требуется. |
| 3.3 | Async client, bounded concurrency, batch, SSE, sync/async benchmark | [Указатель](../homework_3_3/TEACHER_CHECK.md), [benchmark](../scripts/benchmark_results.md), [клиент](../support_copilot/infrastructure/async_llm.py) | Дата не установлена; исторический offline/demo | Реализация/benchmark есть; ускорение demo-provider не является throughput cloud API. |
| 3.4 | FastAPI/Pydantic/DI/lifespan, chat/stream/health/models, ошибки, кеш | [Указатель](../homework_3_4/TEACHER_CHECK.md), [API tests](../tests/test_app_fastapi.py), [service tests](../tests/test_app_llm_service.py) | Историческая дата не установлена; 21.10 — offline regression | Код/проверки входят в финальный набор. |
| 3.5 | Multi-stage/non-root Docker, Compose+Redis, health/readiness, сборка и численные критерии раннего image | [Dockerfile](../Dockerfile), [Compose](../compose.yaml), [указатель](../homework_3_5/TEACHER_CHECK.md), [приёмка](final_acceptance.md) | 21.10.2026; live Docker / локальная сборка | Текущий стек подтверждён. Численные критерии раннего учебного образа не доказаны поздним расширенным image. |
| 3.6 | Phoenix/OTel, request-id JSON logs, usage/latency, PII redaction | [Screenshot](observability/phoenix_trace.png), [указатель](../homework_3_6/TEACHER_CHECK.md), [код](../app/observability/) | Дата trace не установлена; историческое evidence + offline tests | Реализация/screenshot есть. Phoenix запускается optional-профилем. |
| 3.7 | Golden 20+, unit tests, judge relevance/correctness/completeness 1–5, threshold gate | [Указатель](../homework_3_7/TEACHER_CHECK.md), [smoke](../eval/runs/2026-07-07.json), [приёмка](final_acceptance.md) | 07.07 — offline smoke; 21.10 — отдельная live/manual оценка | Dataset/gate есть. Поздняя rubric-оценка не подтверждает требуемый трёхметричный judge-прогон 1–5. |
| 3.8 | Input/output защита, garak baseline/after, 3–5 input→actual-output примеров | [Baseline](security/garak_baseline_2026-07-07.md), [after](security/garak_after_2026-07-07.md), [указатель](../homework_3_8/TEACHER_CHECK.md) | 07.07.2026; исторический live security experiment | Прогоны/остаточный encoding-риск сохранены. Безопасная подборка 3–5 actual-output примеров не оформлена. |
| 4.1 | Repository, JSON+PostgreSQL, миграции, история/контекст, точность token estimation | [Описание](../homework_4_1/README.md), [контракт](../tests/chat/test_repository_contract.py), [приёмка](final_acceptance.md) | 21.10.2026; offline + отдельная live PostgreSQL | 25 contract tests прошли; persistence/изоляция проверены. Сравнение token-count с provider usage не подтверждено. |
| 4.2 | Telegram thin client, команды, backend, streaming, учебный `/ask` FSM | [Описание](../homework_4_2/README.md), [бот](telegram_bot.md), [история FSM](homework_4_2_history.md), [приёмка](final_acceptance.md) | Исторические Git `1432e99`/`b068dc7`; текущая приёмка 21.10 | FSM выполнен исторически, затем заменён свободным вводом. `/ask` не является текущей командой; восстанавливать его не требуется. |
| 4.3 | Media в chat endpoint, история/streaming/notify; достаточно одного media-сценария | [Указатель](../homework_4_3/TEACHER_CHECK.md), [код](../app/chat/media.py), [приёмка](final_acceptance.md) | Историческая дата не установлена; 21.10 — offline media checks | Реализация/tests есть. Новый live media input→output не заявлен; raw images не сохраняются в checkpoint. |
| 4.4 | Moderation, admin auth, feedback/dedup, broadcast, persistence; внешний Moderation API — дополнение | [Указатель](../homework_4_4/TEACHER_CHECK.md), [moderation](../app/moderation/), [admin](../app/admin/) | Историческая дата не установлена; 21.10 — offline regression | Код/tests есть. Реальная массовая рассылка ради проверки не выполнялась. |
| 5.1 | Предметные тройки, embeddings, batching/L2/retry/cache, экономика; live benchmark отдельно от smoke | [8 троек](../tests/eval/mini_benchmark.json), [cache smoke](../homework_5_1/results/cache_smoke.json), [описание](../homework_5_1/README.md) | Дата smoke не установлена; offline / подготовлено | Реализация/экономика/smoke есть. Сравнение embedding-моделей подготовлено, не выдано за выполненное live. |
| 5.2 | Qdrant, 100+ реальных фрагментов, upsert/idempotence, distance, HNSW/payload, фильтры | [Опыт](../homework_5_2/REAL_INDEX_2026-10-21.md), [метрики](../homework_5_2/results/real_index_2026-10-21.json) | 21.10.2026; live, отдельные коллекции | 100 реальных фрагментов, Cosine/Dot, повтор и три фильтра подтверждены; production 14139→14139. Это не оценка retrieval quality. |
| 5.3 | LlamaIndex и bare-metal RAG на одинаковых вопросах, источники/измерения/вывод | [Исторический отчёт](../homework_5_3/historical/2026-09-09_rag_6514f10.md), [raw](../homework_5_3/results/rag_evaluation.json) | Git 09.09.2026, `6514f10`; исторический live | Сравнение восстановлено; старый профиль не выдан за текущий агент. |
| 5.4 | Dataset, три chunking-стратегии, четыре настройки, HR/MRR/Recall, reranker before/after | [Исторический отчёт](../homework_5_4/historical/2026-09-09_chunking_fb594ad.md), [JSON](../homework_5_4/historical/2026-09-09_chunking_fb594ad.json) | 09.09.2026; исторический offline/локальный опыт | 28 кейсов и BGE before/after сохранены. Lexical-hash/BGE не являются текущим runtime. |
| 5.5 | Реальный разнородный корпус, ingestion, provenance, идемпотентность, проверяемый индекс | [Инвентаризация](data_inventory.md), [код](../app/services/ingestion.py), [приёмка](final_acceptance.md) | 21.10.2026; live read-only audit + offline tests | 10 канонических источников и 60 HTML-зеркал; catalog/docstore/Qdrant согласованы. Визуальные/формульные пробелы остаются. |
| 5.6 | Golden 30+, RAGAS, три A/B, thresholds/худшие случаи/traces, citation через `llm.agenerate`; порог допускает разбор недостижения | [RAGAS](rag_evaluation.md), [citation judge](../homework_5_6/CITATION_JUDGE_2026-10-21.md), [указатель](../homework_5_6/TEACHER_CHECK.md) | 12.09 — live A/B; 21.10 — live judge сохранённых ответов | Judge 34/34 complete, false positive сохранён. Citation presence не grounding; новая метрика не подменяет старый RAGAS. |
| 6.1 | Bounded naive loop, allowlist/unknown tools/limits, пять traces | [Traces](agent-naive-traces/), [код](../app/services/agent_naive.py), [указатель](../homework_6_1/TEACHER_CHECK.md) | Дата не установлена; исторический live LLM / учебная KB / stub | Baseline есть; не production RAG и не реальная доставка Telegram. |
| 6.2 | ReAct, строгие tools, deadline/steps/usage, critic/revision, baseline; premium path — дополнение | [Отчёт](agent-react-report.md), [JSON](agent-comparison-results/), [указатель](../homework_6_2/TEACHER_CHECK.md) | Дата не установлена; исторический live LLM | Пять cases/usage/steps/revisions сохранены; premium не обязателен для текущего runtime. |
| 6.3 | Custom/prebuilt graphs, state/tools/topology, визуализация, benchmark 5×3×3 | [Benchmark](../homework_6_3/BENCHMARK_2026-10-21.md), [graph report](agent-graph-report.md), [указатель](../homework_6_3/TEACHER_CHECK.md) | 21.10.2026; live LLM / учебная KB / stub | Все 45 попыток: 44 complete, 1 timeout. Completion не correctness; разные бюджеты и неполная timeout usage ограничивают выводы. |
| 6.4 | Saver, restart/history, interrupt/approve/reject, resume/ownership, time travel, идемпотентность | [Report](agent-persistent-report.md), [указатель](../homework_6_4/TEACHER_CHECK.md), [приёмка](final_acceptance.md) | Исторический report + 21.10; offline / live restart | 40 учебных чатов пережили restart; изоляция проверена, approve/cancel покрыты tests. Telegram принят пользователем отдельно. |
| 6.5 | Single/multi-agent на пяти одинаковых случаях, метрики/quality/выбор; второй framework — дополнение | [Отчёт](multi-agent-report.md), [10 результатов](../experiments/results.json), [граф](architecture-multi-agent.md) | 15.09.2026; исторический live LLM / mock KB | 10 completed и batched judge есть. Выбран single-agent; ошибки out-of-base и пределы mock-корпуса сохранены. |

## Как читать доказательства

43 сценария и 5 расширений финальной приёмки, 34 RAGAS/citation cases,
45 попыток agent benchmark и 10 multi-agent опытов — разные знаменатели.
Тестовый PASS не доказывает смысловую правильность ответа, наличие ссылки
не доказывает entailment, а completion не означает полноту.

Все обязательные ссылки этой таблицы ведут внутрь проекта. Полные private
checkpoint/messages и необязательные raw в ignored `var/` не требуются
для чтения сводок и не заявляются как доступные в сдаваемом ZIP. Отсутствующий
исторический raw не заменяется новым вымышленным результатом.

Следующие материалы итоговой аттестации готовятся отдельно: презентация
на 11 слайдов, демонстрация 5–7 минут и проверка чистого комплекта.
