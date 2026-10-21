# Support Copilot: окончательное техническое описание

Срез для сдачи: 21.10.2026. Функциональная версия опубликована через PR53:
source commit `78daea7`, merge в `main` — `c484959`. Первоначальная база
приёмки — `df96941` / PR51; проверенные изменения уже закоммичены и слиты.
[История публикации](publication_2026-10-21.md).
Пользователь проверил Telegram и принял результат.
Точные хэши кода, образа и индекса, измерения и ограничения — в
[протоколе приёмки](final_acceptance.md). PR52 не включён; исследовательская
S0-005 остаётся открытой. Секреты в документ не включены.

Все материалы: [указатель](INDEX.md), [покрытие 34 ДЗ](homework_coverage.md).
Схема исполняемого агента:
[финальный граф](final_agent_graph.md), [SVG](final-agent-graph.svg),
[Mermaid](final-agent-graph.mmd).

## Назначение и состав системы

Инженер поддержки задаёт вопрос в Telegram, получает ответ по мануалам со
ссылками и страницами, продолжает разговор и готовит черновик сообщения.
По явной просьбе отправить сообщение агент показывает адресата и весь текст;
внешняя отправка требует подтверждения. Инженер проверяет факты и принимает
решение по обращению.

| Компонент | Ответственность | Реализация |
|---|---|---|
| Telegram / aiogram | Admission, меню, вывод ответа, источники, feedback, кнопки согласия | `bot/` |
| FastAPI chat facade | Auth/ownership, multipart extraction, moderation, request budgets, SSE | `app/chat/routes.py`, `app/chat/service.py` |
| ChatAgent | Приватный thread, восстановление контекста, pending/resume, expansion | `app/chat/agent.py`, `app/chat/expansion.py` |
| Persistent LangGraph | Один model/tool loop, bounded state, подготовка отправки и interrupt | `app/services/agent_persistent.py` |
| RAGService | Evidence-only retrieval, продуктовые фильтры, dense/lexical recall, gates и provenance | `app/services/rag.py` |
| Ingestion | Чтение PDF/CHM/HTML и других поддерживаемых форматов, chunking, embeddings, UPSERT | `app/services/ingestion.py` |
| Qdrant и docstore | Векторы/metadata и состояние инкрементальной индексации | Collection + JSON docstore |
| Chat repository / saver | Пользовательская история и отдельно состояние исполнения графа | JSON/PostgreSQL и SQLite/PostgreSQL |
| Notifier / bot guard | Отправка одобренного текста и защита от повторной доставки | `app/services/notifier.py`, `bot/web.py` |

Telegram всегда использует persistent agent через `/chats`; переключателя
технических режимов нет. CLI, прямой RAG API и учебные агенты сохраняют
самостоятельные пути и не являются дополнительными агентами этого разговора.
`POST /rag/retrieve` и одноимённый внутренний метод возвращают evidence без
генерации ответа; финальную формулировку Telegram делает агент. Retrieval может
отдельно вызывать модель для преобразования уточняющего вопроса при включённом
LLM condense, поэтому весь пользовательский запрос не равен одному LLM-вызову.

Общий retrieval-first путь и структурный каталог уже развёрнуты: перед ответом
по документации `prepare_knowledge` вызывает `RAGService.retrieve`, затем
модель получает точный пакет evidence и может выполнить прицельный поиск.
Неопределённый первый вопрос о параметрах получает локальное уточнение без
retrieval/LLM. Редактура, smalltalk, действия и expanded имеют свои условия
пропуска подготовки. Полнота всего корпуса этим не гарантируется.

Корпус текущей версии — мануалы GTS NX и Civil: десять канонических источников
(девять PDF и одна CHM). 60 производных HTML-зеркал исключены из индекса.
В каталоге 1936 locations, 1934 текстовых документа; Qdrant — 14139 точек.
Источники, docstore и полный scroll согласованы, переиндексация не требуется.
Внутренние инструкции установки/лицензирования, тарифы и FAQ ещё не добавлены.
При этом CHM содержит manufacturer Troubleshooting, ошибки и предупреждения.
Синтетический `support_documents.json` в основной поиск не входит.
Состав и пробелы: [data_inventory.md](data_inventory.md).

## Профиль сдаваемой версии

| Настройка | Принятая конфигурация |
|---|---|
| Telegram generation | `gpt-6-luna`, Responses, brief/expanded `medium`, tier `priority` |
| RAG answer / condense | `gpt-6-luna`, `medium` / `none` |
| Visible answer target | brief 800 / expanded 1500; reasoning reserve 8192 |
| Таймауты | generation 60 с / graph turn 90 с; аварийные пределы, не SLA |
| Prepared evidence | включён; исходный пакет до 12К символов + до 6К на прицельные дополнения |
| Дополнительные поиски | независимые dense-варианты с concurrency 2; backup generation выключена |
| Индекс | `rag_eval_chunk_1024`, large/1024d, Cosine, recursive 1024/64, top-k 10, reranker выключен |
| Persistence в Compose | PostgreSQL для истории и checkpoints; Redis для cache/дедупликации доставки |
| Startup принятого стека | готовый индекс, `RAG_INGEST_ON_START=false`; на чистом стеке первая индексация отдельно необходима |

Это фактический профиль, а не все defaults шаблона. `.env.example` содержит
консервативные `low`/пустой service tier и выключенный RAG; точные публичные
переопределения для сдачи и команда запуска приведены в [README](../README.md).

## 1. Модель генерации

**Выбор →** OpenAI `gpt-6-luna`, Responses API, reasoning `medium` для brief и
expanded, tier `priority`. `APP_DEFAULT_MODEL` задаёт модель агента,
`RAG_LLM_MODEL` отдельно задаёт генерацию обычного RAG API. Адаптер `AgentModel`
сохраняет `AIMessage`/tools-контракт графа; GPT-6 использует provider-default
temperature. Ограничение — шесть model turns. `store=false`; opaque reasoning
текущего tool loop поддерживается адаптером, старые reasoning items из прошлых
turns не отправляются. Backup generation выключена. Цепочка OpenAI → Ollama
из раннего CLI не является fallback финального графа.

**Альтернативы →** Прежняя `gpt-5.4-mini`; Luna с none/low/high/xhigh;
локальная `qwen3:1.7b` и OpenRouter из ранних учебных сравнений.

**Причина выбора →** Датированный опыт 15.10 сравнил 384 ответа: на одинаковых
фрагментах medium сохранил средний rubric score полноты mini 0.906, а в пяти
расширениях дал 0.900/1.000 по полноте/grounding. Low был быстрее, но менее
полон; high/xhigh не дали устойчивого выигрыша на обоих режимах. Это небольшой
набор и модельная оценка, а не доказательство общего превосходства. Priority
принят пользователем после инфраструктурных экспериментов 20.10; его влияние
на весь workflow не изолировано. Финальный прогон 21.10 и ручная проверка
пользователя подтверждают пригодность принятой версии с сохранёнными
ограничениями. Текущие цены в этом документе не рассчитываются.

Основания: [Luna reasoning eval](luna_runtime_evaluation.md),
[подготовка и Fast](telegram_preparation_2026-10-20.md),
[новая приёмка](final_acceptance.md), [настройки](../app/core/config.py),
[runtime adapter](../app/services/runtime_model.py).
Новые model IDs, актуальные цены и новые provider limits этим документом не
назначаются; тарифы старых расчётов действуют только на дату тех расчётов.

## 2. Embeddings

**Выбор →** `text-embedding-3-large`, размерность 1024. Модель и размерность
задаются через `RAG_EMBEDDING_MODEL` и `RAG_EMBEDDING_DIMENSIONS`; при построении
индекса и embedding запроса используется согласованная конфигурация.

**Альтернативы →** `text-embedding-3-small` с 1536 измерениями; полная
размерность large (3072) из исходного сравнения; локальные embeddings как
отдельный путь без облачной передачи документов.

**Причина выбора →** Для преимущественно русского корпуса приоритет отдан
качеству multilingual retrieval. Сокращённая размерность ограничивает объём
векторов и вычислений: для float32 1024 измерения занимают 4096 байт на вектор
без metadata и индексов, втрое меньше 3072. Это документированная инженерная
мотивация, а не доказанная победа large над small на финальном корпусе:
сравнительный production benchmark embeddings не заявляется.

Основания: [ДЗ 5.1: выбор и датированные источники](../homework_5_1/README.md),
[config](../app/core/config.py), [ingestion](../app/services/ingestion.py).
Cloud embedding требует предусмотренного приложением согласия на внешний
egress; смена модели/размерности требует согласованной переиндексации.

## 3. Векторное хранилище

**Выбор →** Qdrant, dense-векторы с cosine distance и metadata для продукта,
документа, chunk и страницы. Выбранная collection — `rag_eval_chunk_1024`;
docstore — `var/rag_eval_chunk_1024_docstore.json`. В deployment Qdrant работает
как отдельный сервис, а PostgreSQL сохраняет историю и checkpoints.

**Альтернативы →** pgvector в уже имеющемся PostgreSQL; встроенный/in-memory
vector store для локального эксперимента.

**Причина выбора →** Проект уже интегрировал Qdrant с LlamaIndex, фильтрацией
metadata и последующими экспериментами retrieval. Второй backend потребовал
бы отдельной реализации и проверки без текущей продуктовой необходимости.
In-memory store удобен для тестов, но не заменяет постоянный индекс сервиса.
Cosine явно задаёт семантическую метрику; учебное сравнение с Dot на
нормализованных векторах совпало 5/5 и не показало превосходства одной метрики.

Основания: [vector_store.md](vector_store.md),
[production ingestion](../app/services/ingestion.py), [RAG](rag.md).
Учебная collection `documents`, синтетические 120 фрагментов и explicit HNSW
`m=16/ef_construct=100` относятся к отдельному адаптеру. Production ingestion
не задаёт эти HNSW-параметры явно; их нельзя выдавать за финальную настройку.

## 4. Chunking и выбор evidence

**Выбор →** `recursive`, реализованный `SentenceSplitter`: chunk size 1024
токена, overlap 64; initial `top_k=10`, score threshold 0,45, reranker выключен
по умолчанию. В одном retrieval возвращается не более пяти выбранных fragments;
бюджет распределяется между источниками с обрезкой по целым смысловым единицам.
Prepared-путь ограничивает исходный пакет 12 000 символов; прицельные
дополнения — ещё 6000, сохраняя прежние citation identities.
`top_k=10` — настройка исходного поиска, а не обещание десяти цитат в ответе.

**Альтернативы →** Fixed-size и semantic splitting; прежний recursive 512/64;
`top_k=5` или 20; включение optional reranker.

**Причина выбора →** Offline сравнение fixed/recursive/semantic сначала выбрало
512/64. В последующем облачном прогоне 12.09.2026 на 34 вопросах увеличение
размера до 1024 улучшило faithfulness 0,8641 → 0,8724, relevancy 0,6910 → 0,7009
и precision 0,8529 → 0,8663. Recall снизился 0,9559 → 0,9461, а наличие цитаты —
33/34 → 32/34: компромисс сохранён явно. Top-K5 ухудшил relevancy, precision
и recall; Top-K20 не дал выигрыша в раннем offline сравнении. Reranker не
требовался для выбранного baseline и не включён без отдельного подтверждения
его пользы измерениями. Снижение latency нельзя целиком приписать chunking.

Основания: [offline эксперимент](chunking_experiment.md),
[облачный RAG eval](rag_evaluation.md), [config](../app/core/config.py).
Collection/docstore разделены между вариантами, чтобы UPSERT не пропускал
переиндексацию и не смешивал chunks разных размеров. Сентябрьский отчёт —
обоснование параметров, а не новая оценка качества финального агента.

## 5. Single-agent

**Выбор →** Один persistent ReAct-подобный LangGraph-агент с инструментами
`search_knowledge_base`, `get_current_time`, `send_telegram_message`.
Узлы подготовки/подтверждения и RAGService — части его исполнения, а не
самостоятельные LLM-агенты. Точный граф: [final_agent_graph.md](final_agent_graph.md).

**Альтернативы →** Последовательный supervisor → researcher → writer;
обычный RAG question-answer без model/tool loop.

**Причина выбора →** Поддержка использует общий контекст и один корпус;
отдельные researcher/writer добавляют передачи контекста. На пяти одинаковых
mock-KB вопросах multi-agent увеличил tokens на 33,63%, p50 latency на 77,36%,
LLM calls с 2 до 3 и снизил оценку одного LLM judge с 5 до 3. Этот узкий
эксперимент не доказывает универсального превосходства single-agent, но не даёт
основания усложнять текущий workflow. Обычный RAG API сохранён; агент выбран
для единого разговора, выбора инструментов и контролируемого внешнего действия.

Основание: [multi-agent-report.md](multi-agent-report.md).
Возврат к нескольким агентам требует самостоятельного сценария с независимыми
направлениями поиска и нового eval, а не переноса учебного supervisor в default.

## 6. История и состояние исполнения

**Выбор →** Разделить пользовательскую историю и checkpoints графа.
Chat repository хранит сообщения, sources и feedback: JSON/JSONL локально,
PostgreSQL в Compose. LangGraph saver хранит protocol messages, tool results,
evidence, pending draft и expansion claims: SQLite локально, PostgreSQL в
Compose; memory saver используется для коротких tests/demo.

**Альтернативы →** Только память процесса; одна chat history без checkpoints;
вся история в каждом model request; LLM summary вместо sliding window.

**Причина выбора →** История нужна для просмотра и feedback, checkpoint — для
точного interrupt/resume после перезапуска. Одна история сообщений не хранит
позицию исполнения. Memory теряет ожидание подтверждения при рестарте.
Разделение позволяет использовать существующий repository и saver без второй
БД в Compose. Bounded sliding window предсказуем и не требует платного summary:
до пяти human-turn groups, 24 tool results; provider input дополнительно
ограничен `CHAT_CONTEXT_WINDOW` (default 10 преобразуется в лимит до пяти
human groups) и token budget с резервом для ответа. Обрезаются целые human/tool groups.

Приватный thread определяется interface + owner + chat. При пустом checkpoint
агент восстанавливает до десяти сообщений chat history. `/clear` удаляет обе
части под одним turn lock и инвалидирует прежние approval IDs. Изображения в
сыром виде в checkpoint не сохраняются; extracted text/transcripts доступны.

Основания: [ChatAgent](../app/chat/agent.py), [chat storage](chat.md),
[saver и ограничения интеграции](unified_agent.md).
Repository и saver имеют отдельные транзакции: авария между graph completion
и append может оставить историю отстающей. Single API worker остаётся условием
корректной сериализации текущих process-local locks.

## 7. Подтверждения и расход токенов

**Выбор →** Два отдельных механизма явного согласия:

- Отправка: сохранить exact recipient/text, draft UUID и idempotency key;
  `interrupt()` перед внешним эффектом; resume только по exact approval ID
  владельца этого чата. Подтверждение исполняет сохранённый текст, отмена
  завершает turn без доставки. В Telegram роль всегда `write-with-approve`.
- «Подробнее»: preview уведомления о дополнительном расходе; strict
  `confirmed: true` для конкретного assistant UUID. До генерации записывается
  одноразовый durable claim. Один expanded ответ использует прежние evidence,
  целевой размер видимого ответа 1500 токенов и `tool_choice=none`;
  preview/отмена не вызывают LLM.

**Альтернативы →** Немедленная отправка по model tool call; текстовое «да» без
привязки к draft; подтверждение с изменяемым recipient/text; автоматическая
регенерация оборванного ответа; постоянный ceiling 2000 без отдельного согласия.

**Причина выбора →** Ошибка адресата или текста наблюдаема вне системы, а
дополнительная генерация расходует токены. Пользователь должен видеть предмет
согласия. Ownership, UUID/latest checks и общий lock не позволяют старой или
чужой кнопке одобрить другой draft/ответ. Краткий default 800 токенов и локальное
детерминированное сокращение ограничивают расход без автоматического repair
call. Результат отправки формируется из tool result без ещё одного LLM-вызова.

Bot guard дедуплицирует idempotency key через Redis. Неизвестный исход доставки
не объявляется успешным и не повторяется автоматически. Это защита доставки,
а не общая exactly-once транзакция всех хранилищ. Административный raw agent API
поддерживает роль `full` с обходом interrupt; Telegram её не использует.

Основания: [persistent graph](../app/services/agent_persistent.py),
[expansion orchestration](../app/chat/expansion.py),
[unified_agent.md](unified_agent.md), где зафиксирована действующая политика
Accepted `DEC-2026-10-13-004`.
800/1500 — целевые размеры видимого текста. При reasoning общий provider cap
включает ещё 8192 токена резерва; отдельный жёсткий потолок видимого текста
не гарантируется. Эти бюджеты не ограничивают суммарную стоимость input + output.

## Достоверность и проверка

Citation grounding проверяет существование и нумерацию sources, но не
семантическое соответствие каждого утверждения фрагменту. Prompt требует
поиска для технических уточнений и маркировки интерпретаций; модель может
нарушить это правило. `no_evidence` и `unavailable` имеют разные ответы;
подмена отсутствующих знаний синтетической KB не выполняется.

Исправление от 14.10.2026 объединяет marker, chunk/source/page и полный текст
в одну model-facing запись `evidence`. Полный текст сохраняется в checkpoint
для подробного ответа; прежние tool-результаты и номера цитат не подставляются
как доказательства нового уточнения. Смысл диалога, технические литералы и
публичный формат sources сохраняются. Детали и legacy recovery описаны в
[unified_agent.md](unified_agent.md).

Основные offline доказательства контрактов:
[unified agent](../tests/chat/test_unified_agent.py),
[persistent graph](../tests/test_agent_persistent.py),
[expansion](../tests/chat/test_expansion.py),
[citation contracts](../tests/chat/test_agent_citations.py),
[ingestion](../tests/unit/test_ingestion.py), [retrieval](../tests/unit/test_rag.py).
Команды и датированные результаты: [unified_agent.md](unified_agent.md).
Они проверяют инварианты с заглушками, а не качество реального ответа модели.
Актуальная приёмка от 21.10: 1522 passed / 12 skipped; отдельно PostgreSQL
contract — 25 passed. Завершены 43 сценария и 5 расширений. Для 34 эталонных
вопросов: 27 полных ответов, 7 частичных, mean rubric score 0.897; это не
доля правильных ответов. Grounding применим к 33 основным ответам, все получили
1.0; отказ о цене исключён из этого знаменателя. Расширения: 4 полных и одно
частичное. Все 154 использования источников проверены по оригиналам.

HTTP median 7.63 с, p95 14.27 с, max 24.03 с — не время доставки Telegram и
не доказательство SLA. Эти 43+5 ответы собраны до узкого исправления
неопределённого первого вопроса. После коррекции выполнены полный offline
набор и focused live regression: уточнение без retrieval/model, корректная
история и sources. Пользователь выполнил предложенные Telegram-проверки и
сообщил, что результат устраивает. [Протокол и доказательства](final_acceptance.md).

Текущие ограничения: неполнота части DOF/Mohr–Coulomb ответов, 3968
`visual_unextracted`, 3414 нераспознанных областей изображений; раздельные
транзакции history/checkpoint, один API worker и неизвестный исход доставки
без автоматического повтора. Полный research gate и SLA S0-005 не закрыты.
Существующее принятие рисков зависимостей сохранено в
[dependency-risk-acceptance.md](dependency-risk-acceptance.md).

Предыдущая [оценка 15.10](telegram_agent_evaluation.md), сентябрьский RAGAS,
ранние CLI/модельные benchmark и [45 учебных запусков 6.3](../homework_6_3/BENCHMARK_2026-10-21.md)
имеют собственные конфигурации и знаменатели. Они служат историей выбора,
а не новой оценкой текущего production RAG. Обязательные и дополнительные
результаты, даты и намеренно оставленные пробелы — в
[матрице всех 34 ДЗ](homework_coverage.md).

[Исторический архитектурный PDF](architecture-historical-cli.pdf) сохранён
как учебный CLI-снимок без обновления содержимого. Его будущие компоненты,
нагрузочные assumptions и схема не являются описанием финального графа.
