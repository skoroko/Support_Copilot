# Задержки Telegram: исправление 19.10.2026

## Причины и выбранная архитектура

Основа — `fce0f33`, ветка `hotfix/telegram-turn-latency`. Runtime остаётся
`gpt-6-luna`, Responses API, `medium` для brief и expanded. Evidence budgets,
corpus/index, citation grounding, moderation и human approval не уменьшены.

История кода не подтверждает полезность полного отката на 15 октября.
`1afeba4` увеличил generation timeout до 120с; `2e704b9` добавил SDK retries,
которые могли повторять полный timeout. `3f8e792` поставил перевод перед первым
поиском: auxiliary timeout8с с SDK retries2 превращался примерно в28с.
У embedding adapter библиотеки по умолчанию было10 повторов.
При этом `d72aced` ускорял локальную подготовку CHM с сохранением evidence;
этот выигрыш сохранён. Старые AI messages содержали opaque encrypted reasoning,
которое попадало в следующие вопросы вместе с видимым диалогом.

Теперь один monotonic deadline ограничивает весь graph turn:90с. Один model
invoke вместе со всеми retries/backoff —60с. Бот ограничивает ожидание backend
105с независимо от SSE heartbeats. Таймер активен только во время чтения,
поэтому не отменяет отправку уже готового ответа потребителем. Принятый HTTP
stream не переигрывается после обрыва; повтор соединения до принятия сохранён.
Ошибки времени передаются через существующий SSE error contract с кодом
`chat_turn_timeout`; выдуманный ответ вместо ошибки не создаётся.

Перевод и исходный поиск начинаются одновременно. Полная английская версия
по-прежнему участвует в поиске, когда перевод успешен; провал перевода сохраняет
исходный recall. Вспомогательные перевод/embedding не имеют скрытых SDK retries.
Query embedding ограничен15с и остатком общего бюджета; timeout ingestion не
уменьшен. Thread workers наследуют deadline через ContextVar/copy_context.
TLS connect timeout —10с; pool idle expiry —60с вместо стандартных5с.
Клиент модели создаётся один раз на lifespan приложения и закрывается при остановке.
Shared clients не мутируют: каждый вызов получает собственный timeout.
Прошлый visible dialogue сохраняется, opaque provider continuation очищается
только у прошлых ответов; текущий tool loop сохраняет reasoning.

Lexical BM25 index и tokenizer прогреваются до readiness. Docker image содержит
проверенные tiktoken данные, поэтому первый вопрос не скачивает их из сети.
Existing-stack profile загружает готовый accepted index без повторного
ingestion при каждом restart. Data-файлы не менялись; fresh base compose
сохраняет startup ingestion, uploads продолжают использовать ingestion service.
Холодный запуск финального контейнера занял около194с до события готового
BM25; start_period300с учитывает этот прогрев. Это время старта процесса,
не дополнительное ожидание каждого вопроса. Ускорение холодного старта
отключением ingestion в этом прогоне отдельно не доказано.
Установка зависимостей ограничена по параллелизму и отделена от README/env
примеров: их изменение больше не инвалидирует dependency layer.

## Проверка скорости и качества

Проверки используют только публичные вопросы `tests/eval/verification_set.json`,
отдельный in-memory graph и production corpus/provider configuration. История
Telegram не пересылается, checkpoints/сообщения не записываются; Telegram send
запрещён внедрённым адаптером. Это время до полного grounded ответа, не TTFT и
не время реальной Telegram доставки.

На прежнем image вопросы golden-01,10,16 не завершились в пределах70с каждый.
Это цензурированные timeout наблюдения, а не измеренные длительности ответов;
их нельзя использовать как точный baseline P50/P95. Первый кандидат завершил
их за55.17/10.89/24.65с, но golden-32 упёрся в60с. Повторный прогон выявил
дополнительные остановки на60с; поэтому конфигурация60с отклонена как слишком
жёсткая для полноты сложного поиска. Следующий кандидат45с/90с завершил golden-01 за45.29с и golden-07 за15.47с,
но10/16/32 дали ошибки. Trace подтвердил TLS ConnectTimeout при новом connect3с;
после трёх попыток получался ранний отказ примерно через10.3с. Ещё один валидный
stream был прерван на45с. Поэтому финальный кандидат использует connect10с,
60с на invoke,90с на turn, keepalive60с и query embedding15с. Прерванный
сбоем Engine прогон не считается успешным. Увеличение45→60 защищает полноту,
само по себе ускорением не является.

Размещение приватных технических артефактов: ignored
`var/telegram_latency_fix/`. Содержимое окружения и секреты не выгружаются.

### Финальный образ: фактические результаты

Основной прогон с окончательными настройками, fresh graph на каждый вопрос:

| Public case | Полное время, с | Результат |
|---|---:|---|
| golden-01 |32.09|ответ, 3 источника |
| golden-07 |26.44|ответ, 3 источника |
| golden-10 |63.34|timeout одного модельного вызова на60с |
| golden-16 |33.78|ответ, 5 источников |
| golden-32 |49.17|ответ, 7 источников |

Итого4/5, а не5/5 успешных попыток. Отдельный повтор для диагностики provider
events завершил golden-10 за13.17с и golden-16 за75.24с на том же коде/image.
В последнем случае финальный model invoke занял57.91с: первый provider chunk
пришёл на0.959с, первый текст —32.925с, последний chunk —57.907с. Значит,
по крайней мере этот длинный хвост возникает после начала потока провайдера;
нового скачивания tokenizer или многоминутных SDK-повторов здесь нет.
Это не доказывает отсутствие сетевых задержек или гарантированный SLA.
Неуспешная попытка включена в результаты и не заменена удачным повтором.

Во всех завершённых ответах номера цитат разрешаются в выданные источники.
Ручная выборочная сверка не заменяет семантический acceptance: у golden-01
детализация осей r/θ/z не дана явно в доступном текстовом фрагменте. Это
ограничение качества остаётся открытым; равенство семантики на всех вопросах
и отсутствие любых ошибок не заявляются. Модель, reasoning, prompts, полный
evidence budget и контур проверки источников сохранены. Сводные обезличенные
замеры: [JSON](telegram_latency_fix_2026-10-19.json).

### Verify и deployment

- `.venv/Scripts/python -m pytest tests -q --tb=short --junitxml=var/telegram_latency_fix/full_final.xml`
  — PASS:1432 passed,12 skipped,160.71с, RAG_ENABLED=false/TRACING_ENABLED=false
  для офлайн startup; тесты с включённым RAG используют собственные fixtures.
  Это прогон рабочего дерева, включающий23 теста отдельного незакоммиченного
  model benchmark; его код и отчёты исключены из этого hotfix.
- Последующая проверка `tests/chat/test_routes.py` — PASS:26 tests, включая
  два новых безопасных SSE timeout contracts после полного прогона.
- После замечания Astra `tests/unit/test_retrieval_planner.py` и
  `tests/unit/test_turn_latency_budget.py` — PASS:33 tests. Явный max_retries=0
  закреплён тестом создания адаптера планировщика; default planner выключен.
- Lifecycle/runtime/bot/Docker focused suite — PASS:88 tests.
- Ruff по новым модулям, runtime adapter, bot client и regression tests — PASS.
- `git diff --check` — PASS.
- Docker Compose build — PASS. App и bot пересозданы из одного image
  `sha256:c6431c9f8237148745f05ecdf1c6f1b8b9dbaf2385fef848b53b006cf305c444`.
  Публичные provider-замеры выполнены на предшествующем image
  `sha256:55b701cf39136c4ae48ec4c82f2ccf41c56644e076e1ff2aa7872405e381e29b`;
  последующая runtime правка — retries выключенного planner; ещё изменены
  только лишние пустые строки EOF. Existing-stack override явно отключает
  ingestion на старте, как уже было задано в измерительном harness.
- `/ready` — HTTP200, status ok, redis up, rag up; bot OpenAPI — HTTP200;
  оба контейнера healthy. `check_deployment_source.py` —20/20 файлов совпали
  в каждом контейнере, source mismatch отсутствует.

Все72 frozen corpus-suite files побайтово совпадают с HEAD и index. Устранён
только CRLF checkout на Windows, manifests/эталоны не редактировались. Новый
`.gitattributes -text` сохраняет hash-bound bytes при следующих checkout.
Публичные схемы и data migrations не менялись. Топология agent graph и workflow
approval остаются прежними; добавлены бюджеты I/O и перекрытие независимых работ.

## Review архитектурных границ

SLOC: tokenizer Python; пустые строки/comments не учитываются.

| Файл | До | После | Решение |
|---|---:|---:|---|
| agent_persistent.py |1072|1042|split: policy истории вынесена в agent_history.py |
| rag.py |2478|2400|split: перевод/cache/контракт вынесены в retrieval_translation.py |
| config.py |514|517|keep: один связанный параметр deadline, существующая настройка lifecycle |

Для `rag.py` остаётся существующий legacy exception >1500SLOC. Владелец follow-up:
maintainer meshnavigator; повторный review границы retrieval/selection — при
следующем изменении retrieval, не позднее2026-11-02. Массовый перенос ранжирования
в этом hotfix добавил бы риск для evidence/provenance. Для config.py разрешено
локальное превышение +3SLOC: deadline входит в существующую ответственность
runtime configuration, отдельный generic wrapper не создаётся. Повторный review
при следующей настройке agent budgets, владелец/срок те же.

Независимая read-only проверка обнаружила и закрыла два дефекта первого
кандидата: timer across yield мог отменить consumer, sync embedding/mini
translation игнорировали остаток бюджета. После исправления новых существенных
замечаний не обнаружено.

Независимый согласованный аудит Astra High: выпуск ограниченного latency
hotfix разумен, блокирующих дефектов в Telegram path не найдено. Найденный
неблокирующий дефект opt-in planner (default retries2 вопреки timeout5с)
исправлен явным max_retries=0 и тестом. Полный возврат прежней стабильной
скорости и отсутствие потерь успешных ответов этот аудит не подтверждает.

Следующие измеримые улучшения: запускать English recall сразу после готового
перевода, сохранив общий concurrency cap и стабильный порядок evidence;
сохранять timings первого event/текста/последнего event и максимальной паузы
также при неуспехе. Не повышать лимиты вслепую и не урезать reasoning/evidence.
Ответственный — maintainer meshnavigator; scope review до2026-11-02.

## Операционные ограничения

Override `compose.luna.yaml` рассчитан на существующий принятый индекс.
После изменения корпуса, embedding-модели или chunking сначала нужна отдельная
индексация/миграция. `/ready` проверяет доступность, но не свежесть и полноту
индекса; Qdrant, каталог и исходные файлы должны оставаться согласованными.
Для инкрементального обновления совместимой коллекции текущим конфигом:

```powershell
docker compose -f compose.yaml -f compose.luna.yaml run --rm --no-deps -e RAG_INGEST_ON_START=true app python -c "from app.core.config import get_settings; from app.services.rag import RAGConfig,RAGService; service=RAGService(RAGConfig.from_settings(get_settings())); service.build(); service.close()"
docker compose -f compose.yaml -f compose.luna.yaml up -d --no-deps --no-build --pull never --force-recreate --wait --wait-timeout 360 app bot
```

Эта команда использует существующее разрешение external transfer и может
вызвать embedding-провайдера; при hotfix она не выполнялась, корпус не менялся.
При смене размерности/модели embeddings нужен отдельный план новой коллекции,
простое включение ingestion не является миграцией несовместимого индекса.

Универсальное равенство семантического качества на всех будущих вопросах не
доказано. Профиль модели, содержательные правила и evidence budgets сохранены;
проверки latency не заменяют экспертную оценку корпуса. При длительном сбое
провайдера возможна явная ошибка вместо ответа: ожидание больше не бесконечно.
Синхронный worker может кратко завершать уже начатый I/O после отмены graph;
provider timeout ограничен остатком deadline, новые вызовы после него запрещены.

Первая сборка прервалась Bus error при свободном месте C: около0.7ГБ; затем
Docker Engine не запускался с WSL_E_USER_VHD_ALREADY_ATTACHED. Очищены только
неиспользуемые uv cache entries(106.8MiB), выполнен штатный перезапуск Docker/WSL.
VHD, образы, volumes, базы, corpus и диалоги не удалялись.
