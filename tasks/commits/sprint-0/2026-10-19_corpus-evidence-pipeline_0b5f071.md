# S0-005 — проверка пакета извлечения и поиска перед merge

Дата: 19.10.2026. Проверенный исходный HEAD: `0b5f071`.
Пакет: восемь коммитов `7da2367..0b5f071`, база сравнения —
`origin/main` на `e3645bc` (PR #47). GitHub issue не привязан.
Задача: S0-005, общий процесс извлечения, поиска и ответов по корпусу.
**Слияние реализации не означает прохождение приёмки S0-005 или production rollout.**

## Что изменено

- `app/services/corpus*.py`, `structured_readers.py`, `chm_reader.py`,
  `ingestion.py`: структурированный каталог PDF/CHM/HTML, сохранение таблиц,
  расположения и происхождения фрагментов; поддержка проверенных визуальных
  расшифровок. Производное HTML-зеркало CHM исключается из индекса по умолчанию.
- `app/services/rag.py`, `lexical.py`, `parallel_retrieval.py`,
  `related_evidence.py`, `retrieval_planner.py`: поиск по чистому тексту через
  BM25, раздельная нормализация русского и английского, английский вариант
  запроса, ограниченный параллельный dense retrieval, независимые цитаты
  дополнительных фрагментов. Общий бюджет дополнений уменьшен до 6000 символов.
  Необязательный planner выключен по умолчанию.
- `app/services/agent_persistent.py`, `prepared_evidence.py`,
  `answer_quality.py`, `runtime_model.py`: компактный пакет доказательств,
  устойчивые ссылки, обработка неполного ответа и обозначений единиц,
  настраиваемые повторы временных ошибок. В шаблоне обычный ответ использует
  reasoning low, расширенный — medium; лимиты ответа по умолчанию 800/1500.
- `app/core/sse.py`, streaming routes, `bot/handlers/_common.py`,
  `bot/services/backend_client.py`: heartbeat, статус ожидания и ограниченные
  повторы временных ошибок до получения данных; обычные изменяющие запросы
  после HTTP-ответа автоматически не повторяются.
- Checkpoints: при старте удаляются состояния потоков, неактивных больше
  168 часов; ожидающие подтверждения внутри них также истекают. Значение
  `AGENT_CHECKPOINT_TTL_HOURS=0` отключает очистку. Это не очистка SQL-истории чата.
- `app/eval/`, `scripts/`, `tests/eval/corpus_suite/`: отдельные dev/holdout,
  проверки происхождения и утверждения эталонов, бюджет экспериментов,
  candidate runtime и инструменты измерения/разбора ошибок.

По сохранённому отчёту предыдущего этапа локальный индекс перестроен без
CHM-зеркала, применены 558 проверенных визуальных расшифровок. Эти локальные
данные не переносятся в другое окружение одним Git merge.

## Исправления при этой проверке

- Три параметризованные проверки в `test_retrieval_diversity.py` обновлены
  под независимое добавление уже найденных фрагментов. Ранее они запрещали
  присутствие другого источника вообще. Теперь проверяют точный текст,
  сохранение версии/хеша/location_id обоих источников, отдельное происхождение
  и пустой `related_to`. Production-отбор для исправления тестов не менялся.
- README и описание графа, включая SVG, согласованы с действующим лимитом
  800 токенов и дефолтной коллекцией `rag_eval_chunk_1024`.
- Удалены лишние пустые строки в конце трёх файлов.

## Сохранённые замеры: улучшения и ухудшения

Источник — `tests/eval/results/baseline_v0/summary.json` и
`tests/eval/results/after_block5b/summary.json`. По 52 завершённых graph replay,
одна попытка на вопрос, без измерения доставки в Telegram. Новый платный
benchmark при этой проверке не запускался.

| Метрика | Исходная версия | Последний сохранённый прогон |
|---|---:|---:|
| Медиана полного времени | 8,86 с | 7,45 с |
| p90 полного времени, nearest rank | 21,85 с | 16,70 с |
| Ответы дольше 15 секунд | 7/52 | 8/52 |
| Медиана времени retrieval | 1,52 с | 2,92 с |
| Медиана времени модели | 7,05 с | 3,77 с |
| Медиана времени отбора | 0,525 с | 0,192 с |
| Медиана суммы входных токенов на вопрос | 7556,5 | 8889 |
| Всего вызовов основной модели | 63 | 67 |
| Медиана golden-группы | 7,96 с | 8,52 с |

Общая медиана уменьшилась примерно на 16%, но retrieval стал медленнее,
входной объём вырос примерно на 18%, а число медленных ответов не уменьшилось.
Нельзя делать вывод об экономии денег только по входным токенам: одновременно
изменились reasoning и выходные бюджеты. Это наблюдения разных одиночных
прогонов, а не гарантия скорости или причинный A/B эксперимент.

Все фактически встречающиеся номера цитат в последнем прогоне разрешаются;
для одного ожидаемого отказа цитат нет. Поэтому поле `all_citations_resolve`
истинно для 51/52, а не означает одну сломанную ссылку. Все сохранённые источники
имеют соответствующий непустой evidence. Это проверка структуры, не смысла.

Последний exact-quote диагностический счётчик — 13/21 обязательных anchors
в retrieved/evidence тексте. Он не измеряет правильность утверждений ответа.
Quotes перенастраивались после изменения сериализации корпуса, поэтому его
рост относительно ранних прогонов не доказывает рост точности.
Независимые смысловая приёмка, holdout и повторная end-to-end доставка остаются
открытыми; S0-005 не закрывается этим отчётом.

## Verify

- Первый полный `python -m pytest tests -q --tb=short` — 1391 passed,
  12 skipped, 3 failed: описанные выше устаревшие ожидания.
- `python -m pytest tests/unit/test_retrieval_diversity.py tests/unit/test_retrieved_complement_priority.py -q --tb=short`
  после исправления — PASS, 25 passed.
- Повторный `.venv/Scripts/python -X utf8 -m pytest tests -q --tb=short --junitxml=var/merge_review_20261019/pytest_final.xml`
  — PASS, 1394 passed, 12 skipped, 96 warnings за 151,68 с.
- `uv lock --check --offline --python .venv/Scripts/python.exe` — PASS.
- `python -m compileall -q app bot scripts` — PASS; прежний SyntaxWarning
  в docstring `scripts/time_travel_demo.py` не относится к этому пакету.
- `python -m ruff check app bot scripts tests --select E9,F63,F7,F82 --output-format concise`
  — PASS.
- `git diff --check origin/main` — PASS после удаления пустых EOF-строк.
- Полный `ruff check app bot scripts tests --statistics` — advisory FAIL:
  266 diagnostics против 242 на `origin/main`, преимущественно imports/style.
  Это не нулевой lint baseline; в конфигурации нет blocking quality-команды.
  Диагностики не скрыты blanket suppressions.

Использован managed CPython 3.13.15 из `.venv`, не системный Python 3.10.
Sandbox-запуск остановлен: Windows `socketpair` зависал при создании asyncio loop.
Результаты pytest выше получены с доступом к локальным сокетам. Локальные XML
и lint comparison сохранены в игнорируемом `var/merge_review_20261019/`.

## Архитектурное review и исключение размера

SLOC здесь — непустые строки без строк, состоящих только из комментария;
документационные строки включены. Фикстуры и результаты измерений исключены.

| Production-файл | SLOC main → пакет | Решение и граница дальнейшей работы |
|---|---:|---|
| `app/services/rag.py` | 2620 → 2449 | split выполнен: lexical и post-processing вынесены; дальнейшая граница — context selection |
| `app/services/agent_persistent.py` | 994 → 1063 | defer: отдельно рассмотреть lifecycle/retention saver без изменения approval graph |
| `app/services/ingestion.py` | 568 → 872 | defer: отделить применение reviewed transcriptions от index lifecycle |
| `app/core/config.py` | 481 → 514 | keep: единый settings/environment contract; избегать дробления только по числу строк |
| `app/services/corpus.py` | 0 → 514 | keep: единый каталог identities, locations и blocks |
| `app/services/lexical.py` | 0 → 548 | keep: единый lexical ranking/tokenization contract, вынесенный из RAG |
| `app/eval/corpus_suite.py` | 0 → 727 | defer: возможная граница — sealed contract validation / scoring |
| `app/eval/visual_transcribe.py` | 0 → 659 | defer: возможная граница — provider orchestration / source-bound validation |

Временное исключение для перечисленных новых/увеличенных превышений и legacy
RAG >1500 SLOC: задача S0-005, владелец `skoroko`, повторное review 26.10.2026.
Разделение ingestion/graph в этом merge потребовало бы менять согласованные
порядок evidence, source identity и approval lifecycle. Оно отложено до
отдельного пакета с теми же regression checks; это технический долг, не
утверждение, что размер безвреден. Основные доказательства — corpus, source
identity, retrieval companion, prepared evidence и approval regression tests.

## Контракты, данные и обновление

SQL-миграций и изменений chat schema в пакете нет. RAG schema дополнена
metadata/provenance полями. Изменена структура извлечения/индекса, поэтому
новое окружение требует согласованных каталога, docstore, Qdrant collection
и reviewed-transcription sidecar; merge сам их не разворачивает.
Старые `RAG_CONDENSE_LLM_ENABLED` / `RAG_CONDENSE_REASONING_EFFORT` заменены
на настройки перевода: перед обновлением нужно сверить локальную конфигурацию
с `.env.example`. При необходимости отключения перевода используется
`RAG_TRANSLATE_LLM_ENABLED=false`.

Архитектурные решения из этого пакета описаны в `documentation_evidence.md`;
отдельные `IMPLEMENTATION_DECISIONS` / `CURRENT_IMPLEMENTATION` не добавлялись.
Узлы и рёбра графа не менялись при этой проверке; исправлена подпись его лимита.
Merge выполняется через PR в существующую `main`: удалённой `dev` нет.
Содержимое `.env`, корпуса и `TEMP/` при этой проверке не менялось.
