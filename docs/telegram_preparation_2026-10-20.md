# Ускорение Support Copilot — 20 октября 2026

Основа: `461daff` / PR49. Выпускается инфраструктурный пакет с Fast processing.
Модель `gpt-6-luna`, Responses API, medium, output ceilings, evidence budgets,
исходные prompts, prepared-схема, grounding, moderation и approval сохранены.
Это не полный откат последних изменений. Корпус/индекс и история не мигрируют.

## Что выпускается

- Исходный recall запускается до окончания перевода; независимые read-only
  searches выполняются параллельно, объединяются в прежнем порядке, разделяют
  лимит dense workers. Смешанные action batches сохраняют последовательность.
- Translation/embedding clients переиспользуют соединения, учитывают остаток
  времени и закрываются владельцем. Внедрённые адаптеры не закрываются приложением.
  Mutable BM25 metadata копируется отдельно для каждого запроса.
- Direct/related представления одного native блока связываются по source_id/hash.
  Старые compact checkpoints восстанавливаются по строго ограниченному legacy
  alias из сохранённого raw evidence; несовпадающие/неоднозначные редакции не
  заимствуют текст. Полные актуальные записи имеют приоритет над legacy.
- `normalize_response` сохраняет признаки length/content_filter/error, даже если
  provider одновременно сообщил status=completed: обрезанный результат не
  выдаётся за успешное завершение.
- `AGENT_SERVICE_TIER` необязателен и по умолчанию отсутствует. Existing-stack
  `compose.luna.yaml` включает `priority` — alias Fast, подтверждённый returned
  service_tier=fast. Пользователь отдельно разрешил постоянный тариф модели×2.
  Модель, reasoning и число генераций от этого параметра не меняются.
- Одинаковый резервный вызов реализован и протестирован, но выключен в профиле
  (`AGENT_GENERATION_BACKUP_AFTER=0`). Выбор самого быстрого stochastic ответа
  проверяет завершённость провайдером, а не полноту фактов и ссылок. Разрешение
  проверить расходы не заменяет quality gate.

[Официальные Fast docs](https://developers.openai.com/api/docs/guides/priority-processing)
описывают ускоренную обработку, а
[карточка GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)
задаёт тариф Fast×2. Цена всего обращения зависит также от вспомогательных
вызовов, объёма текста, cache reads/writes. Точный расход из одного winner usage
не вычисляется; отменённый backup может тарифицироваться при неизвестном usage.

## Измерения и качество

Только пять публичных вопросов из `tests/eval/verification_set.json`, отдельный
fresh in-memory graph. Нет Telegram sends и записей production history.
Время до полного ответа графа исключает startup и Telegram delivery. Cohorts
последовательные, не рандомизированные, зависят от провайдера и cache. Это не
P95/SLA и не доказательство причинности каждой дельты.

| Кейс | Контроль PR49, с | PR49 + Fast v21, с | Инфраструктура + Fast v22, с |
|---|---:|---:|---:|
| 01: DOF |34.76|34.35|17.49|
| 07: интерфейсные элементы |15.76|8.15|8.37|
| 10: Мора–Кулона |11.50|7.39|8.26|
| 16: нелинейные алгоритмы |16.49|12.68|11.16|
| 32: граничные условия |23.03|21.69|18.90|

V22: пять завершённых ответов, без timeout/iteration-limit; медиана11.16с,
диапазон8.26–18.90с. V21 сохранил все пять ответов, но harness после них завершился
cleanup AttributeError из-за несовместимого метода старого image. Это не успешное
завершение harness; ошибка сохранена отдельно.

На одинаковом frozen evidence выполнен ABBA-пилот финальной генерации:
одни messages/model/reasoning/output ceiling, tool_choice=none для обоих tiers.
07: Fast9.56/7.43с против Default10.05/11.04с;
32: Fast8.30/9.13с против Default14.48/14.79с. Returned tier подтверждён.
Независимый semantic review восьми ответов не нашёл существенной потери Fast:
Kn/Kt и Кулон сохранены во всех07; drain0 и horizontal assumption — во всех32.
Оба Fast32 сохранили distance recommendation; первый Default32 её пропустил.
Первый незавершённый tier-probe разрешал tools, поэтому сравнивал разную работу;
его частичные результаты исключены из сравнения скорости готового ответа.

**Полный pipeline пока не доказал отсутствие потерь качества.** В v22 ответы01
и07 менее полны baseline: DOF5/6 не найдены, Kn/Kt не объяснены. В32 не найдено
тело Draining Condition, а видимое предположение Transmitting не изложено.
Явных новых фактических выдумок review не нашёл. Эти различия нельзя причинно
приписывать Fast по одному cohort; frozen генерация такого ухудшения не показала.
Полнота retrieval и изложения остаётся открытой работой. Нельзя заявлять, что
вся задача «быстрее без потери качества» полностью решена или что каждый ответ
будет готов за10–15с. Аварийные60/90/105с сохранены; внешние ошибки возможны.

## Отклонённые эксперименты

Код кандидатов сохранён отдельно: commit `d62f09f`, ветка
`experiment/telegram-preparation-20261020`. Append-only prepared input, literal
PDF/CHM completion, изменённые prompts и финальный model-turn guard не включены
в production diff. Agent-first оставлен прежним opt-in, в профиле prepared=true.
V19/v20 дали10–26с, но теряли существенные сведения; v20 имел неверную цитату.
Prepared6с повтор v17 дал17–38с, backup12с v18 —19–45с. Packet24К не устранил
тайм-ауты. Один backup оставлял следующий медленный шаг без защиты. Более быстрые
v12/v13 теряли Kn/Kt. Эти варианты не подменяют принятый baseline.

[Машинный отчёт](telegram_preparation_2026-10-20.json) сохраняет22 cohorts,
106 публичных попыток, outcomes, durations, known usage/unknown cancellations,
frozen pilot и ограничения. Первый v12 прерван ради order guard; label случайно
переиспользован, raw answers утрачены. Сохранившиеся stdout timings отмечены
отдельно и исключены из сравнения. Успешные повторы не скрывают неуспехи.

## Verify

- Полный офлайн-прогон инфраструктурного дерева:1460 passed,12 skipped,
  60 warnings (`pytest tests --ignore=tests/unit/test_manual_model_benchmark.py`).
 23 теста отдельного незакоммиченного benchmark вне scope исключены.
- После исправления legacy checkpoint compatibility:106 затронутых tests PASS,
  включая явный pre-deploy hash, ToolMessage recovery, несовпадающие identity
  fields, ambiguous revisions и приоритет актуального текста.
- Экспериментальный snapshot отдельно прошёл1483/12; это не число тестов выпуска.
- Независимый Astra review: найденный compatibility blocker исправлен,
  повторное review блокеров не выявило. Runtime-параметры/semantic limits описаны выше.
- Проверяются Ruff, git diff --check, Docker build, readiness и точные source
  hashes API/бота. Состояние rollout фиксируется в commit report.

Общий release gate S0-005 остаётся открытым; эти пять вопросов не заменяют holdout.
