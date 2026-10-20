# Public source-choice benchmark

`scripts/benchmark_source_routing.py` переносит фиксированный paired experiment
в независимый от Docker и локального корпуса запуск. Offline-команды:

```powershell
uv run python scripts/benchmark_source_routing.py validate
uv run python scripts/benchmark_source_routing.py report
```

Fixture [source_routing.json](../tests/eval/source_routing.json) содержит восемь
публичных RU/EN вопросов, 96 буквальных описаний `{id, description}`, исходный
порядок кандидатов для каждого вопроса и 64 сохранённых результата. Общие
описания хранятся один раз; полный текст таблиц, figure regions и большой
metadata manifest не перенесены. Восстановленные JSON question/candidates
сохраняют все семь исходных SHA-256 входов. Собственный fixture hash защищает
также ожидания и сохранённые результаты. Provenance содержит hash исходных
inputs/results/review, каталога и selector implementation.

Новый эксперимент запускается отдельной явной командой. Она может создавать
платные запросы; при подготовке этого переноса такие запросы не выполнялись.
Ключ и необязательный base URL берутся из существующих переменных окружения;
скрипт не загружает `.env`, не печатает их значения и не принимает ключ в CLI.

```powershell
uv run python scripts/benchmark_source_routing.py run --allow-paid-calls --model YOUR_SELECTOR_MODEL --output var/source_routing_new.json
uv run python scripts/benchmark_source_routing.py report --results var/source_routing_new.json
```

Каждый endpoint call создаёт и закрывает отдельные `TableSelectionService` и
HTTP client: кэш не прогревается, retries отсутствуют, timeout — 5 секунд.
Порядок Decisions/Responses чередуется по вопросам и обращается во втором
повторе (AB/BA). Проверяются одинаковые input hashes, cache miss, supplied ID
и фактическая последовательность вызовов. Файл результата сохраняется после
каждой строки; существующий output не перезаписывается новым запуском.
Неполный запуск нельзя выдать за завершённый report. Ошибка/timeout,
вернувшие `None`, не засчитываются как правильный отрицательный ответ.

Local (`lexical`) и `off` — повтор сохранённых решений original-question-only
baseline. Они не пересчитывают BM25 по коротким описаниям: точный исходный
поиск использовал больше буквальных данных каталога. Новые timings для этих
строк не создаются. Для живого локального сравнения нужен отдельный corpus
replay, которого этот перенос не выполняет.

## Сохранённый результат

Исходный experiment: `gpt-6-luna`, два повтора, GTS NX / unknown,
10 source-validated документов; 97 scoped tables, 96 после whole-table
бюджета 6000 символов. Семантический selector получал все 96 кандидатов.

| Режим | Верно / 16 строк | Реальные вызовы: верно / всего | Медиана service, с | Диапазон service, с |
|---|---:|---:|---:|---:|
| Local lexical | 4/16 | — | — | — |
| Off | 4/16 | — | — | — |
| Decisions | 14/16 | 12/14 | 0.913098 | 0.653583–1.616627 |
| Responses | 16/16 | 14/14 | 2.620379 | 2.140298–3.616002 |

Ошибок транспорта/парсинга в сохранённом опыте не было. Decisions был быстрее
всех 14 реальных пар. Медианы полного client call, включая создание/закрытие:
1.052755 и 2.727728 секунды соответственно. В каждом endpoint-режиме две
строки `fixed-en` обработаны локально без вызова провайдера; 16/16 не означает
16 Responses calls. Семь semantic вопросов повторно используют одну целевую
таблицу Material; отрицательный вопрос — один.

Decisions в обоих порядках ошибся на `paraphrase-ru-2`: выбрал Property
вместо Material. Property перечисляет типы элементов/свойств и применимость;
это не эквивалентный перечень конститутивных моделей грунта. Ошибочный источник
имеет допустимый ID, scope и исходные байты. Такая валидация не доказывает
семантическую релевантность или достаточность. Краткое human review сохранено
в fixture; исходный fixture SHA:
`c8232beddeb1df734814b94c29c5362e46c03ffa694adaee33c8951db50a3dc1`.

Это source-navigation опыт: ответ не генерируется, sufficiency и end-to-end
quality gates не проверяются. Local 4/16 не измеряет production recall:
реальный RAG использует bilingual translation/planning. В исходном v9 RU
таблица была выбрана по lexical anchors после перевода, без Decisions call.
Нельзя приписывать этот успех Decisions. Два повтора, одна целевая таблица и
один negative не обосновывают общую точность, P95/SLA, стоимость или ускорение
конечного ответа. Confidence не сохранён и не откалиброван. Default selector
остаётся `off` до отдельного multi-target routing и полного answer-quality
gate; этот benchmark не меняет runtime-конфигурацию.
