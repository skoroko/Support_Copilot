# Оценка RAG-пайплайна — домашнее задание 5.6

**Исторические эксперименты от 12.09.2026.** Приведённые ниже показатели
относятся к прежнему RAG-пути и не оценивают финальный Telegram-агентный контур.
Актуальный отдельный прогон: [оценка Telegram-агента от 14.10.2026](telegram_agent_evaluation.md).

Дата прогона: 12 сентября 2026 года. Все сравнения выполнены на одном вручную
проверенном golden dataset из 34 вопросов. Генерация ответов использовала
production RAG на дату эксперимента, а оценка — RAGAS 0.4.3 и отдельного судью `gpt-4o-mini`.

## 1. Конфигурация оценки

| Компонент | Значение |
|---|---|
| RAG LLM | `gpt-5.4-mini` |
| Embedding | `text-embedding-3-large`, 1024 измерения |
| Judge LLM | `gpt-4o-mini` |
| Judge embedding | `text-embedding-3-small` |
| Метрики RAGAS | Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall |
| Кастомная метрика | `has_citation`, `@discrete_metric` |
| AnswerRelevancy strictness | 1 |
| Таймаут одной judge-метрики | 180 секунд |
| Параллелизм | 3 кейса; ContextPrecision — не более 2 вызовов |

`ContextPrecision` проверяет все контексты кейса одним structured call, после
чего применяет формулу average precision из RAGAS. Ответ, retrieval-контексты и
latency сначала фиксируются в CSV, а judge-метрики дописываются к тем же входам.
Это исключает повторный retrieval и расхождение входов при возобновлении после
сетевого таймаута. Средняя latency ниже относится только к production RAG и не
включает последующую работу judge.

## 2. Golden dataset

- `tests/eval/golden_dataset_raw.csv` содержит 36 автоматически полученных
  кандидатов.
- `tests/eval/golden_dataset.json` содержит 34 вручную проверенных уникальных
  кейса с обязательными полями `user_input`, `reference`,
  `reference_contexts`.
- Набор включает точечные и обзорные вопросы, составные вопросы, вопрос вне
  базы знаний, терминологию GTS NX, сетки, материалы, граничные условия и
  нелинейные алгоритмы.
- Все три прогона используют один и тот же набор из 34 вопросов; порядок и
  содержание вопросов были проверены перед сравнением.

## 3. Baseline

Baseline: `recursive`, `chunk_size=512`, `overlap=64`, `top_k=10`, без
re-ranker, коллекция `rag_block_03`.

Артефакты:

- `tests/eval/results/2026-09-12_185629_baseline_batched.csv`;
- `tests/eval/results/2026-09-12_185629_baseline_batched.json`.

| Метрика | Значение |
|---|---:|
| Faithfulness | 0.8641 |
| Answer Relevancy | 0.6910 |
| Context Precision | 0.8529 |
| Context Recall | 0.9559 |
| Has Citation | 0.9706 (33/34) |
| Средняя latency | 16 893.93 мс |

Baseline проходит порог faithfulness и citation, но answer relevancy на 0.0090
ниже целевого значения 0.7.

## 4. Эксперимент A — chunking

Изменён ровно один параметр: `chunk_size` увеличен с 512 до 1024 токенов.
`overlap=64`, `top_k=10`, модели и re-ranker не менялись. Чтобы не смешать
фрагменты разного размера, использованы отдельные коллекция
`rag_eval_chunk_1024` и docstore
`var/rag_eval_chunk_1024_docstore.json`.

Артефакты:

- `tests/eval/results/2026-09-12_175022_chunk_1024_inputs.json` — полная
  конфигурация и зафиксированные входы;
- `tests/eval/results/2026-09-12_193616_chunk_1024_complete.csv` — 34/34
  завершённых строк;
- `tests/eval/results/2026-09-12_193616_chunk_1024_complete.json` — агрегаты.

| Метрика | Chunk 1024 | Δ к baseline |
|---|---:|---:|
| Faithfulness | 0.8724 | +0.0084 |
| Answer Relevancy | 0.7009 | +0.0099 |
| Context Precision | 0.8663 | +0.0135 |
| Context Recall | 0.9461 | −0.0098 |
| Has Citation | 0.9412 (32/34) | −0.0294 |
| Средняя latency | 6 565.43 мс | −10 328.50 мс (−61.14%) |

Более крупный chunk дал лучший общий результат: выросли faithfulness,
relevancy и precision, recall снизился менее чем на один процентный пункт.
Наблюдаемое сокращение latency существенно, но его нельзя полностью приписать
chunking: внешняя генерация выполнялась в разное время и подвержена сетевой
вариативности.

## 5. Эксперимент B — retrieval

Изменён ровно один параметр baseline: `top_k` уменьшен с 10 до 5. Chunking,
модели, коллекция и re-ranker не менялись.

Артефакты:

- `tests/eval/results/2026-09-12_173534_top_k_5_inputs.json` — полная
  конфигурация и входы;
- `tests/eval/results/2026-09-12_185235_top_k_5_complete.csv`;
- `tests/eval/results/2026-09-12_185235_top_k_5_complete.json`.

| Метрика | Top K = 5 | Δ к baseline |
|---|---:|---:|
| Faithfulness | 0.8805 | +0.0164 |
| Answer Relevancy | 0.6413 | −0.0498 |
| Context Precision | 0.8099 | −0.0430 |
| Context Recall | 0.9314 | −0.0245 |
| Has Citation | 0.9706 (33/34) | 0.0000 |
| Средняя latency | 2 161.82 мс | −14 732.11 мс (−87.20%) |

`top_k=5` быстрее и немного лучше по faithfulness, но проигрывает сразу по
relevancy, precision и recall. Для финальной конфигурации этот вариант не
выбран.

## 6. Финальная конфигурация и пороги

Выбрана конфигурация эксперимента A:

| Параметр | Финальное значение |
|---|---|
| LLM | `gpt-5.4-mini` |
| Embedding | `text-embedding-3-large`, 1024 измерения |
| Chunking | `recursive`, 1024 / 64 |
| Retrieval | `top_k=10`, score threshold 0.45 |
| Re-ranker | выключен |
| Коллекция | `rag_eval_chunk_1024` |
| Docstore | `var/rag_eval_chunk_1024_docstore.json` |

Конфигурация закреплена в `app/core/config.py` и `.env.example`. Отдельная
пара collection/docstore обязательна: docstore отслеживает хеши документов, а
повторное использование старой пары могло бы пропустить переиндексацию или
смешать chunks размером 512 и 1024.

| Критерий задания | Результат | Статус |
|---|---:|---|
| Faithfulness > 0.7 | 0.8724 | пройден |
| Answer Relevancy > 0.7 | 0.7009 | пройден, запас мал |
| Has Citation > 0.95 | 0.9412 | не пройден |

## 7. Failure analysis

Ниже разобраны пять худших или диагностически важных случаев финального
прогона.

1. **«Сколько стоят лицензии GTS NX?»** — faithfulness 0, relevancy 0,
   recall 0, citation 0. Ответ корректно отказался выдумывать цену. Это
   контрольный abstention, а не галлюцинация: текущие агрегаты штрафуют
   безопасный отказ, а метрика citation считает его в общем знаменателе.
2. **«Меня интересует модель Мора-Кулона»** — faithfulness 0, relevancy 0,
   recall 1.0, top score 0.4517. Нужный контекст был найден, но значение почти
   совпало с confidence threshold 0.45, и пайплайн вернул отказ без источников.
   Это пограничная ошибка confidence/generation, а не отсутствие знания.
3. **AUTO-SOLID: два типа объёмной сетки** — faithfulness 0.1429 при
   relevancy 0.8456, precision 1.0 и recall 1.0. Ответ практически совпадает с
   эталоном и имеет цитату; низкий score указывает на шум claim decomposition
   LLM-судьи. Случай нужен в повторном прогоне и ручной проверке.
4. **Элемент бесконечной области** — faithfulness 0.6000 при relevancy,
   precision и recall 1.0. Ответ добавил несколько поддержанных контекстом
   технических деталей сверх краткого эталона; их стоит сократить и связывать
   с отдельными цитатами.
5. **Обзор граничных условий** — faithfulness 0.6875 при relevancy 0.8885,
   precision и recall 1.0. Retrieval успешен, но обзорный ответ расширился до
   симметричных условий, которых нет в кратком reference. Нужен более строгий
   лимит аспектов или более полный эталон для обзорных вопросов.

## 8. Phoenix и наблюдаемость

Phoenix работает локально по адресу `http://localhost:6006`, проект
`support-copilot-rag`. Проверочный скрипт подтвердил 86 request traces, из них
34 с полным деревом retriever/embedding/LLM, и 108 traces всех типов. В
интерфейсе вручную
проверена трасса `74be5236fd7d106f0689de0bd048ce59`: 13 spans, дерево
`RAGService.answer` → retrieval/embedding → `ChatCompletion`, документы и
similarity scores около 0.53, 0.51 и 0.49, latency около 1.3 секунды и
рассчитанная стоимость менее $0.01.

Трассировка не записывает API-ключи. При отключении
`TRACING_ENABLED=false` instrumentation не регистрируется.

## 9. Известные проблемы и план улучшений

- `has_citation=0.9412` ниже цели 0.95. Два ответа без ссылок — корректный
  отказ по цене и ошибочный отказ по Мора-Кулону. В продуктовой метрике нужно
  отдельно считать citation coverage для фактических ответов и abstention
  rate, а не требовать фиктивную ссылку от безопасного отказа.
- По буквальному шаблону задания `has_citation` должен вызывать
  `llm.agenerate`. В текущей версии он остаётся `@discrete_metric`, но проверяет
  однозначные маркеры `[N]`, имена файлов и «согласно …» локально. Это
  осознанное ограничение: дополнительная передача сохранённых ответов внешнему
  судье не выполнялась без отдельного разрешения.
- Answer Relevancy прошёл порог всего на 0.0009. Следующий прогон должен
  повторить оценку тем же судьёй минимум три раза и показать среднее с
  разбросом, поскольку LLM-as-judge даёт вариативность.
- Для вопроса о Мора-Кулоне нужно откалибровать threshold на golden dataset и
  проверить ветку lexical evidence вокруг 0.45; снижать порог без проверки
  out-of-domain кейсов нельзя.
- Для широких вопросов стоит уменьшить число аспектов ответа и требовать
  отдельную ссылку на каждое утверждение. Дополнительно полезны
  HallucinationEvaluator Phoenix и регрессионная выборка abstention-кейсов.

## 10. Воспроизведение

```powershell
$env:PHOENIX_WORKING_DIR="$PWD\var\phoenix-eval"
.venv-eval\Scripts\python.exe scripts\verify_eval_setup.py
.venv-eval\Scripts\python.exe scripts\verify_phoenix_traces.py --minimum 20
$env:PHOENIX_WORKING_DIR="$PWD\var\phoenix-test"
.venv-eval\Scripts\python.exe -m pytest -q
python -m ruff check .
docker compose config --quiet
```

Live-evaluation выполняется только с явным
`--allow-external-data-transfer`; точные команды трёх прогонов приведены в
`homework_5_6/README.md`.

Финальная локальная проверка: `240 passed, 9 skipped`; Ruff — без замечаний;
`docker compose config --quiet` — успешно; eval setup — `OK`.
