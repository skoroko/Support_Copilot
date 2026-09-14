# Домашнее задание 5.6 — оценка и наблюдаемость RAG

Инкремент добавляет воспроизводимый RAGAS 0.4 evaluation pipeline, 34-кейсный
golden dataset, A/B-сравнение chunking и retrieval, а также трассировку
LlamaIndex/OpenAI в локальный Phoenix.

## Установка eval-окружения

```powershell
python -m venv .venv-eval
.venv-eval\Scripts\python.exe -m pip install -r requirements.txt -r requirements-eval.txt -r requirements-tracing.txt
$env:PHOENIX_WORKING_DIR="$PWD\var\phoenix-eval"
.venv-eval\Scripts\python.exe scripts\verify_eval_setup.py
```

`arize-phoenix==11.38.0` и `arize-phoenix-evals>=2,<3` закреплены
вместе: Phoenix 20.x конфликтует с требуемой заданием веткой
`langchain-community<0.4`, а Phoenix Evals 3.x несовместим с API Phoenix 11.x.
Для `llama-index==0.14.24` используется
`openinference-instrumentation-llama-index>=4.4,<5`: ветка 3.x обращается к
удалённому модулю `llama_index.core.base.agent` и не может включить трейсинг.
`ContextPrecision` проверяет все контексты одного кейса одним structured call,
но применяет к его бинарным verdicts неизменённую формулу average precision
RAGAS. Два таких вызова могут выполняться параллельно
(`RAG_EVAL_CONTEXT_PRECISION_CONCURRENCY=2`); остальные этапы обрабатывают три
кейса одновременно. Ожидание очереди не входит в таймаут самой метрики.
Кастомная `Has Citation` оформлена через `@discrete_metric` и проверяет маркер
`[N]`, имя файла или конструкцию «согласно …» детерминированно: для факта
наличия ссылки LLM-судья не нужен.

## Датасеты

Автоматически сгенерированные кандидаты сохраняются отдельно и никогда не
подменяют проверенный набор:

```powershell
.venv-eval\Scripts\python.exe scripts\generate_testset.py --size 34 --allow-external-data-transfer
```

- `tests/eval/golden_dataset_raw.csv` — сырой результат TestsetGenerator;
- `tests/eval/golden_dataset.json` — вручную проверенные вопросы, ответы и
  эталонные контексты.

## Три сравнимых прогона

Надёжный live-процесс разделён на две фазы: сначала фиксируются ответы,
контексты и latency, затем judge дополняет именно этот CSV и сохраняет
checkpoint после каждой строки. Поэтому сетевой таймаут не запускает retrieval
повторно и не меняет входы эксперимента.

```powershell
# baseline: recursive 512/64, top_k=10
.venv-eval\Scripts\python.exe scripts\run_eval.py --label baseline_inputs `
  --collect-only --allow-external-data-transfer
.venv-eval\Scripts\python.exe scripts\complete_eval_metrics.py `
  --input tests/eval/results/<baseline_inputs>.csv --label baseline `
  --allow-external-data-transfer

# retrieval A/B: тот же индекс, top_k=5
.venv-eval\Scripts\python.exe scripts\run_eval.py --label top_k_5_inputs `
  --top-k 5 --collect-only --allow-external-data-transfer
.venv-eval\Scripts\python.exe scripts\complete_eval_metrics.py `
  --input tests/eval/results/<top_k_5_inputs>.csv --label top_k_5 `
  --allow-external-data-transfer

# chunking A/B: отдельная коллекция и docstore, 1024/64
.venv-eval\Scripts\python.exe scripts\run_eval.py --label chunk_1024_inputs `
  --chunk-size 1024 --chunk-overlap 64 `
  --collection rag_eval_chunk_1024 --docstore var/rag_eval_chunk_1024_docstore.json `
  --ingest --collect-only --allow-external-data-transfer
.venv-eval\Scripts\python.exe scripts\complete_eval_metrics.py `
  --input tests/eval/results/<chunk_1024_inputs>.csv --label chunk_1024 `
  --allow-external-data-transfer
```

Каждый прогон пишет timestamped CSV с результатом по каждому вопросу и JSON с
конфигурацией и средними метриками. Один вызов retrieval возвращает и ответ, и
полные `retrieved_contexts`; повторного поиска с возможным расхождением нет.
При `--reuse-results` повторно используются только метрики с полностью
совпавшими входами (вопрос, ответ, эталон и контексты); изменившиеся варианты
по-прежнему оцениваются судьёй. Для Answer Relevancy применяется один
перефразированный вопрос (`RAG_EVAL_ANSWER_RELEVANCY_STRICTNESS=1`), чтобы
полный учебный прогон оставался практически выполнимым.

Если внешний judge прервал отдельные metric cells по таймауту, та же команда
принимает последний checkpoint и восстанавливает только пустые ячейки без
повторного retrieval. `--force-metric context_precision` явно пересчитывает
выбранную метрику после изменения её реализации:

```powershell
.venv-eval\Scripts\python.exe scripts\complete_eval_metrics.py `
  --input tests/eval/results/<run>.csv --label <run>_complete `
  --allow-external-data-transfer
```

Скрипт после каждой строки перезаписывает timestamped CSV, поэтому уже
завершённые оценки сохраняются и при последующем сетевом сбое.

## Phoenix

```powershell
docker compose up -d --build --wait app bot phoenix
```

Phoenix доступен на <http://localhost:6006>. Проект трасс задаётся
`PHOENIX_PROJECT_NAME` (по умолчанию `support-copilot-rag`). Трассируются
LlamaIndex retrieval и вызовы OpenAI; `TRACING_ENABLED=false` полностью
отключает instrumentation.

После не менее 20 запросов полнота дерева трасс (retriever, embedding и LLM)
проверяется автоматически:

```powershell
.venv-eval\Scripts\python.exe scripts\verify_phoenix_traces.py --minimum 20
```

Итоги экспериментов и failure analysis находятся в
[`docs/rag_evaluation.md`](../docs/rag_evaluation.md).
