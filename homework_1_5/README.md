# Домашнее задание 1.5: open-source модели HuggingFace

Задача: исследовать HuggingFace Hub и сделать рабочее demo для одной NLP-задачи.

## Выбранная задача

Задача: `sentiment analysis` для русских отзывов и обращений пользователей
Support Copilot.

Почему выбрано это направление:

- тональность помогает быстро находить раздраженные и критичные обращения;
- результат можно использовать как дополнительный сигнал для приоритета и
  эскалации;
- задачу легко проверить на небольшом наборе размеченных примеров.

## Тестовый набор

Файл: [`sentiment_cases.json`](sentiment_cases.json).

Подготовлено 10 русских support-примеров с метками:

- `positive`;
- `neutral`;
- `negative`.

## Поиск моделей через HuggingFace Hub

Скрипт [`hf_model_search.py`](hf_model_search.py) использует `HfApi.list_models()`
для поиска моделей по задаче `text-classification` и запросу `sentiment russian`.

Результат поиска сохранен в:

- `results/hf_model_search.json`.

Из найденных и проверенных моделей для benchmark выбраны:

| Модель | Причина выбора | Лицензия по проверке |
| --- | --- | --- |
| `cointegrated/rubert-tiny-sentiment-balanced` | маленькая RuBERT-модель для коротких русских текстов | не указана в model card metadata |
| `blanchefort/rubert-base-cased-sentiment` | RuBERT sentiment baseline для русских текстов | не указана в model card metadata |
| `tabularisai/multilingual-sentiment-analysis` | multilingual sentiment-модель, поддерживает customer feedback и product reviews | `cc-by-nc-4.0` |

Источники model card:

- https://huggingface.co/cointegrated/rubert-tiny-sentiment-balanced
- https://huggingface.co/blanchefort/rubert-base-cased-sentiment
- https://huggingface.co/tabularisai/multilingual-sentiment-analysis

## Benchmark

Запуск:

```powershell
python .\homework_1_5\hf_model_search.py
python .\homework_1_5\benchmark_hf_sentiment.py
```

Результаты:

- `results/benchmark_results.json` - полные предсказания моделей;
- `results/benchmark_report.md` - итоговая таблица;
- `results/hf_model_search.json` - результат поиска через HuggingFace Hub.

Итоговая таблица:

| Провайдер | Модель | Лицензия | Accuracy | Средняя latency |
| --- | --- | --- | ---: | ---: |
| HuggingFace Pipeline | `cointegrated/rubert-tiny-sentiment-balanced` | not specified | 0.600 | 0.0111 сек |
| HuggingFace Pipeline | `blanchefort/rubert-base-cased-sentiment` | not specified | 0.600 | 0.0329 сек |
| HuggingFace Pipeline | `tabularisai/multilingual-sentiment-analysis` | cc-by-nc-4.0 | 0.900 | 0.0208 сек |
| OpenRouter | `openai/gpt-oss-120b:free` | n/a | 1.000 | 3.2853 сек |

## Сравнение с облачным API

Облачный baseline: OpenRouter `openai/gpt-oss-120b:free`.

На тестовом наборе cloud baseline дал `accuracy=1.000`, но средняя latency была
примерно `3.2853 сек`, то есть существенно выше, чем у локальных pipeline-моделей
после загрузки в память. Для встроенного быстрого сигнала тональности в
Support Copilot лучше подходит HuggingFace pipeline, а облачный LLM можно
использовать как quality baseline или fallback для сложных случаев.

## Gradio demo

Файл: [`gradio_demo.py`](gradio_demo.py).

Для Gradio выбрана лучшая HuggingFace-модель по accuracy:
`tabularisai/multilingual-sentiment-analysis`.

Демо использует отдельное окружение Python 3.13: его зависимости не следует
устанавливать поверх основного `uv`-окружения Support Copilot.

Запуск из корня проекта:

```powershell
uv venv --python 3.13 ..\tmp\hw15-venv
uv pip install --python ..\tmp\hw15-venv\Scripts\python.exe -r .\homework_1_5\requirements.txt
..\tmp\hw15-venv\Scripts\python.exe .\homework_1_5\gradio_demo.py
```

Демо загружает модель из локального кеша HuggingFace. На новой машине перед
первым offline-запуском нужно один раз выполнить benchmark или запустить demo с
доступом к интернету, чтобы модель скачалась.

После запуска Gradio покажет локальную ссылку вида:

```text
http://127.0.0.1:7860
```

Историческая локальная проверка от 16.06.2026 относилась к прежним версиям
зависимостей. Для обновлённого набора от 28.09.2026 подтверждено разрешение
зависимостей под Python 3.13; загрузка модели и HTTP smoke test пока не
выполнялись.

Демо слушает только `127.0.0.1`. Публичную ссылку включать только после
отдельного решения о доступе и допустимых данных.

## Вывод

Для локального demo выбрана `tabularisai/multilingual-sentiment-analysis`: она
дала лучший результат среди HuggingFace pipeline-моделей (`accuracy=0.900`) и
осталась быстрой (`0.0208 сек` в среднем после загрузки). Важно учитывать
лицензию `cc-by-nc-4.0`: модель подходит для учебного некоммерческого demo, но
для коммерческого использования нужно выбрать модель с подходящей лицензией или
получить разрешение. OpenRouter показал лучший accuracy, но для простого сигнала
тональности он медленнее и требует внешнего API.
