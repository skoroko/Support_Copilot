# Краткая сводка для проверки ДЗ 1.5

В этой папке выполнено задание `1-block-05-opensource-huggingface-homework.pdf`:
исследование HuggingFace Hub, benchmark 3 моделей через pipeline, сравнение с
облачным API и Gradio demo.

## Что смотреть

- `README.md` - основной отчет с выбором задачи, таблицей моделей, benchmark и выводом.
- `sentiment_cases.json` - 10 тестовых примеров.
- `hf_model_search.py` - поиск моделей через `HfApi.list_models()`.
- `benchmark_hf_sentiment.py` - benchmark HuggingFace pipeline и OpenRouter.
- `gradio_demo.py` - Gradio demo для лучшей HF-модели.
- `requirements.txt` - зависимости для воспроизведения benchmark и Gradio demo.
- `results/hf_model_search.json` - результат поиска моделей на HuggingFace Hub.
- `results/benchmark_results.json` - все предсказания и latency.
- `results/benchmark_report.md` - итоговая таблица accuracy/latency.

## Что отправить на проверку

Минимально отправить:

1. `homework_1_5/README.md`;
2. `homework_1_5/sentiment_cases.json`;
3. `homework_1_5/results/benchmark_report.md`;
4. `homework_1_5/gradio_demo.py`;
5. `homework_1_5/requirements.txt`.

Если нужен полный воспроизводимый набор, добавить:

- `homework_1_5/hf_model_search.py`;
- `homework_1_5/benchmark_hf_sentiment.py`;
- `homework_1_5/results/benchmark_results.json`;
- `homework_1_5/results/hf_model_search.json`.

Не отправлять `.env` и локальные кеши моделей.
