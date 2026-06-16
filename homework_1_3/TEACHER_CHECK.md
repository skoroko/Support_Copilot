# Краткая сводка для проверки ДЗ 1.3

## Что сделано

- Установлены/проверены локальные модели Ollama:
  - `qwen3:1.7b`;
  - `smollm2:1.7b`;
  - `gemma3:1b`;
  - кастомная `support-copilot-qwen3-1.7b:latest` из `Modelfile`.
- Реализован Python benchmark на 3 задачах:
  - classification;
  - summary;
  - escalation.
- Создан `Modelfile` с system prompt для Support Copilot.
- Реализован structured output для классификации обращения в JSON.
- Результаты сохранены в `results/`.

## Что смотреть

- `benchmark_ollama.py` - основной Python-скрипт.
- `Modelfile` - кастомная Ollama-модель с system prompt проекта.
- `results/benchmark_log.txt` - лог запуска benchmark.
- `results/benchmark_results.json` - JSON-результаты benchmark и structured output.
- `results/benchmark_summary.md` - итоговая таблица и вывод.
