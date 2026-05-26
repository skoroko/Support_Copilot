# Homework 1.3 - Ollama Local Models

Цель: проверить локальные модели Ollama для учебного проекта Support Copilot и
выбрать маленькую модель для сценариев техподдержки.

## Выбор модели

Основной кандидат: `qwen3:1.7b`.

Почему он выбран:

- маленький размер: около 1.4 GB в Ollama;
- семейство Qwen хорошо подходит для мультиязычных задач и русского языка;
- модель достаточно легкая для ноутбука с 32 GB RAM и RTX 2060 Max-Q 4 GB;
- подходит для классификации обращений, кратких ответов и structured output.

Модели для сравнения:

- `qwen3:1.7b` - основной кандидат;
- `smollm2:1.7b` - очень компактная модель 1.7B для on-device задач;
- `gemma3:1b` - ультралегкая модель 1B.

## Установка Ollama и моделей

1. Установить Ollama с https://ollama.com/.
2. Перезапустить терминал, чтобы команда `ollama` появилась в `PATH`.
3. Скачать модели:

```powershell
ollama pull qwen3:1.7b
ollama pull smollm2:1.7b
ollama pull gemma3:1b
```

4. Создать кастомную модель для проекта:

```powershell
cd "C:\00.MIDAS\2026\DEV\PROJECTS\19. HelpDesk_Assistant\HelpDesk_Assistant\homework_1_3"
ollama create support-copilot-qwen3-1.7b -f .\Modelfile
```

## Запуск benchmark

Скрипт использует HTTP API Ollama, поэтому дополнительных Python-зависимостей
не требует.

```powershell
python .\benchmark_ollama.py
```

Запуск только structured output:

```powershell
python .\benchmark_ollama.py --structured-only
```

Запуск с явным списком моделей:

```powershell
python .\benchmark_ollama.py --models qwen3:1.7b smollm2:1.7b gemma3:1b
```

## Что измеряется

Benchmark проверяет 3 задачи:

1. `classification` - классификация обращения техподдержки;
2. `summary` - краткая суммаризация обращения;
3. `escalation` - определение необходимости эскалации.

Для каждой модели считается:

- `latency_sec` - полное время ответа;
- `approx_chars_per_sec` - грубая скорость генерации по символам;
- `quality_score` - простая автоматическая оценка по ключевым словам;
- `answer_preview` - начало ответа для ручной проверки.

Structured output возвращает JSON с полями:

- `category`: `faq`, `technical`, `complaint` или `escalation`;
- `priority`: `low`, `medium`, `high` или `critical`;
- `summary`: краткое резюме обращения;
- `needs_human`: нужен ли специалист;
- `suggested_action`: рекомендуемое действие.

## Ожидаемый вывод

Если Ollama не запущена, скрипт завершится с понятной ошибкой и подскажет
команды установки/запуска. Если модели не скачаны, скрипт покажет команду
`ollama pull <model>`.

Для сдачи ДЗ нужно запустить benchmark локально и приложить скриншот или лог
вывода.
