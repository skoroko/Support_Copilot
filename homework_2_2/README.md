# Homework 2.2 - Prompt engineering

Цель: подготовить production-ready system prompt для Support Copilot и
проверить его автотестами без сетевых вызовов.

## Что реализовано

- `prompt_engineering_demo.py` содержит Jinja2-шаблон system prompt по РРФО.
- В prompt встроены 4 few-shot пары `вопрос -> ответ`.
- `tiktoken` считает длину prompt; полный few-shot prompt должен попадать в
  диапазон 300-800 токенов.
- Есть 6 автотестов `should_contain` / `should_not_contain`, включая prompt
  injection.
- Есть сравнение 3 запросов с few-shot и без few-shot.
- Бонус: параметризация через Jinja2 и безопасный режим краткого обоснования
  вместо раскрытия полного chain-of-thought.

## Подготовка

Из корня `HelpDesk_Assistant`:

```powershell
python -m pip install tiktoken jinja2
```

## Запуск

```powershell
python .\homework_2_2\prompt_engineering_demo.py --preview-chars 1200
```

Проверить prompt без few-shot:

```powershell
python .\homework_2_2\prompt_engineering_demo.py --no-few-shot --preview-chars 1200
```

Проверить режим краткого обоснования:

```powershell
python .\homework_2_2\prompt_engineering_demo.py --reasoning-mode brief_rationale --preview-chars 1200
```

## Результаты

Лог проверенного запуска сохранен в `results/test_run_log.txt`.

Тесты не вызывают внешний LLM API: используется детерминированный offline runner,
который проверяет формат, ожидаемые маркеры и защиту от prompt injection. Это
оставляет ДЗ воспроизводимым без API-ключей и без записи секретов в проект.
