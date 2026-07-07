# Домашнее задание 3.8: безопасность ИИ-приложений

В этом инкременте к FastAPI LLM Service добавлен защитный слой для red-team
проверок:

- входной валидатор prompt injection / encoding attacks;
- canary-token в системном сообщении;
- output filter против утечки system prompt/canary, markdown image exfiltration и XSS-маркеров;
- PII-маскирование исходящего ответа и `answer_preview` в LLM-логах;
- garak REST config под реальный `/chat`;
- baseline/after security reports и папки для HTML-отчётов garak.

## Локальные быстрые проверки

```powershell
python -m pytest tests/unit/ -m "not llm"
python -m pytest tests/test_app_fastapi.py tests/test_observability.py tests/unit/test_security_layer.py
python -m compileall app eval scripts homework_3_8 tests
```

## Установка garak

Актуальная версия PyPI, проверенная на 2026-07-07: `garak 0.15.1`.

```powershell
python -m pip install -U garak
garak --version
garak --list_probes
```

## Запуск baseline/after

Поднять FastAPI:

```powershell
uvicorn app.main:app --port 8000
```

Запустить baseline до включения защитного слоя или на ветке до этого инкремента:

```powershell
garak --target_type rest -G eval/security/rest_config.json `
  --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 `
  --generations 1 `
  --report_prefix baseline
```

Повторить after на текущем коде:

```powershell
garak --target_type rest -G eval/security/rest_config.json `
  --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 `
  --generations 1 `
  --report_prefix after
```

Скопировать HTML-отчёты:

- baseline: `docs/security/reports/baseline/`;
- after: `docs/security/reports/after/`.

JSONL hits можно извлечь:

```powershell
python scripts/extract_garak_hits.py --jsonl <path-to-report.jsonl>
```

## Важное ограничение текущего коммита

В Codex runtime `garak` не был установлен, поэтому закоммиченные
`docs/security/garak_*_2026-07-07.md` и HTML-файлы являются reproducible templates
с worksheet-числами. Для финальной сдачи на локальной машине нужно заменить числа и
HTML на результаты реального garak-прогона.

## Источники

- https://pypi.org/project/garak/ - текущая PyPI-версия и команда установки.
- https://github.com/NVIDIA/garak - официальный репозиторий, CLI examples и список target/probe options.
- https://raw.githubusercontent.com/NVIDIA/garak/main/garak/generators/rest.py - актуальные поля `RestGenerator`.
