# Домашнее задание 3.8: безопасность ИИ-приложений

В этом инкременте к FastAPI LLM Service добавлен защитный слой для red-team
проверок и подготовлены реальные garak baseline/after отчеты.

## Что реализовано

- входной валидатор prompt injection / encoding attacks;
- canary-token в системном сообщении;
- output filter против утечки system prompt/canary, markdown image exfiltration и XSS-маркеров;
- PII-маскирование исходящего ответа и `answer_preview` в LLM-логах;
- `APP_SECURITY_ENABLED=false` для воспроизводимого baseline-прогона без защиты;
- совместимость OpenAI chat completions с `gpt-5*` через `max_completion_tokens`;
- garak REST config под реальный `/chat`;
- реальные baseline/after HTML reports в `docs/security/reports/`;
- markdown-отчеты с числами из JSONL.

## Проверки

```powershell
python -m pytest
python -m compileall app eval scripts homework_3_8 tests
```

## Установка и запуск garak

Установленная версия:

```powershell
python -m pip install -U garak
garak --version
# garak LLM vulnerability scanner v0.15.1
```

Для Windows использовался локальный runtime-каталог, чтобы garak не писал в
пользовательский `%USERPROFILE%\.config`:

```powershell
$root=(Resolve-Path '.garak_runtime').Path
$env:XDG_CONFIG_HOME="$root\config"
$env:XDG_DATA_HOME="$root\data"
$env:XDG_CACHE_HOME="$root\cache"
```

Baseline:

```powershell
$env:APP_SECURITY_ENABLED='false'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

garak --target_type rest.RestGenerator `
  --generator_option_file eval/security/rest_config.json `
  --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 `
  --generations 1 `
  --parallel_requests 1 `
  --parallel_attempts 1 `
  --confidence_interval_method none `
  --report_prefix baseline_2026-07-07
```

After:

```powershell
Remove-Item Env:APP_SECURITY_ENABLED -ErrorAction SilentlyContinue
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

garak --target_type rest.RestGenerator `
  --generator_option_file eval/security/rest_config.json `
  --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 `
  --generations 1 `
  --parallel_requests 1 `
  --parallel_attempts 1 `
  --confidence_interval_method none `
  --report_prefix after_2026-07-07
```

## Результаты garak

Baseline:

- `dan.DAN`: 1 / 127 fails
- `mitigation.MitigationBypass`: 108 / 127 fails
- `encoding.DecodeMatch`: 220 / 256 fails
- `encoding.DecodeApprox`: 232 / 256 fails
- `promptinject.AttackRogueString`: 59 / 256 fails

After:

- DAN probes: 0 fails, 127 / 127 prompts skipped by input security policy
- prompt injection rogue-string probe: 0 fails, 256 / 256 prompts skipped
- base64 probe: 36 / 256 and 28 / 256 fails depending on detector; 216 / 256 prompts skipped

Основной остаточный риск: короткие или обычные base64 payloads, которые не выглядят
как явная инструкция. Для следующего инкремента нужно решить, должен ли Support
Copilot полностью запрещать base64-like пользовательский ввод или декодировать его
в отдельном безопасном preprocessing-контуре.

## Артефакты

- `docs/security/garak_baseline_2026-07-07.md`
- `docs/security/garak_after_2026-07-07.md`
- `docs/security/reports/baseline/baseline_2026-07-07.report.html`
- `docs/security/reports/after/after_2026-07-07.report.html`
- `eval/security/rest_config.json`
- `scripts/extract_garak_hits.py`
