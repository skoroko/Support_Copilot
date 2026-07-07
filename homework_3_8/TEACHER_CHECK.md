# Краткая сводка для проверки ДЗ 3.8

## Что добавлено

- `app/services/security/input_validator.py` - длина, blocklist prompt injection, base64/encoding и Unicode эвристики.
- `app/services/security/output_filter.py` - canary, system prompt leakage, markdown image/XSS markers, PII redaction.
- `app/main.py` - генерация `CANARY_<hex>` при старте и хранение security system prompt в `app.state`.
- `app/routers/chat.py` - входная проверка до LLM, добавление security system prompt в `/chat` и `/chat/stream`.
- `app/services/llm.py` - output filter до записи в cache, безопасный `answer_preview`.
- `app/observability/pii.py` - добавлен доменный placeholder `[ORDER_ID]`.
- `eval/security/rest_config.json` - garak REST target под фактический `/chat`.
- `docs/security/garak_baseline_2026-07-07.md` и `docs/security/garak_after_2026-07-07.md`.
- `docs/security/reports/{baseline,after}/` - места для HTML-отчётов garak.
- `tests/unit/test_security_layer.py` - unit-тесты input/output security и canary leak.

## Команды проверки

```powershell
python -m pytest
python -m compileall app eval scripts homework_3_8 tests
```

## Команды garak

```powershell
python -m pip install -U garak
garak --version
garak --list_probes
garak --target_type rest -G eval/security/rest_config.json --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 --generations 1 --report_prefix baseline
garak --target_type rest -G eval/security/rest_config.json --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 --generations 1 --report_prefix after
```

## Ограничение

В текущей Codex-среде `garak --version` вернул command not found, поэтому реальные
HTML/JSONL garak-артефакты не были получены здесь. Отчёты закоммичены как
reproducible templates; перед финальной сдачей их нужно заменить локальными
результатами garak-прогона.
