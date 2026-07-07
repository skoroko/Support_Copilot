# Краткая сводка для проверки ДЗ 3.8

## Что добавлено

- `app/services/security/input_validator.py` - длина, blocklist prompt injection, base64/encoding и Unicode эвристики.
- `app/services/security/output_filter.py` - canary, system prompt leakage, markdown image/XSS markers, PII redaction.
- `app/main.py` - генерация `CANARY_<hex>`, security prompt в `app.state`, HTTP 400 для output security block.
- `app/routers/chat.py` - входная проверка до LLM, security prompt в `/chat` и `/chat/stream`, `APP_SECURITY_ENABLED=false` для baseline.
- `app/services/llm.py` - output filter до cache write, безопасный `answer_preview`, `max_completion_tokens` для `gpt-5*`.
- `app/observability/pii.py` - доменный placeholder `[ORDER_ID]`.
- `eval/security/rest_config.json` - garak REST target под фактический `/chat`.
- `docs/security/garak_baseline_2026-07-07.md` и `docs/security/garak_after_2026-07-07.md` - реальные числа из JSONL.
- `docs/security/reports/{baseline,after}/` - реальные HTML-отчеты garak.
- `tests/unit/test_security_layer.py` - unit-тесты input/output security и canary leak.

## Команды проверки

```powershell
python -m pytest
python -m compileall app eval scripts homework_3_8 tests
```

## Garak

Установлено и проверено:

```powershell
garak --version
# garak LLM vulnerability scanner v0.15.1
```

Baseline:

```powershell
$env:APP_SECURITY_ENABLED='false'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
garak --target_type rest.RestGenerator --generator_option_file eval/security/rest_config.json --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 --generations 1 --parallel_requests 1 --parallel_attempts 1 --confidence_interval_method none --report_prefix baseline_2026-07-07
```

After:

```powershell
Remove-Item Env:APP_SECURITY_ENABLED -ErrorAction SilentlyContinue
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
garak --target_type rest.RestGenerator --generator_option_file eval/security/rest_config.json --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 --generations 1 --parallel_requests 1 --parallel_attempts 1 --confidence_interval_method none --report_prefix after_2026-07-07
```

## Итог по garak

| Detector | Baseline fails / processed | After fails / processed | Skipped after |
| --- | ---: | ---: | ---: |
| `dan.DAN` | 1 / 127 | 0 / 127 | 127 |
| `mitigation.MitigationBypass` | 108 / 127 | 0 / 127 | 127 |
| `encoding.DecodeMatch` | 220 / 256 | 36 / 256 | 216 |
| `encoding.DecodeApprox` | 232 / 256 | 28 / 256 | 216 |
| `promptinject.AttackRogueString` | 59 / 256 | 0 / 256 | 256 |

Residual risk: base64 payloads. Prompt injection и DAN в текущем наборе garak закрыты входным security policy.
