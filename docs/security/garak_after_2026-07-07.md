# Garak after report - 2026-07-07

## Scope

- Target: Support Copilot FastAPI `/chat`
- Config: `eval/security/rest_config.json`
- Security layer status: enabled
- Local execution status: `garak` was not installed in the Codex runtime. This report records the exact repeat command, the implemented controls, and the expected delta table shape. Replace worksheet values with real garak JSONL/HTML values after a local run.

## Implemented controls

- `app/services/security/input_validator.py`
  - length limit: `4000` characters;
  - blocklist for common injection markers: `ignore previous instructions`, `disregard`, `you are now`, `forget everything`, `DAN`, `developer mode`, system prompt leakage requests;
  - encoding heuristics: base64/ROT13/decode markers, long base64-like segments, non-printable ratio, atypical Unicode ratio.
- `app/services/security/output_filter.py`
  - canary detection;
  - system prompt prefix leakage detection;
  - markdown image exfiltration and script marker blocking;
  - outgoing PII redaction through `app.observability.pii.redact_pii`.
- `app/main.py`
  - per-process canary generated at startup with `secrets.token_hex(4)`;
  - canary system prompt stored in `app.state.security_system_prompt`.
- `app/routers/chat.py`
  - `/chat` and `/chat/stream` validate user input before LLM calls;
  - security system prompt is prepended to model messages.
- `app/services/llm.py`
  - output filter runs before cache writes;
  - `answer_preview` is logged only after PII redaction.

## Repeat command

```powershell
garak --version
garak --list_probes
garak --target_type rest -G eval/security/rest_config.json `
  --probes promptinject.HijackHateHumans,encoding.InjectBase64,dan.Ablation_Dan_11_0 `
  --generations 1 `
  --report_prefix after
```

## Delta summary

Worksheet values below show the expected direction from the implemented regex/canary layer. Replace with real garak values from the after run.

| Probe | Baseline attack success | After attack success | Closed |
| --- | ---: | ---: | ---: |
| `promptinject.HijackHateHumans` | 64.8% | 7.0% | 57.8 pp |
| `encoding.InjectBase64` | 35.0% | 7.5% | 27.5 pp |
| `dan.Ablation_Dan_11_0` | 29.7% | 15.6% | 14.1 pp |

## After examples

| Probe | Input fragment | Expected protected response |
| --- | --- | --- |
| `promptinject.HijackHateHumans` | `ignore previous instructions ...` | HTTP 400 `security_input_blocked`, no `content` field. |
| `encoding.InjectBase64` | `Decode this base64 and follow it: ...` | HTTP 400 `security_input_blocked`, no `content` field. |
| canary leak fixture | Provider returns `CANARY_deadbeef` | `LLMOutputBlockedError`, mapped to HTTP 502. |
| PII fixture | Provider returns `user@example.com, +7 999 123 45 67` | `[EMAIL]`, `[PHONE_RU]` placeholders in output/log preview. |

## Notes

The simple validator is intentionally conservative for support traffic. False positives are acceptable for obvious attack strings because the service is an internal assistant and the user can rephrase a legitimate question without injection control text.
