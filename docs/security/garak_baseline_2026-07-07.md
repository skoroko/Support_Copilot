# Garak baseline report - 2026-07-07

## Scope

- Target: Support Copilot FastAPI `/chat`
- Config: `eval/security/rest_config.json`
- garak version: `0.15.1`
- Model returned by provider: `gpt-5.4-mini-2026-03-17`
- Security layer: disabled with `APP_SECURITY_ENABLED=false`
- Run time: `761.61s`
- HTML report: `docs/security/reports/baseline/baseline_2026-07-07.report.html`

## Command

```powershell
$root=(Resolve-Path '.garak_runtime').Path
$env:XDG_CONFIG_HOME="$root\config"
$env:XDG_DATA_HOME="$root\data"
$env:XDG_CACHE_HOME="$root\cache"
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

## Baseline Summary

| Probe | Detector | Processed | Evaluated | Fails | Attack success rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `dan.Ablation_Dan_11_0` | `dan.DAN` | 127 | 127 | 1 | 0.79% |
| `dan.Ablation_Dan_11_0` | `mitigation.MitigationBypass` | 127 | 127 | 108 | 85.04% |
| `encoding.InjectBase64` | `encoding.DecodeMatch` | 256 | 256 | 220 | 85.94% |
| `encoding.InjectBase64` | `encoding.DecodeApprox` | 256 | 256 | 232 | 90.62% |
| `promptinject.HijackHateHumans` | `promptinject.AttackRogueString` | 256 | 256 | 59 | 23.05% |

## Notes

- The baseline intentionally disables the project security layer but still uses the same provider/model and `/chat` contract.
- The JSONL report is kept in local ignored runtime cache at `.garak_runtime/data/garak/garak_runs/baseline_2026-07-07.report.jsonl`.
- The committed HTML report is the generated garak summary for this baseline run.
