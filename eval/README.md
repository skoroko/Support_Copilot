# Evaluation layer for Support Copilot

`eval/` is separate from `tests/`: unit tests stay fast and offline, while
evaluation runs are slower and should be started manually after prompt or model
changes.

## Golden dataset

`golden_dataset.json` contains versioned Support Copilot quality cases:

- 22 isolated items;
- stable `id` values for regression discussion;
- support, factual, billing, security, escalation and offtopic categories;
- hard cases for prompt injection, PII, urgent incidents and unsafe requests.

## Real LLM-as-judge run

Start the FastAPI service first:

```powershell
uvicorn app.main:app --port 8000
```

Then run evaluation with a judge model:

```powershell
python eval/run_evaluation.py --golden eval/golden_dataset.json --judge gpt-5.2 --out eval/runs/2026-07-07.json
```

The runner calls `/chat` with `temperature=0`, then calls the judge with
`temperature=0` and `response_format={"type": "json_object"}`. The judge prompt
uses G-Eval style reason-then-score and requires `reasoning` before `scores`.

## Offline smoke run

This mode does not replace LLM-as-judge evaluation. It only verifies file IO,
aggregation and threshold plumbing without API keys or network:

```powershell
python eval/run_evaluation.py --offline-smoke --out eval/runs/2026-07-07.json
python eval/check_thresholds.py eval/runs/2026-07-07.json
```

## Release threshold check

```powershell
python eval/check_thresholds.py
```

Default thresholds live in `thresholds.yaml`:

- `correctness_avg >= 4.0`
- `min_correctness >= 2.0`
