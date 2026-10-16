"""Offline, message-free publication of the completed Standard/Fast experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "var/luna_fast_ab"
PUBLIC = ROOT / "tests/eval/results/2026-10-16_luna_fast"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    manifest = read(PRIVATE / "manifest.json")
    quality = read(PRIVATE / "quality.json")
    rows = []
    for slot in range(12):
        path = PRIVATE / f"sample_{slot:02}.json"
        sample = read(path)
        calls = sample["model_calls"]
        assert sample["status"] == "completed" and sample["tier_valid"]
        assert all(c["service_tier"] == sample["profile"] for c in calls)
        totals = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
        cost = 0
        for call in calls:
            usage = call["usage"]
            inp, out = usage["input_tokens"], usage["output_tokens"]
            detail = usage.get("input_token_details", {})
            cached, written = detail.get("cache_read", 0), detail.get("cache_creation", 0)
            assert 0 <= cached + written <= inp
            multiplier = 2 if call["service_tier"] == "fast" else 1
            # Reasoning is already included in output_tokens. Cache writes replace
            # ordinary input pricing; they are not an additional full charge.
            cost += multiplier * ((inp - cached - written) * .1 + cached * .01
                                  + written * .125 + out * .5) / 1_000_000
            for key, value in {"input": inp, "output": out, "cache_read": cached,
                               "cache_write": written,
                               "reasoning": usage.get("output_token_details", {}).get("reasoning", 0)}.items():
                totals[key] += value
        model_seconds = sum(c["seconds"] for c in calls)
        retrieval_seconds = sum(r["seconds"] for r in sample["retrievals"])
        rows.append({
            "slot": slot, "question": sample["question"], "repeat": sample["repeat"],
            "tier": sample["profile"], "total_seconds": sample["total_seconds"],
            "model_seconds": model_seconds, "retrieval_seconds": retrieval_seconds,
            "preparation_seconds": sum(sample["selection_seconds"]),
            "residual_seconds": sample["total_seconds"] - model_seconds - retrieval_seconds,
            "tokenizer_initialization_seconds": sample["tokenizer_initialization_seconds"],
            "rag_initialization_seconds": sample["rag_initialization_seconds"],
            "model_calls": len(calls), "retrieval_calls": len(sample["retrievals"]),
            "served_tiers": [c["service_tier"] for c in calls], "usage": totals,
            "estimated_model_cost_usd": cost, "quality": quality[str(slot)],
            "private_sample_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    grouped = []
    for question in ("0801f604b07c", "a7b5f7c44544"):
        for tier in ("default", "fast"):
            samples = [r for r in rows if (r["question"], r["tier"]) == (question, tier)]
            times = [s["total_seconds"] for s in samples]
            grouped.append({"question": question, "tier": tier, "times": times,
                            "median": median(times), "range": [min(times), max(times)],
                            "model_median": median(s["model_seconds"] for s in samples),
                            "cost_usd": sum(s["estimated_model_cost_usd"] for s in samples),
                            "useful": sum(s["quality"]["useful"] for s in samples)})
    summary = {"manifest": manifest, "samples": rows, "groups": grouped,
               "cost_scope": "All agent Responses calls; excludes embedding and infrastructure costs",
               "pricing_date": "2026-10-16", "pricing_url": "https://developers.openai.com/api/docs/pricing",
               "failed_attempts": [p.name for p in PRIVATE.glob("attempt_*.json") if read(p)["status"] != "completed"],
               "limitations": "Two questions, three repetitions per tier; not an SLA or significance claim"}
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(grouped, indent=2))


if __name__ == "__main__":
    main()
