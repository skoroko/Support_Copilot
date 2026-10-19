"""Publish timings and manually reviewed quality, without answers or manual text."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(private: Path) -> dict:
    manifest = read(private / "manifest.json")
    quality = read(private / "quality.json")
    rows = []
    for spec in manifest["schedule"]:
        path = private / f"sample_{spec['slot']:02}.json"
        sample = read(path)
        assert all(sample[k] == spec[k] for k in ("slot", "question", "repeat", "arm"))
        calls = sample["model_calls"]
        usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cache_read_tokens": 0}
        for call in calls:
            tokens = call.get("usage") or {}
            usage["input_tokens"] += tokens.get("input_tokens", 0)
            usage["output_tokens"] += tokens.get("output_tokens", 0)
            usage["reasoning_tokens"] += (tokens.get("output_token_details") or {}).get("reasoning", 0)
            usage["cache_read_tokens"] += (tokens.get("input_token_details") or {}).get("cache_read", 0)
        rows.append({**spec, "status": sample["status"], "error_type": sample.get("error_type"),
                     "total_seconds": sample.get("total_seconds"), "route": sample.get("route"),
                     "answer_incomplete": (sample.get("answer") or {}).get("answer_info", {}).get("incomplete"),
                     "model_calls": len(calls), "retrieval_calls": len(sample["retrievals"]),
                     "model_seconds": sum(c.get("seconds", 0) for c in calls),
                     "retrieval_seconds": sum(r["seconds"] for r in sample["retrievals"]),
                     "selection_seconds": sum(sample["selection_seconds"]),
                     "evidence_count": len(sample.get("citation_evidence") or []),
                     "evidence_chars": sum(len(e["text"]) for e in sample.get("citation_evidence") or []),
                     "served_tiers": [c.get("service_tier") for c in calls], "usage": usage,
                     "usage_incomplete": any(c["status"] != "completed" for c in calls),
                     "quality": quality[str(spec["slot"])],
                     "private_sample_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    groups = {}
    for arm in ("ordinary", "pilot"):
        selected = [r for r in rows if r["arm"] == arm]
        completed = [r for r in selected if r["status"] == "completed"]
        times = [r["total_seconds"] for r in completed]
        groups[arm] = {"attempted": len(selected), "completed": len(times),
                       "median_seconds": median(times) if times else None,
                       "range_seconds": [min(times), max(times)] if times else None,
                       "times_seconds": times,
                       "model_seconds_median": median(r["model_seconds"] for r in completed) if times else None,
                       "retrieval_seconds_median": median(r["retrieval_seconds"] for r in completed) if times else None,
                       "input_tokens_median": median(r["usage"]["input_tokens"] for r in completed) if times else None,
                       "reasoning_tokens_median": median(r["usage"]["reasoning_tokens"] for r in completed) if times else None,
                       "verified_count": sum(r["route"] == "verified" for r in selected),
                       "factually_correct": sum(r["quality"]["accuracy_pass"] for r in selected),
                       "required_completeness_pass": sum(r["quality"]["required_complete"] for r in selected),
                       "citation_pass": sum(r["quality"]["citations_pass"] for r in selected),
                       "model_calls": sum(r["model_calls"] for r in selected),
                       "retrieval_calls": sum(r["retrieval_calls"] for r in selected)}
    pairs = []
    for question in (case["id"] for case in manifest["cases"]):
        for repeat in range(3):
            before, after = [next(r for r in rows if (r["question"], r["repeat"], r["arm"])
                                  == (question, repeat, arm)) for arm in ("ordinary", "pilot")]
            complete = before["status"] == after["status"] == "completed"
            pairs.append({"question": question, "repeat": repeat, "both_completed": complete,
                          "pilot_faster": bool(complete and after["total_seconds"] < before["total_seconds"]),
                          "pilot_no_less_complete": all(a >= b for a, b in zip(
                              after["quality"]["facets"], before["quality"]["facets"], strict=True))
                          if complete else None})
    faster = sum(p["pilot_faster"] for p in pairs)
    ordinary_median, pilot_median = (groups[a]["median_seconds"] for a in ("ordinary", "pilot"))
    improvement = 1 - pilot_median / ordinary_median if ordinary_median and pilot_median else None
    pilot = [r for r in rows if r["arm"] == "pilot"]
    gate = {
        "all_samples_completed": all(r["status"] == "completed" for r in rows),
        "latency": improvement is not None and improvement >= .2 and faster >= 7,
        "verified_one_call": all(r["route"] == "verified" and r["model_calls"] == 1 for r in pilot),
        "complete_responses": all(r["answer_incomplete"] is False for r in pilot),
        "accuracy": all(r["quality"]["accuracy_pass"] for r in pilot),
        "completeness": all(r["quality"]["required_complete"] for r in pilot)
                        and all(p["pilot_no_less_complete"] for p in pairs),
        "citations": all(r["quality"]["citations_pass"] for r in pilot),
    }
    return {"manifest": manifest, "candidate_image": read(private / "image.json")
            if (private / "image.json").exists() else None,
            "samples": rows, "groups": groups, "pairs": pairs,
            "median_improvement_fraction": improvement, "pilot_faster_pairs": faster,
            "release_gate": gate, "release_allowed": all(gate.values()),
            "usage_scope": "Agent generation calls only; reasoning included in output tokens. Embeddings excluded; failed-call usage may be unknown.",
            "quality_method": "Independent manual source-based review of all answers; primary agent spot checks and release decision. Delivered marker to displayed source to chunk_id to complete evidence text.",
            "limitations": "Three HSS formulations, three repeats per arm; synthetic histories, not a general benchmark or SLA. Failures retained; no replacement attempts."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.private)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("groups", "median_improvement_fraction", "pilot_faster_pairs",
                                            "release_gate", "release_allowed")}, ensure_ascii=False, indent=2))
