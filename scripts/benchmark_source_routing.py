"""Portable public source-choice benchmark; validate/report are entirely offline.

Only `run --allow-paid-calls` can call providers. Local/off decisions are frozen
baseline observations, not a fresh corpus/BM25 evaluation. No RAG is built and
no support answers are generated. Credentials are read only from named env vars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests/eval/source_routing.json"
MODES = ("lexical", "off", "decisions", "responses")


class BenchmarkError(ValueError):
    """An experiment invariant failed; error details never include credentials."""


def encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def payload(fixture: dict, case: dict) -> dict:
    candidates = [deepcopy(fixture["candidates"][index]) for index in case["candidate_order"]]
    for candidate in candidates:
        candidate["description"] = case.get("description_overrides", {}).get(candidate["id"], candidate["description"])
    return {"question": case["question"],
            "candidates": candidates}


def pair_order(repeat: int, case_number: int) -> list[str]:
    return ["decisions", "responses"] if (repeat + case_number) % 2 == 0 else ["responses", "decisions"]


def validate_fixture(fixture: dict) -> None:
    if (fixture.get("schema_version") != 1 or fixture.get("experiment") != "public-source-routing-v1"
            or digest({key: value for key, value in fixture.items() if key != "fixture_sha256"}) != fixture.get("fixture_sha256")):
        raise BenchmarkError("FixtureHashMismatch")
    candidates, cases = fixture["candidates"], fixture["cases"]
    if not 1 <= len(candidates) <= 100 or not 1 <= len(cases) <= 100:
        raise BenchmarkError("FixtureBounds")
    identities = set()
    for row in candidates:
        if (set(row) != {"id", "description"} or not isinstance(row["id"], str) or not row["id"].strip()
                or row["id"].casefold() == "none" or row["id"] in identities
                or not isinstance(row["description"], str) or len(row["description"]) > 1500):
            raise BenchmarkError("CandidateContract")
        identities.add(row["id"])
    case_ids = set()
    for case in cases:
        order = case["candidate_order"]
        if (not isinstance(case["id"], str) or case["id"] in case_ids
                or not isinstance(case["question"], str) or not case["question"].strip()
                or len(case["question"]) > 40_000
                or type(case["semantic_invoked"]) is not bool
                or type(case["requires_zero_anchor_fallback"]) is not bool
                or not isinstance(order, list) or len(order) != len(set(order))
                or any(type(index) is not int or not 0 <= index < len(candidates) for index in order)
                or case["expected_id"] is not None and case["expected_id"] not in identities
                or case["lexical_selected_id"] is not None and case["lexical_selected_id"] not in identities):
            raise BenchmarkError("CaseContract")
        case_ids.add(case["id"])
        overrides = case.get("description_overrides", {})
        if (not isinstance(overrides, dict)
                or any(identity not in {candidates[index]["id"] for index in order}
                       or not isinstance(description, str) or len(description) > 1500
                       for identity, description in overrides.items())):
            raise BenchmarkError("DescriptionOverrideContract")
        if case["semantic_invoked"]:
            shared = encoded(payload(fixture, case))
            if not order or len(shared.decode("utf-8")) > 40_000 or hashlib.sha256(shared).hexdigest() != case["input_sha256"]:
                raise BenchmarkError("SharedInputMismatch")
            if case["lexical_selected_id"] is not None:
                raise BenchmarkError("FallbackContract")
        elif order or case["input_sha256"] is not None:
            raise BenchmarkError("LexicalSkipContract")
        if case["requires_zero_anchor_fallback"] and not case["semantic_invoked"]:
            raise BenchmarkError("FallbackContract")
    if fixture["recorded_results"] is not None:
        validate_results(fixture, fixture["recorded_results"])


def validate_results(fixture: dict, results: dict) -> None:
    if not isinstance(results, dict):
        raise BenchmarkError("ResultsRequired")
    if results.get("fixture_sha256", fixture["fixture_sha256"]) != fixture["fixture_sha256"]:
        raise BenchmarkError("ResultFixtureMismatch")
    repeats = results["repeats"]
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise BenchmarkError("RepeatBounds")
    cases = {case["id"]: (number, case) for number, case in enumerate(fixture["cases"])}
    seen = set()
    actual_order = []
    for row in results["rows"]:
        number, case = cases[row["id"]]
        key = (row["repeat"], row["id"], row["mode"])
        if (key in seen or type(row["repeat"]) is not int or not 0 <= row["repeat"] < repeats
                or row["mode"] not in MODES or row["pair_order"] != pair_order(row["repeat"], number)
                or row["input_sha256"] != case["input_sha256"]
                or row["human_expected_id"] != case["expected_id"]):
            raise BenchmarkError("ResultPairMismatch")
        seen.add(key)
        actual_order.append(key)
        called = row["mode"] in {"decisions", "responses"} and case["semantic_invoked"]
        if row["provider_called"] is not called or called and row["cache"] != "miss":
            raise BenchmarkError("FreshCallContract")
        allowed = {item["id"] for item in payload(fixture, case)["candidates"]} if called else {case["lexical_selected_id"]}
        if row["selected_id"] is not None and row["selected_id"] not in allowed:
            raise BenchmarkError("InventedSelection")
        expected_correct = None if row["error_type"] else row["selected_id"] == case["expected_id"]
        if row["correct_choice"] is not expected_correct:
            raise BenchmarkError("IncorrectScoring")
        for field in ("duration_ms", "service_duration_ms"):
            value = row.get(field)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise BenchmarkError("InvalidTiming")
            if called and not row["error_type"] and value is None:
                raise BenchmarkError("InvalidTiming")
    if len(seen) != repeats * len(cases) * len(MODES):
        raise BenchmarkError("IncompleteResults")
    expected_order = [(repeat, case["id"], mode) for repeat in range(repeats)
                      for number, case in enumerate(fixture["cases"])
                      for mode in ("lexical", "off", *pair_order(repeat, number))]
    if actual_order != expected_order:
        raise BenchmarkError("CallOrderMismatch")


def load_fixture(path: Path) -> dict:
    if path.stat().st_size > 512_000:
        raise BenchmarkError("FixtureBounds")
    fixture = json.loads(path.read_text(encoding="utf-8"))
    validate_fixture(fixture)
    return fixture


def summarize(fixture: dict, results: dict) -> dict:
    validate_results(fixture, results)
    summary = {"fixture_sha256": fixture["fixture_sha256"], "model": results["model"],
               "repeats": results["repeats"], "source_choice_only": True, "modes": []}

    def timing(rows: list[dict], field: str):
        values = [row[field] / 1000 for row in rows if row.get(field) is not None and not row["error_type"]]
        return ({"median": round(statistics.median(values), 6), "min": round(min(values), 6),
                 "max": round(max(values), 6)} if values else None)

    for mode in MODES:
        rows = [row for row in results["rows"] if row["mode"] == mode]
        calls = [row for row in rows if row["provider_called"]]
        summary["modes"].append({"mode": mode, "rows": len(rows),
            "correct": sum(row["correct_choice"] is True for row in rows),
            "errors": sum(bool(row["error_type"]) for row in rows), "provider_calls": len(calls),
            "provider_correct": sum(row["correct_choice"] is True for row in calls),
            "lexical_skips": len(rows) - len(calls) if mode in {"decisions", "responses"} else None,
            "service_seconds": timing(calls, "service_duration_ms"),
            "total_client_seconds": timing(calls, "duration_ms")})
    pairs = {}
    for row in results["rows"]:
        if row["provider_called"] and not row["error_type"]:
            pairs.setdefault((row["repeat"], row["id"]), {})[row["mode"]] = row
    complete = [pair for pair in pairs.values() if set(pair) == {"decisions", "responses"}]
    summary["paired_service_timing"] = {"pairs": len(complete), "decisions_faster": sum(
        pair["decisions"]["service_duration_ms"] < pair["responses"]["service_duration_ms"] for pair in complete)}
    return summary


def run_benchmark(fixture: dict, *, model: str, base_url: str, api_key: str, repeats: int = 2,
                  service_factory=None, checkpoint=None) -> dict:
    """Explicit live entry point; a new selector/client is closed per call."""
    validate_fixture(fixture)
    if not model.strip() or not api_key or type(repeats) is not int or not 1 <= repeats <= 20:
        raise BenchmarkError("RunConfiguration")
    if service_factory is None:
        sys.path.insert(0, str(ROOT))
        from app.services.table_selection import TableSelectionService
        service_factory = TableSelectionService
    results = {"fixture_sha256": fixture["fixture_sha256"], "model": model, "timeout_seconds": 5.,
               "repeats": repeats, "baseline": "captured original-question-only local/off decisions",
               "selector_sha256": hashlib.sha256((ROOT / "app/services/table_selection.py").read_bytes()).hexdigest(),
               "rows": []}
    for repeat in range(repeats):
        for number, case in enumerate(fixture["cases"]):
            order = pair_order(repeat, number)
            for mode in ("lexical", "off", *order):
                called = mode in {"decisions", "responses"} and case["semantic_invoked"]
                report = {"error_type": None, "cache": "not_used"}
                duration = None
                selected = case["lexical_selected_id"]
                if called:
                    started = time.perf_counter()
                    service = service_factory(base_url=base_url, api_key=api_key, mode=mode,
                                              model=model, timeout_seconds=5.)
                    try:
                        data = payload(fixture, case)
                        selected = service.select(data["question"], data["candidates"], diagnostics=report)
                        if report.get("input_sha256") != case["input_sha256"] or report.get("cache") != "miss":
                            raise BenchmarkError("FreshInputContract")
                        if selected is not None and selected not in {row["id"] for row in data["candidates"]}:
                            raise BenchmarkError("InventedSelection")
                    finally:
                        service.close()
                    duration = round((time.perf_counter() - started) * 1000, 3)
                results["rows"].append({"id": case["id"], "repeat": repeat, "mode": mode,
                    "pair_order": order, "duration_ms": duration,
                    "service_duration_ms": report.get("duration_ms"), "error_type": report.get("error_type"),
                    "selected_id": selected, "human_expected_id": case["expected_id"],
                    "correct_choice": None if report.get("error_type") else selected == case["expected_id"],
                    "input_sha256": case["input_sha256"], "cache": report.get("cache"), "provider_called": called})
                if checkpoint is not None:
                    checkpoint(results)
    validate_results(fixture, results)
    return results


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "report", "run"))
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--results", type=Path, help="Report a completed run instead of bundled public observations")
    parser.add_argument("--output", type=Path, help="New output JSON for run; existing files are refused")
    parser.add_argument("--model", help="Explicit source-selector model for a prospective run")
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--allow-paid-calls", action="store_true")
    args = parser.parse_args(argv)
    previous_logging_threshold = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        fixture = load_fixture(args.fixture)
        if args.command == "validate":
            output = {"valid": True, "fixture_sha256": fixture["fixture_sha256"],
                      "cases": len(fixture["cases"]), "candidates": len(fixture["candidates"])}
        elif args.command == "report":
            results = json.loads(args.results.read_text(encoding="utf-8")) if args.results else fixture["recorded_results"]
            if args.results and results.get("fixture_sha256") != fixture["fixture_sha256"]:
                raise BenchmarkError("ResultFixtureMismatch")
            output = summarize(fixture, results)
        else:
            if not args.allow_paid_calls:
                raise BenchmarkError("PaidCallsNotEnabled")
            if not args.output or args.output.exists() or not args.model:
                raise BenchmarkError("NewOutputAndModelRequired")
            results = run_benchmark(fixture, model=args.model,
                base_url=os.environ.get(args.base_url_env) or "https://api.openai.com/v1",
                api_key=os.environ.get(args.api_key_env, ""), repeats=args.repeats,
                checkpoint=lambda value: save(args.output, value))
            output = summarize(fixture, results)
        print(json.dumps(output, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI errors must never expose provider messages or credentials
        # Never print exception messages, HTTP bodies, env values or tracebacks.
        print(json.dumps({"error_type": type(exc).__name__}))
        return 1
    finally:
        logging.disable(previous_logging_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
