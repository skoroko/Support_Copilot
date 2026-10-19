"""Metered, journaled evaluation inside the isolated candidate container.

Reuses the production eval adapters. Each provider attempt is checkpointed;
existing/uncertain runs are never implicitly repeated. Holdout opens only with
an explicit final-evaluation flag and a previously saved candidate freeze.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("/suite"))
    parser.add_argument("--dev-candidate-dir", type=Path)
    parser.add_argument("--holdout-reference-dir", type=Path)
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--unlock-holdout-final-evaluation", action="store_true")
    parser.add_argument("--candidate-freeze", type=Path)
    parser.add_argument("--case-ids", help="Comma-separated subset; all remaining cases stay unmeasured")
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--layers", default="retrieval,generation")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-external-data-transfer", action="store_true")
    args = parser.parse_args()
    if os.environ.get("S0005_CANDIDATE") != "true" or not args.allow_external_data_transfer:
        parser.error("explicit isolated candidate and external transfer opt-in required")
    if args.output.exists() or args.output.with_suffix(".attempts.jsonl").exists():
        parser.error("Existing attempts must not be repeated automatically")
    layers = set(args.layers.split(","))
    if not layers or not layers <= {"retrieval", "generation", "e2e"}:
        parser.error("Only retrieval,generation,e2e adapters are permitted")
    if args.repeats < 1:
        parser.error("Positive repeats required")
    if args.split == "holdout" and (not args.unlock_holdout_final_evaluation or not args.candidate_freeze):
        parser.error("Holdout requires explicit unlock and a saved candidate freeze")
    if args.split == "holdout" and args.dev_candidate_dir:
        parser.error("Dev references cannot replace holdout")
    if args.holdout_reference_dir and args.split != "holdout":
        parser.error("Holdout references require explicit final holdout mode")
    logging.disable(logging.CRITICAL)
    from app.core.config import get_settings
    from app.eval.acceptance_budget import install_acceptance_budget
    from app.eval.candidate_identity import candidate_identity, validate_isolation
    from app.eval.corpus_suite import digest, evaluate, load_suite, read_json, write_new
    from scripts.evaluate_corpus_suite import LiveAdapters

    settings = get_settings()
    validate_isolation(settings)
    if urlsplit(settings.openai_base_url or "https://api.openai.com/v1").hostname != "api.openai.com":
        raise ValueError("Only explicitly authorized official OpenAI endpoint permitted")
    ledger = install_acceptance_budget(Path(os.environ["S0005_BUDGET_LEDGER"]), limit_usd=10.)
    identity = candidate_identity(settings, ROOT)
    if "e2e" in layers:
        import httpx
        response = httpx.get("http://127.0.0.1:8000/__candidate_identity", headers={
            "X-Internal-Token": settings.internal_token.get_secret_value()}, timeout=10)
        response.raise_for_status()
        runtime = response.json()
        if (runtime.get("candidate_identity_sha256") != identity["candidate_identity_sha256"]
                or runtime.get("loaded_source_sha256") != identity["source_sha256"]):
            raise ValueError("Candidate API must be restarted on the measured source/artifacts")
    if args.candidate_freeze:
        freeze = read_json(args.candidate_freeze)
        if freeze["candidate_identity_sha256"] != identity["candidate_identity_sha256"]:
            raise ValueError("Candidate changed after final freeze")
    manifest, cases, contexts = load_suite(args.suite, args.split,
        unlock_holdout=args.unlock_holdout_final_evaluation)
    contract = None
    if args.dev_candidate_dir:
        from app.eval.corpus_suite import load_dev_candidate
        manifest, cases, contexts, contract = load_dev_candidate(args.dev_candidate_dir, manifest, cases)
    if args.holdout_reference_dir:
        from app.eval.holdout_reference_freeze import load_holdout_reference_freeze
        contexts, contract = load_holdout_reference_freeze(args.holdout_reference_dir, args.suite, cases,
            candidate_identity_sha256=identity["candidate_identity_sha256"],
            catalog_sha256=identity["artifacts_sha256"]["rag_evidence_catalog_path"])
        manifest = {**manifest, "holdout_reference_contract_sha256": digest(contract),
                    "holdout_contexts_sha256": digest(contexts)}
    total_cases = len(cases)
    if args.case_ids:
        selected = args.case_ids.split(",")
        cases = [case for case in cases if case["id"] in selected]
        if len(cases) != len(selected) or len(selected) != len(set(selected)):
            raise ValueError("Unknown or duplicate requested case")
    profiles = read_json(args.profiles) if args.profiles else {"configured_default": {}}
    if not isinstance(profiles, dict) or not profiles or any(
        not isinstance(p, dict) or set(p) - {"model", "effort", "api"} for p in profiles.values()
    ):
        raise ValueError("Invalid profile configuration")
    for profile in profiles.values():
        if profile.get("model", settings.default_model) not in {"gpt-6-luna", "gpt-5.4-mini"}:
            raise ValueError("Only authorized compact benchmark models permitted")
    # Same container and source bytes as the candidate; no working-service alias.
    live = LiveAdapters("http://127.0.0.1:8000", enable_e2e="e2e" in layers)
    live.settings = settings
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal = args.output.with_suffix(".attempts.jsonl")
    counts: dict[tuple, int] = {}

    def wrap(layer):
        adapter = getattr(live, layer)

        def call(case, frozen, profile):
            key = (layer, case["id"], digest(profile))
            repetition = counts.get(key, 0)
            counts[key] = repetition + 1
            row = {"layer": layer, "case_id": case["id"], "profile_sha256": digest(profile),
                   "repetition": repetition, "context_sha256": frozen["fingerprint"],
                   "status": "in_flight", "budget_before": ledger.summary()}
            with journal.open("a", encoding="utf8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            started = time.perf_counter()
            try:
                output = adapter(case, frozen, profile)
                row.update(status="completed", output=output)
                return output
            except BaseException as exc:
                row.update(status="failed", error_type=type(exc).__name__)
                raise
            finally:
                row.update(adapter_wall_ms=(time.perf_counter() - started) * 1000,
                           budget_after=ledger.summary())
                with journal.open("a", encoding="utf8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                print(json.dumps({k: row[k] for k in ("case_id", "layer", "status", "adapter_wall_ms")} ), flush=True)

        return call

    before = ledger.summary()
    try:
        run = evaluate(cases, contexts, split=args.split, manifest=manifest,
            adapters={layer: wrap(layer) for layer in layers}, profiles=profiles,
            repeats=args.repeats, policy=read_json(args.suite / "release_policy.json"),
            reference_contract=contract)
    finally:
        live.close()
    run.update(candidate_identity=identity, budget_before=before, budget_after=ledger.summary(),
               selected_cases=len(cases), full_split_cases=total_cases,
               subset_only=len(cases) != total_cases,
               native_polling_measured=False, actual_telegram_delivery_measured=False)
    write_new(args.output, run)
    print(json.dumps({"status": run["summary"]["release_status"],
                      "selected_cases": len(cases), "budget": ledger.summary()}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - credentials/SDK exception bodies never reach output
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
