"""Offline sealed-reference audit. Default reads metadata only, never holdout content.

After root's explicit final unlock, bind original anchors to real source blocks.
Answers are never generated. A separate machine review file may supply source-bound,
query-relative reference candidates, which remain pending independent acceptance.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.eval.corpus_suite import digest, file_sha, load_suite, read_json

BUILDER_ID = "offline_holdout_binding_auditor_v1"
TEXT_KINDS = {"paragraph", "heading", "table_row", "table", "formula_candidate", "formula", "definition", "figure_transcription"}


def metadata(suite: Path) -> dict:
    manifest = read_json(suite / "manifest.json")
    sealed = read_json(suite / "holdout_manifest.json")
    return {"mode": "metadata_only", "holdout_content_opened": False,
            "suite_manifest_file_sha256": file_sha(suite / "manifest.json"),
            "case_count": manifest["split_counts"]["holdout"],
            "group_counts": manifest["group_counts"]["holdout"],
            "references_pending_review": sealed["references_pending_review"],
            "sealed_files_sha256": {name: manifest["files_sha256"][name] for name in ("holdout_cases.json", "holdout_contexts.json")},
            "assurance": "Metadata describes sealed original anchors; it does not establish sufficient reference answers."}


def literal_spans(blocks: list[dict], quote: str) -> list[tuple[dict, str]]:
    """Bind the WHOLE anchor with whitespace-only normalization, preserving case."""
    norm = lambda value: "".join(value.split())
    text = "".join(norm(b.get("text", "")) for b in blocks)
    target = norm(quote)
    start = text.find(target)
    if not target or start < 0 or text.find(target, start + 1) >= 0:
        return []
    end, offset, result = start + len(target), 0, []
    for block in blocks:
        raw = block.get("text", "")
        positions = [i for i, char in enumerate(raw) if not char.isspace()]
        length = len(positions)
        if offset < end and offset + length > start:
            left, right = max(0, start-offset), min(length, end-offset)
            result.append((block, raw[positions[left]:positions[right-1]+1]))
        offset += length
    return result


def validate_review(case, review, catalog, allowed_sources, catalog_sha, cases_sha):
    """Prove identities/provenance only; semantic sufficiency remains a declared review."""
    if review.get("method") != "machine_source_review" or review.get("human_expert_approval") is not False:
        raise ValueError("Review must be explicitly machine, non-human")
    if review.get("reviewer") in {None, "", BUILDER_ID} or not review.get("review_id") or not review.get("reviewed_at"):
        raise ValueError("Independent review identity required")
    if (review.get("case_sha256") != digest(case) or review.get("catalog_sha256") != catalog_sha
            or review.get("sealed_cases_file_sha256") != cases_sha):
        raise ValueError("Review freeze mismatch")
    if (not review.get("reference_answer") or review.get("query_relative_coverage_verified") is not True
            or not review.get("sufficiency_rationale") or review.get("acceptance_status") != "pending_independent_review"):
        raise ValueError("Query-relative candidate review required; no self acceptance")
    if review.get("applicability_status") != "source_specific_supported" or review.get("required_visual_status") not in {"not_required", "approved_literal_bound"}:
        raise ValueError("Applicability or required visual gap remains pending")
    blocks = {b["block_id"]: b for b in catalog["blocks"]}
    locations = {l["location_id"]: l for l in catalog["locations"]}
    sources = {s["source_id"]: s for s in catalog["sources"]}
    if len(blocks) != len(catalog["blocks"]) or len(locations) != len(catalog["locations"]) or len(sources) != len(catalog["sources"]):
        raise ValueError("Catalog identity is ambiguous")
    slots = review.get("claim_slots", [])
    if allowed_sources and not slots:
        raise ValueError("Technical reference requires curated factual claim slots")
    covered = set()
    identities = []
    visual_ids = set()
    for slot in slots:
        if not slot.get("claim") or not slot.get("slot_id") or not slot.get("alternatives"):
            raise ValueError("Empty claim slot")
        for alt in slot["alternatives"]:
            evidence = alt.get("evidence", [])
            if not evidence:
                raise ValueError("Empty evidence conjunction")
            for e in evidence:
                block = blocks.get(e.get("block_id"), {})
                loc = locations.get(block.get("location_id"), {})
                src = sources.get(block.get("source_id"), {})
                if (src.get("source_hash") not in allowed_sources or e.get("source_hash") != src.get("source_hash")
                        or e.get("relative_path") != src.get("relative_path") or e.get("location_id") != loc.get("location_id")
                        or e.get("page") != loc.get("page") or e.get("chm_member") != loc.get("chm_member")
                        or not e.get("quote") or e["quote"] not in block.get("text", "")
                        or e.get("quote_sha256") != digest(e["quote"]) or block.get("kind") not in TEXT_KINDS):
                    raise ValueError("Exact source/location/block/quote binding failed")
                attrs = block.get("attributes", {})
                if attrs.get("method") == "reviewed_visual_transcription":
                    if attrs.get("verified") is not True or not attrs.get("review_id") or e.get("source_review") != attrs:
                        raise ValueError("Approved visual provenance missing or changed")
                    visual_ids.add(block["block_id"])
                covered.add(src["source_hash"])
                identities.append((loc["location_id"], block["block_id"]))
    if covered != allowed_sources:
        raise ValueError("Reference does not cover every required original source")
    # No-answer behavior still requires an independent reviewed answer, never inferred
    # from a lexical miss or a blanket assertion about licensing/error catalog coverage.
    if not allowed_sources and slots:
        raise ValueError("Behavioral case cannot silently introduce external sources")
    required_visuals = {a["slot_id"]+":"+region for a in case["required_evidence_slots"] for region in a.get("expected_visual_regions", [])}
    visual_coverage = review.get("required_visual_evidence", {})
    if required_visuals and (set(visual_coverage) != required_visuals or any(not ids or not set(ids) <= visual_ids for ids in visual_coverage.values())):
        raise ValueError("Every sealed required visual needs exact approved claim evidence")
    if review.get("required_visual_status") == "approved_literal_bound" and not visual_ids:
        raise ValueError("Declared visual recovery has no approved visual evidence")
    return identities


def audit(args):
    # This guard precedes every read/hash of holdout case/context files.
    if not args.unlock_final_holdout:
        return metadata(args.suite)
    if not args.root_freeze_id or not args.sealed_manifest_sha256 or not args.catalog_sha256:
        raise ValueError("Final unlock requires root freeze ID and both frozen artifact hashes")
    if file_sha(args.suite / "manifest.json") != args.sealed_manifest_sha256:
        raise ValueError("Sealed suite manifest changed")
    if not args.catalog or file_sha(args.catalog) != args.catalog_sha256:
        raise ValueError("Candidate catalog freeze mismatch")
    if args.output is None or args.source_root is None:
        raise ValueError("Output and original source root required after unlock")
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory; preserve previous freezes")
    _manifest, cases, _original_contexts = load_suite(args.suite, "holdout", unlock_holdout=True)
    catalog = read_json(args.catalog)
    sources = {s["source_id"]: s for s in catalog["sources"]}
    locations = catalog["locations"]
    byloc = {loc["location_id"]: [] for loc in locations}
    for block in catalog["blocks"]:
        byloc[block["location_id"]].append(block)
    cases_sha = file_sha(args.suite / "holdout_cases.json")
    contexts_sha = file_sha(args.suite / "holdout_contexts.json")
    reviews_sha = file_sha(args.machine_reviews) if args.machine_reviews else None
    review_file = read_json(args.machine_reviews) if args.machine_reviews else {"cases": []}
    reviewed_rows = review_file.get("cases", [])
    if len({r["case_id"] for r in reviewed_rows}) != len(reviewed_rows):
        raise ValueError("Duplicate machine review case ID")
    reviews = {r["case_id"]: r for r in reviewed_rows}
    if set(reviews) - {case["id"] for case in cases}:
        raise ValueError("Machine review includes non-holdout case")
    rows, contexts = [], {}
    source_checks = {}
    for case in cases:
        anchors, selected, gaps = [], {}, []
        allowed = {a["file_sha256"] for a in case["required_evidence_slots"]}
        for anchor in case["required_evidence_slots"]:
            relative = anchor["relative_path"]
            source_path = (args.source_root / relative).resolve()
            source_path.relative_to(args.source_root.resolve())
            current_sha = source_checks.setdefault(relative, file_sha(source_path))
            source_ok = current_sha == anchor["file_sha256"]
            possible = [loc for loc in locations if loc["source_hash"] == anchor["file_sha256"] and (anchor.get("physical_page") is None or loc.get("page") == anchor["physical_page"])]
            bindings = []
            for loc in possible:
                material = [b for b in byloc[loc["location_id"]] if b["kind"] in TEXT_KINDS and b.get("text")]
                spans = literal_spans(material, anchor["exact_quote"])
                if spans:
                    bindings.append((loc, spans))
            exact = source_ok and len(bindings) == 1
            if exact:
                loc, spans = bindings[0]
                selected[loc["location_id"]] = loc
                bound = [{"block_id": block["block_id"], "quote": quote, "quote_sha256": digest(quote)} for block, quote in spans]
            else:
                bound = []
                gaps.append("source_hash_mismatch" if not source_ok else "literal_layout_or_ambiguous_location_gap")
                # PDF page itself can be located even if text layout changed.
                if source_ok and anchor.get("physical_page") is not None and len(possible) == 1:
                    selected[possible[0]["location_id"]] = possible[0]
            anchors.append({"slot_id": anchor["slot_id"], "source_hash_matches": source_ok, "literal_location_matches": len(bindings), "exact_catalog_binding": exact, "bound_fragments": bound})
            if anchor.get("expected_visual_regions"):
                gaps.append("sealed_required_visual_region_needs_independent_review")
            if anchor.get("applicability_status") != "source_specific_supported":
                gaps.append("sealed_applicability_needs_independent_review")
        row = {"case_id": case["id"], "case_sha256": digest(case), "candidate_case_sha256": digest(case), "source_group": case["source_group"], "task_type": case["task_type"], "method": "machine_source_review", "reviewer": BUILDER_ID, "acceptance_status": "pending_independent_review", "sufficiency": "pending", "reference_answer": case.get("reference"), "claim_slots": [], "anchor_checks": anchors, "pending_reasons": sorted(set(gaps + ["query_relative_reference_review_missing"])), "human_expert_approval": False}
        review = reviews.get(case["id"])
        if review:
            try:
                identity = validate_review(case, review, catalog, allowed, args.catalog_sha256, cases_sha)
                if any(not a["source_hash_matches"] for a in anchors):
                    raise ValueError("Original source changed")
                for lid, _ in identity:
                    selected[lid] = next(loc for loc in locations if loc["location_id"] == lid)
                row.update(sufficiency="machine_reviewed_sufficient_candidate", reference_answer=review["reference_answer"], claim_slots=review["claim_slots"], reviewer=review["reviewer"], query_relative_review_sha256=digest(review), pending_reasons=["independent_reference_acceptance_required"], repaired_original_anchor_gaps=sorted(set(gaps)))
            except (KeyError, TypeError, ValueError) as exc:
                row["pending_reasons"].append("machine_review_invalid:" + type(exc).__name__)
                row["review_failure_reason"] = str(exc)
        records, unresolved = [], []
        for loc in selected.values():
            material = [b for b in byloc[loc["location_id"]] if b["kind"] in TEXT_KINDS and b.get("text")]
            src = sources[loc["source_id"]]
            records.append({"source": {"relative_path": src["relative_path"], "file_sha256": src["source_hash"], "location_id": loc["location_id"], "physical_page": loc.get("page"), "chm_member": loc.get("chm_member"), "product": loc["product"], "version": loc["version"]}, "text": "\n".join(b["text"] for b in material), "block_ids": [b["block_id"] for b in material]})
            unresolved += [b["block_id"] for b in byloc[loc["location_id"]] if b["kind"] == "visual_unextracted" and not b.get("attributes", {}).get("approved_transcription_id")]
        row["unresolved_visual_block_ids"] = unresolved
        contexts[case["id"]] = {"records": records, "fingerprint": digest(records)}
        rows.append(row)
    # Original cases/questions/references remain byte-identical; candidate answers live
    # only in the independently versioned contract, never back-written into originals.
    report = {"schema_version": 1, "split": "holdout", "method": "machine_source_review", "cases_sha256": digest(cases), "root_freeze_id": args.root_freeze_id, "sealed_manifest_sha256": args.sealed_manifest_sha256, "sealed_cases_file_sha256": cases_sha, "sealed_contexts_file_sha256": contexts_sha, "catalog_sha256": args.catalog_sha256, "machine_review_file_sha256": reviews_sha, "builder_sha256": file_sha(Path(__file__)), "questions_modified": 0, "source_hashes": source_checks, "summary": dict(Counter(r["sufficiency"] for r in rows)), "human_expert_approval": False, "semantic_release": "blocked until independent reference acceptance and all required evaluation gates", "provider_calls": 0, "cases": rows}
    if file_sha(args.catalog) != args.catalog_sha256 or file_sha(args.suite / "manifest.json") != args.sealed_manifest_sha256:
        raise ValueError("Input freeze changed during audit")
    if (file_sha(args.suite / "holdout_cases.json") != cases_sha or file_sha(args.suite / "holdout_contexts.json") != contexts_sha
            or (args.machine_reviews and file_sha(args.machine_reviews) != reviews_sha)):
        raise ValueError("Sealed content or review changed during audit")
    args.output.mkdir(parents=True)
    outputs = {"holdout_binding_contract.json": report, "holdout_candidate_contexts.json": contexts}
    for name, value in outputs.items():
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in {"cases", "source_hashes"}}
    summary["artifacts"] = {name: file_sha(args.output/name) for name in outputs}
    (args.output/"manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"mode": "unlocked_offline_audit", "summary": report["summary"], "output": str(args.output), "provider_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "tests/eval/corpus_suite")
    parser.add_argument("--unlock-final-holdout", action="store_true")
    parser.add_argument("--root-freeze-id")
    parser.add_argument("--sealed-manifest-sha256")
    parser.add_argument("--catalog-sha256")
    for name in ("catalog", "source-root", "machine-reviews", "output"):
        parser.add_argument("--"+name, type=Path)
    print(json.dumps(audit(parser.parse_args()), ensure_ascii=True, indent=2))
