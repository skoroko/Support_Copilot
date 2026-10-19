"""Offline dev-only source-bound candidate; never opens holdout files."""
import argparse
import re
import copy
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from app.eval.corpus_suite import digest, file_sha, read_json, write_new, validate_reference_contract

def build(source_root, catalog_path, destination, previous_candidate=None):
    suite = ROOT / "tests/eval/corpus_suite"
    cases = read_json(suite / "dev_cases.json")
    manifest = read_json(suite / "manifest.json")
    catalog_bytes = catalog_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    catalog_sha = hashlib.sha256(catalog_bytes).hexdigest()
    decisions = read_json(suite / "review/dev_review_decisions_v2.json")["decisions"]
    sources = {s["source_id"]: s for s in catalog["sources"]}
    by_hash = {}
    for loc in catalog["locations"]:
        by_hash.setdefault(loc["source_hash"], []).append(loc)
    by_location = {}
    for block in catalog["blocks"]:
        by_location.setdefault(block["location_id"], []).append(block)
    strip = lambda text: "".join(text.casefold().split())
    def locate(anchor):
        locs = by_hash.get(anchor["file_sha256"], [])
        page = anchor.get("physical_page")
        if page is not None:
            return [l for l in locs if l.get("page") == page]
        if anchor["relative_path"].endswith(".html"):
            return locs
        quote = anchor["exact_quote"]
        body = quote[quote.find("Home >"):]
        title = strip(quote.split("Home >")[0].strip())
        exact = [l for l in locs if strip(l.get("title", "")) == title
                 and any(b.get("text", "").startswith("Home >")
                         and strip(b["text"]) in strip(body) for b in by_location[l["location_id"]])]
        if len(exact) > 1:
            # Real CHM aliases may contain byte-identical extracted bodies.
            signatures = {digest([b["text"] for b in by_location[l["location_id"]]
                                  if b["kind"] not in {"link", "image", "visual_unextracted"}]) for l in exact}
            if len(signatures) == 1:
                canonical = min(exact, key=lambda l: (len(l.get("chm_member", "")), l.get("chm_member", "")))
                canonical = dict(canonical, equivalent_body_alias_location_ids=[l["location_id"] for l in exact])
                return [canonical]
        if len(exact) == 1:
            return exact
        # Literal prefix of breadcrumb/body, never query similarity or result tuning.
        target = strip(body[:250])
        matches = [l for l in locs if target in strip(" ".join(b.get("text", "") for b in by_location[l["location_id"]]))]
        if len(matches) != 1:
            target = strip(body.split("▒")[0].split(" - ")[0])
            matches = [l for l in locs if target in strip(" ".join(b.get("text", "") for b in by_location[l["location_id"]]))]
        return matches
    selectors = read_json(suite / "review/curated_claim_windows_v2.json")["selectors"]
    origins = {c["required_evidence_slots"][0]["document_record_id"]: i
               for i, c in enumerate(cases[:54], 1)}
    for case_number, slot_number, selector in [(55, 1, 55), (57, 1, 56), (65, 1, 57), (70, 1, 58), (72, 1, 59)]:
        origins[cases[case_number-1]["required_evidence_slots"][slot_number]["document_record_id"]] = selector
    def curated_blocks(blocks, selector):
        result = []
        compact = "".join(strip(b["text"]) for b in blocks)
        offsets, current = [], 0
        for b in blocks:
            offsets.append((current, current + len(strip(b["text"])), b))
            current = offsets[-1][1]
        for start, end in selectors.get(str(selector), []):
            left = compact.find(strip(start))
            right_start = compact.find(strip(end), left) if end and left >= 0 else len(compact)
            if left < 0 or right_start < 0:
                result.append({"window": [start, end], "status": "literal_window_not_found", "blocks": []})
                continue
            right = right_start + len(strip(end)) if end else len(compact)
            selected = []
            for lo, hi, b in offsets:
                if hi <= left or lo >= right:
                    continue
                indexes = [i for i, char in enumerate(b["text"]) if not char.isspace()]
                a, z = max(0, left - lo), min(hi - lo, right - lo)
                quote = b["text"][indexes[a]: indexes[z-1]+1]
                # Column labels are extraction structure, never physical-source quotes.
                # Preserve every real cell as a separate literal substring in the same block.
                cells = re.split(r"(?:^|\s*\|\s*)Column \d+=", quote)
                for cell in cells:
                    cell = cell.strip()
                    if cell and cell != "(empty)":
                        selected.append(dict(b, quote=cell))
            result.append({"window": [start, end], "status": "matched", "blocks": selected})
        return result
    contexts, rows, v2cases = {}, [], []
    source_proofs = {}
    for case, decision in zip(cases, decisions, strict=True):
        if int(case["id"].rsplit("-", 1)[1]) != decision["case_number"]:
            raise ValueError("Decision identity mismatch")
        candidate = copy.deepcopy(case)
        if decision["question_revision"]:
            candidate["question"] = decision["question_revision"]
        candidate["reference"] = decision["reference_answer"]
        candidate["reference_status"] = "pending_review"
        candidate["reference_note"] = "Machine source-reviewed candidate; independent acceptance required. Original dev freeze preserved."
        v2cases.append(candidate)
        records, spans, checks, unresolved = [], [], [], []
        selected_origins = {}
        selected_locations = []
        for anchor_index, anchor in enumerate(case["required_evidence_slots"]):
            path = source_root / "data" / anchor["relative_path"]
            actual = source_proofs.setdefault(anchor["relative_path"], file_sha(path))
            if actual != anchor["file_sha256"]:
                raise ValueError("Original source changed")
            matches = locate(anchor)
            checks.append({"original_slot_id": anchor["slot_id"], "source_hash_matches": True,
                           "location_match_count": len(matches), "equivalent_body_alias_location_ids": matches[0].get("equivalent_body_alias_location_ids", []) if len(matches) == 1 else []})
            if len(matches) != 1:
                continue
            selected_locations.extend(matches)
            selected_origins[matches[0]["location_id"]] = (60 if decision["case_number"] == 56 and "Узел" in anchor["relative_path"] else origins.get(anchor["document_record_id"]))
            if str(anchor_index) in decision.get("slot_selectors", {}):
                selected_origins[matches[0]["location_id"]] = decision["slot_selectors"][str(anchor_index)]
            if anchor.get("physical_page") and (anchor is case["required_evidence_slots"][0] or "Узел" in anchor["relative_path"]):
                extra_locations = [l for l in by_hash[anchor["file_sha256"]]
                                   if l.get("page") in (decision["extra_pages"] + ([4] if "Узел" in anchor["relative_path"] and anchor.get("physical_page") == 3 else []))]
                selected_locations.extend(extra_locations)
                for extra in extra_locations:
                    selector = decision.get("extra_page_selectors", {}).get(str(extra.get("page")))
                    if selector is not None:
                        selected_origins[extra["location_id"]] = selector
        seen = set()
        for loc in selected_locations:
            if loc["location_id"] in seen:
                continue
            seen.add(loc["location_id"])
            source = sources[loc["source_id"]]
            blocks = by_location[loc["location_id"]]
            unresolved.extend({"source_hash": loc["source_hash"], "page": loc.get("page"),
                               "location_id": loc["location_id"], "block_id": b["block_id"],
                               "bbox": b.get("bbox"), "image_sha256": b.get("attributes", {}).get("image_sha256")}
                              for b in blocks if b["kind"] == "visual_unextracted"
                              and not b.get("attributes", {}).get("approved_transcription_id"))
            textblocks = [b for b in blocks if b.get("text", "").strip()
                          and b["kind"] in {"paragraph", "heading", "table_row", "table", "formula_candidate", "definition", "formula", "figure_transcription"}]
            # Keep full source location for generation. No unseen retrieval output is used.
            record_source = {"relative_path": source["relative_path"], "file_sha256": source["source_hash"],
                             "physical_page": loc.get("page"), "section_identity": loc.get("title"),
                             "location_id": loc["location_id"], "product": loc.get("product"),
                             "version": loc.get("version"), "chm_member": loc.get("chm_member"), "applicability_status": "source_specific; cross_version_not_approved"}
            records.append({"source": record_source, "text": "\n".join(b["text"] for b in textblocks),
                            "block_ids": [b["block_id"] for b in textblocks]})
            for group in curated_blocks(textblocks, selected_origins.get(loc["location_id"])):
                if group["status"] != "matched":
                    checks.append({"curated_window": group["window"], "location_match_count": 0,
                                   "error": "literal_window_not_found"})
                    continue
                evidence = [{"relative_path": source["relative_path"], "source_hash": source["source_hash"],
                             "location_id": loc["location_id"], "block_id": b["block_id"], "page": loc.get("page"),
                             "quote": b["quote"], "quote_sha256": digest(b["quote"]), "kind": b["kind"], "bbox": b.get("bbox"),
                             "source_review": b.get("attributes", {}) if b.get("attributes", {}).get("method") == "reviewed_visual_transcription" else None}
                            for b in group["blocks"]]
                spans.append({"slot_id": "fact-" + str(len(spans) + 1),
                              "claim": " ".join(b["quote"] for b in group["blocks"]),
                              "claim_boundary": group["window"],
                              "mapping_status": "curated_machine_factual_group; independent_review_required",
                              "alternatives": [{"evidence": evidence}]})
        sufficiency = decision["sufficiency"]
        limitations = list(decision["limitations"])
        if any(c["location_match_count"] != 1 for c in checks):
            sufficiency = "pending"
            limitations.append("Nonunique/missing current catalog locator")
        contexts[case["id"]] = {"records": records, "fingerprint": digest(records),
                               "assurance": "full source locations; untranscribed images omitted explicitly"}
        rows.append({"case_id": case["id"], "case_sha256": digest(case), "candidate_case_sha256": digest(candidate),
                     "reviewer": "Codex / source-review agent", "method": "machine_source_review",
                     "acceptance_status": "pending_independent_review", "sufficiency": sufficiency,
                     "reference_answer": decision["reference_answer"], "limitations": limitations,
                     "question_revision": decision["question_revision"], "claim_slots": spans,
                     "locator_checks": checks, "unresolved_visual_regions": unresolved,
                     "production_extraction_assurance": "exact frozen catalog blocks verified; remaining visual regions listed separately; production retrieval coverage must be measured independently"})
    contract = {"schema_version": 2, "method": "machine_source_review", "human_expert_approval": False,
                "dev_cases_sha256": digest(cases), "catalog_sha256": catalog_sha,
                "base_manifest_sha256": digest(manifest), "cases": rows,
                "claim_mapping_assurance": "curated factual windows, each conjunction of exact block quotes; independent query-relative review pending"}
    validate_reference_contract(contract, cases, catalog, catalog_sha256=catalog_sha)
    destination.mkdir(parents=True, exist_ok=False)
    files = {"dev_cases_v2.json": v2cases, "dev_contexts_v2.json": contexts,
             "dev_reference_contract_v2.json": contract}
    for name, value in files.items():
        write_new(destination / name, value)
    audit = {"schema_version": 2, "split": "dev", "holdout_opened": False,
             "base_manifest_sha256": digest(manifest), "original_dev_cases_sha256": digest(cases),
             "original_dev_contexts_file_sha256": file_sha(suite / "dev_contexts.json"),
             "catalog_sha256": catalog_sha, "source_root": str(source_root.resolve()), "source_hashes": source_proofs,
             "artifacts": {name: {"file_sha256": file_sha(destination/name), "canonical_sha256": digest(value)} for name,value in files.items()},
             "review_summary": dict(Counter(r["sufficiency"] for r in rows)),
             "review_decisions_sha256": file_sha(suite / "review/dev_review_decisions_v2.json"),
             "curated_claim_windows_sha256": file_sha(suite / "review/curated_claim_windows_v2.json"),
             "builder_sha256": file_sha(Path(__file__)),
             "changed_questions": [r["case_id"] for r in rows if r["question_revision"]],
             "semantic_release": "blocked; independent reference acceptance and result review required",
             "api_budget_ledger": {"task_user_budget_usd": 10, "quota_usd": 0,
                 "quota_authorized_by_root": False, "attempts": [], "actual_calls": 0, "actual_cost_usd": 0},
             "claim_coverage_metric": "strict source+location+block+literal quote; mechanical only; not answer correctness"}
    if previous_candidate is not None:
        previous_manifest = previous_candidate / "manifest_v2.json"
        previous = read_json(previous_manifest)
        audit["previous_candidate_audit"] = {
            "manifest_file_sha256": file_sha(previous_manifest),
            "manifest_canonical_sha256": digest(previous),
            "catalog_sha256": previous["catalog_sha256"],
            "status": "preserved; superseded candidate, never independently accepted"}
    write_new(destination / "manifest_v2.json", audit)
    print(audit["review_summary"])

if __name__ == "__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--catalog", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    p.add_argument("--previous-candidate", type=Path)
    a=p.parse_args()
    build(a.source_root, a.catalog, a.destination, a.previous_candidate)
