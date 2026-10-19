"""Bind four repaired dev references to approved literal blocks; holdout never read."""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from app.eval.corpus_suite import digest, file_sha, read_json, write_new, validate_reference_contract


def build(args):
    prior = ROOT / "tests/eval/corpus_suite/dev_v2_8"
    original_cases = read_json(ROOT / "tests/eval/corpus_suite/dev_cases.json")
    cases = read_json(prior / "dev_cases_v2.json")
    contract = read_json(prior / "dev_reference_contract_v2.json")
    contexts = read_json(prior / "dev_contexts_v2.json")
    drafts_path = ROOT / "tests/eval/corpus_suite/review/pending_v2_9_originals/reference_answer_drafts.json"
    drafts = read_json(drafts_path)
    approvals = read_json(args.approvals)
    if file_sha(args.approvals) != drafts["literal_approval_file_sha256"]:
        raise ValueError("Literal approval freeze changed")
    if len(approvals) != 24 or any(r["approval_status"] != "approved" or r.get("human_expert_approval") is not False for r in approvals):
        raise ValueError("Independent literal review scope mismatch")
    catalog = read_json(args.catalog)
    sources = {s["source_id"]: s for s in catalog["sources"]}
    locs = {l["location_id"]: l for l in catalog["locations"]}
    blocks = catalog["blocks"]
    byloc = {lid: [] for lid in locs}
    for block in blocks:
        byloc[block["location_id"]].append(block)
    norm = lambda value: "".join(value.casefold().split())
    textkinds = {"paragraph", "heading", "table_row", "table", "formula_candidate", "definition", "formula", "figure_transcription"}

    def evidence(block, quote=None):
        loc = locs[block["location_id"]]
        src = sources[block["source_id"]]
        quote = block["text"] if quote is None else quote
        return {"relative_path": src["relative_path"], "source_hash": src["source_hash"], "location_id": loc["location_id"], "block_id": block["block_id"], "page": loc.get("page"), "quote": quote, "quote_sha256": digest(quote), "kind": block["kind"], "bbox": block.get("bbox"), "source_review": block["attributes"] if block.get("attributes", {}).get("method") == "reviewed_visual_transcription" else None}

    def native_window(selected, first, last):
        native = [b for loc in selected for b in byloc[loc["location_id"]] if b.get("text") and b["kind"] in textkinds and b.get("attributes", {}).get("method") != "reviewed_visual_transcription"]
        compact = "".join(norm(b["text"]) for b in native)
        a = compact.find(norm(first)); z = compact.find(norm(last), a)
        if a < 0 or z < 0:
            raise ValueError("Missing literal native claim window")
        z += len(norm(last)); offset = 0; result = []
        for block in native:
            length = len(norm(block["text"]))
            if offset < z and offset + length > a:
                indexes = [i for i, ch in enumerate(block["text"]) if not ch.isspace()]
                left, right = max(0, a-offset), min(length, z-offset)
                result.append(evidence(block, block["text"][indexes[left]:indexes[right-1]+1]))
            offset += length
        return result

    def approved(case_id, prefix):
        records = [r for r in approvals if case_id in r["case_ids"] and r["text"].startswith(prefix)]
        if len(records) != 1:
            raise ValueError("Nonunique approved literal selector")
        record = records[0]
        matched = [b for b in blocks if b.get("attributes", {}).get("review_id") == record["review_id"] and b["text"] == record["text"] and b["location_id"] == record["location_id"] and sources[b["source_id"]]["source_hash"] == record["source_hash"]]
        if len(matched) != 1 or matched[0].get("attributes", {}).get("verified") is not True:
            raise ValueError("Approved literal is not uniquely indexed")
        return evidence(matched[0])

    updates = []
    for draft in drafts["cases"]:
        cid = draft["case_id"]
        case = next(c for c in cases if c["id"] == cid)
        row = next(r for r in contract["cases"] if r["case_id"] == cid)
        source_hash = case["required_evidence_slots"][0]["file_sha256"]
        selected = [l for l in catalog["locations"] if l["source_hash"] == source_hash and (l.get("page") in draft.get("native_context_pages", []) or l.get("chm_member") == draft.get("native_context_member", "--no-member--"))]
        selected.sort(key=lambda l: l.get("page", 0))
        if not selected:
            raise ValueError("Required source location missing")
        def window(first, last):
            return native_window(selected, first, last)
        if cid.endswith("012"):
            groups = [window("Твердотельные элементы часто", "и т.п."), window("В глобальной системе координат", "вдоль осей .") + [approved(cid, "x,"), approved(cid, r"\mathbf{u}_i")]]
        elif cid.endswith("014"):
            groups = [window("Отношение между относительным перемещением", "следующей формулой:") + [approved(cid, r"\mathbf{t}=")], [approved(cid, r"\mathbf{D}=")] + window("Величина", "по каждому из направлений."), window("Перемещения верхней/нижней поверхности", "использованием функции формы.")]
        elif cid.endswith("024"):
            groups = [window("При начальном нагружении", "следующей скелетной кривой") + window("Кривая гистерезиса определяется", "следующим образом:"), [approved(cid, r"G_o\gamma=")] + window("начальная жесткость (модуль сдвига)", "константа максимального демпфирования"), [approved(cid, r"G_o\left(")]]
        else:
            ui = approved(cid, "Create Hinge")
            first = copy.deepcopy(ui); first["quote"] = ui["quote"].split("Name: Hinge")[0].strip(); first["quote_sha256"] = digest(first["quote"])
            second = copy.deepcopy(ui); second["quote"] = ui["quote"][ui["quote"].index("Name: Hinge"):]; second["quote_sha256"] = digest(second["quote"])
            groups = [[first], [second]]
        row["claim_slots"] = [{"slot_id": f"fact-{i+1}", "claim": draft["required_factual_groups"][i], "mapping_status": "curated_machine_factual_group; independent_review_required", "alternatives": [{"evidence": group}]} for i, group in enumerate(groups)]
        row.update(reference_answer=draft["reference_answer"], sufficiency="machine_reviewed_sufficient_candidate", limitations=["Independent engineering semantic acceptance pending; literal source review only."], unresolved_visual_regions=[])
        case["reference"] = draft["reference_answer"]
        row["candidate_case_sha256"] = digest(case)
        records = []
        unresolved = []
        for loc in selected:
            src = sources[loc["source_id"]]
            material = [b for b in byloc[loc["location_id"]] if b.get("text") and b["kind"] in textkinds]
            records.append({"source": {"relative_path": src["relative_path"], "file_sha256": src["source_hash"], "physical_page": loc.get("page"), "section_identity": loc.get("title"), "location_id": loc["location_id"], "product": loc["product"], "version": loc["version"], "chm_member": loc.get("chm_member"), "applicability_status": "source_specific; cross_version_not_approved"}, "text": "\n".join(b["text"] for b in material), "block_ids": [b["block_id"] for b in material]})
            unresolved += [{"source_hash": loc["source_hash"], "page": loc.get("page"), "location_id": loc["location_id"], "block_id": b["block_id"], "bbox": b.get("bbox"), "image_sha256": b.get("attributes", {}).get("image_sha256"), "required_for_current_question": False} for b in byloc[loc["location_id"]] if b["kind"] == "visual_unextracted" and not b.get("attributes", {}).get("approved_transcription_id")]
        row["unresolved_visual_regions"] = unresolved
        contexts[cid] = {"records": records, "fingerprint": digest(records), "assurance": "Full source locations plus independently literal-reviewed visual material; unneeded unresolved visual regions explicitly retained in contract."}
        updates.append(cid)
    catalog_sha = file_sha(args.catalog)
    contract["catalog_sha256"] = catalog_sha
    validate_reference_contract(contract, original_cases, catalog, catalog_sha256=catalog_sha)
    # Every required fragment must exist under exact identity in frozen context.
    for row in contract["cases"]:
        for slot in row["claim_slots"]:
            for alt in slot["alternatives"]:
                for e in alt.get("evidence", [alt]):
                    if not any(e["block_id"] in r["block_ids"] and e["location_id"] == r["source"]["location_id"] and e["source_hash"] == r["source"]["file_sha256"] and e["quote"] in r["text"] for r in contexts[row["case_id"]]["records"]):
                        raise ValueError("Required claim absent from frozen context")
    args.destination.mkdir(parents=True, exist_ok=False)
    artifacts = {"dev_cases_v2.json": cases, "dev_contexts_v2.json": contexts, "dev_reference_contract_v2.json": contract}
    for name, value in artifacts.items():
        write_new(args.destination / name, value)
    manifest = {"schema_version": 2, "split": "dev", "holdout_opened": False, "base_manifest_sha256": contract["base_manifest_sha256"], "original_dev_cases_sha256": digest(original_cases), "catalog_sha256": catalog_sha, "previous_candidate_audit": {"manifest_file_sha256": file_sha(prior / "manifest_v2.json"), "manifest_canonical_sha256": digest(read_json(prior / "manifest_v2.json")), "status": "preserved"}, "literal_approval_file_sha256": file_sha(args.approvals), "draft_file_sha256": file_sha(drafts_path), "builder_sha256": file_sha(Path(__file__)), "updated_reference_cases": updates, "questions_changed_from_previous": [], "review_summary": dict(Counter(r["sufficiency"] for r in contract["cases"])), "human_expert_approval": False, "semantic_release": "blocked pending independent engineering semantic reference acceptance, all layers and holdout", "artifacts": {name: {"file_sha256": file_sha(args.destination/name), "canonical_sha256": digest(value)} for name, value in artifacts.items()}, "provider_calls": 0}
    superseded = ROOT / "tests/eval/corpus_suite/dev_v2_9/manifest_v2.json"
    if superseded.exists():
        manifest["superseded_candidate_audit"] = {"manifest_file_sha256": file_sha(superseded), "status": "preserved; missing original dev binding field; runner rejected before API"}
    write_new(args.destination / "manifest_v2.json", manifest)
    print(json.dumps(manifest["review_summary"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    for name in ("catalog", "approvals", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    build(parser.parse_args())
