"""Immutable metadata-only repair; never rebuild source material or open holdout."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from app.eval.corpus_suite import digest, file_sha, read_json, write_new


def repair(candidate, destination):
    if destination.exists():
        raise FileExistsError("Preserve earlier freezes")
    cases = read_json(candidate / "dev_cases_v2.json")
    old_contract = read_json(candidate / "dev_reference_contract_v2.json")
    manifest = read_json(candidate / "manifest_v2.json")
    contract = copy.deepcopy(old_contract)
    contract["dev_cases_sha256"] = digest(cases)
    changed = [key for key in set(old_contract) | set(contract) if old_contract.get(key) != contract.get(key)]
    if changed != ["dev_cases_sha256"]:
        raise ValueError("Expected exactly one stale global case binding field")
    for name in ("dev_cases_v2.json", "dev_contexts_v2.json", "dev_reference_contract_v2.json"):
        if file_sha(candidate/name) != manifest["artifacts"][name]["file_sha256"]:
            raise ValueError("Parent freeze changed")
    destination.mkdir(parents=True)
    for name in ("dev_cases_v2.json", "dev_contexts_v2.json"):
        (destination/name).write_bytes((candidate/name).read_bytes())
    write_new(destination / "dev_reference_contract_v2.json", contract)
    manifest["artifacts"]["dev_reference_contract_v2.json"] = {"file_sha256": file_sha(destination/"dev_reference_contract_v2.json"), "canonical_sha256": digest(contract)}
    manifest["metadata_repair"] = {"method": "global_candidate_case_hash_only", "parent_manifest_file_sha256": file_sha(candidate/"manifest_v2.json"), "parent_contract_file_sha256": file_sha(candidate/"dev_reference_contract_v2.json"), "parent_contract_canonical_sha256": digest(old_contract), "new_contract_canonical_sha256": digest(contract), "changed_contract_fields": changed, "old_dev_cases_sha256": old_contract["dev_cases_sha256"], "new_dev_cases_sha256": contract["dev_cases_sha256"], "catalog_sha256_preserved": manifest["catalog_sha256"], "cases_bytes_preserved": True, "contexts_bytes_preserved": True, "all_case_rows_preserved": old_contract["cases"] == contract["cases"], "implementation_sha256": file_sha(Path(__file__)), "provider_calls": 0, "holdout_opened": False}
    write_new(destination / "manifest_v2.json", manifest)
    print(json.dumps({"manifest_file_sha256": file_sha(destination/"manifest_v2.json"), "contract_file_sha256": file_sha(destination/"dev_reference_contract_v2.json"), "contract_canonical_sha256": digest(contract), "changed_contract_fields": changed, "catalog_sha256": manifest["catalog_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    repair(args.candidate, args.destination)
