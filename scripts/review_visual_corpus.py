"""Offline PDF image inventory and selective original-crop review packets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.visual_recovery import (
    build_visual_inventory,
    make_visual_contact_sheets,
    render_review_packet,
    render_unique_visual_batch,
    select_dev_pages,
    verify_candidate_locators,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--build-inventory", action="store_true")
    parser.add_argument("--dev-cases", type=Path)
    parser.add_argument("--case-ids", nargs="+", default=[f"corpus-dev-{number:03d}" for number in (19, 21, 30, 58, 60, 63)])
    parser.add_argument("--selection", type=Path, help="Explicit source_hash/page list, reusable without eval data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--unique-batch-output", type=Path, help="Prepare unique full crops/classifier thumbnails; never calls OCR/API")
    parser.add_argument("--batch-limit", type=int, help="Optional development-only group count limit")
    parser.add_argument("--contact-sheets", type=Path, help="Save compact text-presence triage contact sheets")
    parser.add_argument("--batch-manifest", type=Path, help="Existing complete unique batch manifest for contact sheets")
    parser.add_argument("--verify-candidates", type=Path, help="Mechanically verify candidate source locators/crop hashes, without semantic approval")
    parser.add_argument("--verification-report", type=Path)
    args = parser.parse_args()
    if args.build_inventory:
        if not args.catalog:
            parser.error("--build-inventory requires --catalog")
        inventory = build_visual_inventory(args.catalog, args.data_dir)
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf8")
    else:
        inventory = json.loads(args.inventory.read_text(encoding="utf8"))
    print(json.dumps(inventory["summary"], ensure_ascii=True, indent=2))
    if args.output:
        if args.selection:
            selection = json.loads(args.selection.read_text(encoding="utf8"))
        elif args.dev_cases:
            selection = select_dev_pages(inventory, args.dev_cases, args.case_ids)
        else:
            parser.error("Rendering requires --selection or explicit --dev-cases")
        packet = render_review_packet(inventory, args.data_dir, selection, args.output, scale=args.scale)
        (args.output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf8")
        print(json.dumps({"rendered_pages": len(packet["pages"]), "pending_candidates": packet["candidate_count"]}))
    if args.unique_batch_output:
        batch = render_unique_visual_batch(inventory, args.data_dir, args.unique_batch_output,
                                           scale=args.scale, limit=args.batch_limit)
        print(json.dumps({"prepared_unique_groups": batch["prepared_groups"], "prepared_occurrences": batch["prepared_occurrences"],
                          "all_inventory_groups_prepared": batch["prepared_groups"] == batch["total_inventory_groups"]}))
    if args.contact_sheets:
        manifest_path = args.batch_manifest or (args.unique_batch_output / "batch_manifest.json" if args.unique_batch_output else None)
        if not manifest_path:
            parser.error("Contact sheets require --batch-manifest or --unique-batch-output")
        contact = make_visual_contact_sheets(manifest_path, args.contact_sheets)
        print(json.dumps({"triage_sheets": len(contact["sheets"]), "triage_tiles": contact["tile_count"]}))
    if args.verify_candidates:
        candidates = json.loads(args.verify_candidates.read_text(encoding="utf8"))
        report = verify_candidate_locators(inventory, candidates)
        if args.verification_report:
            args.verification_report.parent.mkdir(parents=True, exist_ok=True)
            args.verification_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
        print(json.dumps({"candidate_locators_checked": len(report["records"]), "all_locators_match": report["all_locators_match"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
