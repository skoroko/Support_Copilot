"""Local, selective PDF visual-review tooling; never approves its own output.

Pixel hashes group byte-identical raster payloads. They do not establish that
two contextual uses have the same meaning, applicability, or transcription.
Geometric triage classes are review hints, not semantic image classifiers.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.services.corpus import CorpusCatalog, file_hash, stable_id


def _geometry_class(bbox: list[float], page_bbox: list[float]) -> str:
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    page_area = (page_bbox[2] - page_bbox[0]) * (page_bbox[3] - page_bbox[1])
    if width * height >= page_area * 0.85:
        return "page_sized_raster_candidate"
    if width <= 32 and height <= 32:
        return "small_symbol_or_decoration_candidate"
    if height <= 80 and width / max(height, 0.01) >= 2:
        return "wide_strip_candidate"
    return "diagram_screenshot_or_other_candidate"


def build_visual_inventory(catalog_path: Path, data_root: Path) -> dict[str, Any]:
    """Bind every saved image region to actual unchanged PDF image geometry."""
    import pdfplumber

    catalog = CorpusCatalog.load(catalog_path)
    sources = {s["source_id"]: s for s in catalog.sources}
    locations = {loc["location_id"]: loc for loc in catalog.locations}
    wanted: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in catalog.blocks:
        if item["kind"] == "visual_unextracted":
            wanted[item["source_id"]][locations[item["location_id"]]["page"]].append(item)
    records = []
    for source_id, pages in wanted.items():
        source = sources[source_id]
        path = (data_root / source["relative_path"]).resolve()
        if not path.is_relative_to(data_root.resolve()) or source["format"] != "pdf":
            raise ValueError("Visual inventory source must be a permitted PDF inside data root")
        if file_hash(path) != source["source_hash"]:
            raise ValueError("Source changed since catalog extraction")
        with pdfplumber.open(path) as pdf:
            for number, items in sorted(pages.items()):
                page = pdf.pages[number - 1]
                images = page.images
                used = set()
                for item in items:
                    loc = locations[item["location_id"]]
                    candidates = [(index, image) for index, image in enumerate(images)
                                  if index not in used and [round(float(image[k]), 4) for k in ("x0", "top", "x1", "bottom")] == item["bbox"]]
                    matches = []
                    for index, image in candidates:
                        try:
                            digest = hashlib.sha256(image["stream"].get_data()).hexdigest()
                        except Exception:  # noqa: BLE001 - retain failed payloads, never call them deduplicated
                            digest = None
                        if digest == item["attributes"].get("image_sha256"):
                            matches.append((index, image, digest))
                    if not matches:
                        raise ValueError("Saved visual region no longer matches original PDF image")
                    index, image, digest = matches[0]
                    used.add(index)
                    pixels = list(image.get("srcsize", []))
                    bbox = item["bbox"]
                    records.append({
                        "region_id": item["block_id"], "source_id": source_id,
                        "source_hash": source["source_hash"], "relative_path": source["relative_path"],
                        "location_id": item["location_id"], "page": number, "bbox": bbox,
                        "image_sha256": digest, "source_pixel_size": pixels,
                        "bits_per_component": image.get("bits"), "is_image_mask": bool(image.get("imagemask")),
                        "drawn_size_pt": [round(bbox[2] - bbox[0], 4), round(bbox[3] - bbox[1], 4)],
                        "page_bbox": list(page.bbox), "product": loc.get("product", "unknown"),
                        "version": loc.get("version", "unknown"),
                        "triage_class": _geometry_class(bbox, list(page.bbox)),
                        "semantic_class": "unreviewed", "review_status": "unreviewed",
                        "duplicate_group_id": stable_id("raster", digest, pixels, image.get("bits"), bool(image.get("imagemask"))) if digest else None,
                    })
                page.close()
    groups: dict[str, list[str]] = defaultdict(list)
    for item in records:
        if item["duplicate_group_id"]:
            groups[item["duplicate_group_id"]].append(item["region_id"])
    return {"schema_version": 1, "mode": "local_read_only_sources", "regions": records,
            "duplicate_groups": [{"group_id": group_id, "region_ids": ids} for group_id, ids in sorted(groups.items())],
            "summary": inventory_summary(records)}


def inventory_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"regions": len(records), "sources": len({r["source_hash"] for r in records}),
            "unique_decoded_hashes": len({r["image_sha256"] for r in records if r["image_sha256"]}),
            "unique_raster_geometry_groups": len({r["duplicate_group_id"] for r in records if r["duplicate_group_id"]}),
            "hash_unavailable": sum(not r["image_sha256"] for r in records),
            "triage_classes": dict(sorted(Counter(r["triage_class"] for r in records).items())),
            "per_source": dict(sorted(Counter(r["relative_path"] for r in records).items())),
            "semantic_classification_complete": False,
            "important_text_bearing_total": None,
            "limitation": "Geometry and duplicate hashes do not identify all important text-bearing images; unreviewed groups remain gaps."}


def select_dev_pages(inventory: dict[str, Any], dev_cases_path: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    """Use ONLY explicitly selected dev slots, then cover every PDF source."""
    if "holdout" in dev_cases_path.name.casefold():
        raise ValueError("Holdout files must not guide visual recovery")
    cases = json.loads(dev_cases_path.read_text(encoding="utf8"))
    chosen = [c for c in cases if c["id"] in case_ids]
    if len(chosen) != len(case_ids) or any(c.get("split") != "dev" for c in chosen):
        raise ValueError("Every requested case must be an explicit dev case")
    pages: dict[tuple[str, int], dict[str, Any]] = {}
    for case in chosen:
        for slot in case["required_evidence_slots"]:
            key = (slot["file_sha256"], slot["physical_page"])
            pages.setdefault(key, {"source_hash": key[0], "page": key[1], "case_ids": [], "reason": "selected_dev_source_slot"})["case_ids"].append(case["id"])
    regions = inventory["regions"]
    for source_hash in sorted({r["source_hash"] for r in regions} - {key[0] for key in pages}):
        counts: Counter[int] = Counter()
        for region in regions:
            if region["source_hash"] == source_hash:
                counts[region["page"]] += 3 if region["triage_class"] == "wide_strip_candidate" else 1
        page = min(counts, key=lambda number: (-counts[number], number))
        pages[(source_hash, page)] = {"source_hash": source_hash, "page": page, "case_ids": [], "reason": "deterministic_remaining_source_geometry_sample"}
    return [pages[key] for key in sorted(pages)]


def render_review_packet(inventory: dict[str, Any], data_root: Path, selected_pages: list[dict[str, Any]],
                         output: Path, *, scale: float = 2.5) -> dict[str, Any]:
    """Render original pages/crops and create PENDING transcription records."""
    import pypdfium2 as pdfium

    if not 1 <= scale <= 6:
        raise ValueError("Render scale must be between 1 and 6")
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    page_records = []
    for selection in selected_pages:
        regions = [r for r in inventory["regions"] if r["source_hash"] == selection["source_hash"] and r["page"] == selection["page"]]
        if not regions:
            page_records.append({**selection, "status": "no_catalog_image_regions"})
            continue
        source = (data_root / regions[0]["relative_path"]).resolve()
        if not source.is_relative_to(data_root.resolve()) or file_hash(source) != selection["source_hash"]:
            raise ValueError("Selected source path/hash is not the unchanged allowed original")
        prefix = f'{selection["source_hash"][:12]}-p{selection["page"]:04d}'
        with pdfium.PdfDocument(str(source)) as pdf:
            page = pdf[selection["page"] - 1]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil().copy()
            width, height = page.get_size()
            expected = regions[0]["page_bbox"]
            if abs(width - (expected[2] - expected[0])) > 0.1 or abs(height - (expected[3] - expected[1])) > 0.1:
                raise ValueError("Renderer/canonical page coordinates differ; manual mapping required")
            bitmap.close()
            page.close()
        page_path = output / f"{prefix}-page.png"
        image.save(page_path)
        page_records.append({**selection, "status": "rendered", "relative_path": regions[0]["relative_path"],
                             "page_png": str(page_path.resolve()), "regions": len(regions), "render_scale": scale})
        for region in regions:
            bbox = region["bbox"]
            rectangle = (max(0, math.floor((bbox[0] - expected[0]) * scale)),
                         max(0, math.floor((bbox[1] - expected[1]) * scale)),
                         min(image.width, math.ceil((bbox[2] - expected[0]) * scale)),
                         min(image.height, math.ceil((bbox[3] - expected[1]) * scale)))
            if rectangle[2] <= rectangle[0] or rectangle[3] <= rectangle[1]:
                raise ValueError("Image region lies outside rendered page")
            crop_path = output / f'{prefix}-{region["region_id"][:12]}.png'
            image.crop(rectangle).save(crop_path)
            candidates.append({key: region[key] for key in ("region_id", "source_hash", "page", "bbox", "image_sha256", "product", "version")} | {
                "approval_status": "pending", "kind": "figure_transcription", "text": "",
                "reviewer": "", "reviewed_at": "", "review_id": "", "candidate_method": "manual_from_original_render",
                "case_ids": selection.get("case_ids", []), "triage_class": region["triage_class"],
                "relative_path": region["relative_path"], "crop_path": str(crop_path.resolve()),
                "page_png": str(page_path.resolve()), "crop_sha256": file_hash(crop_path),
                "transcription_status": "not_started", "semantic_class": "unreviewed",
            })
        image.close()
    packet = {"schema_version": 1, "sources_read_only": True, "approval_performed": False,
              "pages": page_records, "candidate_count": len(candidates),
              "limitation": "Selected pages only; candidates are not searchable evidence until independently approved."}
    (output / "manifest.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf8")
    (output / "candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf8")
    return packet


def render_unique_visual_batch(inventory: dict[str, Any], data_root: Path, output: Path,
                               *, scale: float = 3, thumbnail_side: int = 512,
                               limit: int | None = None) -> dict[str, Any]:
    """Prepare all unique rasters for bounded external/local triage, WITHOUT OCR.

    A representative crop is not approval to propagate its transcription into
    every occurrence. The manifest retains each original contextual locator.
    Only these local source renders are written; no provider is contacted.
    """
    import pypdfium2 as pdfium
    from PIL import Image, ImageStat

    if not 1 <= scale <= 6 or not 64 <= thumbnail_side <= 1024:
        raise ValueError("Unsupported batch render scale or thumbnail size")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for region in inventory["regions"]:
        if region.get("duplicate_group_id"):
            groups[region["duplicate_group_id"]].append(region)
    group_ids = sorted(groups)
    if limit is not None:
        if limit < 1:
            raise ValueError("Batch limit must be positive")
        group_ids = group_ids[:limit]
    representatives: dict[str, dict[int, list[tuple[str, dict[str, Any]]]]] = defaultdict(lambda: defaultdict(list))
    for group_id in group_ids:
        representative = min(groups[group_id], key=lambda r: (r["relative_path"], r["page"], r["region_id"]))
        representatives[representative["relative_path"]][representative["page"]].append((group_id, representative))
    full_dir, thumb_dir = output / "full", output / "classifier"
    full_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1, "source_read_only": True, "ocr_performed": False, "provider_calls": 0,
        "approval_performed": False, "complete": False, "requested_groups": len(group_ids),
        "total_inventory_groups": len(groups), "render_scale": scale, "classifier_max_side": thumbnail_side,
        "groups": [], "limitation": "Classifier thumbnails can lose tiny symbols; use full crop and source page before literal approval. Duplicate uses require contextual review.",
    }

    def checkpoint() -> None:
        target = output / "batch_manifest.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
        temporary.replace(target)

    completed_pages = 0
    for relative_path, pages in sorted(representatives.items()):
        path = (data_root / relative_path).resolve()
        expected_hash = next(iter(pages.values()))[0][1]["source_hash"]
        if not path.is_relative_to(data_root.resolve()) or file_hash(path) != expected_hash:
            raise ValueError("Batch source path/hash is not the unchanged permitted original")
        with pdfium.PdfDocument(str(path)) as pdf:
            for number, items in sorted(pages.items()):
                page = pdf[number - 1]
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil().copy()
                width, height = page.get_size()
                bitmap.close()
                page.close()
                for group_id, representative in items:
                    page_bbox = representative["page_bbox"]
                    if abs(width - (page_bbox[2] - page_bbox[0])) > 0.1 or abs(height - (page_bbox[3] - page_bbox[1])) > 0.1:
                        raise ValueError("Batch renderer/canonical page coordinates differ")
                    bbox = representative["bbox"]
                    rectangle = (max(0, math.floor((bbox[0] - page_bbox[0]) * scale)),
                                 max(0, math.floor((bbox[1] - page_bbox[1]) * scale)),
                                 min(image.width, math.ceil((bbox[2] - page_bbox[0]) * scale)),
                                 min(image.height, math.ceil((bbox[3] - page_bbox[1]) * scale)))
                    if rectangle[2] <= rectangle[0] or rectangle[3] <= rectangle[1]:
                        raise ValueError("Batch image region lies outside page")
                    crop = image.crop(rectangle)
                    profile = f"s{scale:g}-t{thumbnail_side}"
                    full_path, thumb_path = full_dir / f"{group_id}-{profile}.png", thumb_dir / f"{group_id}-{profile}.png"
                    crop.save(full_path)
                    grayscale = crop.convert("L")
                    extrema = grayscale.getextrema()
                    pixel_stats = {"grayscale_min": extrema[0], "grayscale_max": extrema[1],
                                   "grayscale_mean": round(ImageStat.Stat(grayscale).mean[0], 3),
                                   "uniform_rendered_pixels": extrema[0] == extrema[1]}
                    grayscale.close()
                    full_size = list(crop.size)
                    crop.thumbnail((thumbnail_side, thumbnail_side), Image.Resampling.LANCZOS)
                    crop.save(thumb_path)
                    crop.close()
                    manifest["groups"].append({
                        "group_id": group_id, "image_sha256": representative["image_sha256"],
                        "source_pixel_size": representative["source_pixel_size"],
                        "triage_class": representative["triage_class"], "semantic_class": "unreviewed",
                        "representative_region_id": representative["region_id"],
                        "full_crop_path": str(full_path.resolve()), "classifier_crop_path": str(thumb_path.resolve()),
                        "full_crop_sha256": file_hash(full_path), "classifier_crop_sha256": file_hash(thumb_path),
                        "full_render_pixel_size": full_size, "rendered_pixel_statistics": pixel_stats,
                        "occurrences": [{key: region[key] for key in (
                            "region_id", "source_hash", "source_id", "relative_path", "location_id", "page", "bbox", "image_sha256", "product", "version"
                        )} for region in groups[group_id]],
                    })
                image.close()
                completed_pages += 1
                if completed_pages % 25 == 0:
                    checkpoint()
        checkpoint()
    manifest["complete"] = len(manifest["groups"]) == len(group_ids)
    manifest["prepared_groups"] = len(manifest["groups"])
    manifest["prepared_occurrences"] = sum(len(g["occurrences"]) for g in manifest["groups"])
    checkpoint()
    return manifest


def make_visual_contact_sheets(batch_manifest: Path, output: Path, *, columns: int = 4,
                               rows: int = 5, tile_width: int = 280, tile_height: int = 220) -> dict[str, Any]:
    """Economical text-presence triage sheets with stable IDs, never OCR output.

    Sheet thumbnails cannot support approval of a formula or numeric table.
    Full crops and the parent page must be used for literal transcription.
    """
    from PIL import Image, ImageDraw

    if not 1 <= columns <= 8 or not 1 <= rows <= 8 or tile_width < 128 or tile_height < 128:
        raise ValueError("Unsupported contact sheet geometry")
    batch = json.loads(batch_manifest.read_text(encoding="utf8"))
    if not batch.get("complete"):
        raise ValueError("Contact sheets require a complete batch manifest")
    output.mkdir(parents=True, exist_ok=True)
    groups = sorted(batch["groups"], key=lambda group: group["group_id"])
    manifest = {"schema_version": 1, "batch_manifest_sha256": file_hash(batch_manifest), "sheets": [],
                "purpose": "text_presence_triage_only", "ocr_performed": False,
                "approval_performed": False, "tile_count": len(groups),
                "limitation": "Downsampled tiles cannot verify formulas, indices, units or numeric values."}
    per_sheet = columns * rows
    for offset in range(0, len(groups), per_sheet):
        sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#dddddd")
        draw = ImageDraw.Draw(sheet)
        records = []
        for index, group in enumerate(groups[offset:offset + per_sheet]):
            tile_id = f"V{offset + index + 1:05d}"
            x, y = (index % columns) * tile_width, (index // columns) * tile_height
            draw.rectangle((x + 2, y + 2, x + tile_width - 3, y + tile_height - 3), fill="white", outline="black")
            draw.text((x + 7, y + 7), f"{tile_id}  {group['group_id'][:10]}", fill="black")
            with Image.open(group["classifier_crop_path"]) as original:
                thumb = original.convert("RGB")
                thumb.thumbnail((tile_width - 14, tile_height - 34), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (x + (tile_width - thumb.width) // 2, y + 29 + (tile_height - 34 - thumb.height) // 2))
                thumb.close()
            records.append({"tile_id": tile_id, "group_id": group["group_id"], "representative_region_id": group["representative_region_id"],
                            "sheet_bbox": [x, y, x + tile_width, y + tile_height],
                            "full_crop_path": group["full_crop_path"], "occurrences": group["occurrences"]})
        path = output / f"triage-sheet-{offset // per_sheet + 1:04d}.png"
        sheet.save(path)
        sheet.close()
        manifest["sheets"].append({"path": str(path.resolve()), "sha256": file_hash(path), "tiles": records})
    (output / "contact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    return manifest


def verify_candidate_locators(inventory: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Mechanical independent mapping; explicitly does not judge transcript text."""
    originals = {r["region_id"]: r for r in inventory["regions"]}
    records = []
    for candidate in candidates:
        original = originals.get(candidate.get("region_id"))
        keys = ("source_hash", "page", "bbox", "image_sha256", "product", "version")
        locator_matches = original is not None and all(candidate.get(key) == original.get(key) for key in keys)
        crop = Path(candidate["crop_path"])
        crop_matches = crop.is_file() and file_hash(crop) == candidate.get("crop_sha256")
        records.append({"region_id": candidate.get("region_id"), "locator_matches_original_inventory": locator_matches,
                        "crop_hash_matches": crop_matches, "transcript_semantically_verified": False,
                        "source": {key: original[key] for key in ("relative_path", *keys)} if original else None})
    return {"schema_version": 1, "mode": "mechanical_locator_verification_only", "records": records,
            "all_locators_match": all(r["locator_matches_original_inventory"] and r["crop_hash_matches"] for r in records),
            "semantic_approval_performed": False}
