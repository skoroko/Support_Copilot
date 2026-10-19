"""Source-bound literal review of CHM assets and PDF render regions.

These locators extend raster-image review without inventing PDF images, CHM
pages or engineering meaning. Local source verification never calls a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from app.services.corpus import (
    apply_reviewed_relations,
    file_hash,
    source_identity,
    stable_id,
)

_KINDS = {"formula", "table_row", "figure_transcription", "definition"}
_NATIVE = {"paragraph", "definition", "formula_candidate", "table_row", "table_header"}
_SCOPE = "literal_source_transcription_only"
_RECIPE = "pdf_vector_pdftoppm300_margin2_v1"
_ROUNDING = "floor min; ceil max; bbox expanded by margin_points before scaling"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _bounds(value: Any) -> bool:
    return bool(isinstance(value, list) and len(value) == 4
        and all(type(v) in {float, int} and math.isfinite(v) for v in value)
        and value[0] < value[2] and value[1] < value[3])


def _contains(outer: list[float], inner: list[float]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def native_snapshot(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: b[key] for key in ("block_id", "bbox", "text")} for b in blocks]


def _native_contributors(locator: dict, metadata: dict, blocks: dict[str, dict]) -> list[dict]:
    ids = locator.get("contributing_native_block_ids")
    bbox = locator.get("bbox")
    page_bbox = metadata.get("page_bbox")
    if (type(locator.get("page")) is not int or locator["page"] < 1
            or locator["page"] != metadata.get("page") or not _bounds(bbox)
            or not _bounds(page_bbox) or not _contains(page_bbox, bbox)
            or not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
            or any(not isinstance(i, str) or i not in blocks for i in ids)):
        raise ValueError("PDF review requires exact physical page and bounded unique native blocks")
    selected = [blocks[identity] for identity in ids]
    if any(b.get("kind") not in _NATIVE or b.get("attributes", {}).get("method") == "reviewed_visual_transcription"
           or b.get("source_id", metadata.get("source_id")) != metadata.get("source_id")
           or b.get("location_id", metadata.get("location_id")) != metadata.get("location_id")
           or not _bounds(b.get("bbox")) or not _contains(bbox, b["bbox"]) for b in selected):
        raise ValueError("PDF crop must contain every exact same-page native contributor")
    if locator.get("contributing_native_blocks_sha256") != canonical_sha256(native_snapshot(selected)):
        raise ValueError("Contributing native block text or geometry changed")
    return selected


def pdf_pixel_crop(bbox: list[float], *, dpi: int = 300, margin_points: int = 2) -> list[int]:
    scale = dpi / 72
    x, y = math.floor((bbox[0] - margin_points) * scale), math.floor((bbox[1] - margin_points) * scale)
    return [x, y, math.ceil((bbox[2] + margin_points) * scale) - x,
            math.ceil((bbox[3] + margin_points) * scale) - y]


def _validate_recipe(locator: dict, page_bbox: list[float]) -> list[int]:
    recipe = locator.get("recipe")
    if (not isinstance(recipe, dict) or set(recipe) != {"renderer", "version", "dpi", "margin_points",
            "pixel_crop", "pixel_rounding", "recipe_version"}
            or recipe.get("renderer") != "pdftoppm" or recipe.get("recipe_version") != _RECIPE
            or type(recipe.get("dpi")) is not int or recipe["dpi"] != 300
            or type(recipe.get("margin_points")) is not int or recipe["margin_points"] != 2
            or recipe.get("pixel_rounding") != _ROUNDING
            or not isinstance(recipe.get("version"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}", recipe["version"])):
        raise ValueError("Unsupported exact local render recipe")
    # This bounded recipe uses top-left page coordinates and unrotated zero-origin
    # pages. Other page-coordinate conventions require their own reviewed recipe.
    if page_bbox[:2] != [0, 0]:
        raise ValueError("Render recipe requires zero-origin page coordinates")
    crop = pdf_pixel_crop(locator["bbox"])
    if (recipe["pixel_crop"] != crop or any(type(v) is not int for v in recipe["pixel_crop"])
            or min(crop) < 0 or crop[2] * crop[3] > 40_000_000):
        raise ValueError("Render pixel crop differs from the reviewed bounded geometry")
    return crop


def render_pdf_region_sha256(source: Path, locator: dict, *, page_bbox: list[float]) -> str:
    """Reproduce a bounded original Poppler crop. No source extraction or OCR."""
    from pypdf import PdfReader

    crop = _validate_recipe(locator, page_bbox)
    original_page = PdfReader(source).pages[locator["page"] - 1]
    if (original_page.rotation != 0 or list(original_page.mediabox.lower_left) != [0, 0]
            or abs(float(original_page.mediabox.width) - page_bbox[2]) > .1
            or abs(float(original_page.mediabox.height) - page_bbox[3]) > .1):
        raise ValueError("Original PDF page coordinates differ from the reviewed render recipe")
    version = subprocess.run(["pdftoppm", "-v"], capture_output=True, check=True, timeout=15)
    match = re.search(rb"pdftoppm version ([^\s]+)", version.stdout + version.stderr)
    if not match or match.group(1).decode("ascii") != locator["recipe"]["version"]:
        raise ValueError("Original reviewed renderer version is unavailable")
    with tempfile.TemporaryDirectory(prefix="support-copilot-region-") as temporary:
        target = Path(temporary) / "region"
        page = str(locator["page"])
        subprocess.run(["pdftoppm", "-f", page, "-l", page, "-r", "300",
            "-x", str(crop[0]), "-y", str(crop[1]), "-W", str(crop[2]), "-H", str(crop[3]),
            "-png", "-singlefile", str(source), str(target)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=120)
        return file_hash(target.with_suffix(".png"))


def _member(value: Any) -> str:
    if (not isinstance(value, str) or not value.startswith("/") or value.startswith("//")
            or "\\" in value or "\x00" in value or ":" in value
            or any(piece in {"", ".", ".."} for piece in value[1:].split("/"))
            or unquote(value) != value):
        raise ValueError("Exact canonical archive member required")
    return value


def _asset_href(member: str, href: str) -> str:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or "\\" in href or "\x00" in href:
        raise ValueError("CHM image must reference a local archive asset")
    path = unquote(parsed.path)
    stack = [] if path.startswith("/") else list(PurePosixPath(member).parent.parts[1:])
    for piece in path.split("/"):
        if piece in {"", "."}:
            continue
        if piece == "..":
            if not stack:
                raise ValueError("Archive asset href escapes original archive root")
            stack.pop()
        else:
            stack.append(piece)
    return _member("/" + "/".join(stack))


def original_chm_asset_sha256(source: Path, locator: dict, image: dict) -> str:
    """Read exact original topic and asset; never substitute a similar filename."""
    from bs4 import BeautifulSoup
    from pylibmspack import ChmArchive

    from app.services.chm_reader import CHMReader

    member, asset = _member(locator["member"]), _member(locator["asset_member"])
    if _asset_href(member, image.get("attributes", {}).get("src", "")) != asset:
        raise ValueError("Persisted image block does not reference the exact approved asset")
    reader = CHMReader()
    with tempfile.TemporaryDirectory(prefix="support-copilot-reviewed-chm-") as temporary:
        archive = ChmArchive(str(reader._ascii_compatible_path(source, Path(temporary))))
        members = {str(row["name"]): row for row in archive.files(include_system=False)}
        if member not in members or asset not in members:
            raise ValueError("Original CHM topic or image asset absent")

        def exact_read(name: str) -> bytes:
            try:
                return archive.read(name, max_size=16 * 1024 * 1024)
            except FileNotFoundError:
                if name.isascii():
                    raise
                return reader._read_exact_member_ascii_output(archive, name, members[name].get("size"))

        html = BeautifulSoup(reader._decode_html(exact_read(member)), "html.parser")
        found = False
        for tag in html.find_all("img"):
            try:
                found |= _asset_href(member, str(tag.get("src") or "")) == asset
            except ValueError:
                continue
        if not found:
            raise ValueError("Original CHM member does not reference the approved image")
        return hashlib.sha256(exact_read(asset)).hexdigest()


def validate_visual_locator(document: Any, record: dict, data_root: Path) -> tuple[dict, list[dict]]:
    """Validate original locators even for pending review packets; never approves text."""
    metadata = document.metadata
    locator = record.get("visual_locator")
    if not isinstance(locator, dict):
        raise ValueError("Exact visual locator required")  # noqa: TRY004 - malformed sidecar schema
    keys = ("source_id", "source_hash", "location_id", "product", "version")
    if any(record.get(key) != metadata.get(key) or not record.get(key) for key in keys):
        raise ValueError("Visual review source/location/product/version mismatch")
    relative = record.get("relative_path")
    if (not isinstance(relative, str) or "\\" in relative or ":" in relative
            or relative.startswith("/") or any(piece in {"", ".", ".."} for piece in relative.split("/"))):
        raise ValueError("Safe relative corpus source path required")
    source = (data_root.resolve() / relative).resolve()
    if (not source.is_relative_to(data_root.resolve()) or not source.is_file()
            or source_identity(source, data_root)[0] != metadata["source_id"]
            or file_hash(source) != metadata["source_hash"]):
        raise ValueError("Original visual source bytes or permitted data path changed")
    blocks = {b["block_id"]: b for b in metadata.get("evidence_blocks", [])}
    if locator.get("kind") == "chm_asset":
        if (set(locator) != {"kind", "member", "asset_member", "asset_sha256", "source_image_block_id"}
                or source.suffix.lower() != ".chm" or locator.get("member") != metadata.get("chm_member")
                or metadata["location_id"] != stable_id("location", metadata["source_id"], locator["member"])
                or record.get("page") is not None or record.get("bbox") is not None
                or "image_sha256" in record):
            raise ValueError("CHM review requires exact original topic, never synthetic PDF/image geometry")
        _member(locator["member"])
        _member(locator["asset_member"])
        image = blocks.get(locator.get("source_image_block_id"))
        if not image or image.get("kind") != "image":
            raise ValueError("Exact source CHM image block required")
        if original_chm_asset_sha256(source, locator, image) != locator.get("asset_sha256"):
            raise ValueError("Original CHM asset hash differs from reviewed asset")
        selected = [image]
    elif locator.get("kind") == "pdf_render_region":
        if (set(locator) != {"kind", "page", "bbox", "render_sha256", "recipe",
                            "contributing_native_block_ids", "contributing_native_blocks_sha256"}
                or source.suffix.lower() != ".pdf" or "image_sha256" in record
                or record.get("page") != locator.get("page") or record.get("bbox") != locator.get("bbox")):
            raise ValueError("PDF render review requires exact source/page/crop, never a synthetic image hash")
        selected = _native_contributors(locator, metadata, blocks)
        if render_pdf_region_sha256(source, locator, page_bbox=metadata["page_bbox"]) != locator.get("render_sha256"):
            raise ValueError("Original PDF rendered crop differs from reviewed crop")
    else:
        raise ValueError("Unsupported visual locator kind")
    if file_hash(source) != metadata["source_hash"]:
        raise ValueError("Original visual source changed during verification")
    return deepcopy(locator), selected


def valid_literal_render_relation(relation: dict, transcript: dict, metadata: dict, blocks: dict) -> bool:
    """Offline catalog proof for source derivation, not mathematical interpretation."""
    try:
        attrs = transcript["attributes"]
        locator = attrs["visual_locator"]
        selected = _native_contributors(locator, metadata, blocks)
        _validate_recipe(locator, metadata["page_bbox"])
        ids = [r.get("to_id") for r in transcript.get("relations", []) if r.get("kind") == "literal_render_of"]
        return bool(relation.get("kind") == "literal_render_of"
            and relation.get("to_id") in {b["block_id"] for b in selected}
            and len(ids) == len(selected) and set(ids) == {b["block_id"] for b in selected}
            and transcript.get("block_id") == stable_id(metadata["location_id"], "reviewed_visual_region", canonical_sha256(locator))
            and locator.get("kind") == "pdf_render_region"
            and re.fullmatch(r"[0-9a-f]{64}", locator.get("render_sha256", ""))
            and attrs.get("method") == "reviewed_visual_transcription" and attrs.get("verified") is True
            and attrs.get("verification_scope") == _SCOPE and attrs.get("semantic_verified") is False
            and transcript.get("bbox") == locator["bbox"] and transcript.get("kind") in _KINDS
            and all(attrs.get(key) == metadata.get(key) and metadata.get(key) is not None
                    for key in ("source_id", "source_hash", "location_id", "page", "product", "version"))
            and all(isinstance(attrs.get(key), str) and attrs[key].strip()
                    for key in ("review_id", "reviewer", "reviewed_at")))
    except (KeyError, TypeError, ValueError):
        return False


def apply_reviewed_visual_region(document: Any, record: dict, data_root: Path) -> bool:
    """Add one approved literal block, idempotently, retaining every original block."""
    if (record.get("approval_status") != "approved" or record.get("semantic_verified") is not False
            or record.get("verification_scope") != _SCOPE or record.get("kind") not in _KINDS
            or any(not isinstance(record.get(key), str) or not record[key].strip()
                   for key in ("text", "review_id", "reviewer", "reviewed_at"))):
        raise ValueError("Separately approved literal-only source review required")
    locator, selected = validate_visual_locator(document, record, data_root)
    metadata = document.metadata
    identity = stable_id(metadata["location_id"], "reviewed_visual_region", canonical_sha256(locator))
    existing = next((b for b in metadata["evidence_blocks"] if b["block_id"] == identity), None)
    attrs = {"method": "reviewed_visual_transcription", "verified": True, "verification_scope": _SCOPE,
        "semantic_verified": False, "visual_locator": locator,
        **{key: metadata[key] for key in ("source_id", "source_hash", "location_id", "product", "version")},
        **{key: record[key] for key in ("review_id", "reviewer", "reviewed_at")}}
    if locator["kind"] == "pdf_render_region":
        attrs["page"] = locator["page"]
    else:
        attrs["chm_member"] = locator["member"]
        rows = record.get("reviewed_relations", [])
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("reviewed_relations must be a list of separately reviewed associations")
        if any(row.get("approval_status") == "approved" for row in rows):
            raise ValueError("Reviewed semantic associations are bounded to physical PDF pages")
    if existing:
        if (existing.get("kind") != record["kind"] or existing.get("text") != record["text"]
                or existing.get("attributes") != attrs or existing.get("bbox") != locator.get("bbox")):
            raise ValueError("Conflicting approved visual region; explicit revision review required")
        return apply_reviewed_relations(document, existing, record)
    relation_kind = "literal_render_of" if locator["kind"] == "pdf_render_region" else "transcription_of"
    section = selected[0].get("section_path", [])
    if any(b.get("section_path", []) != section for b in selected):
        section = []
    transcript = {"block_id": identity, "kind": record["kind"], "text": record["text"],
        "ordinal": max((b.get("ordinal", -1) for b in metadata["evidence_blocks"]), default=-1) + 1,
        "section_path": deepcopy(section), "anchor": None, "bbox": locator.get("bbox"), "attributes": attrs,
        "relations": [{"kind": relation_kind, "to_id": b["block_id"]} for b in selected]}
    apply_reviewed_relations(document, transcript, record)
    metadata["evidence_blocks"].append(transcript)
    metadata.setdefault("relations", []).extend({"from_id": identity, **r} for r in transcript["relations"]
        if r["kind"] == relation_kind)
    document.set_content(document.text + "\n\n" + record["text"])
    metadata["extraction_status"] = "text"
    return True
