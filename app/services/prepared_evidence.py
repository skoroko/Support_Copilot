"""Topic-neutral preparation of retrieved evidence, never a sufficiency verdict.

Retrieval scores select recall candidates. Only the answering model can decide
whether their actual contents support the question or need a targeted search.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_MODEL_SOURCE_FIELDS = frozenset({
    "chunk_id", "document_key", "file_name", "source_locator", "page", "printed_page",
    "product", "version", "language", "chm_member", "section", "section_path",
    "title", "heading", "block_type", "atomic_evidence", "origin", "relation",
    "related_to", "extraction_warnings",
})
_MODEL_QUALITY_FIELDS = frozenset({"header_status", "quality_status", "review_status", "visual_review_required",
    "verified", "transcription", "reading_order", "layout", "approved_transcription_id", "method",
    "verification_scope", "semantic_verified"})


def source_for_model(source: dict[str, Any]) -> dict[str, Any]:
    """Project runtime metadata; full geometry remains in API/catalog sources.

    Source scope is retained; service telemetry (absolute paths, hashes,
    geometry, extraction diagnostics) stays out of the model packet.  The
    document binding survives as a short ``document_key`` fingerprint so
    citation identity still separates same-named files without leaking host
    paths.  Full chunk membership stays in the saved API evidence, where it
    can be verified mechanically. Glyph arrays, image inventories,
    duplicated cells and page-relation graphs are extraction artifacts, not
    generation text. This projection is shared by initial/tools/expanded.
    """
    compact = {key: value for key, value in source.items() if key in _MODEL_SOURCE_FIELDS}
    document_key = _document_key(source)
    if document_key:
        compact["document_key"] = document_key
    attributes = source.get("block_attributes") or {}
    compact["block_attributes"] = {key: value for key, value in attributes.items()
                                    if key in _MODEL_QUALITY_FIELDS and isinstance(value, (str, bool, int, float, type(None)))}
    if "figure_regions" in source:
        regions = source.get("figure_regions") or []
        compact["visual_region_count"] = len(regions)
        compact["untranscribed_visual_regions_present"] = any(
            not region.get("approved_transcription_id") for region in regions)
    else:
        compact["visual_region_count"] = source.get("visual_region_count", 0)
        compact["untranscribed_visual_regions_present"] = bool(
            source.get("untranscribed_visual_regions_present", False))
    compact["model_metadata_projection"] = "compact-v1"
    return compact


def should_prepare_evidence(question: Any) -> bool:
    """Skip conversational/editing/action requests, without a topic registry."""
    if not isinstance(question, str) or not question.strip():
        return False
    text = " ".join(question.casefold().replace("ё", "е").split()).strip(" .!?…")
    if re.fullmatch(r"(?:привет|здравствуй(?:те)?|доброе утро|добрый (?:день|вечер)|"
                    r"спасибо(?: большое)?|благодарю|пока|до свидания|ок(?:ей)?|понятно|"
                    r"как дела|hello|hi|thanks?(?: you)?|bye|good (?:morning|evening)|ok)", text):
        return False
    # An imperative requests an effect. Questions about how to perform an
    # operation remain knowledge questions and are deliberately not excluded.
    if re.search(r"(?:^|[.!?;]\s*|\b(?:и|and)\s+)(?:пожалуйста[, ]+)?(?:отправь(?:те)?|пошли(?:те)?|"
                 r"перешли(?:те)?|удали(?:те)?|сохрани(?:те)?|опубликуй(?:те)?|"
                 r"выполни(?:те)?|запусти(?:те)?|создай(?:те)?|"
                 r"send|delete|save|publish|execute|run|create)\b", text):
        return False
    if re.search(r"^(?:можешь|мог бы|можете|can you|could you)\s+(?:пожалуйста\s+)?"
                 r"(?:отправить|удалить|сохранить|опубликовать|выполнить|запустить|"
                 r"send|delete|save|publish|execute|run)\b", text):
        return False
    return not re.search(r"^(?:пожалуйста[, ]+)?(?:перепиши(?:те)?|перефразируй(?:те)?|"
                 r"отредактируй(?:те)?|переведи(?:те)?|сократи(?:те)?|исправь(?:те)?|"
                 r"rewrite|rephrase|edit|translate|shorten|proofread)\b", text)


def _document_key(source: dict[str, Any]) -> str | None:
    """Short portable document fingerprint shared by raw and compact evidence."""

    key = source.get("document_key")
    if isinstance(key, str) and key:
        return key
    # Direct/related routes can carry different legacy parents for one block.
    # Bind raw and projected identities to the actual corpus source revision.
    source_id, source_hash = source.get("source_id"), source.get("source_hash")
    if all(isinstance(value, str) and value.strip() for value in (source_id, source_hash)):
        binding = json.dumps(["corpus-v1", source_id, source_hash], ensure_ascii=False)
        return hashlib.sha256(binding.encode("utf-8")).hexdigest()[:16]
    document_id = source.get("document_id")
    if not document_id:
        return None
    return hashlib.sha256(str(document_id).encode("utf-8")).hexdigest()[:16]


def evidence_identity(source: dict[str, Any]) -> str:
    """Citation identity shared by raw and compact saved evidence.

    The document binding is its fingerprint in both forms: raw records hash
    ``document_id`` on the fly, compact records carry the precomputed
    ``document_key``.  Per-block provenance fields stay out of the identity so
    a raw and a projected record for the same chunk still match.
    """
    fields = ("product", "version", "source_locator", "file_name",
              "chm_member", "page", "printed_page", "section")
    return json.dumps([source.get("chunk_id") or source.get("id"), _document_key(source),
                       *(source.get(key) for key in fields)], sort_keys=True, ensure_ascii=False)


def bounded_evidence(text: str, limit: int) -> str:
    """Keep complete rows/lines, and never leave a partial structured table."""
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    lines = text.splitlines(keepends=True)
    if len(lines) > 1 or re.match(r"Column \d+=", text) or "|" in text or "\t" in text:
        kept: list[str] = []
        used = 0
        table_start: int | None = None
        for line in lines:
            is_table = bool(re.match(r"\s*Column \d+=", line) or "|" in line or "\t" in line)
            if is_table and table_start is None:
                table_start = len(kept)
            elif not is_table:
                table_start = None
            if used + len(line) > limit:
                if is_table and table_start is not None:
                    kept = kept[:table_start]
                break
            kept.append(line)
            used += len(line)
        return "".join(kept).rstrip()
    prefix = text[:limit]
    endings = list(re.finditer(r'[.!?。！？][\]"»”)]*(?=\s|$)', prefix))
    if endings:
        return prefix[:endings[-1].end()].rstrip()
    return prefix.rsplit(" ", 1)[0].rstrip() if " " in prefix else ""


def evidence_limitations(result: dict[str, Any], sources: list[dict[str, Any]], *, truncated: bool = False) -> list[str]:
    """Use the same diagnostics for initial recall and subsequent tool calls."""
    telemetry = result.get("telemetry") or {}
    limitations = list(result.get("limitations") or [])
    planning = telemetry.get("retrieval_planning") or {}
    if planning.get("status") in {"failed", "deadline_exhausted", "invalid_input"}:
        limitations.append("query_planning_" + planning["status"])
    if planning.get("translated_query_omitted"):
        limitations.append("translated_query_budget_exhausted")
    if planning.get("history_truncated"):
        limitations.append("planning_history_truncated")
    for field, label in (("retrieval_partial_failure", "retrieval_partial_failure"),
                         ("related_evidence_truncated", "structural_expansion_truncated"),
                         ("evidence_dropped_for_budget", "evidence_dropped_for_budget")):
        if telemetry.get(field):
            limitations.append(label)
    if any((source.get("block_attributes") or {}).get("visual_review_required") for source in sources):
        limitations.append("unverified_formula_or_layout")
    if any(source.get("figure_regions") or source.get("untranscribed_visual_regions_present") for source in sources):
        limitations.append("visual_regions_not_transcribed")
    if result.get("status", "unavailable") not in {"ok", "no_evidence"}:
        limitations.append("retrieval_unavailable")
    if truncated or telemetry.get("context_truncated"):
        limitations.append("context_truncated")
    return list(dict.fromkeys(limitations))


def prepare_evidence(result: dict[str, Any], *, query: str, budget: int = 24_000) -> dict[str, Any]:
    """Package already gated recall and preserve miss/outage/truncation status."""
    from app.services.rag import MAX_SOURCE_SNIPPET_CHARS, RAGService

    status = str(result.get("status", "unavailable"))
    entries: dict[str, tuple[dict[str, Any], str]] = {}
    if status == "ok":
        for index, source in enumerate(result.get("sources") or []):
            contexts = result.get("contexts") or []
            if not isinstance(source, dict) or index >= len(contexts):
                continue
            text = contexts[index]
            if not isinstance(text, str) or not text.strip():
                continue
            identity = evidence_identity(source)
            previous = entries.get(identity)
            if previous is None:
                entries[identity] = (dict(source), text)
            elif text not in previous[1]:
                entries[identity] = (previous[0], text if previous[1] in text else
                                     previous[1] + "\n\n" + text)
    values = list(entries.values())
    allocations = RAGService._context_allocations([text for _, text in values], max(0, budget))
    sources, records = [], []
    truncated = bool((result.get("telemetry") or {}).get("context_truncated"))
    for (source, text), allocation in zip(values, allocations, strict=True):
        bounded = ((text if len(text) <= allocation else "") if source.get("atomic_evidence")
                   else bounded_evidence(text, allocation))
        truncated |= bounded != text
        if not bounded:
            continue
        number = len(records) + 1
        sources.append({**source, "snippet": bounded[:MAX_SOURCE_SNIPPET_CHARS]})
        records.append({"citation_number": number, "marker": f"[{number}]",
                        "source": source_for_model(source),
                        "text": bounded})
    knowledge_status = "ok" if records else ("no_evidence" if status in {"ok", "no_evidence"} else status)
    return {"sources": sources, "citation_evidence": records,
            "knowledge_status": knowledge_status,
            "packet": {"contract": "prepared_evidence_v1", "evidence": records,
                       "searches": [{"query": query, "status": status}],
                       "context_truncated": truncated,
                       "limitations": evidence_limitations(result, sources, truncated=truncated)}}
