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
    "verification_scope", "semantic_verified", "structure", "header_inference", "group_row_count", "literal_column_counts"})


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
        compact["untranscribed_visual_regions_present"] = bool(
            source.get('untranscribed_visual_regions_present')
            or attributes.get('untranscribed_visual_regions_present')
            or any(not region.get("approved_transcription_id") for region in regions))
    else:
        compact["visual_region_count"] = source.get("visual_region_count", 0)
        compact["untranscribed_visual_regions_present"] = bool(
            source.get("untranscribed_visual_regions_present", False)
            or attributes.get('untranscribed_visual_regions_present'))
    compact["model_metadata_projection"] = "compact-v1"
    return compact


def navigation_reference(identity: str) -> str:
    """Short, stable model locator; server bindings retain the complete ID."""
    return 'n_' + hashlib.sha256(identity.encode('utf-8')).hexdigest()[:10]


def bounded_navigation(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Keep raw authorization bindings and their visible hints in exact sync."""
    accepted, visible, identities = [], [], set()
    aliases: dict[str, set[str]] = {}
    for item in items:
        aliases.setdefault(navigation_reference(item['id']), set()).add(item['id'])
    collisions = {alias for alias, ids in aliases.items() if len(ids) > 1}
    truncated = bool(collisions)
    for item in items:
        alias = navigation_reference(item['id'])
        if item['id'] in identities or alias in collisions:
            continue
        compact = {key: item[key] for key in ('id', 'page', 'chm_member', 'file_name', 'hint', 'preview', 'not_citable') if key in item}
        compact['id'] = alias
        if len(visible) >= 100 or len(json.dumps([*visible, compact], ensure_ascii=False)) > 10_000:
            truncated = True
            break
        accepted.append(item)
        visible.append(compact)
        identities.add(item['id'])
    return accepted, visible, truncated


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


def atomic_unit_fingerprint(source: dict[str, Any], text: str) -> str | None:
    """Bind a complete visible unit to exact server provenance and text bytes.

    This is only a packing exclusion within one turn. It cannot authorize a
    read, merge citations, deduplicate other locations or assert sufficiency.
    Missing provenance and partial text never match a newly resolved unit.
    """
    fields = ('source_id', 'source_hash', 'location_id', 'block_id', 'product', 'version')
    if (source.get('atomic_evidence') is not True or not isinstance(text, str) or not text
            or not all(isinstance(source.get(key), str) and source[key] for key in fields)):
        return None
    binding = json.dumps([*(source[key] for key in fields), text], ensure_ascii=False)
    return hashlib.sha256(binding.encode('utf-8')).hexdigest()


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


def allocate_search_batch(
    existing: list[dict[str, Any]], sources: list[dict[str, Any]], contexts: list[str], *, budget: int,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Allocate fresh evidence together, retaining every previously visible fact.

    Allocation is over *increments*, so repeated evidence cannot consume the
    new-search quota a second time. Atomic records remain whole; prose shares
    the remaining space. Omitted titles are navigation hints, never evidence.
    """
    from app.services.rag import RAGService

    previous = {evidence_identity(row['source']): str(row['text']) for row in existing}
    candidates: dict[str, tuple[dict[str, Any], str]] = {}
    for index, source in enumerate(sources):
        text = str(contexts[index]) if index < len(contexts) else str(source.get('snippet') or source.get('quote') or '')
        identity = evidence_identity(source)
        old = candidates.get(identity, (source, previous.get(identity, '')))[1]
        if text in old:
            combined = old
        elif old in text:
            combined = text
        else:
            combined = old + ('\n\n[Отдельный фрагмент того же источника]\n' if old else '') + text
        candidates[identity] = (source, combined)
    remaining = max(0, budget - sum(map(len, previous.values())))
    entries = list(candidates.items())
    needs = [max(0, len(text) - len(previous.get(identity, ''))) for identity, (_source, text) in entries]
    grants = [0] * len(entries)
    ordinary = []
    for index, (_identity, (source, _text)) in enumerate(entries):
        if source.get('atomic_evidence'):
            if needs[index] <= remaining:
                grants[index] = needs[index]
                remaining -= needs[index]
        else:
            ordinary.append(index)
    for index, grant in zip(ordinary, RAGService._context_allocations(
            [' ' * needs[index] for index in ordinary], remaining), strict=True):
        grants[index] = grant
    accepted, omitted = {}, []
    for index, (identity, (source, text)) in enumerate(entries):
        old = previous.get(identity, '')
        capacity = len(old) + grants[index]
        if len(text) <= capacity:
            bounded = text
        elif source.get('atomic_evidence'):
            bounded = old
        elif old and text.startswith(old):
            # Bound only the added material. Never trim the old citation body.
            suffix = text[len(old):]
            bounded = old + bounded_evidence(suffix, grants[index])
        else:
            bounded = bounded_evidence(text, capacity)
            if old and old not in bounded:
                bounded = old
        accepted[identity] = bounded
        if bounded != text:
            omitted.append({'source': source_for_model(source),
                            'heading_hint': next((line.strip()[:240] for line in text.splitlines() if line.strip()), ''),
                            'reason': 'evidence_budget', 'not_citable': True})
    return accepted, omitted


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
    if any(source.get("figure_regions") or source.get("untranscribed_visual_regions_present")
           or (source.get('block_attributes') or {}).get('untranscribed_visual_regions_present') for source in sources):
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
    _, navigation, navigation_truncated = bounded_navigation(result.get('_navigation', []))
    return {"sources": sources, "citation_evidence": records,
            "knowledge_status": knowledge_status,
            "packet": {"contract": "prepared_evidence_v1", "evidence": records,
                       "navigation": navigation,
                       "navigation_truncated": bool(result.get('navigation_truncated')) or navigation_truncated,
                       "searches": [{"query": query, "status": status}],
                       "context_truncated": truncated,
                       "limitations": evidence_limitations(result, sources, truncated=truncated)}}
