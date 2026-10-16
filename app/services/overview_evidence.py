"""Conservative overview recall plans and deterministic evidence packaging.

Plans contain search terms, never answers. Unknown, compound and procedural
questions stay on the ordinary agent path. Coverage is a hint, not permission
to bypass grounding or disable a necessary follow-up search.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.services.rag import MAX_SOURCE_SNIPPET_CHARS, RAGService

_MODELS = {
    "Hardening Soil": ("hardening soil", "харденинг соил", "харденинг сойл"),
    "Hardening Soil small strain": ("hardening soil small strain", "hs small"),
    "Mohr Coulomb": ("mohr coulomb", "мора кулона", "мор кулон"),
    "Modified Mohr Coulomb": ("modified mohr coulomb", "модифицированная мора кулона"),
    "Soft Soil": ("soft soil", "софт соил"),
    "Soft Soil Creep": ("soft soil creep",),
    "Modified Cam Clay": ("modified cam clay",),
    "Hoek Brown": ("hoek brown", "хоека брауна", "хоука брауна"),
    "Drucker Prager": ("drucker prager", "друкера прагера"),
    "Jardine": ("jardine", "жардина"),
}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[‐‑–—-]", " ", text.casefold())).strip(" .?!")


@dataclass(frozen=True)
class OverviewPlan:
    kind: str
    subject: str
    queries: tuple[tuple[str, str], ...]


def plan_overview(question: str) -> OverviewPlan | None:
    """Recognize only a complete, explicitly named, bounded overview request."""
    if not isinstance(question, str) or len(question) > 250 or "\n" in question:
        return None
    text = _normalized(question)
    text = re.sub(r"\s+(?:в|in)\s+(?:midas\s+)?gts\s*nx$", "", text)
    if re.fullmatch(r"(?:какие граничные условия (?:есть|доступны)|"
                    r"(?:расскажи(?: мне)? (?:про|о)|обзор) граничны[ех] услови[яй]|"
                    r"what (?:boundary conditions are available|are the boundary conditions))", text):
        return OverviewPlan("boundaries", "Boundary Condition", (
            ("Граничные условия GTS NX: общая классификация и виды анализа",
             "GTS NX boundary condition classification categories and analysis types"),
            ("Граничные условия GTS NX: назначение и область применения",
             "GTS NX boundary conditions purpose scope and analysis types"),
        ))
    match = re.fullmatch(
        r"(?:расскажи(?:те)?(?: мне)?(?: кратко)? (?:про|о)|"
        r"дай(?: краткий)? обзор|обзор|что такое|describe|tell me about|what is) "
        r"(?:модель |модели |model )?(.+)", text,
    )
    if match is None:
        return None
    subject = next((name for name, aliases in _MODELS.items()
                    if match[1] in aliases), None)
    if subject is None:
        return None
    return OverviewPlan("material", subject, (
        (f"Модель {subject} GTS NX: описание поведения и область применения",
         f"GTS NX {subject} model behavior description and scope"),
        (f"Модель {subject} GTS NX: основные параметры и их определения",
         f"GTS NX {subject} model material parameters and definitions"),
    ))


def _identity(source: dict[str, Any]) -> str:
    # Only stable chunk identities can safely deduplicate two independently
    # retrieved records; an incomplete metadata tuple is not sufficient.
    return str(source.get("chunk_id") or source.get("id") or "")


def can_skip_second_facet(plan: OverviewPlan, prepared: dict[str, Any]) -> bool:
    """Skip redundant prefetch only for explicit, visible overview facets.

    This never limits the model's normal follow-up tools, and says nothing
    about evidence that might exist elsewhere in the corpus.
    """
    packet = prepared["packet"]
    if prepared["failed_queries"] or not packet["evidence"]:
        return False
    if plan.kind == "boundaries":
        return all(packet["coverage_checks"].values())
    if plan.subject not in {"Hardening Soil", "Hardening Soil small strain"}:
        return False
    texts = [_normalized(r["text"]) for r in packet["evidence"]]
    description = any("hardening soil" in t and
        ("modified mohr coulomb" in t or "модифицирован" in t) and
        ("stiffness" in t or "жестк" in t or "жёстк" in t) for t in texts)
    compact = [re.sub(r"\s+", "", t) for t in texts]
    small_parameters = any("g0ref" in t and ("γ0.7" in t or "gamma0.7" in t) for t in compact)
    stiffness_table = any(all(term in t for term in ("e50ref", "eoedref", "eurref"))
        and "smallstrain" in t for t in compact)
    return description and small_parameters and stiffness_table


def package_overview(
    plan: OverviewPlan, results: list[dict[str, Any]], *, budget: int = 24_000,
) -> dict[str, Any]:
    """Merge already gated RAG evidence; bound rows before assigning citations."""
    entries: dict[str, tuple[dict[str, Any], str]] = {}
    # Interleave independent facets so budget pressure cannot starve the second.
    batches = [list(zip(r.get("sources", []), r.get("contexts", []), strict=True))
               if r.get("status") == "ok" else [] for r in results]
    for position in range(max(map(len, batches), default=0)):
        for facet, batch in enumerate(batches):
            if position >= len(batch):
                continue
            source, text = batch[position]
            if not isinstance(text, str) or not text.strip():
                continue
            key = _identity(source) or f"anonymous-{facet}-{position}"
            previous = entries.get(key)
            if previous is None:
                entries[key] = (dict(source), text)
            elif text not in previous[1]:
                # Query-specific CHM projections can expose complementary rows
                # of the same chunk. Length alone cannot establish containment.
                merged = text if previous[1] in text else (
                    previous[1] + "\n\n[Отдельный фрагмент того же источника]\n" + text)
                entries[key] = (previous[0], merged)
    values = list(entries.values())
    allocations = RAGService._context_allocations([text for _, text in values], budget)
    records = []
    sources = []
    for (source, text), allocation in zip(values, allocations, strict=True):
        bounded = RAGService._bounded_context(text, allocation)
        if not bounded:
            continue
        number = len(records) + 1
        sources.append({**source, "snippet": bounded[:MAX_SOURCE_SNIPPET_CHARS]})
        records.append({"citation_number": number, "marker": f"[{number}]",
                        "source": {k: v for k, v in source.items() if k not in {"snippet", "quote"}},
                        "text": bounded})
    texts = [_normalized(record["text"]) for record in records]
    if plan.kind == "boundaries":
        # Draining Condition and a common menu breadcrumb are NOT proof of
        # seepage-analysis applicability or of the complete classification.
        checks = {
            "mechanical_category": any("dof" in t and "stress analysis" in t for t in texts),
            "seepage_category": any("seepage boundary" in t and "seepage analysis" in t for t in texts),
            "dynamic_category": any("boundary element" in t and "equivalent linear" in t for t in texts),
        }
    else:
        subject = _normalized(plan.subject)
        own = [t for t in texts if subject in t or (
            plan.subject == "Hardening Soil small strain" and "hardening soil" in t and "small strain" in t)]
        checks = {
            "named_model_description": any(any(word in t for word in
                ("behavior", "behaviour", "поведени", "жестк", "жёстк", "stiffness")) for t in own),
            "named_model_parameters": any(any(word in t for word in
                ("parameter", "параметр")) for t in own),
        }
    # Successful retrieval is not a completeness claim; partial outages retain
    # useful evidence but explicitly request validation/additional recall.
    missing = [name for name, covered in checks.items() if not covered]
    failed = sum(r.get("status") not in {"ok", "no_evidence"} for r in results)
    if failed:
        missing.append("retrieval_unavailable")
    searches = [{"query": query, "english_query": english, "status": result.get("status", "unavailable")}
                for (query, english), result in zip(plan.queries, results, strict=False)]
    return {"sources": sources, "citation_evidence": records,
            "knowledge_status": "ok" if records else "unused",
            "packet": {"kind": plan.kind, "subject": plan.subject,
                       "coverage_checks": checks, "missing_aspects": missing, "searches": searches,
                       "evidence": records},
            "failed_queries": failed, "context_chars": sum(len(r["text"]) for r in records)}
