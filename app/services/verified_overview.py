"""A narrow, source-attributed overview contract before tool-free generation."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Any

from app.services.overview_evidence import OverviewPlan, _normalized, package_overview

PILOT_SUBJECT = "Hardening Soil small strain"
_DIGEST = re.compile(r"[0-9a-f]{64}")


def supports_verified_overview(plan: OverviewPlan | None) -> bool:
    # Base HS, boundary classifications, other versions and compound questions
    # need their own structural contracts before entering this pilot.
    return plan is not None and plan.kind == "material" and plan.subject == PILOT_SUBJECT


def focused_verified_plan(plan: OverviewPlan) -> OverviewPlan:
    """Recall the actual stiffness rows before deciding evidence is complete.

    Live evaluation showed a generic parameter query repeatedly retrieved only
    descriptions. Explicit parameter names recall the table without weakening
    the exact subject/source gates. A description-only facet remains a fallback.
    """
    if not supports_verified_overview(plan):
        raise ValueError("Unsupported verified overview")
    return replace(plan, queries=((
        "Hardening Soil small strain GTS NX: E50ref Eoedref Eurref, секущая трёхосная и одометрическая жёсткость, разгрузка и повторное нагружение",
        "GTS NX Hardening Soil small strain stiffness nonlinear parameters E50ref Eoedref Eurref secant triaxial oedometer unloading reloading reference stiffness",
    ), plan.queries[0]))


def prepare_verified_overview(
    plan: OverviewPlan, results: list[dict[str, Any]], *, budget: int = 8_000,
) -> dict[str, Any]:
    """Select exact-scope evidence, then check required facets AFTER bounding.

    Facet checks establish a limited input contract, not semantic correctness
    of a future model response. Unknown scope is never repaired by body words.
    """
    filtered = []
    rejected = 0
    seen: set[str] = set()
    versions: set[str | None] = set()
    for result in results:
        sources, contexts = [], []
        if result.get("status") == "ok":
            for source, text in zip(result.get("sources", []), result.get("contexts", []), strict=True):
                scope = source.get("evidence_scope") or {}
                anchor = scope.get("title_anchor") or {}
                span = scope.get("normalized_span") or {}
                if not (supports_verified_overview(plan)
                        and scope.get("status") == "exact"
                        and scope.get("subject") == plan.subject
                        and scope.get("product") == "gts_nx"
                        and source.get("confident") is True
                        and scope.get("chunk_id") == source.get("chunk_id")
                        and scope.get("file_name") == source.get("file_name")
                        and isinstance(text, str) and text.strip()
                        and _DIGEST.fullmatch(str(scope.get("chunk_sha256", "")))
                        and _DIGEST.fullmatch(str(anchor.get("member_sha256", "")))
                        and isinstance(anchor.get("title"), str) and anchor["title"]
                        and isinstance(span.get("start"), int) and span["start"] >= 0
                        and isinstance(span.get("end"), int) and span["end"] > span["start"]):
                    rejected += 1
                    continue
                versions.add(scope.get("document_version"))
                identity = str(scope.get("document_version")) + "\0" + _normalized(text)
                if identity in seen:
                    continue
                seen.add(identity)
                sources.append(source)
                contexts.append(text)
        filtered.append({"status": result.get("status"), "sources": sources, "contexts": contexts})
    prepared = package_overview(plan, filtered, budget=budget)
    records = prepared["citation_evidence"]
    texts = [_normalized(record["text"]) for record in records]
    compact = [re.sub(r"\s+", "", text) for text in texts]
    checks = {
        "description_and_basis": any(
            ("modified mohr coulomb" in text or "модифицированной модели мора кулона" in text)
            and any(term in text for term in ("stiffness", "жестк", "жёстк")) for text in texts),
        "stiffness_parameters": any(all(term in text for term in ("e50ref", "eoedref", "eurref"))
                                    and ("column 2=" in original or "описани" in original)
                                    for text, original in zip(compact, texts, strict=True)),
        "small_strain_parameters": any("g0ref" in text and ("γ0.7" in text or "gamma0.7" in text)
                                       and ("column 2=" in original or "modulus" in original)
                                       for text, original in zip(compact, texts, strict=True)),
        "consistent_document_version": bool(records) and len(versions) == 1,
    }
    missing = [name for name, covered in checks.items() if not covered]
    if prepared["failed_queries"]:
        missing.append("retrieval_unavailable")
    for record in records:
        record["text_sha256"] = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
    prepared["packet"].update(coverage_checks=checks, missing_aspects=missing,
                              contract="verified_hss_overview_v1")
    prepared.update(ready=bool(records) and not missing, rejected_sources=rejected)
    return prepared
