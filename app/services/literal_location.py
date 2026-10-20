"""Spend only unused context space on exact, independently scoped page text."""
from __future__ import annotations

import hashlib
import json


def complete_selected_atoms(sources, contexts, resolver, *, budget):
    """Replace an atom by its literal page group, preserving every old text.

    The source slot count, confidence gate, ranking and total budget are fixed.
    A group denotes co-location, never a semantic relationship or verification.
    """
    sources, contexts = list(sources), list(contexts)
    remaining = max(0, budget - sum(map(len, contexts)))
    completed = 0
    for index, (source, original) in enumerate(zip(sources, contexts, strict=True)):
        if source.get("block_type") not in {"definition", "formula", "formula_candidate", "paragraph", "heading"}:
            continue
        group = resolver.literal_atom_group(source, original, max_chars=len(original) + remaining)
        if not group:
            group = resolver.literal_member_prefix(source, original, max_chars=len(original) + remaining)
        if not group:
            continue
        text, atoms = group["text"], group["blocks"]
        if original.strip() not in text or len(text) - len(original) > remaining:
            continue
        ids = [str(atom["block_id"]) for atom in atoms]
        binding = json.dumps([source.get("source_id"), source.get("source_hash"),
                              source.get("location_id"), ids], ensure_ascii=False)
        identity = group.get("identity_prefix", "literal-page-") + hashlib.sha256(binding.encode()).hexdigest()
        attributes = [atom.get("attributes") or {} for atom in atoms]
        shared = {key: attributes[0][key] for key in attributes[0]
                  if all(key in a and a[key] == attributes[0][key] for a in attributes)}
        flags = {**shared,
                 "verified": all(a.get("verified") is True for a in attributes),
                 "semantic_verified": all(a.get("semantic_verified") is True for a in attributes),
                 "visual_review_required": any(a.get("visual_review_required") for a in attributes),
                 "verification_scope": group.get("scope", "literal_page_text"), "method": "literal_location_completion",
                 "literal_blocks": [{key: atom.get(key) for key in
                    ("block_id", "kind", "text", "bbox", "attributes", "section_path")} for atom in atoms]}
        sources[index] = {**source, "id": identity, "chunk_id": identity,
            "block_id": None, "block_ids": ids, "block_bbox": None,
            "block_type": "literal_location_context", "atomic_evidence": True,
            "origin": "literal_location_context", "relation": None, "related_to": [],
            "block_attributes": flags, "block_relations": [],
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "snippet": text[:700]}
        sources[index].pop("evidence_scope", None)
        contexts[index] = text
        remaining -= len(text) - len(original)
        completed += 1
    return sources, contexts, completed
