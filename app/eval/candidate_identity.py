"""Safe byte identity and isolation checks for an explicitly isolated runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit


def source_identity(root: Path) -> str:
    digest = hashlib.sha256()
    for directory in ("app", "bot"):
        for path in sorted((root / directory).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def candidate_identity(settings, root: Path) -> dict:
    # Explicit allowlist: URLs, credential fields, users, document text never leave.
    profile = {name: getattr(settings, name) for name in (
        "default_model", "agent_llm_api", "agent_reasoning_effort",
        "agent_expanded_reasoning_effort", "agent_reasoning_token_reserve",
        "chat_response_tokens", "chat_expanded_response_tokens", "rag_collection",
        "rag_embedding_model", "rag_embedding_dimensions", "agent_prepared_evidence_enabled",
        "agent_verified_overview_enabled", "agent_generation_timeout", "rag_enabled",
        "rag_chunk_strategy", "rag_chunk_size", "rag_chunk_overlap", "rag_top_k",
        "rag_product", "rag_product_scope", "rag_retrieval_concurrency", "rag_min_score",
        "rag_reranker_enabled", "rag_reranker_model", "rag_reranker_device",
        "rag_reranker_batch_size", "rag_reranker_top_n", "rag_condense_enabled",
        "rag_condense_llm_enabled", "rag_condense_reasoning_effort", "rag_lexical_retrieval_enabled",
        "rag_planner_enabled", "rag_planner_model", "rag_planner_effort", "rag_planner_timeout",
        "rag_lexical_term_weight", "rag_lexical_pair_weight", "rag_llm_model",
        "rag_reasoning_effort", "rag_generation_timeout",
    )}
    files = {}
    for name in ("rag_docstore_path", "rag_evidence_catalog_path", "rag_visual_transcriptions_path"):
        path = getattr(settings, name, None)
        files[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None
    payload = {"source_sha256": source_identity(root), "profile": profile, "artifacts_sha256": files}
    payload["candidate_identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def validate_isolation(settings) -> None:
    if settings.rag_collection != "s0005_candidate_v1":
        raise ValueError("candidate collection required")
    if urlsplit(settings.qdrant_url or "").port != 16333:
        raise ValueError("isolated candidate Qdrant port required")
    if urlsplit(settings.chat_database_url).hostname != "postgres":
        raise ValueError("candidate Compose database service required")
    if urlsplit(settings.chat_database_url).path != "/candidate":
        raise ValueError("candidate database required")
    if urlsplit(settings.redis_url).hostname != "redis":
        raise ValueError("candidate Compose Redis service required")
    if settings.rag_ingest_on_start or settings.rag_rebuild_on_start:
        raise ValueError("candidate startup must not ingest or rebuild")
    if settings.bot_url != "http://127.0.0.1:9":
        raise ValueError("outbound action transport must remain disconnected")

