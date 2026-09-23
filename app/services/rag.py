from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    Snowball,
    SnowballLanguage,
    SnowballParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
)

from app.services.ingestion import IngestionConfig, IngestionService, canonical_product, infer_product
from app.observability.logging import get_logger
from app.observability.pii import prompt_hash
from app.observability.tracing import trace_rag_query
from app.services.reranker import BGEReranker, RerankCandidate


RAG_SYSTEM_PROMPT = """Ты — помощник технической поддержки MIDAS GTS NX.
Отвечай только на русском языке и только по предоставленному контексту.
Документы могут быть на русском или английском: переводи найденный ответ на русский.
Если контекст не содержит надёжного ответа, честно сообщи, что сведений в базе нет.
Не используй нерелевантные документы и не придумывай цены, SLA, функции или регламенты.
Отвечай только на запрошенное свойство или действие; не подменяй его сведениями о соседних темах из контекста.
Если найден прямой ответ, не добавляй альтернативные трактовки, условные ответы «если вопрос о...» или предложение продолжить.
Если сведений для ответа нет, верни дословно: «В предоставленной базе знаний нет достаточно надёжных сведений для ответа на этот вопрос.» Не добавляй к этой фразе ссылки.
После каждого утверждения указывай ссылки на использованные фрагменты в формате [1], [2].
"""
NO_ANSWER = "В предоставленной базе знаний нет достаточно надёжных сведений для ответа на этот вопрос."
MAX_CONTEXT_SOURCES = 5
MAX_OVERVIEW_CONTEXT_SOURCES = 3
FOLLOWUP_LEXICAL_CONTEXT_SOURCES = 3
MAX_RAW_CANDIDATES = 30
DIALOG_HISTORY_MESSAGES = 6
DIALOG_MESSAGE_CHARS = 600
FOLLOWUP_CONTEXT_CHARS = 400
MAX_LEXICAL_QUERY_VARIANTS = 8
MAX_DENSE_FALLBACK_QUERIES = 4
RAW_CANDIDATE_MULTIPLIER = 3
RRF_RANK_CONSTANT = 60
LEXICAL_CONTEXT_RESERVE = 1
LEXICAL_PAYLOAD_FIELD = "_node_content"
NEAR_DUPLICATE_RATIO = 0.9
# ``snippet`` is sent through the public API, persisted with the chat message
# and used as a compact UI preview.  It must never be reused as the generation
# context: a selected node can legitimately contain an answer after its first
# 700 characters (for example, after a page header and introduction).
MAX_SOURCE_SNIPPET_CHARS = 700
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_LEGACY_PAGE_REF_RE = re.compile(r"_part_(\d+)$")
_PRINTED_PAGE_RE = re.compile(r"(?m)^[ \t]*(\d{1,4})[ \t]*$")
_LEXICAL_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_LEXICAL_STOP_WORDS = {
    "а",
    "без",
    "бы",
    "в",
    "во",
    "для",
    "и",
    "из",
    "или",
    "как",
    "какая",
    "какие",
    "какой",
    "кратко",
    "на",
    "не",
    "мне",
    "можешь",
    "но",
    "о",
    "по",
    "пожалуйста",
    "подробнее",
    "подробно",
    "про",
    "при",
    "с",
    "со",
    "что",
    "ты",
    "знаешь",
    "такое",
    "такую",
    "таком",
    "это",
    "about",
    "are",
    "describe",
    "explain",
    "is",
    "me",
    "please",
    "tell",
    "what",
    "gts",
    "midas",
    "nx",
}
_LEXICAL_IGNORED_PREFIXES = (
    "использ",
    "объясн",
    "опиш",
    "поясн",
    "расскаж",
)
_OVERVIEW_PREFIXES = (
    "дай обзор ",
    "дай краткий обзор ",
    "кратко расскажи ",
    "можешь рассказать ",
    "обзор ",
    "объясни ",
    "опиши ",
    "подробно расскажи ",
    "расскажи ",
    "расскажи подробнее ",
    "что известно о ",
    "что такое ",
    "что ты знаешь о ",
    "describe ",
    "explain ",
    "overview ",
    "tell me about ",
    "what are ",
    "what is ",
)
_RUSSIAN_SUFFIXES = tuple(
    sorted(
        {
            "иями",
            "ыми",
            "ими",
            "ами",
            "ями",
            "ого",
            "его",
            "ому",
            "ему",
            "иях",
            "ах",
            "ях",
            "ов",
            "ев",
            "ей",
            "ий",
            "ый",
            "ой",
            "ая",
            "яя",
            "ое",
            "ее",
            "ие",
            "ые",
            "ую",
            "юю",
            "ам",
            "ям",
            "ом",
            "ем",
            "а",
            "я",
            "ы",
            "и",
            "е",
            "о",
            "у",
            "ю",
        },
        key=len,
        reverse=True,
    )
)
_FOLLOWUP_REFERENCE_TERMS = {
    "его",
    "ее",
    "её",
    "ей",
    "ему",
    "ею",
    "им",
    "ими",
    "него",
    "нее",
    "неё",
    "ней",
    "нем",
    "нём",
    "них",
    "ними",
    "их",
    "он",
    "она",
    "оно",
    "они",
    "та",
    "так",
    "такая",
    "такие",
    "таким",
    "такими",
    "таких",
    "такого",
    "такое",
    "такой",
    "такому",
    "такую",
    "там",
    "те",
    "тем",
    "теми",
    "тех",
    "того",
    "той",
    "тому",
    "тот",
    "ту",
    "этого",
    "этой",
    "этом",
    "этому",
    "эта",
    "эти",
    "этим",
    "этими",
    "этих",
    "это",
    "этот",
    "эту",
    "it",
    "its",
    "that",
    "their",
    "them",
    "these",
    "they",
    "this",
    "those",
}
# Question words that open a follow-up when an anaphora sits near the head
# ("как мне настроить этот параметр?", "почему это не работает?").
_FOLLOWUP_QUESTION_WORDS = {
    "где",
    "зачем",
    "как",
    "какая",
    "какие",
    "каким",
    "какими",
    "каких",
    "какого",
    "какое",
    "какой",
    "каком",
    "какому",
    "какую",
    "когда",
    "кем",
    "кого",
    "кому",
    "кто",
    "куда",
    "насколько",
    "откуда",
    "отчего",
    "почему",
    "сколько",
    "чего",
    "чей",
    "чем",
    "что",
    "чье",
    "чьё",
    "чьи",
    "чьим",
    "чьих",
    "чья",
    "how",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
}
# Single-token openers that mark a conversational continuation.  They are
# matched on the first token, so trailing punctuation cannot hide them.
_FOLLOWUP_HEAD_MARKERS = {
    "а",
    "и",
    "но",
    "или",
    "если",
    "иначе",
    "тогда",
    "значит",
    "подробнее",
    "поподробнее",
    "конкретнее",
    "например",
    "уточни",
    "уточните",
    "and",
    "but",
    "or",
    "so",
    "then",
}
# Multi-word openers matched on the token-joined text ("то есть это работает?").
_ELLIPTICAL_FOLLOWUP_PHRASES = (
    "то есть ",
    "что насчет ",
    "what about ",
    "how about ",
    "for example ",
    "for instance ",
    "more detail",
)
# Bare one-word turns that continue the prior exchange ("Подробнее", "Почему?").
_FOLLOWUP_SINGLE_WORDS = {
    "дальше",
    "еще",
    "ещё",
    "зачем",
    "почему",
    "пример",
    "примеры",
    "уточнение",
    "details",
    "example",
    "examples",
    "more",
    "why",
}
# Short fragments opening with a preposition are elliptical ("для 2D?").
_FOLLOWUP_FRAGMENT_PREPOSITIONS = {
    "без",
    "в",
    "вместо",
    "во",
    "для",
    "до",
    "за",
    "из",
    "к",
    "ко",
    "кроме",
    "между",
    "на",
    "над",
    "о",
    "об",
    "обо",
    "около",
    "от",
    "по",
    "под",
    "после",
    "при",
    "про",
    "против",
    "с",
    "со",
    "у",
    "через",
    "about",
    "for",
    "in",
    "of",
    "on",
    "to",
    "with",
    "without",
}
# Leading discourse particles are stripped before marker checks
# ("ладно, а для них?" -> "а для них?").
_FOLLOWUP_LEAD_PARTICLE_RE = re.compile(
    r"^(?:(?:okay|ладно|итак|вот|ну|ок|ok)\b[\s,;:—–-]*)+"
)

logger = get_logger("rag-service")

HistoryItem = Mapping[str, Any] | Any
CondenseHook = Callable[[str, Sequence[HistoryItem]], str | None]


@dataclass(frozen=True)
class RetrievalPlan:
    ranking_question: str
    queries: tuple[str, ...]
    mode: str


@dataclass(frozen=True)
class RAGConfig:
    data_dir: Path
    qdrant_url: str
    qdrant_api_key: str | None
    collection_name: str
    baremetal_collection_name: str
    llm_model: str
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int = 1024
    chunk_overlap: int = 64
    top_k: int = 10
    target_product: str = "gts_nx"
    min_score: float = 0.3
    rerank_enabled: bool = False
    rerank_top_n: int = 5
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str | None = None
    reranker_batch_size: int = 4
    condense_enabled: bool = True
    lexical_retrieval_enabled: bool = True
    allow_external_data_transfer: bool = False
    ingest_on_start: bool = True
    docstore_path: Path | None = None
    rebuild_on_start: bool = False
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    request_timeout: float = 60.0

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k < 3:
            raise ValueError("top_k must be at least 3")
        if not canonical_product(self.target_product):
            raise ValueError("target_product must not be blank")
        if self.rerank_top_n < 1:
            raise ValueError("rerank_top_n must be positive")
        if self.reranker_batch_size < 1:
            raise ValueError("reranker_batch_size must be positive")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be between 0 and 1")
        if self.collection_name == self.baremetal_collection_name:
            raise ValueError("LlamaIndex and bare-metal collections must be different")

    @classmethod
    def from_settings(cls, settings: Any) -> "RAGConfig":
        data_dir = Path(getattr(settings, "rag_data_dir", "data"))
        if not data_dir.is_absolute():
            data_dir = Path(__file__).resolve().parents[2] / data_dir
        qdrant_api_key = getattr(settings, "qdrant_api_key", None)
        api_key = qdrant_api_key.get_secret_value() if qdrant_api_key else None
        openai_api_key = getattr(settings, "openai_api_key", None)
        score_threshold = getattr(settings, "rag_score_threshold", getattr(settings, "rag_min_score", 0.3))
        docstore_path = Path(
            getattr(
                settings,
                "rag_docstore_path",
                Path(__file__).resolve().parents[2] / "var" / "rag_eval_chunk_1024_docstore.json",
            )
        )
        if not docstore_path.is_absolute():
            docstore_path = Path(__file__).resolve().parents[2] / docstore_path
        return cls(
            data_dir=data_dir,
            qdrant_url=getattr(settings, "qdrant_url", None) or "http://localhost:6333",
            qdrant_api_key=api_key,
            collection_name=getattr(settings, "rag_collection", "rag_eval_chunk_1024"),
            baremetal_collection_name=getattr(settings, "rag_baremetal_collection", "rag_block_03_baremetal"),
            llm_model=getattr(settings, "rag_llm_model", "gpt-5.4-mini"),
            embedding_model=getattr(settings, "rag_embedding_model", "text-embedding-3-small"),
            embedding_dimensions=getattr(settings, "rag_embedding_dimensions", 1536),
            chunk_size=getattr(settings, "rag_chunk_size", 1024),
            chunk_overlap=getattr(settings, "rag_chunk_overlap", 64),
            top_k=getattr(settings, "rag_top_k", 10),
            target_product=getattr(settings, "rag_product", "gts_nx"),
            min_score=float(score_threshold),
            rerank_enabled=bool(
                getattr(settings, "rag_reranker_enabled", getattr(settings, "rag_rerank_enabled", False))
            ),
            rerank_top_n=int(
                getattr(settings, "rag_reranker_top_n", getattr(settings, "rag_rerank_top_n", 5))
            ),
            reranker_model=getattr(settings, "rag_reranker_model", "BAAI/bge-reranker-v2-m3"),
            reranker_device=getattr(settings, "rag_reranker_device", None) or None,
            reranker_batch_size=int(getattr(settings, "rag_reranker_batch_size", 4)),
            condense_enabled=bool(getattr(settings, "rag_condense_enabled", True)),
            lexical_retrieval_enabled=bool(
                getattr(settings, "rag_lexical_retrieval_enabled", True)
            ),
            allow_external_data_transfer=bool(getattr(settings, "rag_allow_external_data_transfer", False)),
            ingest_on_start=bool(getattr(settings, "rag_ingest_on_start", True)),
            docstore_path=docstore_path,
            rebuild_on_start=getattr(settings, "rag_rebuild_on_start", False),
            openai_api_key=openai_api_key.get_secret_value() if openai_api_key else None,
            openai_base_url=getattr(settings, "openai_base_url", None),
            request_timeout=float(getattr(settings, "request_timeout", 60.0)),
        )


class RAGService:
    """LlamaIndex implementation of the Support Copilot RAG pipeline."""

    def __init__(
        self,
        config: RAGConfig,
        *,
        client: Any | None = None,
        embed_model: Any | None = None,
        llm: Any | None = None,
        reranker: Any | None = None,
        condense_hook: CondenseHook | None = None,
        ingestion_service: Any | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self._embed_model = embed_model
        self._llm = llm
        self._reranker = reranker
        self._condense_hook = condense_hook
        self._ingestion_service = ingestion_service
        self._index: Any | None = None
        self._retriever: Any | None = None
        self._lexical_available = False

    @property
    def ingestion_service(self) -> Any | None:
        """The shared incremental-ingestion service after :meth:`build`.

        The FastAPI composition root can put this object in ``app.state`` for
        upload handlers, keeping one document store and Qdrant client for both
        startup indexing and background uploads.
        """

        return self._ingestion_service

    def build(self) -> "RAGService":
        if self._retriever is not None:
            return self
        if self.client is None:
            self.client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout=self.config.request_timeout,
            )
        # ``rag_rebuild_on_start`` is deliberately ignored here.  The homework
        # requires repeatable UPSERT ingestion, and deleting a shared
        # collection at application start would make that safety guarantee
        # impossible.
        if self.config.ingest_on_start:
            self._validate_corpus()
            self._ingest_incrementally()
        else:
            # Keep the ingestion service available for later document uploads,
            # but do not read the corpus or make embedding requests at startup.
            self._ensure_ingestion_service()
        vector_store = QdrantVectorStore(client=self.client, collection_name=self.config.collection_name)
        self._validate_collection(self.config.collection_name)
        self._lexical_available = self._ensure_lexical_index()
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self._embedding_model(),
        )
        # Retrieval is deliberately separated from generation.  The score guard
        # below must run before an LLM sees any retrieved text.
        self._retriever = self._index.as_retriever(similarity_top_k=self._raw_candidate_limit())
        return self

    def answer(
        self,
        question: str,
        *,
        history: Sequence[HistoryItem] | None = None,
        standalone_question: str | None = None,
    ) -> dict[str, Any]:
        """Return the stable production response contract."""

        with trace_rag_query(question):
            return self._answer(
                question,
                history=history,
                standalone_question=standalone_question,
                include_retrieved_contexts=False,
            )

    def evaluate_inputs(
        self,
        question: str,
        *,
        history: Sequence[HistoryItem] | None = None,
        standalone_question: str | None = None,
    ) -> dict[str, Any]:
        """Return one production answer plus the complete contexts used for it.

        Evaluation must not issue a second retrieval request or score the short
        public ``snippet`` previews.  This method therefore executes the same
        guarded production path once and adds only ``retrieved_contexts``.
        """

        return self._answer(
            question,
            history=history,
            standalone_question=standalone_question,
            include_retrieved_contexts=True,
        )

    def _answer(
        self,
        question: str,
        *,
        history: Sequence[HistoryItem] | None = None,
        standalone_question: str | None = None,
        include_retrieved_contexts: bool,
    ) -> dict[str, Any]:
        """Answer a question after guarded retrieval.

        ``history`` is supplied by the caller; the RAG service intentionally
        has no dependency on a particular chat repository.  A caller that has
        already condensed the dialogue can pass ``standalone_question``.
        """
        if not question.strip():
            raise ValueError("question must not be blank")
        if self._retriever is None:
            raise RuntimeError("RAG index is not built")

        self._require_external_transfer_allowed("embed a retrieval question")
        started_at = time.perf_counter()
        plan = self._retrieval_plan(question, history, standalone_question)
        retrieved_by_query = [
            list(self._retriever.retrieve(query))[: self._raw_candidate_limit()]
            for query in plan.queries
        ]
        product_rankings = [
            [item for item in ranking if self._matches_target_product(item)]
            for ranking in retrieved_by_query
        ]
        lexical_queries = self._lexical_queries(question, history, plan)
        lexical_query: str | None = None
        lexical_candidates: list[Any] = []
        if self._lexical_available:
            for candidate_query in lexical_queries:
                candidates = [
                    item
                    for item in self._retrieve_lexical_candidates(candidate_query)
                    if self._matches_target_product(item)
                ]
                if candidates:
                    lexical_query = candidate_query
                    lexical_candidates = candidates
                    break
        dense_fallback_queries: list[str] = []
        initial_dense_score = max(
            (float(item.score or 0.0) for ranking in product_rankings for item in ranking),
            default=0.0,
        )
        if not lexical_candidates and initial_dense_score < self.config.min_score:
            for fallback_query in lexical_queries[1:]:
                if fallback_query in plan.queries:
                    continue
                fallback_ranking = list(self._retriever.retrieve(fallback_query))[
                    : self._raw_candidate_limit()
                ]
                fallback_product_ranking = [
                    item for item in fallback_ranking if self._matches_target_product(item)
                ]
                retrieved_by_query.append(fallback_ranking)
                product_rankings.append(fallback_product_ranking)
                dense_fallback_queries.append(fallback_query)
                fallback_score = max(
                    (float(item.score or 0.0) for item in fallback_product_ranking),
                    default=0.0,
                )
                if fallback_score >= self.config.min_score:
                    break
                if len(dense_fallback_queries) >= MAX_DENSE_FALLBACK_QUERIES:
                    break
        if lexical_candidates:
            product_rankings.append(lexical_candidates)
        product_candidates = self._fuse_retrieval_candidates(product_rankings)
        lexical_order = list(
            dict.fromkeys(self._node_identity(item) for item in lexical_candidates)
        )
        lexical_ids = set(lexical_order)
        canonical_candidates = {
            self._node_identity(item): item for item in product_candidates
        }
        for node_id in lexical_ids:
            if candidate := canonical_candidates.get(node_id):
                candidate.node.metadata["_lexical_match"] = True
        # Rank the complete product-scoped pool before removing duplicates or
        # applying the final context limit.  An earlier per-file cap discarded
        # exact evidence whenever two weaker dense matches from the same manual
        # happened to arrive first.
        ranked = self._rank_by_lexical_evidence(plan.ranking_question, product_candidates)
        # Keep the complete union until lexical evidence has had a chance to
        # rescue a candidate returned by only one query variant.  Capping the
        # RRF union first can discard that exact passage behind candidates
        # duplicated across both dense rankings.
        ranked = ranked[: self._raw_candidate_limit()]
        ranked = self._rerank(plan.ranking_question, ranked)
        ranked_candidate_count = len(ranked)
        overview_question = self._is_overview_question(question)
        best_dense_score = max((float(item.score or 0.0) for item in ranked), default=0.0)
        reserve_lexical = not overview_question or best_dense_score < self.config.min_score
        reserved = self._rank_fulltext_evidence(
            question,
            [canonical_candidates[node_id] for node_id in lexical_order if node_id in canonical_candidates],
            retrieval_query=lexical_query,
        )[: min(LEXICAL_CONTEXT_RESERVE, self._context_limit())] if reserve_lexical else []
        for candidate in reserved:
            candidate.node.metadata["_lexical_focus_query"] = lexical_query
        ranked = self._prepend_reserved(reserved, ranked)
        if plan.mode == "follow_up" and reserved:
            # The reserved lexical fragment still leads, but a single-fragment
            # context made the answer hostage to one all-token match: a wrong
            # hit left the generator no corrective dense evidence at all.
            context_limit = min(self._context_limit(), FOLLOWUP_LEXICAL_CONTEXT_SOURCES)
        elif overview_question:
            context_limit = min(self._context_limit(), MAX_OVERVIEW_CONTEXT_SOURCES)
        else:
            context_limit = self._context_limit()
        ranked = self._deduplicate(
            ranked,
            limit=context_limit,
            max_per_file=1 if overview_question else None,
        )
        sources = [self._source_from_node(item) for item in ranked]
        # Cosine similarity remains the confidence signal even when a
        # cross-encoder changes presentation order.  Cross-encoder scores are
        # not calibrated to the embedding threshold.
        top_score = max((source["score"] for source in sources), default=0.0)
        telemetry = {
            "question_hash": prompt_hash(question.strip()),
            "plan_mode": plan.mode,
            "query_variant_count": len(plan.queries),
            "raw_candidate_counts": [len(items) for items in retrieved_by_query],
            "product_candidate_counts": [len(items) for items in product_rankings],
            "fused_candidate_count": len(product_candidates),
            "lexical_candidate_count": len(lexical_candidates),
            "lexical_query_variant_count": len(lexical_queries),
            "dense_fallback_query_count": len(dense_fallback_queries),
            "lexical_selected_count": sum(
                self._node_identity(item) in lexical_ids for item in ranked
            ),
            "ranked_candidate_count": ranked_candidate_count,
            "selected_candidate_ids": [source["id"] for source in sources],
            "selected_candidate_count": len(sources),
            "answer_mode": "overview" if overview_question else "focused",
            "top_score": round(top_score, 6),
            "retrieval_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
        has_lexical_evidence = any(self._node_identity(item) in lexical_ids for item in ranked)
        retrieved_contexts = [self._context_text(item) for item in ranked]
        if top_score < self.config.min_score and not has_lexical_evidence:
            logger.info("rag_retrieval_completed", **telemetry, outcome="low_score", cited_source_count=0)
            result: dict[str, Any] = {
                "answer": NO_ANSWER,
                "top_score": top_score,
                "sources": [],
            }
            if include_retrieved_contexts:
                result["retrieved_contexts"] = retrieved_contexts
            return result
        answer, cited_sources = self._generate_answer(
            question.strip(),
            sources,
            ranked,
            overview_question=overview_question,
            history=history,
        )
        outcome = "grounded" if cited_sources else "missing_valid_citation"
        logger.info(
            "rag_retrieval_completed",
            **telemetry,
            outcome=outcome,
            cited_source_count=len(cited_sources),
        )
        result = {"answer": answer, "top_score": top_score, "sources": cited_sources}
        if include_retrieved_contexts:
            result["retrieved_contexts"] = retrieved_contexts
        return result

    def query(
        self,
        question: str,
        *,
        history: Sequence[HistoryItem] | None = None,
        standalone_question: str | None = None,
    ) -> dict[str, Any]:
        """Alias for :meth:`answer` used by HTTP/chat integrations."""

        return self.answer(question, history=history, standalone_question=standalone_question)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def _validate_corpus(self) -> None:
        if not self.config.data_dir.is_dir():
            raise FileNotFoundError(f"RAG_DATA_DIR does not exist: {self.config.data_dir}")
        supported = {".pdf", ".docx", ".html", ".htm", ".md", ".txt", ".chm"}
        files = [path for path in self.config.data_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported]
        if len(files) < 30:
            raise ValueError(f"RAG corpus must contain at least 30 supported files; found {len(files)}")

    def _validate_collection(self, name: str) -> None:
        vectors = self.client.get_collection(name).config.params.vectors
        if isinstance(vectors, dict):
            raise ValueError(f"RAG collection {name!r} must use one unnamed dense vector")
        if int(vectors.size) != self.config.embedding_dimensions or vectors.distance != Distance.COSINE:
            raise ValueError(
                f"RAG collection {name!r} must use Cosine/{self.config.embedding_dimensions} dimensions"
            )

    def _ensure_lexical_index(self) -> bool:
        """Create one online full-text payload index without rebuilding vectors."""

        if not self.config.lexical_retrieval_enabled or self.client is None:
            return False
        try:
            info = self.client.get_collection(self.config.collection_name)
            schema = getattr(info, "payload_schema", {}) or {}
            if LEXICAL_PAYLOAD_FIELD not in schema:
                self.client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=LEXICAL_PAYLOAD_FIELD,
                    field_schema=TextIndexParams(
                        type=TextIndexType.TEXT,
                        tokenizer=TokenizerType.MULTILINGUAL,
                        lowercase=True,
                        phrase_matching=True,
                        stemmer=SnowballParams(
                            type=Snowball.SNOWBALL,
                            language=SnowballLanguage.RUSSIAN,
                        ),
                    ),
                    wait=True,
                )
            return True
        except Exception as exc:
            logger.warning(
                "rag_lexical_index_unavailable",
                error_type=type(exc).__name__,
                collection=self.config.collection_name,
            )
            return False

    def _ingest_incrementally(self) -> None:
        self._ensure_ingestion_service().ingest(
            self.config.data_dir,
            allow_external_data_transfer=self.config.allow_external_data_transfer,
        )

    def _ensure_ingestion_service(self) -> Any:
        if self._ingestion_service is None:
            docstore_path = self.config.docstore_path or (
                Path(__file__).resolve().parents[2] / "var" / "ingestion_docstore.json"
            )
            self._ingestion_service = IngestionService(
                IngestionConfig(
                    data_dir=self.config.data_dir,
                    qdrant_url=self.config.qdrant_url,
                    qdrant_api_key=self.config.qdrant_api_key,
                    collection_name=self.config.collection_name,
                    embedding_model=self.config.embedding_model,
                    embedding_dimensions=self.config.embedding_dimensions,
                    openai_api_key=self.config.openai_api_key,
                    openai_base_url=self.config.openai_base_url,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    default_product=self.config.target_product,
                    docstore_path=docstore_path,
                ),
                client=self.client,
                embed_model=self._embed_model,
            )
        return self._ingestion_service

    def _embedding_model(self) -> Any:
        if self._embed_model is None:
            self._embed_model = OpenAIEmbedding(
                model=self.config.embedding_model, dimensions=self.config.embedding_dimensions,
                api_key=self.config.openai_api_key, api_base=self.config.openai_base_url,
                timeout=self.config.request_timeout,
            )
        return self._embed_model

    def _require_external_transfer_allowed(self, action: str) -> None:
        # An injected embedding implementation is used by offline tests and by
        # explicitly configured local providers.  The built-in adapter is the
        # OpenAI embedding client, so it always needs the user's opt-in.
        if self._embed_model is None and not self.config.allow_external_data_transfer:
            raise RuntimeError(
                f"Refusing to {action}: set RAG_ALLOW_EXTERNAL_DATA_TRANSFER=true "
                "only after approving transfer of document fragments to the embedding provider."
            )

    def _language_model(self) -> Any:
        if self._llm is None:
            self._llm = LlamaOpenAI(
                model=self.config.llm_model, api_key=self.config.openai_api_key,
                api_base=self.config.openai_base_url, timeout=self.config.request_timeout,
                temperature=0, system_prompt=RAG_SYSTEM_PROMPT,
            )
        return self._llm

    def _rerank(self, question: str, source_nodes: list[Any]) -> list[Any]:
        if not self.config.rerank_enabled or not source_nodes:
            return source_nodes
        if self._reranker is None:
            self._reranker = BGEReranker(
                model_name=self.config.reranker_model,
                device=self.config.reranker_device,
                batch_size=self.config.reranker_batch_size,
            )
        candidates = [
            RerankCandidate(
                text=self._node_text(source_node),
                doc_id=str(index),
                source=str(self._source_from_node(source_node)["file_name"]),
                score=float(source_node.score or 0.0),
            )
            for index, source_node in enumerate(source_nodes)
        ]
        reranked = self._reranker.rerank(
            question,
            candidates,
            # Ask the cross-encoder for a complete ordering.  Its configured
            # context limit is applied only after duplicate removal, otherwise
            # duplicate top results could leave too little usable context.
            top_n=len(candidates),
        )
        return [source_nodes[int(candidate.doc_id)] for candidate in reranked]

    def _matches_target_product(self, source_node: Any) -> bool:
        """Filter both new product metadata and legacy payloads consistently."""

        metadata = source_node.node.metadata or {}
        explicit = canonical_product(str(metadata.get("product") or ""))
        if explicit:
            product = explicit
        else:
            product = infer_product(
                (
                    metadata.get("file_name"),
                    metadata.get("source"),
                    metadata.get("file_path"),
                    metadata.get("category"),
                ),
                default_product=self.config.target_product,
            )
        return product == canonical_product(self.config.target_product)

    def _raw_candidate_limit(self) -> int:
        return min(MAX_RAW_CANDIDATES, max(MAX_CONTEXT_SOURCES, self.config.top_k * RAW_CANDIDATE_MULTIPLIER))

    def _context_limit(self) -> int:
        if self.config.rerank_enabled:
            return min(MAX_CONTEXT_SOURCES, self.config.rerank_top_n)
        return MAX_CONTEXT_SOURCES

    def _retrieval_plan(
        self,
        question: str,
        history: Sequence[HistoryItem] | None,
        standalone_question: str | None,
    ) -> RetrievalPlan:
        """Plan one normal query or two complementary follow-up queries.

        A context-dependent turn uses the user's exact wording for narrow
        recall plus a history-expanded wording for disambiguation.  Explicit
        caller rewrites and custom hooks remain authoritative single queries.
        Independent questions do not pay for, or get polluted by, a second
        retrieval request.
        """

        focused = self._standalone_question(question, history, standalone_question)
        current = question.strip()
        if standalone_question and standalone_question.strip():
            return RetrievalPlan(focused, (focused,), "caller_standalone")
        if self._condense_hook is not None and focused != current:
            return RetrievalPlan(focused, (focused,), "condense_hook")
        if focused == current:
            return RetrievalPlan(current, (current,), "independent")
        return RetrievalPlan(focused, (current, focused), "follow_up")

    def _fuse_retrieval_candidates(self, rankings: Sequence[Sequence[Any]]) -> list[Any]:
        """Merge variant rankings with deterministic reciprocal-rank fusion.

        Dense scores from two different query embeddings are not directly
        comparable.  RRF uses only rank positions, rewards nodes found by both
        variants and avoids introducing an unevaluated score weight.  The
        original best dense score is retained on the returned node for the
        existing confidence threshold.
        """

        fused: dict[str, dict[str, Any]] = {}
        first_seen = 0
        for ranking in rankings:
            seen_in_ranking: set[str] = set()
            for rank, source_node in enumerate(ranking, start=1):
                node_id = self._node_identity(source_node)
                if node_id in seen_in_ranking:
                    continue
                seen_in_ranking.add(node_id)
                entry = fused.get(node_id)
                if entry is None:
                    entry = {
                        "node": source_node,
                        "rrf": 0.0,
                        "best_score": float(source_node.score or 0.0),
                        "first_seen": first_seen,
                    }
                    fused[node_id] = entry
                    first_seen += 1
                score = float(source_node.score or 0.0)
                if score > entry["best_score"]:
                    entry["node"] = source_node
                    entry["best_score"] = score
                entry["rrf"] += 1.0 / (RRF_RANK_CONSTANT + rank)

        ordered = sorted(
            fused.items(),
            key=lambda item: (
                -float(item[1]["rrf"]),
                -float(item[1]["best_score"]),
                int(item[1]["first_seen"]),
                item[0],
            ),
        )
        return [entry["node"] for _, entry in ordered]

    def _retrieve_lexical_candidates(self, query: str) -> list[Any]:
        """Return strict all-token matches as a complementary recall path."""

        if self.client is None:
            return []
        try:
            records, _ = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key=LEXICAL_PAYLOAD_FIELD,
                            match=MatchText(text=query),
                        )
                    ]
                ),
                limit=self._raw_candidate_limit(),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning("rag_lexical_retrieval_failed", error_type=type(exc).__name__)
            return []

        candidates: list[Any] = []
        for record in records:
            try:
                node = metadata_dict_to_node(record.payload or {})
            except Exception:
                logger.warning("rag_lexical_point_invalid", point_id=str(record.id))
                continue
            candidates.append(NodeWithScore(node=node, score=0.0))
        return candidates

    @classmethod
    def _lexical_query(
        cls,
        question: str,
        history: Sequence[HistoryItem] | None,
        plan: RetrievalPlan,
    ) -> str | None:
        """Build an all-token query, carrying only the prior turn's subject."""

        terms = cls._fulltext_terms(question)[:4]
        if plan.mode == "follow_up" and history:
            previous_user_question = next(
                (
                    content
                    for item in reversed(history)
                    if cls._history_value(item, "role") == "user"
                    and (content := str(cls._history_value(item, "content", "")).strip())
                    and content != question.strip()
                ),
                "",
            )
            terms.extend(cls._fulltext_terms(previous_user_question)[:2])
        unique = list(dict.fromkeys(terms))
        is_specific_pair = len(unique) == 2 and sum(map(len, unique)) >= 9
        return " ".join(unique) if len(unique) >= 3 or is_specific_pair else None

    @classmethod
    def _lexical_queries(
        cls,
        question: str,
        history: Sequence[HistoryItem] | None,
        plan: RetrievalPlan,
    ) -> tuple[str, ...]:
        """Build a strict query plus bounded entity-window fallbacks.

        Conversational lead-ins are open-ended (for example "меня интересует"
        or "подскажи насчёт"), so a finite stop-word list cannot cover them.
        Short windows are tried from both ends towards the middle only when
        the stricter primary query has no product-scoped matches. This covers
        both a conversational prefix and a trailing phrase after the subject.
        """

        queries: list[str] = []
        if primary := cls._lexical_query(question, history, plan):
            queries.append(primary)
        current_terms = cls._fulltext_terms(question)
        for width in (3, 2):
            if len(current_terms) <= width:
                continue
            left = 0
            right = len(current_terms) - width
            starts: list[int] = []
            while left <= right:
                starts.append(left)
                if right != left:
                    starts.append(right)
                left += 1
                right -= 1
            for start in starts:
                window = list(dict.fromkeys(current_terms[start : start + width]))
                is_specific_pair = len(window) == 2 and sum(map(len, window)) >= 9
                if len(window) >= 3 or is_specific_pair:
                    queries.append(" ".join(window))
                unique_queries = tuple(dict.fromkeys(queries))
                if len(unique_queries) >= MAX_LEXICAL_QUERY_VARIANTS:
                    return unique_queries
        return tuple(dict.fromkeys(queries))

    @staticmethod
    def _fulltext_terms(text: str) -> list[str]:
        return [
            token
            for token in _LEXICAL_TOKEN_RE.findall(text.casefold().replace("ё", "е"))
            if len(token) >= 3
            and token not in _LEXICAL_STOP_WORDS
            and token not in _FOLLOWUP_REFERENCE_TERMS
            and not any(token.startswith(prefix) for prefix in _LEXICAL_IGNORED_PREFIXES)
        ]

    @classmethod
    def _rank_fulltext_evidence(
        cls,
        question: str,
        source_nodes: Sequence[Any],
        *,
        retrieval_query: str | None = None,
    ) -> list[Any]:
        """Prefer compact direct evidence over incidental all-token matches."""

        query_terms = cls._lexical_terms(retrieval_query or question)
        if not query_terms:
            return list(source_nodes)
        query_unique = set(query_terms)
        query_pairs = set(zip(query_terms, query_terms[1:]))
        current_terms = set(cls._lexical_terms(" ".join(cls._fulltext_terms(question))))

        def rank_key(indexed_node: tuple[int, Any]) -> tuple[float, float, float, float, float, int]:
            index, source_node = indexed_node
            full_terms = cls._lexical_terms(cls._node_text(source_node))
            lead_terms = cls._lexical_terms(cls._node_text(source_node)[:400])
            full_coverage = len(query_unique.intersection(full_terms)) / len(query_unique)
            current_lead_coverage = (
                len(current_terms.intersection(lead_terms)) / len(current_terms)
                if current_terms
                else 0.0
            )
            pair_coverage = 0.0
            if query_pairs:
                full_pairs = set(zip(full_terms, full_terms[1:]))
                pair_coverage = len(query_pairs.intersection(full_pairs)) / len(query_pairs)
            span = cls._minimum_cover_span(full_terms, query_unique)
            proximity = 1.0 / span if span is not None else 0.0
            dense_score = float(source_node.score or 0.0)
            return (
                full_coverage,
                current_lead_coverage,
                proximity,
                pair_coverage,
                dense_score,
                -index,
            )

        return [
            source_node
            for _, source_node in sorted(
                enumerate(source_nodes),
                key=rank_key,
                reverse=True,
            )
        ]

    @staticmethod
    def _minimum_cover_span(terms: Sequence[str], required: set[str]) -> int | None:
        """Return the shortest token window containing every required term."""

        if not required:
            return None
        counts: dict[str, int] = {}
        missing = len(required)
        left = 0
        best: int | None = None
        for right, term in enumerate(terms):
            if term not in required:
                continue
            counts[term] = counts.get(term, 0) + 1
            if counts[term] == 1:
                missing -= 1
            while missing == 0:
                best = min(best or len(terms), right - left + 1)
                left_term = terms[left]
                left += 1
                if left_term not in required:
                    continue
                counts[left_term] -= 1
                if counts[left_term] == 0:
                    missing += 1
        return best

    @classmethod
    def _prepend_reserved(cls, reserved: Sequence[Any], ranked: Sequence[Any]) -> list[Any]:
        result: list[Any] = []
        seen: set[str] = set()
        for item in (*reserved, *ranked):
            node_id = cls._node_identity(item)
            if node_id in seen:
                continue
            seen.add(node_id)
            result.append(item)
        return result

    def _rank_by_lexical_evidence(self, question: str, source_nodes: Sequence[Any]) -> list[Any]:
        """Boost literal query evidence while retaining dense similarity.

        Only informative query terms contribute.  Matching both the terms and
        their adjacent pairs rewards passages that contain the user's actual
        concept (for example ``конструкционный расчёт``), rather than generic
        manual boilerplate that happens to repeat one common word.
        """

        query_terms = self._lexical_terms(question)
        if not query_terms:
            return list(source_nodes)
        query_unique = set(query_terms)
        query_pairs = set(zip(query_terms, query_terms[1:]))

        def rank_key(indexed_node: tuple[int, Any]) -> tuple[float, float, int]:
            index, source_node = indexed_node
            text_terms = self._lexical_terms(self._node_text(source_node))
            text_unique = set(text_terms)
            term_coverage = len(query_unique & text_unique) / len(query_unique)
            pair_coverage = 0.0
            if query_pairs:
                text_pairs = set(zip(text_terms, text_terms[1:]))
                pair_coverage = len(query_pairs & text_pairs) / len(query_pairs)
            lexical_relevance = 0.7 * term_coverage + 0.3 * pair_coverage
            dense_score = float(source_node.score or 0.0)
            # Dense and lexical evidence have equal bounded influence.  The
            # original cosine score remains untouched for confidence checks.
            return dense_score + lexical_relevance, dense_score, -index

        return [
            source_node
            for _, source_node in sorted(
                enumerate(source_nodes),
                key=rank_key,
                reverse=True,
            )
        ]

    @classmethod
    def _lexical_terms(cls, text: str) -> list[str]:
        terms: list[str] = []
        for raw_token in _LEXICAL_TOKEN_RE.findall(text.casefold().replace("ё", "е")):
            if raw_token in _LEXICAL_STOP_WORDS or any(
                raw_token.startswith(prefix) for prefix in _LEXICAL_IGNORED_PREFIXES
            ):
                continue
            token = cls._light_stem(raw_token)
            if len(token) >= 3:
                terms.append(token)
        return terms

    @staticmethod
    def _light_stem(token: str) -> str:
        if not re.search(r"[а-я]", token) or len(token) < 5:
            return token
        for suffix in _RUSSIAN_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                return token[: -len(suffix)]
        return token

    def _deduplicate(
        self,
        source_nodes: Sequence[Any],
        *,
        limit: int,
        max_per_file: int | None = None,
    ) -> list[Any]:
        """Remove duplicate chunks after ranking and cap the LLM context."""

        selected: list[Any] = []
        selected_texts: list[str] = []
        file_counts: dict[str, int] = {}
        for source_node in source_nodes:
            normalized_text = self._normalized_chunk_text(self._node_text(source_node))
            if any(self._near_duplicate(normalized_text, previous) for previous in selected_texts):
                continue
            file_name = str(self._source_from_node(source_node)["file_name"])
            if max_per_file is not None and file_counts.get(file_name, 0) >= max_per_file:
                continue
            selected.append(source_node)
            selected_texts.append(normalized_text)
            file_counts[file_name] = file_counts.get(file_name, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _normalized_chunk_text(text: str) -> str:
        return " ".join(re.findall(r"\w+", text.casefold()))

    @staticmethod
    def _near_duplicate(candidate: str, previous: str) -> bool:
        if not candidate or not previous:
            return candidate == previous
        if candidate == previous:
            return True
        # Short snippets frequently share boilerplate while carrying a
        # different key/value; exact matching is the safer rule for them.
        if min(len(candidate), len(previous)) < 80:
            return False
        return SequenceMatcher(None, candidate, previous, autojunk=False).ratio() >= NEAR_DUPLICATE_RATIO

    def _standalone_question(
        self,
        question: str,
        history: Sequence[HistoryItem] | None,
        standalone_question: str | None,
    ) -> str:
        if standalone_question and standalone_question.strip():
            return standalone_question.strip()
        if not history or not self.config.condense_enabled:
            return question.strip()
        if self._condense_hook is not None:
            condensed = self._condense_hook(question, history)
            if condensed and condensed.strip():
                return condensed.strip()
        if not self._is_context_dependent_followup(question):
            return question.strip()
        current = question.strip()
        previous_user_question = next(
            (
                content
                for item in reversed(history)
                if self._history_value(item, "role") == "user"
                and (content := self._followup_context_text(
                    self._history_value(item, "content", "")
                ))
                and content != current
            ),
            "",
        )
        previous_assistant_answer = next(
            (
                content
                for item in reversed(history)
                if self._history_value(item, "role") == "assistant"
                and (content := self._followup_context_text(
                    self._history_value(item, "content", ""), assistant=True
                ))
            ),
            "",
        )
        # Merge the latest exchange, not only the last user question: the
        # assistant's answer often introduces the referent a follow-up points
        # to, and a chained follow-up would otherwise keep pointing at an
        # elliptical predecessor.  Keep the user's wording, but omit synthetic
        # labels such as "контекст предыдущего вопроса": those labels dilute
        # the embedding and do not add support-domain meaning.
        context_parts = [
            part for part in (previous_user_question, previous_assistant_answer) if part
        ]
        if not context_parts:
            return current
        return " ".join((*context_parts, current))

    @classmethod
    def _followup_context_text(cls, content: Any, *, assistant: bool = False) -> str:
        """Compact a prior turn for the merged follow-up retrieval query.

        Assistant answers carry citation markers that belong to an older
        fragment numbering and abstention sentences that add no topical
        terms, so both are removed before the bounded merge.
        """

        text = cls._history_text(content)
        if assistant:
            text = _CITATION_RE.sub("", text)
            text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = " ".join(text.split())
        if assistant and (not text or cls._is_abstention_answer(text)):
            return ""
        if len(text) > FOLLOWUP_CONTEXT_CHARS:
            text = text[:FOLLOWUP_CONTEXT_CHARS].rsplit(" ", 1)[0].rstrip(",;:")
        return text

    @staticmethod
    def _is_context_dependent_followup(question: str) -> bool:
        """Recognize conversational ellipses and questions anchored in prior context.

        The detector intentionally stays conservative, but covers the common
        support-dialog shapes: continuation openers ("а", "и", "но", "если",
        "подробнее"), one-word continuation markers ("почему", "дальше"),
        short prepositional fragments ("для 2D?") and anaphoric pronouns in
        short questions or near the head of a wh-question ("как мне настроить
        этот параметр?").  A false negative still runs the exact current
        question; a false positive adds one bounded query variant.
        """

        normalized = _FOLLOWUP_LEAD_PARTICLE_RE.sub(
            "", " ".join(question.casefold().replace("ё", "е").split())
        )
        tokens = _LEXICAL_TOKEN_RE.findall(normalized)
        if not tokens:
            return False
        if (
            tokens[0] in _FOLLOWUP_HEAD_MARKERS
            or " ".join(tokens).startswith(_ELLIPTICAL_FOLLOWUP_PHRASES)
        ):
            return True
        if len(tokens) == 1 and tokens[0] in _FOLLOWUP_SINGLE_WORDS:
            return True
        if len(tokens) <= 4 and tokens[0] in _FOLLOWUP_FRAGMENT_PREPOSITIONS:
            return True
        reference_positions = [
            index
            for index, token in enumerate(tokens)
            if token in _FOLLOWUP_REFERENCE_TERMS
            # "что такое/такой/такие" and "кто такой" are the "what/who is"
            # idiom, not a reference to a prior turn.
            and not (index > 0 and tokens[index - 1] in {"что", "кто"} and token.startswith("так"))
        ]
        if not reference_positions:
            return False
        # Short pronominal questions ("Какие из них?", "Для них?") are
        # dependent, and so are medium questions that open with a wh-word or
        # a reference to the prior turn.  A pronoun buried late in a complete
        # technical question is not enough on its own: adding unrelated
        # history would pollute recall.
        return (
            len(tokens) <= 4
            or (
                tokens[0] in _FOLLOWUP_QUESTION_WORDS
                and reference_positions[0] <= 4
                and len(tokens) <= 12
            )
            or (reference_positions[0] <= 2 and len(tokens) <= 8)
        )

    def _generate_answer(
        self,
        question: str,
        sources: list[dict[str, Any]],
        source_nodes: Sequence[Any],
        *,
        overview_question: bool = False,
        history: Sequence[HistoryItem] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate from complete selected chunks, not their API previews.

        ``sources`` and ``source_nodes`` are built from the same ranked list;
        their positions deliberately stay aligned with citation numbers.  A
        compact ``snippet`` remains sufficient for source APIs and Telegram,
        while the LLM receives the complete chunk selected by retrieval.
        """

        if len(sources) != len(source_nodes):
            raise ValueError("sources and source_nodes must have the same length")
        context = "\n\n".join(
            f"[{index}] {self._context_source_label(source)}\n"
            f"{self._context_text(source_node)}"
            for index, (source, source_node) in enumerate(zip(sources, source_nodes, strict=True), start=1)
        )
        overview_instruction = (
            "\nЭто широкий обзорный вопрос. Сначала дай определение в одном предложении, "
            "затем выбери не более трёх главных аспектов и свяжи их общей логикой. "
            "Не пересказывай каждый фрагмент контекста и не включай узкие настройки, "
            "численные детали или частные алгоритмы, если о них не спрашивали. "
            "Если контекст относится к разным типам расчёта и единого ответа нет, "
            "кратко назови эти типы и задай один уточняющий вопрос о сценарии пользователя.\n"
            if overview_question
            else ""
        )
        prompt = (
            f"{RAG_SYSTEM_PROMPT}\n\n"
            f"{self._dialog_context_section(history)}"
            f"Контекст:\n{context}\n\n"
            f"Вопрос пользователя: {question}\n\n"
            f"{overview_instruction}"
            f"Дай краткий проверяемый ответ. Используй только контекст и ссылайся только на "
            f"существующие номера от [1] до [{len(sources)}]."
        )
        response = self._language_model().complete(prompt)
        answer = str(getattr(response, "text", response) or "").strip() or NO_ANSWER
        return self._ground_answer(answer, sources)

    @classmethod
    def _dialog_context_section(cls, history: Sequence[HistoryItem] | None) -> str:
        """Render a bounded recent-dialog block placed before the retrieved context.

        The generation prompt previously received only the raw current wording,
        so an elliptical follow-up ("а какие из них?") was unreadable for the
        LLM even though retrieval had already resolved its topic.  Citations
        inside previous assistant answers refer to an older context numbering
        and are stripped so they cannot collide with [1]..[N] fragments.
        """

        if not history:
            return ""
        lines: list[str] = []
        for item in history[-DIALOG_HISTORY_MESSAGES:]:
            label = {"user": "Пользователь", "assistant": "Ассистент"}.get(
                str(cls._history_value(item, "role"))
            )
            if label is None:
                continue
            text = cls._history_text(cls._history_value(item, "content", ""))
            if label == "Ассистент":
                text = _CITATION_RE.sub("", text)
            text = " ".join(text.split())
            if not text:
                continue
            lines.append(f"{label}: {text[:DIALOG_MESSAGE_CHARS]}")
        if not lines:
            return ""
        return "Диалог:\n" + "\n".join(lines) + "\n\n"

    @staticmethod
    def _history_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, Mapping) and part.get("type") == "text"
            )
        return str(content or "")

    @staticmethod
    def _context_source_label(source: Mapping[str, Any]) -> str:
        parts = [f"Файл: {source['file_name']}"]
        if source.get("page") is not None:
            parts.append(f"PDF-страница: {source['page']}")
        if source.get("printed_page") is not None:
            parts.append(f"страница документа: {source['printed_page']}")
        if len(parts) == 1:
            parts.append("страница: не указана")
        return "; ".join(parts)

    @classmethod
    def _ground_answer(
        cls,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Validate citations, return only cited sources and keep numbering aligned."""

        if cls._is_abstention_answer(answer):
            return NO_ANSWER, []
        source_count = len(sources)
        cited_numbers = sorted(
            {
                number
                for match in _CITATION_RE.finditer(answer)
                for raw_number in match.group(1).split(",")
                if 1 <= (number := int(raw_number.strip())) <= source_count
            }
        )
        if not cited_numbers:
            return NO_ANSWER, []

        number_map = {old: new for new, old in enumerate(cited_numbers, start=1)}

        def remap(match: re.Match[str]) -> str:
            mapped: list[int] = []
            for raw_number in match.group(1).split(","):
                new_number = number_map.get(int(raw_number.strip()))
                if new_number is not None and new_number not in mapped:
                    mapped.append(new_number)
            return f"[{', '.join(str(number) for number in mapped)}]" if mapped else ""

        grounded_answer = _CITATION_RE.sub(remap, answer)
        cited_sources = [sources[number - 1] for number in cited_numbers]
        return grounded_answer, cited_sources

    @classmethod
    def _is_abstention_answer(cls, answer: str) -> bool:
        """Recognize grounded-generation abstentions and remove false citations.

        The exact canonical sentence is preferred, but providers sometimes
        paraphrase it as "the context does not contain pricing information"
        and append citations.  A scope marker is required so factual negative
        statements such as "the model does not include rotations" remain valid.
        """

        without_citations = _CITATION_RE.sub("", answer).strip()
        if without_citations == NO_ANSWER:
            return True
        normalized = " ".join(without_citations.casefold().replace("ё", "е").split())
        if len(normalized) > 500:
            return False
        has_scope = any(
            marker in normalized
            for marker in (
                "базе знаний",
                "в базе",
                "в контексте",
                "предоставленном контексте",
                "в документе",
                "в документах",
                "в материалах",
            )
        )
        if not has_scope:
            return False
        subject = r"(?:сведен\w*|информац\w*|данн\w*|цен\w*|стоимост\w*)"
        absence = r"(?:нет|не\s+(?:найден\w*|содерж\w*|указан\w*)|отсутств\w*)"
        return bool(
            re.search(rf"{subject}[^.!?]{{0,120}}{absence}", normalized)
            or re.search(rf"{absence}[^.!?]{{0,120}}{subject}", normalized)
        )

    @staticmethod
    def _is_overview_question(question: str) -> bool:
        normalized = " ".join(question.casefold().replace("ё", "е").split())
        return normalized.startswith(_OVERVIEW_PREFIXES)

    @staticmethod
    def _ensure_citations(answer: str, sources: list[dict[str, Any]]) -> str:
        """Keep only citations the model actually used in its answer.

        Retrieval may provide ten candidates, but appending references for all
        of them falsely implies that each one supports the answer.  Telegram
        later resolves the citations that are already present in ``answer``.
        """

        source_count = len(sources)

        def keep_valid(match: re.Match[str]) -> str:
            valid: list[int] = []
            for raw_number in match.group(1).split(","):
                number = int(raw_number.strip())
                if 1 <= number <= source_count and number not in valid:
                    valid.append(number)
            return f"[{', '.join(str(number) for number in valid)}]" if valid else ""

        return _CITATION_RE.sub(keep_valid, answer)

    @staticmethod
    def _history_value(item: HistoryItem, name: str, default: Any = None) -> Any:
        if isinstance(item, Mapping):
            return item.get(name, default)
        return getattr(item, name, default)

    @staticmethod
    def _node_text(source_node: Any) -> str:
        return source_node.node.get_content(metadata_mode=MetadataMode.NONE)

    @classmethod
    def _node_identity(cls, source_node: Any) -> str:
        node = source_node.node
        node_id = str(getattr(node, "node_id", "") or "")
        if node_id:
            return node_id
        metadata = node.metadata or {}
        return "\x00".join(
            (
                str(metadata.get("doc_id") or metadata.get("source") or ""),
                cls._normalized_chunk_text(cls._node_text(source_node)),
            )
        )

    @classmethod
    def _context_text(cls, source_node: Any) -> str:
        """Normalize a node, focusing strict lexical evidence on its tightest passage."""

        text = cls._node_text(source_node)
        metadata = source_node.node.metadata or {}
        focus_query = str(metadata.get("_lexical_focus_query") or "").strip()
        if focus_query:
            required = set(cls._lexical_terms(focus_query))
            if span := cls._minimum_cover_char_span(text, required):
                start, end = span
                window_start = max(0, start - 160)
                window_end = min(len(text), end + 480)
                while window_start and not text[window_start - 1].isspace():
                    window_start -= 1
                while window_end < len(text) and not text[window_end].isspace():
                    window_end += 1
                text = text[window_start:window_end]
        return " ".join(text.split())

    @classmethod
    def _minimum_cover_char_span(cls, text: str, required: set[str]) -> tuple[int, int] | None:
        """Locate the shortest character passage containing all required stems."""

        if not required:
            return None
        occurrences = [
            (stem, match.start(), match.end())
            for match in _LEXICAL_TOKEN_RE.finditer(text.casefold().replace("ё", "е"))
            if (stem := cls._light_stem(match.group())) in required
        ]
        counts: dict[str, int] = {}
        missing = len(required)
        left = 0
        best: tuple[int, int] | None = None
        for right, (term, _start, end) in enumerate(occurrences):
            counts[term] = counts.get(term, 0) + 1
            if counts[term] == 1:
                missing -= 1
            while missing == 0:
                start = occurrences[left][1]
                if best is None or end - start < best[1] - best[0]:
                    best = (start, end)
                left_term = occurrences[left][0]
                left += 1
                counts[left_term] -= 1
                if counts[left_term] == 0:
                    missing += 1
        return best

    def _source_from_node(self, source_node: Any) -> dict[str, Any]:
        node = source_node.node
        metadata = node.metadata or {}
        source = str(metadata.get("file_name") or metadata.get("source") or metadata.get("file_path") or "unknown")
        page = metadata.get("page") or metadata.get("page_label") or metadata.get("page_number")
        if page is None and "#page=" in source:
            _source_name, _, page_text = source.partition("#page=")
            source = _source_name
            try:
                page = int(page_text)
            except ValueError:
                page = None
        if page is None and source.casefold().endswith(".pdf"):
            ref_doc_id = str(
                getattr(node, "ref_doc_id", "")
                or metadata.get("ref_doc_id")
                or metadata.get("document_id")
                or ""
            )
            legacy_match = _LEGACY_PAGE_REF_RE.search(ref_doc_id)
            if legacy_match:
                # Legacy ingestion created one parent document per physical PDF
                # page and suffixed its zero-based position as ``_part_N``.
                page = int(legacy_match.group(1)) + 1
        if page is not None:
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = None
        printed_page = self._printed_page(metadata, self._node_text(source_node))
        score = float(source_node.score or 0.0)
        snippet = self._context_text(source_node)[:MAX_SOURCE_SNIPPET_CHARS]
        return {
            "id": str(metadata.get("doc_id") or node.node_id),
            "file_name": Path(source).name,
            "page": page,
            "printed_page": printed_page,
            "score": score,
            "snippet": snippet,
            "confident": score >= self.config.min_score or bool(metadata.get("_lexical_match")),
        }

    @staticmethod
    def _printed_page(metadata: Mapping[str, Any], text: str) -> int | None:
        explicit = metadata.get("printed_page") or metadata.get("document_page")
        if explicit is not None:
            try:
                return int(explicit)
            except (TypeError, ValueError):
                return None
        # Printed page numbers are recovered only from a unique, isolated line.
        # This avoids treating equation/table numbers embedded in prose as pages.
        # PDF extractors place the running page number in the header.  Limit
        # the scan to that region so an isolated value in a later table is not
        # misreported as the printed page.
        matches = {int(value) for value in _PRINTED_PAGE_RE.findall(text[:400])}
        if len(matches) == 1:
            return matches.pop()
        return None


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Support Copilot RAG query")
    parser.add_argument("question", help="Question about the configured RAG corpus")
    parser.add_argument(
        "--allow-external-data-transfer",
        action="store_true",
        help="Confirm that manual fragments may be sent to the configured embedding and LLM provider",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    if not args.allow_external_data_transfer:
        raise SystemExit(
            "Refusing to send manual fragments externally. Re-run with "
            "--allow-external-data-transfer only after explicit authorization."
        )

    from app.core.config import get_settings

    service = RAGService(RAGConfig.from_settings(get_settings())).build()
    try:
        print(json.dumps(service.answer(args.question), ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
