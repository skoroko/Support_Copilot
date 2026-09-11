from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from app.services.ingestion import IngestionConfig, IngestionService, canonical_product, infer_product
from app.services.reranker import BGEReranker, RerankCandidate


RAG_SYSTEM_PROMPT = """Ты — помощник технической поддержки MIDAS GTS NX.
Отвечай только на русском языке и только по предоставленному контексту.
Документы могут быть на русском или английском: переводи найденный ответ на русский.
Если контекст не содержит надёжного ответа, честно сообщи, что сведений в базе нет.
Не используй нерелевантные документы и не придумывай цены, SLA, функции или регламенты.
После каждого утверждения указывай ссылки на использованные фрагменты в формате [1], [2].
"""
NO_ANSWER = "В предоставленной базе знаний нет достаточно надёжных сведений для ответа на этот вопрос."
MAX_CONTEXT_SOURCES = 5
MAX_RAW_CANDIDATES = 30
RAW_CANDIDATE_MULTIPLIER = 3
NEAR_DUPLICATE_RATIO = 0.9
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
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
    "на",
    "не",
    "но",
    "о",
    "по",
    "при",
    "с",
    "со",
    "что",
    "это",
    "gts",
    "midas",
    "nx",
}
_LEXICAL_IGNORED_PREFIXES = ("использ",)
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

HistoryItem = Mapping[str, Any] | Any
CondenseHook = Callable[[str, Sequence[HistoryItem]], str | None]


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
    chunk_size: int = 512
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
                Path(__file__).resolve().parents[2] / "var" / "ingestion_docstore.json",
            )
        )
        if not docstore_path.is_absolute():
            docstore_path = Path(__file__).resolve().parents[2] / docstore_path
        return cls(
            data_dir=data_dir,
            qdrant_url=getattr(settings, "qdrant_url", None) or "http://localhost:6333",
            qdrant_api_key=api_key,
            collection_name=getattr(settings, "rag_collection", "rag_block_03"),
            baremetal_collection_name=getattr(settings, "rag_baremetal_collection", "rag_block_03_baremetal"),
            llm_model=getattr(settings, "rag_llm_model", "gpt-5.4-mini"),
            embedding_model=getattr(settings, "rag_embedding_model", "text-embedding-3-small"),
            embedding_dimensions=getattr(settings, "rag_embedding_dimensions", 1536),
            chunk_size=getattr(settings, "rag_chunk_size", 512),
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
        retrieval_question = self._standalone_question(question, history, standalone_question)
        retrieved = list(self._retriever.retrieve(retrieval_question))[: self._raw_candidate_limit()]
        product_candidates = [item for item in retrieved if self._matches_target_product(item)]
        # Rank the complete product-scoped pool before removing duplicates or
        # applying the final context limit.  An earlier per-file cap discarded
        # exact evidence whenever two weaker dense matches from the same manual
        # happened to arrive first.
        ranked = self._rank_by_lexical_evidence(retrieval_question, product_candidates)
        ranked = self._rerank(retrieval_question, ranked)
        ranked = self._deduplicate(ranked, limit=self._context_limit())
        sources = [self._source_from_node(item) for item in ranked]
        # Cosine similarity remains the confidence signal even when a
        # cross-encoder changes presentation order.  Cross-encoder scores are
        # not calibrated to the embedding threshold.
        top_score = max((source["score"] for source in sources), default=0.0)
        if top_score < self.config.min_score:
            return {"answer": NO_ANSWER, "top_score": top_score, "sources": sources}
        answer = self._generate_answer(question.strip(), sources)
        return {"answer": answer, "top_score": top_score, "sources": sources}

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

    def _deduplicate(self, source_nodes: Sequence[Any], *, limit: int) -> list[Any]:
        """Remove duplicate chunks after ranking and cap the LLM context."""

        selected: list[Any] = []
        selected_texts: list[str] = []
        for source_node in source_nodes:
            normalized_text = self._normalized_chunk_text(self._node_text(source_node))
            if any(self._near_duplicate(normalized_text, previous) for previous in selected_texts):
                continue
            selected.append(source_node)
            selected_texts.append(normalized_text)
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
        previous_user_question = next(
            (
                content
                for item in reversed(history)
                if self._history_value(item, "role") == "user"
                and (content := str(self._history_value(item, "content", "")).strip())
                and content != question.strip()
            ),
            "",
        )
        if not previous_user_question:
            return question.strip()
        return f"Контекст предыдущего вопроса: {previous_user_question}\nУточняющий вопрос: {question.strip()}"

    def _generate_answer(self, question: str, sources: list[dict[str, Any]]) -> str:
        context = "\n\n".join(
            f"[{index}] Файл: {source['file_name']}; страница: {source['page'] or 'не указана'}\n"
            f"{source['snippet']}"
            for index, source in enumerate(sources, start=1)
        )
        prompt = (
            f"{RAG_SYSTEM_PROMPT}\n\nКонтекст:\n{context}\n\n"
            f"Вопрос пользователя: {question}\n\n"
            f"Дай краткий проверяемый ответ. Используй только контекст и ссылайся только на "
            f"существующие номера от [1] до [{len(sources)}]."
        )
        response = self._language_model().complete(prompt)
        answer = str(getattr(response, "text", response) or "").strip() or NO_ANSWER
        return self._ensure_citations(answer, sources)

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
        if page is not None:
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = None
        score = float(source_node.score or 0.0)
        snippet = " ".join(self._node_text(source_node).split())[:700]
        return {
            "id": str(metadata.get("doc_id") or node.node_id),
            "file_name": Path(source).name,
            "page": page,
            "score": score,
            "snippet": snippet,
            "confident": score >= self.config.min_score,
        }


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
