from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RerankCandidate:
    text: str
    doc_id: str
    source: str
    score: float
    rerank_score: float | None = None


class BGEReranker:
    """Lazy multilingual cross-encoder used only after dense candidate retrieval."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        model: Any | None = None,
        device: str | None = None,
        batch_size: int = 4,
        max_length: int = 512,
        local_files_only: bool = False,
    ) -> None:
        if batch_size < 1 or max_length < 32:
            raise ValueError("reranker batch_size and max_length must be positive")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.local_files_only = local_files_only
        self._model = model

    def rerank(
        self,
        question: str,
        candidates: list[RerankCandidate],
        *,
        top_n: int,
    ) -> list[RerankCandidate]:
        if not question.strip():
            raise ValueError("question must not be blank")
        if top_n < 1:
            raise ValueError("top_n must be positive")
        if not candidates:
            return []

        model = self._load_model()
        scores = model.predict(
            [(question, candidate.text) for candidate in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        rescored = [
            replace(candidate, rerank_score=float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(
            key=lambda item: (
                item.rerank_score
                if item.rerank_score is not None
                else float("-inf")
            ),
            reverse=True,
        )
        return rescored[:top_n]

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                # Text-only CrossEncoder does not need torchaudio. Some shared
                # workstation environments contain an optional torchaudio build
                # that is incompatible with the installed CPU-only torch build;
                # make Transformers treat that broken optional dependency as absent.
                try:
                    import torchaudio  # noqa: F401
                except (ImportError, OSError):
                    import transformers.utils as transformers_utils

                    transformers_utils.is_torchaudio_available = lambda: False

                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for BAAI/bge-reranker-v2-m3"
                ) from exc
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
                local_files_only=self.local_files_only,
            )
        return self._model
