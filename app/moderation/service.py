from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from app.moderation.models import ModerationResult
from app.observability.logging import get_logger
from app.observability.pii import prompt_hash, redact_pii


logger = get_logger("moderation")


class ModerationService:
    def __init__(
        self,
        *,
        keywords_path: Path,
        client: Any | None = None,
        enabled: bool = True,
        openai_enabled: bool = False,
        model: str = "omni-moderation-latest",
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.enabled = enabled
        self.client = client
        self.openai_enabled = openai_enabled
        self.model = model
        self.thresholds = thresholds or {}
        self.patterns = self._load_patterns(keywords_path)

    async def check_input(self, content: str) -> ModerationResult:
        return await self._check(content, direction="input")

    async def check_output(self, content: str) -> ModerationResult:
        return await self._check(content, direction="output")

    def check_output_fragment(self, content: str) -> ModerationResult:
        """Keyword-only screen for one streamed output prefix.

        Runs per provider delta on the accumulated answer, so it stays
        synchronous and never calls the optional moderation API — the
        authoritative check remains ``check_output`` on the completed answer.
        """

        if not self.enabled:
            return ModerationResult()
        result = self._check_keywords(content)
        if not result.allowed:
            self._log_incident(content, result, direction="output")
        return result

    async def _check(self, content: str, *, direction: str) -> ModerationResult:
        if not self.enabled or not content.strip():
            return ModerationResult()

        keyword_result = self._check_keywords(content)
        if not keyword_result.allowed:
            self._log_incident(content, keyword_result, direction=direction)
            return keyword_result

        if not self.openai_enabled or self.client is None:
            return ModerationResult()

        try:
            response = await self.client.moderations.create(model=self.model, input=content)
            result = response.results[0]
            category_flags = self._model_mapping(result.categories)
            category_scores = self._model_mapping(result.category_scores)
            categories = sorted(name for name, flagged in category_flags.items() if bool(flagged))
            reasons = [f"OpenAI flagged category: {name}" for name in categories]

            for name, threshold in self.thresholds.items():
                score = self._score_for(category_scores, name)
                if score is not None and score >= threshold and name not in categories:
                    categories.append(name)
                    reasons.append(f"Category score {name}={score:.4f} reached threshold {threshold:.4f}")

            blocked = bool(result.flagged) or bool(categories)
            moderation_result = ModerationResult(
                allowed=not blocked,
                categories=sorted(categories),
                reasons=reasons,
                blocked_by="openai" if blocked else "none",
            )
            if blocked:
                self._log_incident(content, moderation_result, direction=direction)
            return moderation_result
        except Exception as exc:  # noqa: BLE001 - moderation API is an optional layer
            logger.warning(
                "moderation_api_failed",
                direction=direction,
                text_hash=prompt_hash(content),
                error_type=type(exc).__name__,
            )
            return ModerationResult(reasons=["optional moderation API unavailable"])

    def _check_keywords(self, content: str) -> ModerationResult:
        categories: list[str] = []
        reasons: list[str] = []
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    categories.append(category)
                    reasons.append(f"Matched local moderation rule: {category}")
                    break
        return ModerationResult(
            allowed=not categories,
            categories=sorted(categories),
            reasons=reasons,
            blocked_by="keyword" if categories else "none",
        )

    def _log_incident(self, content: str, result: ModerationResult, *, direction: str) -> None:
        logger.warning(
            "moderation_blocked",
            direction=direction,
            text_hash=prompt_hash(content),
            masked_text=redact_pii(content)[:160],
            categories=result.categories,
            blocked_by=result.blocked_by,
        )

    @staticmethod
    def _load_patterns(path: Path) -> dict[str, list[re.Pattern[str]]]:
        if not path.exists():
            raise FileNotFoundError(f"Moderation keywords file not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        categories = payload.get("categories", {})
        if not isinstance(categories, dict):
            raise ValueError("moderation_keywords.yaml must contain a categories mapping")
        compiled: dict[str, list[re.Pattern[str]]] = {}
        for category, expressions in categories.items():
            if not isinstance(expressions, list):
                raise ValueError(f"Moderation category {category!r} must contain a list of regex patterns")
            compiled[str(category)] = [re.compile(str(expression), re.IGNORECASE | re.UNICODE) for expression in expressions]
        return compiled

    @staticmethod
    def _model_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            return dict(dump(by_alias=True))
        return {}

    @staticmethod
    def _score_for(scores: dict[str, Any], category: str) -> float | None:
        aliases = {
            category,
            category.replace("/", "_"),
            category.replace("-", "_").replace("/", "_"),
        }
        for alias in aliases:
            if alias in scores:
                try:
                    return float(scores[alias])
                except (TypeError, ValueError):
                    return None
        return None
