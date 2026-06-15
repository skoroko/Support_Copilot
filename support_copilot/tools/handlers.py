from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SUPPORT_COPILOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_KB_PATH = SUPPORT_COPILOT_DIR / "data" / "knowledge_base.json"
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


class SearchKnowledgeBaseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=500)
    product: Literal[
        "support_copilot",
        "licensing",
        "installation",
        "calculation",
        "documentation",
        "security",
    ]


def _tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def _entry_text(entry: dict[str, Any]) -> str:
    parts = [
        str(entry.get("id", "")),
        str(entry.get("product", "")),
        str(entry.get("title", "")),
        str(entry.get("summary", "")),
        str(entry.get("answer", "")),
        " ".join(str(tag) for tag in entry.get("tags", [])),
    ]
    return " ".join(parts)


@lru_cache(maxsize=4)
def load_knowledge_base(path: str | Path = DEFAULT_KB_PATH) -> dict[str, Any]:
    kb_path = Path(path)
    return json.loads(kb_path.read_text(encoding="utf-8"))


def search_knowledge_base(query: str, product: str = "support_copilot") -> dict[str, Any]:
    args = SearchKnowledgeBaseArgs.model_validate({"query": query, "product": product})
    kb = load_knowledge_base()
    query_terms = set(_tokenize(args.query))
    scored_entries: list[tuple[int, list[str], dict[str, Any]]] = []

    for entry in kb.get("entries", []):
        entry_product = str(entry.get("product", ""))
        entry_terms = set(_tokenize(_entry_text(entry)))
        matched_terms = sorted(query_terms & entry_terms)
        score = len(matched_terms) * 2

        if args.product == entry_product:
            score += 5
        elif args.product == "support_copilot":
            score += 1
        else:
            score -= 2

        for tag in entry.get("tags", []):
            tag_terms = set(_tokenize(str(tag)))
            if tag_terms & query_terms:
                score += 2

        if score > 0:
            scored_entries.append((score, matched_terms, entry))

    scored_entries.sort(key=lambda item: item[0], reverse=True)
    results = []
    for score, matched_terms, entry in scored_entries[:3]:
        results.append(
            {
                "id": entry.get("id"),
                "product": entry.get("product"),
                "title": entry.get("title"),
                "summary": entry.get("summary"),
                "answer": entry.get("answer"),
                "source": entry.get("source"),
                "score": score,
                "matched_terms": matched_terms,
            }
        )

    return {
        "query": args.query,
        "product": args.product,
        "kb_version": kb.get("version"),
        "kb_scope": kb.get("scope"),
        "results": results,
        "needs_escalation": not results,
        "message": (
            "Найдены релевантные записи в локальной базе знаний."
            if results
            else "В базе знаний нет надежного совпадения; нужна проверка инженером поддержки."
        ),
    }


TOOL_HANDLERS = {
    "search_knowledge_base": search_knowledge_base,
}

