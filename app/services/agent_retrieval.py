"""Read-only knowledge retrieval, separate from ordered citation-state merging."""
from __future__ import annotations

import asyncio
from typing import Any

from app.observability.logging import get_logger
from app.services.turn_budget import remaining_seconds

logger = get_logger("agent-retrieval")


async def fetch_knowledge(args: dict, messages: list[Any], rag: Any, *,
                          max_context_chars: int, unavailable_message: str,
                          track_worker: Any = None, navigation_items: list[dict] | None = None,
                          known_atomic_units: frozenset[str] = frozenset()) -> dict:
    query = args.get("query")
    result: dict[str, Any] = {
        "status": "unavailable",
        "sources": [],
        "contexts": [],
        "message": unavailable_message,
    }
    if isinstance(query, str) and query.strip() and rag is not None:
        english = args.get("english_query")
        retrieval_options = {}
        if known_atomic_units and getattr(rag, 'supports_prepared_evidence', False):
            retrieval_options['known_atomic_units'] = known_atomic_units
        if isinstance(english, str) and english.strip():
            # The model already supplied a standalone translation;
            # a second condense call can discard half of the query.
            retrieval_options["standalone_question"] = query
            retrieval_options["translated_question"] = english.strip()
        history = [
            {
                "role": "user" if item.type == "human" else "assistant",
                "content": item.content,
            }
            for item in messages
            if item.type in {"human", "ai"}
            and not getattr(item, "tool_calls", [])
        ]
        # The current query is not earlier dialogue. Passing it as a
        # one-message "history" makes an independent first turn pay
        # for a redundant condense call. A different targeted query
        # may still need the current human question as its anchor.
        if (history and history[-1]["role"] == "user"
                and isinstance(history[-1]["content"], str)
                and history[-1]["content"].strip() == query.strip()):
            history.pop()
        history = history[-10:]
        try:
            timeout = remaining_seconds(min(
                float(
                    getattr(getattr(rag, "config", None), "request_timeout", 60)
                ),
                60.0,
            ))
            if navigation_items is not None:
                worker = asyncio.create_task(asyncio.to_thread(
                    rag.read_navigation, navigation_items, max_context_chars=max_context_chars))
            else:
                worker = asyncio.create_task(
                    asyncio.to_thread(
                        rag.retrieve,
                        query,
                        history=history,
                        timeout_seconds=timeout,
                        max_context_chars=max_context_chars,
                        **retrieval_options,
                    )
                )
            tracker = track_worker
            if tracker is not None:
                tracker(worker)
            worker.add_done_callback(
                lambda task: task.exception() if not task.cancelled() else None
            )
            result = await asyncio.wait_for(
                asyncio.shield(worker),
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - safe provider boundary
            # Do not leak provider exceptions, request text, or credentials.
            logger.warning(
                "agent_retrieval_unavailable", error_type=type(exc).__name__
            )
    elif not isinstance(query, str) or not query.strip():
        result = {
            "status": "error",
            "sources": [],
            "contexts": [],
            "message": "Нужен непустой query.",
        }
    return result
