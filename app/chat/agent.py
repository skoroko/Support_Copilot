"""Persistent agent adapter for the existing authenticated chat facade."""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app.chat.domain import Chat
from app.services.agent_persistent import APPROVAL_NODE, retrieval_worker_tracker
from app.services.rag import RAGService


class StaleApprovalError(ValueError):
    pass


class ChatAgent:
    def __init__(
        self,
        graph: Any,
        checkpointer: Any,
        track_worker: Any = None,
        settings: Any = None,
    ) -> None:
        self.graph = graph
        self.checkpointer = checkpointer
        self.track_worker = track_worker
        self.settings = settings

    async def _invoke(self, value: Any, chat: Chat, *, expanded: bool = False) -> dict:
        token = retrieval_worker_tracker.set(self.track_worker)
        try:
            config = self.config(chat)
            response_tokens = getattr(self.settings, "chat_expanded_response_tokens", 1500) if expanded else getattr(self.settings, "chat_response_tokens", 500)
            config["configurable"].update(response_tokens=response_tokens, expanded_mode=expanded)
            reasoning_reserve = 0
            if self.settings is not None:
                model = getattr(self.settings, "default_model", "")
                effort = getattr(self.settings, "agent_reasoning_effort", None)
                if expanded:
                    effort = getattr(self.settings, "agent_expanded_reasoning_effort", None) or effort
                if effort is None:
                    effort = "low" if model.startswith("gpt-6") else "none"
                if effort != "none":
                    reasoning_reserve = getattr(self.settings, "agent_reasoning_token_reserve", 8192)
            if self.settings is not None:
                config["configurable"].update(
                    {
                        "context_budget": max(
                            1,
                            self.settings.chat_model_context_tokens
                            - response_tokens
                            - reasoning_reserve
                            - self.settings.chat_safety_margin_tokens,
                        ),
                        "history_window": self.settings.chat_context_window,
                    }
                )
            return await self.graph.ainvoke(value, config)
        finally:
            retrieval_worker_tracker.reset(token)

    @staticmethod
    def config(chat: Chat) -> dict[str, Any]:
        material = f"{chat.interface}\0{chat.owner_external_id}\0{chat.id}".encode()
        return {
            "configurable": {
                "thread_id": "chat-agent-" + hashlib.sha256(material).hexdigest(),
                "user_role": "write-with-approve",
            }
        }

    async def pending(self, chat: Chat) -> dict[str, str] | None:
        snapshot = await self.graph.aget_state(self.config(chat))
        if APPROVAL_NODE in snapshot.next:
            draft = snapshot.values.get("telegram_draft")
            if draft:
                return dict(draft)
        return None

    async def turn(
        self, chat: Chat, content: Any, history: list | None = None
    ) -> dict[str, Any]:
        pending = await self.pending(chat)
        if pending:
            return {
                "answer": "Сначала подтвердите или отмените подготовленную отправку.",
                "sources": [],
                "approval": pending,
            }
        snapshot = await self.graph.aget_state(self.config(chat))
        messages = []
        if not snapshot.values.get("messages"):
            for message in (history or [])[-10:]:
                message_type = HumanMessage if message.role == "user" else AIMessage
                historical_content = message.content
                if message.role == "user" and message.media_refs:
                    parts = message.media_refs.get("parts") or [
                        message.media_refs.get("part")
                    ]
                    extracted = [
                        str(part["text"])
                        for part in parts
                        if isinstance(part, dict)
                        and part.get("type") == "text"
                        and part.get("text")
                    ]
                    historical_content = "\n".join([historical_content, *extracted])[:16_000]
                messages.append(message_type(content=historical_content))
        messages.append(HumanMessage(content=content))
        state = await self._invoke({"messages": messages}, chat)
        return self._result(state, await self.pending(chat))

    async def resume(
        self, chat: Chat, approval_id: str, approved: bool
    ) -> dict[str, Any]:
        pending = await self.pending(chat)
        if pending is None or pending.get("approval_id") != approval_id:
            raise StaleApprovalError("Подтверждение устарело или уже обработано.")
        state = await self._invoke(Command(resume=approved), chat)
        return self._result(state, await self.pending(chat))

    async def expansion_claim(self, chat: Chat, message_id: str) -> str | None:
        snapshot = await self.graph.aget_state(self.config(chat))
        return (snapshot.values.get("expansion_claims") or {}).get(message_id)

    async def claim_expansion(self, chat: Chat, message_id: str, result_id: str) -> None:
        # Caller holds the shared chat turn lock. Commit the one-use claim
        # before a billed call; failures remain claimed across API restarts.
        snapshot = await self.graph.aget_state(self.config(chat))
        claims = dict(snapshot.values.get("expansion_claims") or {})
        claims[message_id] = result_id
        await self.graph.aupdate_state(self.config(chat), {"expansion_claims": dict(list(claims.items())[-20:])})

    async def expand(self, chat: Chat, content: str) -> dict[str, Any]:
        state = await self._invoke({"messages": [HumanMessage(content=content)]}, chat, expanded=True)
        return self._result(state, await self.pending(chat))

    async def redact_output(self, chat: Chat, replacement: str) -> None:
        """Remove a blocked answer from live context without deleting claims."""
        snapshot = await self.graph.aget_state(self.config(chat))
        latest = next((item for item in reversed(snapshot.values.get("messages", []))
                       if isinstance(item, AIMessage)), None)
        update = {"sources": [], "answer_evidence": [], "citation_evidence": [], "knowledge_status": "unused",
                  "answer_incomplete": False, "telegram_draft": None}
        if latest is not None:
            update["messages"] = [latest.model_copy(update={"content": replacement,
                                                           "tool_calls": [], "invalid_tool_calls": []})]
        await self.graph.aupdate_state(self.config(chat), update)

    async def clear(self, chat: Chat) -> None:
        await self.checkpointer.adelete_thread(
            self.config(chat)["configurable"]["thread_id"]
        )

    @staticmethod
    def _result(
        state: dict[str, Any], pending: dict[str, str] | None
    ) -> dict[str, Any]:
        latest = next(
            (
                item
                for item in reversed(state.get("messages", []))
                if isinstance(item, AIMessage)
            ),
            None,
        )
        answer, sources = RAGService.ground_answer(
            str(latest.content or "") if latest else "Не удалось подготовить ответ.",
            list(state.get("sources") or []),
        )
        mode = state.get("answer_mode", "brief")
        return {"answer": answer, "sources": sources, "approval": pending,
                "answer_info": {"mode": mode, "incomplete": bool(state.get("answer_incomplete")),
                                "expandable": mode == "brief" and bool(state.get("answer_evidence"))
                                and state.get("knowledge_status") == "ok" and pending is None}}
