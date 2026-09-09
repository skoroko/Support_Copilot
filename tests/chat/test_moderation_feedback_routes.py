from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

from fastapi.testclient import TestClient

from app.chat.deps import get_chat_service
from app.chat.domain import Chat, ChatMessage
from app.chat.feedback import MessageFeedback
from app.chat.service import ModerationBlockedError
from app.main import app
from app.moderation.models import ModerationResult


class RouteService:
    def __init__(self) -> None:
        self.chat = Chat(owner_external_id="42", interface="telegram")
        self.message = ChatMessage(chat_id=self.chat.id, role="assistant", content="answer")
        self.send_calls = 0

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        return self.chat if chat_id == self.chat.id else None

    async def ensure_input_allowed(self, _chat_id: UUID, _content: str) -> ModerationResult:
        raise ModerationBlockedError(
            ModerationResult(
                allowed=False,
                categories=["violence"],
                reasons=["blocked"],
                blocked_by="keyword",
            )
        )

    async def send_message(self, *_args, **_kwargs) -> AsyncIterator[str]:
        self.send_calls += 1
        yield "must not happen"

    async def submit_feedback(self, chat_id: UUID, message_id: UUID, value: str) -> MessageFeedback:
        assert chat_id == self.chat.id
        assert message_id == self.message.id
        return MessageFeedback(message_id=message_id, owner_external_id="42", value=value)


def test_moderation_route_returns_403_before_llm() -> None:
    service = RouteService()
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(f"/chats/{service.chat.id}/messages", data={"content": "blocked"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "moderation_blocked"
    assert response.json()["error"]["categories"] == ["violence"]
    assert service.send_calls == 0


def test_feedback_route_returns_saved_vote() -> None:
    service = RouteService()
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/chats/{service.chat.id}/messages/{service.message.id}/feedback",
                json={"value": "up"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["message_id"] == str(service.message.id)
    assert response.json()["value"] == "up"
