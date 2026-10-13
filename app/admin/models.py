from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class StatsOut(BaseModel):
    total_messages: int = 0
    active_users: int = 0
    avg_latency_ms: float = 0.0
    moderation_block_rate: float = 0.0
    feedback_up_ratio: float = 0.0


class UserOut(BaseModel):
    owner_external_id: str
    chat_count: int
    last_seen_at: datetime


class BroadcastIn(BaseModel):
    message: str = Field(min_length=1, max_length=4096)
    interface_filter: Literal["telegram"] = "telegram"


class BroadcastRecipient(BaseModel):
    chat_id: int
    status: Literal["pending", "in_flight", "sent", "failed", "unknown"] = "pending"
    attempt_id: UUID | None = None


class BroadcastItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    message: str
    interface: str
    status: Literal["pending", "processing", "sent", "failed"] = "pending"
    recipients: list[BroadcastRecipient] = Field(default_factory=list)
    invalid_recipient_count: int = 0
    lease_id: UUID | None = None
    lease_expires_at: datetime | None = None
    lease_seconds: int = 300
    review_required: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BroadcastClaimIn(BaseModel):
    limit: int = Field(default=10, ge=1, le=100)
    lease_seconds: int = Field(default=300, ge=30, le=3600)


class BroadcastRecipientStateIn(BaseModel):
    lease_id: UUID
    attempt_id: UUID
    status: Literal["in_flight", "sent", "failed", "unknown"]


class BroadcastFinishIn(BaseModel):
    lease_id: UUID
