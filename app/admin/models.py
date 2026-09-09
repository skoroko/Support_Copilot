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


class BroadcastItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    message: str
    interface: str
    status: Literal["pending", "processing", "sent", "failed"] = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BroadcastStatusIn(BaseModel):
    status: Literal["processing", "sent", "failed"]
