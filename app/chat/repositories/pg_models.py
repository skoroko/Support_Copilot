from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, Text, UniqueConstraint, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ChatRow(Base):
    __tablename__ = "chats"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    owner_external_id: Mapped[str] = mapped_column(Text, nullable=False)
    interface: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ChatMessageRow(Base):
    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    chat_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    media_refs: Mapped[dict | None] = mapped_column(JSON)
    rag_sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list, server_default="[]")
    tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


Index(
    "ix_chat_messages_chat_created",
    ChatMessageRow.chat_id,
    ChatMessageRow.created_at.desc(),
    postgresql_where=ChatMessageRow.deleted_at.is_(None),
)


class MessageFeedbackRow(Base):
    __tablename__ = "message_feedback"
    __table_args__ = (
        UniqueConstraint("owner_external_id", "message_id", name="uq_message_feedback_owner_message"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    message_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    owner_external_id: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    shown_sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list, server_default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ModerationEventRow(Base):
    __tablename__ = "moderation_events"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    chat_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
    )
    direction: Mapped[str] = mapped_column(Text, nullable=False)
    allowed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    categories: Mapped[list] = mapped_column(JSON, nullable=False)
    blocked_by: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class BroadcastQueueRow(Base):
    __tablename__ = "broadcast_queue"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    interface: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    recipients: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list, server_default="[]")
    invalid_recipient_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    lease_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_seconds: Mapped[int] = mapped_column(Integer, nullable=False, server_default="300")
    review_required: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


Index("ix_broadcast_queue_status_created", BroadcastQueueRow.status, BroadcastQueueRow.created_at)
