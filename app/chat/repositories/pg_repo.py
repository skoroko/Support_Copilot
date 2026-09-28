from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import and_, distinct, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.broadcast_state import claim_item, finish_item, update_recipient
from app.admin.models import BroadcastItem, StatsOut, UserOut
from app.chat.domain import Chat, ChatMessage
from app.chat.feedback import MessageFeedback
from app.chat.repositories.pg_models import (
    BroadcastQueueRow,
    ChatMessageRow,
    ChatRow,
    MessageFeedbackRow,
    ModerationEventRow,
)
from app.moderation.models import ModerationResult


class PostgresChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        chat = Chat(
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
        )
        self.session.add(
            ChatRow(
                id=chat.id,
                owner_external_id=chat.owner_external_id,
                interface=chat.interface,
                system_prompt=chat.system_prompt,
                created_at=chat.created_at,
            )
        )
        await self.session.commit()
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        row = await self.session.get(ChatRow, chat_id)
        if row is None:
            return None
        return Chat.model_validate(row, from_attributes=True)

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        message = message.model_copy(update={"chat_id": chat_id})
        self.session.add(
            ChatMessageRow(
                id=message.id,
                chat_id=message.chat_id,
                role=message.role,
                content=message.content,
                media_refs=message.media_refs,
                rag_sources=message.rag_sources,
                tokens=message.tokens,
                latency_ms=message.latency_ms,
                created_at=message.created_at,
            )
        )
        await self.session.commit()
        return message

    async def append_turn(
        self,
        chat_id: UUID,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
    ) -> tuple[ChatMessage, ChatMessage]:
        messages = (
            user_message.model_copy(update={"chat_id": chat_id}),
            assistant_message.model_copy(update={"chat_id": chat_id}),
        )
        try:
            for message in messages:
                self.session.add(
                    ChatMessageRow(
                        id=message.id,
                        chat_id=message.chat_id,
                        role=message.role,
                        content=message.content,
                        media_refs=message.media_refs,
                        rag_sources=message.rag_sources,
                        tokens=message.tokens,
                        latency_ms=message.latency_ms,
                        created_at=message.created_at,
                    )
                )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        return messages

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        if limit <= 0:
            return []
        result = await self.session.execute(
            select(ChatMessageRow)
            .where(ChatMessageRow.chat_id == chat_id, ChatMessageRow.deleted_at.is_(None))
            .order_by(ChatMessageRow.created_at.desc())
            .limit(limit)
        )
        rows = list(result.scalars().all())
        return [ChatMessage.model_validate(row, from_attributes=True) for row in reversed(rows)]

    async def get_message(self, chat_id: UUID, message_id: UUID) -> ChatMessage | None:
        result = await self.session.execute(
            select(ChatMessageRow).where(
                ChatMessageRow.id == message_id,
                ChatMessageRow.chat_id == chat_id,
                ChatMessageRow.deleted_at.is_(None),
            )
        )
        row = result.scalar_one_or_none()
        return ChatMessage.model_validate(row, from_attributes=True) if row is not None else None

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        await self.session.execute(
            update(ChatMessageRow)
            .where(ChatMessageRow.chat_id == chat_id, ChatMessageRow.deleted_at.is_(None))
            .values(deleted_at=datetime.now(timezone.utc))
        )
        await self.session.commit()

    async def save_feedback(self, feedback: MessageFeedback) -> MessageFeedback:
        statement = insert(MessageFeedbackRow).values(
            id=uuid4(),
            message_id=feedback.message_id,
            owner_external_id=feedback.owner_external_id,
            value=feedback.value,
            shown_sources=feedback.shown_sources,
            created_at=feedback.created_at,
        )
        statement = statement.on_conflict_do_update(
            constraint="uq_message_feedback_owner_message",
            set_={
                "value": feedback.value,
                "shown_sources": feedback.shown_sources,
                "created_at": feedback.created_at,
            },
        )
        await self.session.execute(statement)
        await self.session.commit()
        return feedback

    async def record_moderation(
        self,
        chat_id: UUID,
        direction: str,
        result: ModerationResult,
    ) -> None:
        self.session.add(
            ModerationEventRow(
                id=uuid4(),
                chat_id=chat_id,
                direction=direction,
                allowed=result.allowed,
                categories=result.categories,
                blocked_by=result.blocked_by,
                created_at=datetime.now(timezone.utc),
            )
        )
        await self.session.commit()

    async def get_admin_stats(self) -> StatsOut:
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        total_messages = await self.session.scalar(
            select(func.count(ChatMessageRow.id)).where(
                ChatMessageRow.created_at >= since,
                ChatMessageRow.deleted_at.is_(None),
            )
        )
        active_users = await self.session.scalar(
            select(func.count(distinct(ChatRow.owner_external_id)))
            .join(ChatMessageRow, ChatMessageRow.chat_id == ChatRow.id)
            .where(ChatMessageRow.created_at >= since, ChatMessageRow.deleted_at.is_(None))
        )
        avg_latency = await self.session.scalar(
            select(func.avg(ChatMessageRow.latency_ms)).where(
                ChatMessageRow.created_at >= since,
                ChatMessageRow.deleted_at.is_(None),
                ChatMessageRow.latency_ms.is_not(None),
            )
        )
        moderation_total = await self.session.scalar(
            select(func.count(ModerationEventRow.id)).where(ModerationEventRow.created_at >= since)
        )
        moderation_blocked = await self.session.scalar(
            select(func.count(ModerationEventRow.id)).where(
                ModerationEventRow.created_at >= since,
                ModerationEventRow.allowed.is_(False),
            )
        )
        feedback_total = await self.session.scalar(
            select(func.count(MessageFeedbackRow.id)).where(MessageFeedbackRow.created_at >= since)
        )
        feedback_up = await self.session.scalar(
            select(func.count(MessageFeedbackRow.id)).where(
                MessageFeedbackRow.created_at >= since,
                MessageFeedbackRow.value == "up",
            )
        )
        moderation_total_value = int(moderation_total or 0)
        feedback_total_value = int(feedback_total or 0)
        return StatsOut(
            total_messages=int(total_messages or 0),
            active_users=int(active_users or 0),
            avg_latency_ms=round(float(avg_latency or 0.0), 2),
            moderation_block_rate=(
                round(int(moderation_blocked or 0) / moderation_total_value, 4)
                if moderation_total_value
                else 0.0
            ),
            feedback_up_ratio=(
                round(int(feedback_up or 0) / feedback_total_value, 4) if feedback_total_value else 0.0
            ),
        )

    async def list_admin_users(self, limit: int = 50, interface: str | None = None) -> list[UserOut]:
        last_seen = func.coalesce(func.max(ChatMessageRow.created_at), func.max(ChatRow.created_at))
        statement = (
            select(
                ChatRow.owner_external_id,
                func.count(distinct(ChatRow.id)).label("chat_count"),
                last_seen.label("last_seen_at"),
            )
            .outerjoin(ChatMessageRow, ChatMessageRow.chat_id == ChatRow.id)
            .group_by(ChatRow.owner_external_id)
            .order_by(last_seen.desc())
            .limit(limit)
        )
        if interface is not None:
            statement = statement.where(ChatRow.interface == interface)
        rows = (await self.session.execute(statement)).all()
        return [
            UserOut(owner_external_id=row.owner_external_id, chat_count=row.chat_count, last_seen_at=row.last_seen_at)
            for row in rows
        ]

    async def enqueue_broadcast(self, message: str, interface: str) -> BroadcastItem:
        item = BroadcastItem(message=message, interface=interface)
        self.session.add(BroadcastQueueRow(**item.model_dump(exclude={"recipients"}), recipients=[]))
        await self.session.commit()
        return item

    async def list_pending_broadcasts(self, limit: int = 10) -> list[BroadcastItem]:
        result = await self.session.execute(
            select(BroadcastQueueRow)
            .where(BroadcastQueueRow.status == "pending")
            .order_by(BroadcastQueueRow.created_at)
            .limit(limit)
        )
        return [BroadcastItem.model_validate(row, from_attributes=True) for row in result.scalars().all()]

    async def list_review_broadcasts(self, limit: int = 10, offset: int = 0) -> list[BroadcastItem]:
        result = await self.session.execute(
            select(BroadcastQueueRow)
            .where(BroadcastQueueRow.review_required.is_(True))
            .order_by(BroadcastQueueRow.created_at, BroadcastQueueRow.id)
            .offset(offset)
            .limit(limit)
        )
        return [BroadcastItem.model_validate(row, from_attributes=True) for row in result.scalars().all()]

    async def get_broadcast(self, broadcast_id: UUID) -> BroadcastItem | None:
        row = await self.session.get(BroadcastQueueRow, broadcast_id)
        return BroadcastItem.model_validate(row, from_attributes=True) if row is not None else None

    async def claim_broadcasts(self, limit: int = 10, lease_seconds: int = 300) -> list[BroadcastItem]:
        now = datetime.now(timezone.utc)
        result = await self.session.execute(
            select(BroadcastQueueRow)
            .where(or_(
                BroadcastQueueRow.status == "pending",
                and_(BroadcastQueueRow.status == "processing", BroadcastQueueRow.lease_expires_at <= now),
            ))
            .order_by(BroadcastQueueRow.created_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        owners_by_interface: dict[str, list[str]] = {}
        claimed: list[BroadcastItem] = []
        for row in result.scalars().all():
            if row.status == "pending" and row.interface not in owners_by_interface:
                owners = await self.session.scalars(
                    select(distinct(ChatRow.owner_external_id)).where(ChatRow.interface == row.interface)
                )
                owners_by_interface[row.interface] = list(owners.all())
            item = claim_item(
                BroadcastItem.model_validate(row, from_attributes=True),
                owners_by_interface.get(row.interface, []), now=now, lease_seconds=lease_seconds,
            )
            if item is not None:
                self._store_broadcast(row, item)
                claimed.append(item)
        await self.session.commit()
        return claimed

    async def update_broadcast_recipient(
        self, broadcast_id: UUID, lease_id: UUID, chat_id: int, attempt_id: UUID, status: str,
    ) -> BroadcastItem | None:
        row = await self._locked_broadcast(broadcast_id)
        if row is None:
            await self.session.rollback()
            return None
        updated = update_recipient(
            BroadcastItem.model_validate(row, from_attributes=True),
            lease_id=lease_id, chat_id=chat_id, attempt_id=attempt_id,
            status=status, now=datetime.now(timezone.utc),
        )
        if updated is None:
            await self.session.rollback()
            return None
        self._store_broadcast(row, updated)
        await self.session.commit()
        return updated

    async def finish_broadcast(self, broadcast_id: UUID, lease_id: UUID) -> BroadcastItem | None:
        row = await self._locked_broadcast(broadcast_id)
        if row is None:
            await self.session.rollback()
            return None
        updated = finish_item(
            BroadcastItem.model_validate(row, from_attributes=True), lease_id=lease_id,
            now=datetime.now(timezone.utc),
        )
        if updated is None:
            await self.session.rollback()
            return None
        self._store_broadcast(row, updated)
        await self.session.commit()
        return updated

    async def _locked_broadcast(self, broadcast_id: UUID) -> BroadcastQueueRow | None:
        result = await self.session.execute(
            select(BroadcastQueueRow).where(BroadcastQueueRow.id == broadcast_id).with_for_update()
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _store_broadcast(row: BroadcastQueueRow, item: BroadcastItem) -> None:
        row.status = item.status
        row.recipients = [recipient.model_dump(mode="json") for recipient in item.recipients]
        row.invalid_recipient_count = item.invalid_recipient_count
        row.lease_id = item.lease_id
        row.lease_expires_at = item.lease_expires_at
        row.lease_seconds = item.lease_seconds
        row.review_required = item.review_required
        row.updated_at = item.updated_at
