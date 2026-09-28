"""Pure broadcast transitions shared by JSON and PostgreSQL repositories."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from app.admin.models import BroadcastItem, BroadcastRecipient
from app.chat.domain import telegram_chat_id_from_owner


def snapshot_recipients(owners: Iterable[str]) -> tuple[list[BroadcastRecipient], int]:
    targets: set[int] = set()
    invalid = 0
    for owner in owners:
        try:
            targets.add(telegram_chat_id_from_owner(owner))
        except (AttributeError, TypeError, ValueError):
            invalid += 1
    return [BroadcastRecipient(chat_id=target) for target in sorted(targets)], invalid


def claim_item(
    item: BroadcastItem,
    owners: Iterable[str],
    *,
    now: datetime,
    lease_seconds: int,
) -> BroadcastItem | None:
    if item.status == "pending":
        recipients, invalid = snapshot_recipients(owners)
    elif item.status == "processing" and item.lease_expires_at is not None and item.lease_expires_at <= now:
        recipients = [
            recipient.model_copy(update={"status": "unknown"})
            if recipient.status == "in_flight" else recipient
            for recipient in item.recipients
        ]
        invalid = item.invalid_recipient_count
    else:
        return None
    return item.model_copy(update={
        "status": "processing",
        "recipients": recipients,
        "invalid_recipient_count": invalid,
        "lease_id": uuid4(),
        "lease_expires_at": now + timedelta(seconds=lease_seconds),
        "lease_seconds": lease_seconds,
        "review_required": item.review_required or invalid > 0 or any(
            recipient.status == "unknown" for recipient in recipients
        ),
        "updated_at": now,
    })


def update_recipient(
    item: BroadcastItem,
    *,
    lease_id: UUID,
    chat_id: int,
    attempt_id: UUID,
    status: str,
    now: datetime,
) -> BroadcastItem | None:
    if item.status != "processing" or item.lease_id != lease_id:
        return None
    if item.lease_expires_at is None or item.lease_expires_at <= now:
        return None
    recipients = list(item.recipients)
    for index, recipient in enumerate(recipients):
        if recipient.chat_id != chat_id:
            continue
        if recipient.status == status and recipient.attempt_id == attempt_id:
            return item.model_copy(update={
                "lease_expires_at": now + timedelta(seconds=item.lease_seconds),
                "updated_at": now,
            })
        if status == "in_flight":
            if recipient.status != "pending":
                return None
        elif status in {"sent", "failed", "unknown"}:
            if recipient.status != "in_flight" or recipient.attempt_id != attempt_id:
                return None
        else:
            return None
        recipients[index] = recipient.model_copy(update={"status": status, "attempt_id": attempt_id})
        return item.model_copy(update={
            "recipients": recipients,
            "lease_expires_at": now + timedelta(seconds=item.lease_seconds),
            "review_required": item.review_required or status in {"failed", "unknown"},
            "updated_at": now,
        })
    return None


def finish_item(item: BroadcastItem, *, lease_id: UUID, now: datetime) -> BroadcastItem | None:
    if item.status != "processing" or item.lease_id != lease_id:
        return None
    if item.lease_expires_at is None or item.lease_expires_at <= now:
        return None
    if any(recipient.status in {"pending", "in_flight"} for recipient in item.recipients):
        return None
    failed = item.invalid_recipient_count > 0 or any(
        recipient.status != "sent" for recipient in item.recipients
    )
    return item.model_copy(update={
        "status": "failed" if failed else "sent",
        "review_required": failed,
        "lease_id": None,
        "lease_expires_at": None,
        "updated_at": now,
    })
