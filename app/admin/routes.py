from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from app.admin.auth import require_admin
from app.admin.deps import AdminServiceDep
from app.admin.models import (
    BroadcastClaimIn, BroadcastFinishIn, BroadcastIn, BroadcastItem,
    BroadcastRecipientStateIn, StatsOut, UserOut,
)


router = APIRouter(
    prefix="/chats/admin",
    tags=["chat-admin"],
    dependencies=[Depends(require_admin)],
)


@router.get("/stats", response_model=StatsOut)
async def stats(service: AdminServiceDep) -> StatsOut:
    return await service.stats()


@router.get("/users", response_model=list[UserOut])
async def users(
    service: AdminServiceDep,
    limit: int = Query(default=50, ge=1, le=1000),
    interface: str | None = Query(default=None),
) -> list[UserOut]:
    return await service.users(limit=limit, interface=interface)


@router.post("/broadcast", response_model=BroadcastItem)
async def broadcast(payload: BroadcastIn, service: AdminServiceDep) -> BroadcastItem:
    return await service.enqueue_broadcast(payload.message, payload.interface_filter)


@router.get("/broadcasts/pending", response_model=list[BroadcastItem])
async def pending_broadcasts(
    service: AdminServiceDep,
    limit: int = Query(default=10, ge=1, le=100),
) -> list[BroadcastItem]:
    return await service.pending_broadcasts(limit=limit)


@router.get("/broadcasts/review", response_model=list[BroadcastItem])
async def review_broadcasts(
    service: AdminServiceDep,
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[BroadcastItem]:
    return await service.review_broadcasts(limit=limit, offset=offset)


@router.get("/broadcasts/{broadcast_id}", response_model=BroadcastItem)
async def get_broadcast(broadcast_id: UUID, service: AdminServiceDep) -> BroadcastItem:
    item = await service.get_broadcast(broadcast_id)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "broadcast_not_found", "message": f"Broadcast {broadcast_id} not found"},
        )
    return item


@router.post("/broadcasts/claim", response_model=list[BroadcastItem])
async def claim_broadcasts(payload: BroadcastClaimIn, service: AdminServiceDep) -> list[BroadcastItem]:
    return await service.claim_broadcasts(limit=payload.limit, lease_seconds=payload.lease_seconds)


@router.post("/broadcasts/{broadcast_id}/recipients/{chat_id}/state", response_model=BroadcastItem)
async def update_broadcast_recipient(
    broadcast_id: UUID,
    chat_id: int,
    payload: BroadcastRecipientStateIn,
    service: AdminServiceDep,
) -> BroadcastItem:
    item = await service.update_broadcast_recipient(
        broadcast_id, payload.lease_id, chat_id, payload.attempt_id, payload.status,
    )
    if item is None:
        raise HTTPException(
            status_code=409,
            detail={"code": "broadcast_transition_rejected", "message": "Lease or recipient state is stale"},
        )
    return item


@router.post("/broadcasts/{broadcast_id}/finish", response_model=BroadcastItem)
async def finish_broadcast(
    broadcast_id: UUID,
    payload: BroadcastFinishIn,
    service: AdminServiceDep,
) -> BroadcastItem:
    item = await service.finish_broadcast(broadcast_id, payload.lease_id)
    if item is None:
        raise HTTPException(
            status_code=409,
            detail={"code": "broadcast_finish_rejected", "message": "Lease or recipient progress is stale"},
        )
    return item
