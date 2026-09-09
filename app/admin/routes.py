from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from app.admin.auth import require_admin
from app.admin.deps import AdminServiceDep
from app.admin.models import BroadcastIn, BroadcastItem, BroadcastStatusIn, StatsOut, UserOut


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


@router.post("/broadcasts/{broadcast_id}/status", response_model=BroadcastItem)
async def update_broadcast_status(
    broadcast_id: UUID,
    payload: BroadcastStatusIn,
    service: AdminServiceDep,
) -> BroadcastItem:
    item = await service.update_broadcast_status(broadcast_id, payload.status)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "broadcast_not_found", "message": f"Broadcast {broadcast_id} not found"},
        )
    return item
