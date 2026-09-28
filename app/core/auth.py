from __future__ import annotations

import asyncio
import secrets
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from app.deps.providers import SettingsDep


@dataclass(frozen=True)
class Principal:
    user_id: str
    interface: str


class InMemoryRequestBudget:
    """Per-process abuse guard for authenticated internal callers.

    A deployment-wide gateway can enforce stricter distributed quotas.  This
    guard still prevents one authenticated principal from exhausting a single
    application worker.
    """

    def __init__(self, *, requests_per_minute: int, max_concurrency: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.max_concurrency = max_concurrency
        self._lock = asyncio.Lock()
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._active: dict[str, int] = defaultdict(int)

    async def acquire(self, principal: str) -> None:
        now = time.monotonic()
        async with self._lock:
            recent = self._requests[principal]
            while recent and recent[0] <= now - 60:
                recent.popleft()
            if len(recent) >= self.requests_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail={"code": "principal_rate_limited", "message": "Request quota exceeded."},
                )
            if self._active[principal] >= self.max_concurrency:
                raise HTTPException(
                    status_code=429,
                    detail={"code": "principal_concurrency_limited", "message": "Too many concurrent requests."},
                )
            recent.append(now)
            self._active[principal] += 1

    async def release(self, principal: str) -> None:
        async with self._lock:
            active = self._active.get(principal, 0)
            if active <= 1:
                self._active.pop(principal, None)
            else:
                self._active[principal] = active - 1


class RequestBudgetLease:
    """Keep one admitted request slot until its stream and workers all exit."""

    def __init__(self, budget: InMemoryRequestBudget, principal: str) -> None:
        self._budget = budget
        self._principal = principal
        self._holds = 0
        self.release_request = self.hold()

    def hold(self) -> Callable[[], Awaitable[None]]:
        self._holds += 1
        released = False

        async def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            self._holds -= 1
            if self._holds == 0:
                await self._budget.release(self._principal)

        return release

    def track_worker(self, task: asyncio.Task[None]) -> None:
        release = self.hold()

        def worker_finished(_task: asyncio.Task[None]) -> None:
            asyncio.create_task(release())

        task.add_done_callback(worker_finished)


def request_budget(request: Request, settings: SettingsDep) -> InMemoryRequestBudget:
    budget = getattr(request.app.state, "request_budget", None)
    if budget is None:
        budget = InMemoryRequestBudget(
            requests_per_minute=settings.api_rate_limit_per_minute,
            max_concurrency=settings.api_max_concurrency_per_principal,
        )
        request.app.state.request_budget = budget
    return budget


def authenticate_principal(
    settings: SettingsDep,
    *,
    x_internal_token: str | None,
    x_user_id: str | None,
    x_user_interface: str | None,
) -> Principal:
    expected = settings.internal_token.get_secret_value()
    if not x_internal_token or not secrets.compare_digest(x_internal_token, expected):
        raise HTTPException(
            status_code=401,
            detail={"code": "service_unauthorized", "message": "Valid service authentication is required."},
        )
    user_id = (x_user_id or "").strip()
    if not user_id or len(user_id) > 200 or any(ord(char) < 32 for char in user_id):
        raise HTTPException(
            status_code=401,
            detail={"code": "principal_required", "message": "A valid authenticated principal is required."},
        )
    interface = (x_user_interface or "service").strip().casefold()
    if interface not in {"telegram", "cli", "web", "service"}:
        raise HTTPException(
            status_code=401,
            detail={"code": "principal_interface_invalid", "message": "A valid principal interface is required."},
        )

    return Principal(user_id=user_id, interface=interface)


async def require_principal(
    request: Request,
    settings: SettingsDep,
    x_internal_token: Annotated[str | None, Header(alias="X-Internal-Token")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    x_user_interface: Annotated[str | None, Header(alias="X-User-Interface")] = None,
) -> AsyncIterator[Principal]:
    principal = authenticate_principal(
        settings,
        x_internal_token=x_internal_token,
        x_user_id=x_user_id,
        x_user_interface=x_user_interface,
    )
    budget = request_budget(request, settings)
    await budget.acquire(principal.user_id)
    lease = RequestBudgetLease(budget, principal.user_id)
    request.state.request_budget_lease = lease
    try:
        yield principal
    finally:
        await lease.release_request()


PrincipalDep = Annotated[Principal, Depends(require_principal)]
