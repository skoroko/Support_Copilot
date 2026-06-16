from __future__ import annotations

from fastapi import APIRouter


router = APIRouter(tags=["health"])


RESPONSES = {
    200: {"description": "Service process is alive"},
    422: {"description": "Validation error"},
    429: {"description": "LLM provider rate limit"},
    502: {"description": "LLM provider/auth error"},
    504: {"description": "LLM provider timeout"},
}


@router.get(
    "/health",
    summary="Health check without external dependencies",
    responses=RESPONSES,
)
async def health() -> dict[str, str]:
    return {"status": "ok"}
