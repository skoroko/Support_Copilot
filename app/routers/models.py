from __future__ import annotations

from fastapi import APIRouter

from app.schemas.models import OPENAI_MODELS, ModelInfo


router = APIRouter(tags=["models"])


RESPONSES = {
    200: {"description": "Static model price list"},
    422: {"description": "Validation error"},
    429: {"description": "LLM provider rate limit"},
    502: {"description": "LLM provider/auth error"},
    504: {"description": "LLM provider timeout"},
}


@router.get(
    "/models",
    response_model=list[ModelInfo],
    summary="List supported OpenAI models and token prices",
    responses=RESPONSES,
)
async def list_models() -> list[ModelInfo]:
    return OPENAI_MODELS
