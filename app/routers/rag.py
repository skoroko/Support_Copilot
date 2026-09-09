from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.schemas.rag import RAGQuery, RAGResponse
from app.services.security.input_validator import validate_input


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGResponse, summary="Answer from the GTS NX knowledge base")
async def query_rag(payload: RAGQuery, request: Request) -> RAGResponse:
    validation = validate_input(payload.question)
    if not validation.ok:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "security_input_blocked",
                "message": "Request blocked by input security policy",
                "rule": validation.rule,
            },
        )
    service = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "rag_unavailable", "message": "RAG is disabled or the index is not ready"},
        )
    result = await run_in_threadpool(service.answer, payload.question)
    return RAGResponse.model_validate(result)
