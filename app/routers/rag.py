from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.core.auth import PrincipalDep
from app.schemas.rag import RAGQuery, RAGResponse
from app.services.security.input_validator import validate_input


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGResponse, summary="Answer from the GTS NX knowledge base")
async def query_rag(payload: RAGQuery, request: Request, _principal: PrincipalDep) -> RAGResponse:
    for text in [
        payload.question,
        payload.standalone_question,
        *(message.content for message in payload.history or [] if message.role == "user"),
    ]:
        if text is None:
            continue
        validation = validate_input(text)
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
    kwargs = {}
    if payload.history:
        kwargs["history"] = payload.history
    if payload.standalone_question:
        kwargs["standalone_question"] = payload.standalone_question
    method = getattr(service, "query", None) or service.answer
    result = await run_in_threadpool(method, payload.question, **kwargs)
    return RAGResponse.model_validate(result)
