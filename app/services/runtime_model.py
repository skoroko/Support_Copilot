"""Provider compatibility for the persistent agent and small RAG completions."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

from app.services.turn_budget import remaining_seconds

REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


def uses_responses(model: str, api: str = "auto") -> bool:
    return api == "responses" or (api == "auto" and model.startswith("gpt-6"))


def visible_text(content: Any) -> str:
    """Extract only user-facing text; never stringify reasoning/tool blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        str(block.get("text", ""))
        for block in content or []
        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
    )


def normalize_response(message: AIMessage) -> AIMessage:
    metadata = dict(message.response_metadata)
    if metadata.get("status") == "incomplete":
        detail = metadata.get("incomplete_details") or {}
        metadata["finish_reason"] = (
            "length"
            if detail.get("reason") == "max_output_tokens"
            else "content_filter"
        )
    elif metadata.get("status") == "completed":
        metadata["finish_reason"] = "tool_calls" if message.tool_calls else "stop"
    return message.model_copy(
        update={"content": visible_text(message.content), "response_metadata": metadata}
    )


class AgentModel:
    """Keep Responses details behind the graph's existing AIMessage contract.

    max_tokens remains the desired answer size. The provider's total generation
    cap additionally reserves space for reasoning, without promising a separate
    hard limit on reasoning tokens.
    """

    def __init__(self, settings: Any, tools: Sequence[Any], *, client: Any = None):
        self.settings = settings
        self._http_async_client: httpx.AsyncClient | None = None
        self.model = settings.default_model
        self.responses = uses_responses(
            self.model, getattr(settings, "agent_llm_api", "auto")
        )
        self.effort = getattr(settings, "agent_reasoning_effort", None)
        if self.effort is None:
            self.effort = "low" if self.model.startswith("gpt-6") else "none"
        self.expanded_effort = (
            getattr(settings, "agent_expanded_reasoning_effort", None) or self.effort
        )
        if (
            self.model.startswith("gpt-6")
            and not self.responses
            and (self.effort != "none" or self.expanded_effort != "none")
        ):
            raise ValueError("GPT-6 reasoning with tools requires Responses API")
        if client is None:
            # Retrieval often takes longer than the SDK's five-second idle
            # expiry. Reuse the TLS connection across the current tool loop.
            self._http_async_client = httpx.AsyncClient(limits=httpx.Limits(
                max_connections=1000, max_keepalive_connections=100,
                keepalive_expiry=60.0,
            ))
            options = {
                "model": self.model,
                "api_key": settings.openai_api_key.get_secret_value(),
                "base_url": settings.openai_base_url,
                # SDK retries only transient failures (429/5xx/timeouts) with
                # bounded exponential backoff; the final error still escalates.
                "max_retries": getattr(settings, "llm_max_retries", 2),
                "timeout": httpx.Timeout(
                    getattr(settings, "agent_generation_timeout", settings.request_timeout),
                    connect=10.0, pool=5.0,
                ),
                # Aggregate the provider stream privately. The existing graph,
                # grounding and moderation still publish only a finished answer.
                "streaming": True,
                "http_async_client": self._http_async_client,
            }
            if self.responses:
                options.update(
                    use_responses_api=True,
                    output_version="v0",
                    temperature=None,
                    store=False,
                    include=["reasoning.encrypted_content"],
                )
            else:
                options.update(temperature=0, store=False)
            client = ChatOpenAI(**options).bind_tools(tools)
        self.client = client

    async def aclose(self) -> None:
        if self._http_async_client is not None:
            await self._http_async_client.aclose()

    def output_budget(self, answer_tokens: int, *, expanded: bool = False) -> int:
        effort = self.expanded_effort if expanded else self.effort
        reserve = (
            getattr(self.settings, "agent_reasoning_token_reserve", 8192)
            if effort != "none"
            else 0
        )
        return answer_tokens + reserve

    async def ainvoke(
        self,
        messages: list,
        *,
        max_tokens: int = 500,
        expanded: bool = False,
        **kwargs: Any,
    ) -> AIMessage:
        effort = self.expanded_effort if expanded else self.effort
        kwargs["max_tokens"] = self.output_budget(max_tokens, expanded=expanded)
        if self.responses:
            kwargs["reasoning"] = {"effort": effort}
        elif self.model.startswith("gpt-6") or effort != "none":
            kwargs["reasoning_effort"] = effort
            kwargs["temperature"] = None
        timeout = remaining_seconds(getattr(self.settings, "agent_generation_timeout", 60.0))
        # Includes every SDK retry and backoff; retries cannot reset this clock.
        async with asyncio.timeout(timeout):
            response = await self.client.ainvoke(messages, **kwargs)
        return normalize_response(response)


class ResponsesTextLLM:
    """Small synchronous interface used by RAG, without a stale model registry."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None,
        timeout: float,
        effort: str = "none",
        max_output_tokens: int = 2048,
        system_prompt: str = "",
        max_retries: int = 2,
        client: Any = None,
    ):
        self.model, self.effort, self.timeout = model, effort, timeout
        self.max_output_tokens, self.system_prompt = max_output_tokens, system_prompt
        self.client = client or OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )

    def complete(self, prompt: str) -> Any:
        from types import SimpleNamespace

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            instructions=self.system_prompt or None,
            reasoning={"effort": self.effort},
            max_output_tokens=self.max_output_tokens,
            store=False,
            timeout=remaining_seconds(self.timeout),
        )
        if response.status != "completed":
            raise RuntimeError("RAG generation did not complete")
        return SimpleNamespace(text=response.output_text, raw=response)

    def close(self) -> None:
        self.client.close()

    def stream_complete(self, prompt: str):
        from types import SimpleNamespace

        with self.client.responses.stream(
            model=self.model,
            input=prompt,
            instructions=self.system_prompt or None,
            reasoning={"effort": self.effort},
            max_output_tokens=self.max_output_tokens,
            store=False,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield SimpleNamespace(delta=event.delta)
            if stream.get_final_response().status != "completed":
                raise RuntimeError("RAG generation did not complete")
