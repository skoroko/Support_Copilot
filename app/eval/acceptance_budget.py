"""Persistent, process-safe spend accounting for opt-in acceptance runs.

An unacknowledged/failed request keeps its entire reservation. No SDK retry is
hidden from this ledger. This is a conservative local cap, not provider billing.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

# Official Standard USD/1M rates fetched 2026-10-16, <=272K input tokens.
# https://developers.openai.com/api/docs/pricing
RATES = {
    "gpt-6-luna": (.10, .01, .125, .50),
    "gpt-6.1-sol": (2., .10, 2.50, 10.),
    "gpt-5.4-mini": (.75, .075, .75, 4.50),
    "text-embedding-3-small": (.02, .02, .02, 0.),
    "text-embedding-3-large": (.13, .13, .13, 0.),
}


class BudgetExceeded(RuntimeError):
    """No provider request may start after this exception."""


def micros(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError("Cost must be finite and nonnegative")
    return math.ceil(value * 1_000_000)


def usage_cost(model: str, usage: dict) -> float:
    incoming = usage.get("input_tokens", usage.get("prompt_tokens"))
    outgoing = usage.get("output_tokens", usage.get("completion_tokens", 0))
    if type(incoming) is not int or type(outgoing) is not int or min(incoming, outgoing) < 0:
        raise ValueError("Missing or invalid provider usage")
    rate, cache, write, output = RATES[model]
    details = usage.get("input_tokens_details") or usage.get("input_token_details") or usage.get("prompt_tokens_details") or {}
    read_count = details.get("cached_tokens", details.get("cache_read", 0)) or 0
    write_count = details.get("cache_write_tokens", details.get("cache_creation", 0)) or 0
    if any(type(x) is not int or x < 0 for x in (read_count, write_count)) or read_count + write_count > incoming:
        raise ValueError("Invalid cached token usage")
    if incoming > 272_000 and not model.startswith("text-embedding-"):
        if model == "gpt-5.4-mini":
            raise ValueError("Unsupported long-context rate")
        rate, cache, write, output = rate * 2, cache * 2, write * 2, output * 1.5
    return ((incoming - read_count - write_count) * rate + read_count * cache
            + write_count * write + outgoing * output) / 1_000_000


def upper_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cache writes may cost more than normal input; reserve the greater rate."""
    rate, _, write, output = RATES[model]
    if min(input_tokens, output_tokens) < 0:
        raise ValueError("Negative token bound")
    if input_tokens > 272_000 and not model.startswith("text-embedding-"):
        if model == "gpt-5.4-mini":
            raise ValueError("Unsupported long-context input")
        rate, write, output = rate * 2, write * 2, output * 1.5
    return (input_tokens * max(rate, write) + output_tokens * output) / 1_000_000


class BudgetLedger:
    def __init__(self, path: Path, *, limit_usd: float = 10.):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        limit = micros(limit_usd)
        if not limit:
            raise ValueError("Budget must be positive")
        with closing(self._connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS limits (id INTEGER PRIMARY KEY CHECK(id=1), micros INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, kind TEXT, model TEXT, fingerprint TEXT, "
                       "reserved INTEGER NOT NULL, charged INTEGER NOT NULL, status TEXT NOT NULL, usage TEXT, started REAL NOT NULL)")
            db.execute("INSERT OR IGNORE INTO limits VALUES (1, ?)", (limit,))
            if db.execute("SELECT micros FROM limits WHERE id=1").fetchone()[0] != limit:
                raise ValueError("Existing ledger budget cannot be changed implicitly")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def reserve(self, *, kind: str, model: str, amount: float, fingerprint: str = "") -> str:
        charge = micros(amount)
        if model not in RATES or kind not in {"embedding", "generation", "auxiliary", "review", "vision"}:
            raise ValueError("Unregistered provider operation")
        identity = uuid4().hex
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            total = db.execute("SELECT COALESCE(SUM(charged),0) FROM calls").fetchone()[0]
            limit = db.execute("SELECT micros FROM limits WHERE id=1").fetchone()[0]
            if total + charge > limit:
                raise BudgetExceeded("Acceptance API budget exhausted")
            db.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?,NULL,?)",
                       (identity, kind, model, fingerprint, charge, charge, "reserved", time.time()))
        return identity

    def settle(self, identity: str, *, usage: dict | None = None, error: bool = False) -> None:
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT model,reserved,status FROM calls WHERE id=?", (identity,)).fetchone()
            if not row or row[2] != "reserved":
                raise ValueError("Unknown or already settled reservation")
            if error or usage is None:
                # An exception does not establish that the provider charged zero.
                db.execute("UPDATE calls SET status='unknown_cost' WHERE id=?", (identity,))
                return
            actual = micros(usage_cost(row[0], usage))
            if actual > row[1]:
                # Preserve observed spend before surfacing a broken upper bound.
                db.execute("UPDATE calls SET charged=?,status='bound_exceeded',usage=? WHERE id=?",
                           (actual, json.dumps(usage), identity))
                db.commit()
                raise BudgetExceeded("Provider usage exceeded reserved upper bound")
            db.execute("UPDATE calls SET charged=?,status='completed',usage=? WHERE id=?",
                       (actual, json.dumps(usage), identity))

    def summary(self) -> dict:
        with closing(self._connect()) as db, db:
            limit = db.execute("SELECT micros FROM limits WHERE id=1").fetchone()[0]
            rows = db.execute("SELECT kind,status,charged FROM calls").fetchall()
        total = sum(r[2] for r in rows)
        return {"limit_usd": limit / 1e6, "accounted_upper_usd": total / 1e6,
                "remaining_usd": max(0, limit - total) / 1e6, "calls": len(rows),
                "unknown_cost_calls": sum(r[1] in {"reserved", "unknown_cost", "bound_exceeded"} for r in rows),
                "by_kind": {kind: sum(r[2] for r in rows if r[0] == kind) / 1e6 for kind in sorted({r[0] for r in rows})}}

    def reconcile_usage(self) -> dict:
        """Reprice saved provider usage after a usage-schema adapter correction.

        Unknown requests retain their full reservation. No provider is called.
        """
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT id,model,usage,reserved FROM calls WHERE status='completed'").fetchall()
            for identity, model, usage, reserved in rows:
                actual = micros(usage_cost(model, json.loads(usage)))
                db.execute("UPDATE calls SET charged=?,status=? WHERE id=?", (
                    actual, "completed" if actual <= reserved else "bound_exceeded", identity))
        return self.summary()


def open_budget(path: Path, *, limit_usd: float = 10.):
    """Docker must use one Windows ledger owner, never cross-OS SQLite locks."""
    if os.environ.get("S0005_BUDGET_URL"):
        from app.core.config import get_settings
        from app.eval.budget_rpc import RemoteBudgetLedger
        return RemoteBudgetLedger(path, url=os.environ["S0005_BUDGET_URL"],
            token=get_settings().internal_token.get_secret_value(), limit_usd=limit_usd)
    return BudgetLedger(path, limit_usd=limit_usd)


def content_bound(value) -> int:
    """A conservative BPE token bound for text-only JSON, including overhead."""
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")) + 2048


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def metered_create(create, ledger: BudgetLedger, *, kind: str, **kwargs):
    if kwargs.get("stream"):
        raise ValueError("Acceptance metering requires final provider usage")
    if kwargs.get("previous_response_id") or kwargs.get("conversation"):
        raise ValueError("Unbounded server-side history is forbidden")
    if any(tool.get("type") != "function" for tool in kwargs.get("tools", [])):
        raise ValueError("Hosted tool charges are outside this text/vision budget")
    model = kwargs["model"]
    maximum = kwargs.get("max_output_tokens", kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 0)))
    if kind != "embedding" and not maximum:
        raise ValueError("Unbounded generation is forbidden in acceptance runs")
    if kind != "embedding":
        if kwargs.get("service_tier") not in {None, "default"}:
            raise ValueError("Only Standard provider pricing is permitted")
        kwargs["service_tier"] = "default"
    # Include system instructions, schemas, tool descriptions and all request
    # text. Base64 length alone is not a valid bound for a low-entropy image.
    token_bound = content_bound(kwargs)
    if kind == "vision":
        # Reserve the documented maximum input for this model, irrespective
        # of image encoding/patch tokenization. Actual usage releases the rest.
        token_bound = max(token_bound, {"gpt-6-luna": 922_000, "gpt-5.4-mini": 272_000}[model])
    identity = ledger.reserve(kind=kind, model=model, amount=upper_cost(model, token_bound, maximum),
                              fingerprint=fingerprint(kwargs))
    received = False
    try:
        result = create(**kwargs)
        usage = result.usage.model_dump() if getattr(result, "usage", None) else None
        received = True
        ledger.settle(identity, usage=usage)
        return result
    finally:
        if not received:
            ledger.settle(identity, error=True)


def build_embedding(*, ledger: BudgetLedger, model: str, dimensions: int, api_key: str | None,
                    api_base: str | None = None, cache_path: Path | None = None):
    """Injectable BaseEmbedding with durable exact-text cache and no SDK retry."""
    import asyncio

    from llama_index.core.base.embeddings.base import BaseEmbedding
    from openai import OpenAI
    from pydantic import PrivateAttr

    from app.services.embeddings import SQLiteEmbeddingCache

    class MeteredEmbedding(BaseEmbedding):
        _provider = PrivateAttr()
        _cache = PrivateAttr()

        def __init__(self):
            super().__init__(model_name=model, embed_batch_size=64)
            self._provider = OpenAI(api_key=api_key, base_url=api_base, timeout=60, max_retries=0)
            self._cache = SQLiteEmbeddingCache(cache_path or ledger.path.with_name("embeddings.sqlite3"))

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            keys = [fingerprint([model, dimensions, text]) for text in texts]
            cached = self._cache.get_many(keys)
            if any(len(vector) != dimensions or not all(math.isfinite(v) for v in vector)
                   for vector in cached.values()):
                raise ValueError("Invalid cached embedding dimensions or values")
            misses = {key: text for key, text in zip(keys, texts, strict=True) if key not in cached}
            if misses:
                response = metered_create(self._provider.embeddings.create, ledger, kind="embedding",
                                          model=model, dimensions=dimensions, input=list(misses.values()))
                entries = sorted(response.data, key=lambda value: value.index)
                if len(entries) != len(misses) or [v.index for v in entries] != list(range(len(misses))):
                    raise ValueError("Embedding response lost input alignment")
                fresh = dict(zip(misses, (entry.embedding for entry in entries), strict=True))
                if any(len(vector) != dimensions or not all(math.isfinite(v) for v in vector) for vector in fresh.values()):
                    raise ValueError("Invalid embedding dimensions or values")
                self._cache.set_many(fresh)
                cached.update(fresh)
            return [cached[key] for key in keys]

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._get_text_embeddings([query])[0]

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._get_text_embeddings([text])[0]

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return await asyncio.to_thread(self._get_query_embedding, query)

        async def _aget_text_embedding(self, text: str) -> list[float]:
            return await asyncio.to_thread(self._get_text_embedding, text)

        async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(self._get_text_embeddings, texts)

    return MeteredEmbedding()


def install_acceptance_budget(path: Path, *, limit_usd: float = 10.) -> BudgetLedger:
    """Install process-local adapters before candidate app startup, never globally.

    The main application's entrypoint never imports/calls this function.
    Only the dedicated acceptance ASGI entrypoint opts into these adapters.
    """
    from app.services import agent_persistent, rag, runtime_model

    ledger = open_budget(path, limit_usd=limit_usd)
    original_model = runtime_model.AgentModel
    original_text = runtime_model.ResponsesTextLLM

    class MeteredAgentModel(original_model):
        async def ainvoke(self, messages, *, max_tokens=500, expanded=False, **kwargs):
            # Includes encrypted history/tool schema, which a text length bound
            # cannot cover reliably. The graph separately enforces this ceiling.
            maximum_input = {"gpt-6-luna": 922_000, "gpt-5.4-mini": 272_000}[self.model]
            identity = ledger.reserve(kind="generation", model=self.model,
                amount=upper_cost(self.model, maximum_input, self.output_budget(max_tokens, expanded=expanded)),
                fingerprint=fingerprint([self.model, self.effort, messages]))
            received = False
            try:
                kwargs["service_tier"] = "default"
                result = await super().ainvoke(messages, max_tokens=max_tokens, expanded=expanded, **kwargs)
                trace_dir = os.environ.get("S0005_PROVIDER_TRACE_DIR")
                if trace_dir:
                    directory = Path(trace_dir)
                    directory.mkdir(parents=True, exist_ok=True)
                    # One immutable file per reservation avoids shared append
                    # corruption across concurrent candidate processes. No SDK
                    # headers, settings or credentials are serialized.
                    trace = {"ledger_call_id": identity, "model": self.model, "effort": self.effort,
                             "received_at": time.time(), "content": result.content,
                             "tool_calls": result.tool_calls,
                             "response_metadata": {key: result.response_metadata[key] for key in (
                                 "id", "model_name", "model_provider", "finish_reason", "status",
                                 "service_tier", "created_at") if key in result.response_metadata
                                 and isinstance(result.response_metadata[key], (str, int, float, bool, type(None)))},
                             "usage": dict(result.usage_metadata or {})}
                    with (directory / f"{identity}.json").open("x", encoding="utf-8") as handle:
                        json.dump(trace, handle, ensure_ascii=False)
                received = True
                ledger.settle(identity, usage=dict(result.usage_metadata or {}))
                return result
            finally:
                if not received:
                    ledger.settle(identity, error=True)

    class MeteredResponses:
        def __init__(self, resource):
            self.resource = resource

        def create(self, **kwargs):
            return metered_create(self.resource.create, ledger, kind="auxiliary", **kwargs)

    class MeteredTextLLM(original_text):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.client.responses = MeteredResponses(self.client.responses)

        def stream_complete(self, prompt):
            raise ValueError("Candidate streaming without final usage metering is forbidden")

    def embedding_factory(**kwargs):
        return build_embedding(ledger=ledger, model=kwargs["model"], dimensions=kwargs["dimensions"],
                               api_key=kwargs.get("api_key"), api_base=kwargs.get("api_base"))

    # The auxiliary text adapter exposes the same .complete(). Use its configured
    # model through Responses with an explicit cap so no paid call escapes metering.
    def legacy_text_factory(**kwargs):
        return MeteredTextLLM(model=kwargs["model"], api_key=kwargs.get("api_key"),
            base_url=kwargs.get("api_base"), timeout=kwargs.get("timeout", 30),
            effort="none", max_output_tokens=512)

    runtime_model.AgentModel = agent_persistent.AgentModel = MeteredAgentModel
    runtime_model.ResponsesTextLLM = MeteredTextLLM
    rag.OpenAIEmbedding = embedding_factory
    rag.LlamaOpenAI = legacy_text_factory
    return ledger
