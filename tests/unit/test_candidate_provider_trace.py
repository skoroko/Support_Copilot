"""Candidate raw responses survive incomplete status without leaking settings."""
import json

import pytest
from langchain_core.messages import AIMessage

from app.eval.acceptance_budget import install_acceptance_budget
from app.services import agent_persistent, rag, runtime_model


@pytest.mark.asyncio
async def test_generation_trace_is_immutable_reservation_bound_and_keeps_partial(tmp_path, monkeypatch):
    for module, name in [(agent_persistent, "AgentModel"), (runtime_model, "ResponsesTextLLM"),
                         (rag, "OpenAIEmbedding"), (rag, "LlamaOpenAI")]:
        monkeypatch.setattr(module, name, getattr(module, name))
    class Model:
        model, effort = "gpt-6-luna", "none"
        def output_budget(self, tokens, **kwargs):
            return tokens
        async def ainvoke(self, *args, **kwargs):
            return AIMessage(content="partial source-bound answer",
                response_metadata={"status": "incomplete", "finish_reason": "length",
                                   "headers": {"authorization": "never-store-this"},
                                   "unknown_provider_field": "never-store-this"},
                usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
    monkeypatch.setattr(runtime_model, "AgentModel", Model)
    monkeypatch.delenv("S0005_BUDGET_URL", raising=False)
    monkeypatch.setenv("S0005_PROVIDER_TRACE_DIR", str(tmp_path / "traces"))
    ledger = install_acceptance_budget(tmp_path / "budget.sqlite3", limit_usd=1.)
    result = await runtime_model.AgentModel().ainvoke([], max_tokens=100)
    paths = list((tmp_path / "traces").glob("*.json"))
    assert len(paths) == 1 and ledger.summary()["unknown_cost_calls"] == 0
    trace = json.loads(paths[0].read_text(encoding="utf-8"))
    assert paths[0].stem == trace["ledger_call_id"]
    assert trace["content"] == result.content and trace["usage"] == result.usage_metadata
    assert trace["response_metadata"]["status"] == "incomplete"
    assert "never-store-this" not in paths[0].read_text(encoding="utf-8")
    assert set(trace) == {"ledger_call_id", "model", "effort", "received_at", "content", "tool_calls",
                          "response_metadata", "usage"}
