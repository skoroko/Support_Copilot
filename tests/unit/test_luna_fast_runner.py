"""Offline checks of the experiment's only variable and frozen sample design."""
import asyncio
from collections import Counter
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.services.runtime_model import AgentModel
from scripts.compare_luna_fast import PROFILES, SCHEDULE


def test_balanced_design_and_same_inputs():
    assert len(SCHEDULE) == 12
    assert set(Counter((q, mode) for q, _, mode in SCHEDULE).values()) == {3}
    assert PROFILES["default"] == PROFILES["fast"]


def test_only_tier_differs_in_responses_payload():
    payloads = []
    for tier in ("default", "fast"):
        captured = {}

        class Client:
            async def ainvoke(self, messages, captured=captured, tier=tier, **kwargs):
                captured.update(kwargs)
                return AIMessage(content="ok", response_metadata={"service_tier": tier})

        settings = SimpleNamespace(**PROFILES[tier], agent_reasoning_token_reserve=8192)
        model = AgentModel(settings, [], client=Client())
        result = asyncio.run(model.ainvoke([HumanMessage(content="offline")], service_tier=tier))
        assert result.response_metadata["service_tier"] == tier
        client = ChatOpenAI.model_construct(model_name="gpt-6-luna",
                            use_responses_api=True, temperature=None, store=False)
        payload = client._get_request_payload([HumanMessage(content="offline")], **captured)
        assert payload.pop("service_tier") == tier
        assert payload["reasoning"] == {"effort": "medium"}
        assert payload["max_output_tokens"] == 8692
        assert payload["store"] is False
        payloads.append(payload)
    assert payloads[0] == payloads[1]
