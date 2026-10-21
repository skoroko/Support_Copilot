"""Benchmark-only bridge preserving the historical native ReAct loop contract."""
from types import SimpleNamespace
from typing import Any


class ResponsesChatBridge:
    def __init__(self, client: Any, *, effort: str, service_tier: str | None = None):
        self.client, self.effort, self.service_tier = client, effort, service_tier
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, model, messages, tools=None, **kwargs):
        inputs = []
        for message in messages:
            if hasattr(message, "responses_output"):
                inputs.extend(message.responses_output)
            elif isinstance(message, dict) and message["role"] == "tool":
                inputs.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]})
            else:
                inputs.append(message)
        options = {
            "model": model, "input": inputs, "reasoning": {"effort": self.effort},
            "include": ["reasoning.encrypted_content"], "store": False,
            "max_output_tokens": 8692,
        }
        if self.service_tier:
            options["service_tier"] = self.service_tier
        if tools:
            options["tools"] = [{"type": "function", **tool["function"]} for tool in tools]
            options.update(kwargs)
        response = self.client.responses.create(**options)
        if response.status != "completed":
            raise RuntimeError("Responses generation incomplete")
        calls = [SimpleNamespace(id=item.call_id, function=SimpleNamespace(name=item.name, arguments=item.arguments))
                 for item in response.output if item.type == "function_call"]
        message = SimpleNamespace(
            content=response.output_text, tool_calls=calls,
            responses_output=[item.model_dump(exclude_none=True) for item in response.output],
        )
        usage = response.usage
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=usage.input_tokens, completion_tokens=usage.output_tokens, total_tokens=usage.total_tokens),
        )


class ResponsesStructuredJudge:
    """RAGAS discrete_metric-compatible agenerate using the installed SDK."""
    def __init__(self, client: Any, *, model: str, effort: str = "medium", service_tier: str | None = None):
        self.client, self.model, self.effort, self.service_tier = client, model, effort, service_tier

    async def agenerate(self, prompt: str, response_model: type):
        options = {"model": self.model, "input": prompt, "text_format": response_model,
                   "reasoning": {"effort": self.effort}, "max_output_tokens": 8692, "store": False}
        if self.service_tier:
            options["service_tier"] = self.service_tier
        response = await self.client.responses.parse(**options)
        if response.status != "completed" or response.output_parsed is None:
            raise RuntimeError("Citation judge did not return a completed structured verdict")
        return response.output_parsed
