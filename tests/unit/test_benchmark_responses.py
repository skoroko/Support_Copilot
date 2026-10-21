from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from scripts import bench_agents
from scripts.benchmark_responses import ResponsesChatBridge, ResponsesStructuredJudge


def test_snapshot_keeps_visible_text_and_marks_incomplete_provider_response():
    snapshot = bench_agents._state_snapshot({"messages": [AIMessage(
        content=[{"type": "text", "text": "partial"}, {"type": "reasoning", "text": "private"}],
        response_metadata={"status": "incomplete"},
    )]})
    assert snapshot["answer"] == "partial"
    assert snapshot["error"] == "provider generation incomplete"


def test_native_bridge_preserves_reasoning_tool_calls_usage_and_followup():
    calls = []

    class Item:
        type = "function_call"
        call_id, name, arguments = "c1", "get_current_time", '{}'

        def model_dump(self, **_kwargs):
            return {"type": self.type, "call_id": self.call_id, "name": self.name, "arguments": self.arguments}

    output = [Item()]
    response = SimpleNamespace(status="completed", output=output, output_text="",
                               usage=SimpleNamespace(input_tokens=10, output_tokens=3, total_tokens=13))
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs) or response))
    bridge = ResponsesChatBridge(client, effort="medium", service_tier="priority")
    first = bridge.create(model="existing", messages=[{"role": "user", "content": "time"}],
                          tools=[{"type": "function", "function": {"name": "get_current_time", "parameters": {}}}],
                          parallel_tool_calls=False)
    bridge.create(model="existing", messages=[first.choices[0].message, {"role": "tool", "tool_call_id": "c1", "content": "now"}])
    assert first.usage.total_tokens == 13
    assert first.choices[0].message.tool_calls[0].function.name == "get_current_time"
    assert calls[1]["input"][0] == output[0].model_dump()
    assert calls[1]["input"][1] == {"type": "function_call_output", "call_id": "c1", "output": "now"}
    assert calls[0]["include"] == ["reasoning.encrypted_content"]
    assert calls[0]["store"] is False
    assert calls[0]["reasoning"] == {"effort": "medium"}


@pytest.mark.asyncio
async def test_judge_uses_responses_parse_and_rejects_incomplete_output():
    calls = []

    async def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="incomplete", output_parsed=None)

    judge = ResponsesStructuredJudge(SimpleNamespace(responses=SimpleNamespace(parse=parse)), model="existing")
    with pytest.raises(RuntimeError, match="completed structured"):
        await judge.agenerate("citation", response_model=dict)
    assert calls[0]["text_format"] is dict
    assert calls[0]["model"] == "existing"


@pytest.mark.asyncio
async def test_benchmark_retains_all_45_records_and_saves_after_each(monkeypatch):
    calls, saved = [], []

    def record(case, repeat, implementation, timeout):
        calls.append((case["id"], repeat, implementation, timeout))
        return {"error": "failure" if len(calls) == 2 else None, "answer": "raw"}

    monkeypatch.setattr(bench_agents, "_isolated_record", record)
    rows = await bench_agents.run_benchmark(on_row=lambda rows: saved.append(len(rows)))
    assert len(rows) == len(calls) == 45
    assert saved == list(range(1, 46))
    assert rows[1]["error"] == "failure"
    assert rows[-1]["answer"] == "raw"


def test_isolation_preserves_last_checkpoint_on_timeout_and_terminates_worker(monkeypatch):
    class Receive:
        def __init__(self):
            self.polls = 0

        def poll(self, _timeout):
            self.polls += 1
            return self.polls == 1

        def recv(self):
            return "checkpoint", {"answer": "partial", "total_tokens": 12}

        def close(self):
            pass

    class Process:
        terminated = False

        def start(self):
            pass

        def is_alive(self):
            return True

        def terminate(self):
            self.terminated = True

        def join(self):
            pass

    process = Process()
    context = SimpleNamespace(Pipe=lambda **_: (Receive(), SimpleNamespace(close=lambda: None)),
                              Process=lambda **_: process)
    monkeypatch.setattr(bench_agents.multiprocessing, "get_context", lambda _: context)
    row = bench_agents._isolated_record({"id": "case", "task": "test"}, 1, "custom_langgraph", 10)
    assert row["answer"] == "partial"
    assert row["error"] == "execution timeout"
    assert row["status"] == "failed"
    assert row["total_tokens"] == 12
    assert row["usage_complete"] is False
    assert len(row["checkpoints"]) == 1
    assert process.terminated
