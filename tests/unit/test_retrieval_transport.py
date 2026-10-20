"""Offline checks for retrieval transport reuse, deadlines and ownership."""
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from pydantic import SecretStr

from app.services.rag import RAGConfig, RAGService
from app.services.retrieval_embedding import OpenAIEmbedding
from app.services.retrieval_translation import build_llm_translate_hook
from app.services.runtime_model import ResponsesTextLLM
from app.services.turn_budget import turn_budget


def config(**overrides):
    values = {"openai_api_key": SecretStr("offline-test"),
        "rag_allow_external_data_transfer": True, "rag_llm_model": "gpt-6-luna"}
    return RAGConfig.from_settings(SimpleNamespace(**(values | overrides)))


def test_embedding_reuses_owned_pool_and_cached_sdk_client():
    embedding = OpenAIEmbedding(api_key="offline-test", timeout=30, max_retries=0)
    try:
        first, second = embedding._get_client(), embedding._get_client()
        assert first._client is second._client is embedding._client._client
        assert first._client._transport._pool._keepalive_expiry == 60
        assert first.timeout.read == 30
        assert first.timeout.connect == 10
        assert first.max_retries == 0
    finally:
        embedding.close()
    assert first._client.is_closed


def test_embedding_does_not_close_injected_transport():
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as http:
        embedding = OpenAIEmbedding(api_key="offline-test", http_client=http)
        assert embedding._get_client()._client is http
        embedding.close()
        assert not http.is_closed


def test_embedding_keeps_tcp_connection_past_sdk_default_idle_expiry(monkeypatch):
    """Exercise a real local pool, including idle expiry; no external provider."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")

    class Server(ThreadingHTTPServer):
        accepted = 0

        def get_request(self):
            connection = super().get_request()
            self.accepted += 1
            return connection

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"data": [{"embedding": [1., 2., 3.], "index": 0}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = Server(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    managed = OpenAIEmbedding(api_key="offline-test", api_base=base_url, max_retries=0)
    try:
        with httpx.Client(limits=httpx.Limits(keepalive_expiry=5)) as http:
            baseline = OpenAIEmbedding(api_key="offline-test", api_base=base_url,
                max_retries=0, http_client=http)
            assert managed.get_query_embedding("first managed") == [1., 2., 3.]
            assert baseline.get_query_embedding("first baseline") == [1., 2., 3.]
            assert server.accepted == 2
            time.sleep(6)
            assert managed.get_query_embedding("second managed") == [1., 2., 3.]
            assert server.accepted == 2
            assert baseline.get_query_embedding("second baseline") == [1., 2., 3.]
            assert server.accepted == 3
    finally:
        managed.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.asyncio
async def test_embedding_query_keeps_per_request_deadline_without_mutation():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"embedding": [1., 2., 3.], "index": 0}],
            "model": "text-embedding-3-small", "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        embedding = OpenAIEmbedding(api_key="offline-test", timeout=30,
            max_retries=0, http_client=http)
        async with turn_budget(.5):
            assert await asyncio.to_thread(embedding.get_query_embedding, "public question") == [1., 2., 3.]
        assert 0 < requests[0].extensions["timeout"]["connect"] <= .5
        assert 0 < requests[0].extensions["timeout"]["read"] <= .5
        assert embedding.timeout == 30
        assert embedding._client.timeout == 30


def test_responses_owned_pool_and_injected_sdk_lifecycles():
    llm = ResponsesTextLLM(model="gpt-6-luna", api_key="offline-test", base_url=None,
        timeout=30, max_retries=0)
    assert llm.client._client._transport._pool._keepalive_expiry == 60
    assert llm.client.timeout.connect == 10
    assert llm.client.max_retries == 0
    llm.close()
    llm.close()
    assert llm.client.is_closed()
    client = SimpleNamespace(close=Mock())
    injected = ResponsesTextLLM(model="gpt-6-luna", api_key=None, base_url=None,
        timeout=30, client=client)
    injected.close()
    client.close.assert_not_called()


@pytest.mark.asyncio
async def test_responses_complete_and_stream_use_residual_connect_timeout():
    from unittest.mock import MagicMock

    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.__iter__.return_value = iter([])
    stream.get_final_response.return_value = SimpleNamespace(status="completed")
    responses = SimpleNamespace(create=Mock(return_value=SimpleNamespace(
        status="completed", output_text="public answer")), stream=Mock(return_value=stream))
    llm = ResponsesTextLLM(model="gpt-6-luna", api_key=None, base_url=None,
        timeout=30, client=SimpleNamespace(responses=responses))
    async with turn_budget(.5):
        llm.complete("public prompt")
        list(llm.stream_complete("public prompt"))
    for call in (responses.create.call_args, responses.stream.call_args):
        timeout = call.kwargs["timeout"]
        assert 0 < timeout.connect <= timeout.read <= .5
        assert call.kwargs["reasoning"] == {"effort": "none"}


def test_translation_initialization_is_once_under_concurrent_queries(monkeypatch):
    started, release = Event(), Event()
    adapter = SimpleNamespace(complete=Mock(return_value=SimpleNamespace(text="English question")),
        close=Mock())

    def create(**kwargs):
        started.set()
        assert release.wait(3)
        return adapter

    factory = Mock(side_effect=create)
    monkeypatch.setattr("app.services.runtime_model.ResponsesTextLLM", factory)
    hook = build_llm_translate_hook(config())
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(hook, "Первый вопрос")
        assert started.wait(3)
        second = executor.submit(hook, "Второй вопрос")
        release.set()
        assert first.result() == second.result() == "English question"
    factory.assert_called_once()
    assert hook("Первый вопрос") == "English question"
    assert adapter.complete.call_count == 2  # Existing cache still works.
    hook.close()
    adapter.close.assert_called_once()


def test_translation_injected_adapter_retains_ownership():
    adapter = SimpleNamespace(complete=Mock(return_value="English question"), close=Mock())
    hook = build_llm_translate_hook(config(), llm=adapter)
    assert hook("Вопрос") == "English question"
    hook.close()
    adapter.close.assert_not_called()


def test_legacy_translation_pool_is_lazy_reused_and_closed(monkeypatch):
    adapter = SimpleNamespace(_get_client=Mock(),
        complete=Mock(return_value="English question"))
    factory = Mock(return_value=adapter)
    monkeypatch.setattr("app.services.retrieval_translation.LlamaOpenAI", factory)
    legacy_config = config(rag_llm_model="gpt-5.4-mini")
    hook = build_llm_translate_hook(legacy_config)
    factory.assert_not_called()
    assert hook("Первый вопрос") == hook("Второй вопрос") == "English question"
    factory.assert_called_once()
    http = factory.call_args.kwargs["http_client"]
    assert http._transport._pool._keepalive_expiry == 60
    assert not http.is_closed
    assert factory.call_args.kwargs["max_retries"] == 0
    assert adapter.complete.call_args.kwargs["timeout"].connect == 8
    hook.close()
    assert http.is_closed


def test_rag_serializes_embedding_initialization_and_closes_owned_resources(monkeypatch):
    service = RAGService(config())
    original = OpenAIEmbedding
    started, release = Event(), Event()
    initialized = []
    original_init = original.__init__

    def initialize(self, **kwargs):
        initialized.append(self)
        started.set()
        assert release.wait(3)
        original_init(self, **kwargs)

    monkeypatch.setattr(original, "__init__", initialize)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service._embedding_model)
        assert started.wait(3)
        second = executor.submit(service._embedding_model)
        release.set()
        assert first.result() is second.result()
    assert len(initialized) == 1
    http = service._embed_model._get_client()._client
    service.close()
    assert http.is_closed


def test_rag_injected_resources_close_only_with_explicit_hook_transfer():
    embed, llm, hook = Mock(), Mock(), Mock()
    service = RAGService(config(), embed_model=embed, llm=llm, translate_hook=hook)
    service.close()
    embed.close.assert_not_called()
    llm.close.assert_not_called()
    hook.close.assert_not_called()
    service = RAGService(config(), translate_hook=hook, own_translate_hook=True)
    service.close()
    hook.close.assert_called_once()
