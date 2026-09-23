from __future__ import annotations

from types import SimpleNamespace

from app import main


class FakeRAGService:
    instances: list["FakeRAGService"] = []
    build_error: Exception | None = None

    def __init__(self, config: object, condense_hook: object = None) -> None:
        self.config = config
        self.condense_hook = condense_hook
        self.closed = False
        self.instances.append(self)

    def build(self) -> "FakeRAGService":
        if self.build_error is not None:
            raise self.build_error
        return self

    def close(self) -> None:
        self.closed = True


def test_initialize_rag_returns_built_service(monkeypatch) -> None:
    FakeRAGService.instances.clear()
    FakeRAGService.build_error = None
    monkeypatch.setattr(main.RAGConfig, "from_settings", lambda _settings: "config")
    monkeypatch.setattr(main, "RAGService", FakeRAGService)

    service, error = main._initialize_rag(SimpleNamespace())

    assert service is FakeRAGService.instances[0]
    assert error is None
    assert service.closed is False


def test_initialize_rag_degrades_and_closes_partial_service(monkeypatch) -> None:
    FakeRAGService.instances.clear()
    FakeRAGService.build_error = PermissionError("provider denied the request")
    monkeypatch.setattr(main.RAGConfig, "from_settings", lambda _settings: "config")
    monkeypatch.setattr(main, "RAGService", FakeRAGService)

    service, error = main._initialize_rag(SimpleNamespace())

    assert service is None
    assert error == "PermissionError"
    assert FakeRAGService.instances[0].closed is True


def test_initialize_rag_refuses_external_transfer_without_optin(monkeypatch) -> None:
    # Without the opt-in the built-in embedding adapter would raise RuntimeError
    # on every chat question; startup degrades once instead of per request.
    FakeRAGService.instances.clear()
    FakeRAGService.build_error = None
    monkeypatch.setattr(
        main.RAGConfig,
        "from_settings",
        lambda _settings: SimpleNamespace(allow_external_data_transfer=False),
    )
    monkeypatch.setattr(main, "RAGService", FakeRAGService)

    service, error = main._initialize_rag(SimpleNamespace())

    assert service is None
    assert error == "ExternalDataTransferNotAllowed"
    assert FakeRAGService.instances == []


def test_initialize_rag_builds_when_external_transfer_allowed(monkeypatch) -> None:
    FakeRAGService.instances.clear()
    FakeRAGService.build_error = None
    monkeypatch.setattr(
        main.RAGConfig,
        "from_settings",
        lambda _settings: SimpleNamespace(allow_external_data_transfer=True),
    )
    monkeypatch.setattr(main, "RAGService", FakeRAGService)

    service, error = main._initialize_rag(SimpleNamespace())

    assert service is FakeRAGService.instances[0]
    assert error is None


def test_initialize_rag_injects_llm_condense_hook(monkeypatch) -> None:
    FakeRAGService.instances.clear()
    FakeRAGService.build_error = None
    sentinel_hook = object()
    factory_calls: list[object] = []
    monkeypatch.setattr(
        main.RAGConfig,
        "from_settings",
        lambda _settings: SimpleNamespace(
            allow_external_data_transfer=True,
            condense_enabled=True,
            condense_llm_enabled=True,
        ),
    )
    monkeypatch.setattr(
        main,
        "build_llm_condense_hook",
        lambda config: factory_calls.append(config) or sentinel_hook,
    )
    monkeypatch.setattr(main, "RAGService", FakeRAGService)

    service, error = main._initialize_rag(SimpleNamespace())

    assert error is None
    assert service.condense_hook is sentinel_hook
    assert factory_calls == [service.config]


def test_initialize_rag_skips_condense_hook_when_llm_condense_disabled(monkeypatch) -> None:
    FakeRAGService.instances.clear()
    FakeRAGService.build_error = None
    monkeypatch.setattr(
        main.RAGConfig,
        "from_settings",
        lambda _settings: SimpleNamespace(
            allow_external_data_transfer=True,
            condense_enabled=True,
            condense_llm_enabled=False,
        ),
    )
    monkeypatch.setattr(
        main,
        "build_llm_condense_hook",
        lambda _config: (_ for _ in ()).throw(AssertionError("hook factory must not run")),
    )
    monkeypatch.setattr(main, "RAGService", FakeRAGService)

    service, error = main._initialize_rag(SimpleNamespace())

    assert error is None
    assert service.condense_hook is None


def test_initialize_rag_skips_condense_hook_when_condense_disabled(monkeypatch) -> None:
    FakeRAGService.instances.clear()
    FakeRAGService.build_error = None
    monkeypatch.setattr(
        main.RAGConfig,
        "from_settings",
        lambda _settings: SimpleNamespace(
            allow_external_data_transfer=True,
            condense_enabled=False,
            condense_llm_enabled=True,
        ),
    )
    monkeypatch.setattr(
        main,
        "build_llm_condense_hook",
        lambda _config: (_ for _ in ()).throw(AssertionError("hook factory must not run")),
    )
    monkeypatch.setattr(main, "RAGService", FakeRAGService)

    service, error = main._initialize_rag(SimpleNamespace())

    assert error is None
    assert service.condense_hook is None
