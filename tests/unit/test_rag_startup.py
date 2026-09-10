from __future__ import annotations

from types import SimpleNamespace

from app import main


class FakeRAGService:
    instances: list["FakeRAGService"] = []
    build_error: Exception | None = None

    def __init__(self, config: object) -> None:
        self.config = config
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
