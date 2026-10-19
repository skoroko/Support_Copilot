"""No provider calls: authenticated owner, shared cap and lost-ack reserves."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Thread
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest
from app.eval.acceptance_budget import BudgetExceeded, BudgetLedger

from app.eval import budget_rpc


@contextmanager
def local_owner(tmp_path):
    ledger = BudgetLedger(tmp_path / "owner.sqlite3", limit_usd=10)
    server = budget_rpc.ledger_server(ledger, "fixture-owner-token", port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ledger, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def connect_remote(tmp_path, monkeypatch, address):
    """Keep production endpoint validation; route only fixture I/O to port 0."""
    opener = build_opener(ProxyHandler({}))
    attempts = []
    def transport(request, timeout):
        assert request.full_url.startswith("http://127.0.0.1:16334/")
        attempts.append(request.full_url)
        local_request = Request(address + request.full_url.removeprefix("http://127.0.0.1:16334"),
                                data=request.data, headers=dict(request.header_items()), method=request.method)
        return opener.open(local_request, timeout=timeout)
    monkeypatch.setattr(budget_rpc, "build_opener", lambda *args: SimpleNamespace(open=transport))
    remote = budget_rpc.RemoteBudgetLedger(tmp_path / "never-written.sqlite3",
                                           url="http://127.0.0.1:16334", token="fixture-owner-token")
    return remote, transport, attempts


def test_owner_requires_auth_and_rejects_unknown_actions(tmp_path):
    with local_owner(tmp_path) as (ledger, address):
        opener = build_opener(ProxyHandler({}))
        for token in (None, "wrong-token"):
            headers = {"X-Internal-Token": token} if token else {}
            request = Request(address + "/reserve", data=b"{}", headers=headers, method="POST")
            with pytest.raises(HTTPError) as exc:
                opener.open(request, timeout=5)
            assert exc.value.code == 401
        request = Request(address + "/__dict__", data=b"{}", method="POST",
                          headers={"X-Internal-Token": "fixture-owner-token"})
        with opener.open(request, timeout=5) as result:
            assert json.load(result) == {"error": "invalid_request"}
        assert ledger.summary()["calls"] == 0


def test_concurrent_remote_clients_share_one_hard_cap(tmp_path, monkeypatch):
    with local_owner(tmp_path) as (owner, address):
        remote, _, _ = connect_remote(tmp_path, monkeypatch, address)
        def reserve(_):
            try:
                return remote.reserve(kind="review", model="gpt-6-luna", amount=3)
            except BudgetExceeded:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            reservations = list(pool.map(reserve, range(8)))
        assert len({r for r in reservations if r}) == 3
        assert owner.summary()["accounted_upper_usd"] == 9
        assert remote.summary()["calls"] == 3
        assert not remote.path.exists()  # Client never writes shared SQLite.


def test_lost_reserve_ack_keeps_charge_and_never_retries(tmp_path, monkeypatch):
    with local_owner(tmp_path) as (owner, address):
        remote, transport, attempts = connect_remote(tmp_path, monkeypatch, address)
        def lose_ack(request, timeout):
            with transport(request, timeout) as result:
                result.read()  # Owner committed; acknowledgement lost in transport.
            raise OSError("fixture acknowledgement loss")
        monkeypatch.setattr(remote._opener, "open", lose_ack)
        before = len(attempts)
        with pytest.raises(RuntimeError, match="owner unavailable"):
            remote.reserve(kind="review", model="gpt-6-luna", amount=2)
        assert len(attempts) == before + 1
        assert owner.summary()["accounted_upper_usd"] == 2
        assert owner.summary()["unknown_cost_calls"] == 1


@pytest.mark.parametrize("url", ["https://127.0.0.1:16334", "http://example.invalid:16334",
                                  "http://127.0.0.1:6333", "http://127.0.0.1:16334/proxy",
                                  "http://user:pass@127.0.0.1:16334", "http://127.0.0.1:16334?q=x",
                                  "http://127.0.0.1:16334#target", "file:///tmp/owner"])
def test_remote_rejects_unapproved_endpoint_before_transport(tmp_path, monkeypatch, url):
    def forbidden(*args, **kwargs):
        pytest.fail("Endpoint rejected before transport")
    monkeypatch.setattr(budget_rpc, "build_opener", forbidden)
    with pytest.raises(ValueError, match="Only local"):
        budget_rpc.RemoteBudgetLedger(tmp_path / "none.sqlite3", url=url, token="fixture")


@pytest.mark.parametrize("token", ["", "   "])
def test_empty_owner_token_is_rejected(tmp_path, token):
    ledger = BudgetLedger(tmp_path / "owner.sqlite3", limit_usd=10)
    with pytest.raises(ValueError, match="authentication|token|Token"):
        budget_rpc.ledger_server(ledger, token, port=0)


def test_remote_disables_proxy_and_rejects_redirects(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://unapproved.invalid:9999")
    monkeypatch.setattr(budget_rpc.RemoteBudgetLedger, "summary", lambda self: {"limit_usd": 10})
    remote = budget_rpc.RemoteBudgetLedger(tmp_path / "none", url="http://127.0.0.1:16334", token="fixture")
    # urllib omits an empty ProxyHandler from its active handlers altogether.
    assert not any(isinstance(h, ProxyHandler) and h.proxies for h in remote._opener.handlers)
    handler = next(h for h in remote._opener.handlers if isinstance(h, budget_rpc.NoRedirect))
    with pytest.raises(ValueError, match="redirect"):
        handler.redirect_request(Request(remote.url, headers={"X-Internal-Token": "fixture"}), None,
                                 302, "redirect", {}, "https://unapproved.invalid/collect")
