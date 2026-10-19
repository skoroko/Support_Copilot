"""Authenticated local ledger owner for Windows/Docker acceptance clients.

SQLite files on Docker Desktop bind mounts must not have writers in two OSes.
Only the Windows owner writes the task ledger; no provider credentials or
document text cross this transport. A lost acknowledgement keeps the reserve.
"""
from __future__ import annotations

import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from app.eval.acceptance_budget import BudgetExceeded, BudgetLedger


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Ledger redirects are forbidden")


class RemoteBudgetLedger:
    def __init__(self, path: Path, *, url: str, token: str, limit_usd: float = 10.):
        if not token or not token.strip():
            raise ValueError("Nonempty authentication required")
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "host.docker.internal"}
                or parsed.port != 16334 or parsed.path not in {"", "/"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Only local acceptance ledger owner :16334 is permitted")
        self.path, self.url, self._token = Path(path), url.rstrip("/"), token
        self._opener = build_opener(ProxyHandler({}), NoRedirect())
        if self.summary()["limit_usd"] != limit_usd:
            raise ValueError("Remote task budget mismatch")

    def _request(self, action: str, payload: dict):
        request = Request(self.url + "/" + action,
            data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json", "X-Internal-Token": self._token})
        try:
            with self._opener.open(request, timeout=35) as response:
                result = json.load(response)
        except Exception:  # noqa: BLE001 - transport errors must never expose authorization headers
            raise RuntimeError("Acceptance ledger owner unavailable") from None
        if result.get("error") == "BudgetExceeded":
            raise BudgetExceeded("Acceptance API budget exhausted")
        if "error" in result:
            raise ValueError("Acceptance ledger request rejected")
        return result["result"]

    def reserve(self, **kwargs) -> str:
        return self._request("reserve", kwargs)

    def settle(self, identity: str, **kwargs) -> None:
        self._request("settle", {"identity": identity, **kwargs})

    def summary(self) -> dict:
        return self._request("summary", {})


def ledger_server(ledger: BudgetLedger, token: str, *, port: int = 16334) -> ThreadingHTTPServer:
    if not token or not token.strip():
        raise ValueError("Nonempty authentication required")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if not secrets.compare_digest(self.headers.get("X-Internal-Token", ""), token):
                self.send_error(401)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                action = self.path.removeprefix("/")
                if action not in {"reserve", "settle", "summary"} or not isinstance(payload, dict):
                    raise ValueError("Invalid action")
                value = getattr(ledger, action)(**payload)
                result = {"result": value}
            except Exception as exc:  # noqa: BLE001 - return a safe error class, never provider/config details
                result = {"error": "BudgetExceeded" if isinstance(exc, BudgetExceeded) else "invalid_request"}
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
