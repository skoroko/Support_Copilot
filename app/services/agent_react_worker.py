from __future__ import annotations

import sys
from multiprocessing.connection import Connection
from typing import Any


def run_tool(tool_name: str, tool_args: dict[str, Any], connection: Connection) -> None:
    """Minimal spawn entry point for one killable allowlisted tool call."""

    from app.services.agent_react_tools import DISPATCH

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        handler = DISPATCH[tool_name]
        connection.send(("ok", str(handler(**tool_args))))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
    connection.close()
