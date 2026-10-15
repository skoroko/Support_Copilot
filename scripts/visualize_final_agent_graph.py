"""Export the production agent topology without providers or persistent storage."""

from __future__ import annotations

import sys
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.agent_persistent import build_agent


def main() -> None:
    # An inert model prevents lazy provider construction; no graph invocation.
    graph = build_agent(InMemorySaver(), model=object())
    destination = PROJECT_ROOT / "docs" / "final-agent-graph.mmd"
    temporary = destination.with_suffix(".mmd.tmp")
    temporary.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    temporary.replace(destination)
    print(destination.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
