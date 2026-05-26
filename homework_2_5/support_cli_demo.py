"""Homework 2.5 demo runner for the interactive Support Copilot CLI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from support_copilot.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--demo",
                "--log-file",
                "homework_2_5/cli_session_log.txt",
            ]
        )
    )
