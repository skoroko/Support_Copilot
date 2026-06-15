from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOMEWORK_DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "results" / "vision_demo_log.txt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from support_copilot.core import vision as _vision  # noqa: E402
from support_copilot.core.vision import *  # noqa: E402,F401,F403


def main() -> int:
    _vision.DEFAULT_LOG_PATH = HOMEWORK_DEFAULT_LOG_PATH
    return _vision.main()


if __name__ == "__main__":
    raise SystemExit(main())
