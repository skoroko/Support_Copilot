"""Report configuration presence without ever printing secret values.

Use this command instead of viewing or searching ``.env`` directly.  The
output contains a fixed allowlist of variable names and only the states
``configured`` or ``missing``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
CHECKED_VARIABLES = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "BOT_TOKEN",
    "BOT_ADMIN_IDS",
    "BOT_ALLOWED_USER_IDS",
    "BOT_ALLOWED_CHAT_IDS",
    "INTERNAL_TOKEN",
    "ADMIN_TOKEN",
)


def configuration_status(env_file: Path) -> dict[str, str]:
    """Return allowlisted presence flags; values never leave this function."""
    configured: set[str] = set()
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, raw_value = stripped.split("=", 1)
            name = name.strip()
            if name not in CHECKED_VARIABLES:
                continue
            value = raw_value.strip().strip("\"'").strip()
            if value:
                configured.add(name)
    return {
        name: "configured" if name in configured else "missing"
        for name in CHECKED_VARIABLES
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely report whether expected provider settings are configured.",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    args = parser.parse_args()
    payload = {
        "env_file_exists": args.env_file.is_file(),
        "variables": configuration_status(args.env_file),
        "note": "Values are intentionally redacted and never printed.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
