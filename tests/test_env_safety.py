from __future__ import annotations

import json

from scripts.check_env_safety import CHECKED_VARIABLES, configuration_status


def test_configuration_status_never_returns_secret_values(tmp_path):
    env_file = tmp_path / ".env"
    secret = "sk-test-super-secret-value"
    env_file.write_text(
        "\n".join(
            [
                f"OPENAI_API_KEY={secret}",
                "OPENAI_BASE_URL=https://api.openai.com/v1",
                "OPENAI_MODEL=\"gpt-5.4-mini\"",
                "BOT_ADMIN_IDS=123456789",
                "BOT_ALLOWED_USER_IDS=123456789,987654321",
                "BOT_ALLOWED_CHAT_IDS=   ",
                "OPENROUTER_API_KEY=   ",
                "UNRELATED_PRIVATE_VALUE=must-not-be-listed",
            ]
        ),
        encoding="utf-8",
    )

    status = configuration_status(env_file)
    serialized = json.dumps(status)

    assert tuple(status) == CHECKED_VARIABLES
    assert status["OPENAI_API_KEY"] == "configured"
    assert status["OPENROUTER_API_KEY"] == "missing"
    assert status["BOT_ADMIN_IDS"] == "configured"
    assert status["BOT_ALLOWED_USER_IDS"] == "configured"
    assert status["BOT_ALLOWED_CHAT_IDS"] == "missing"
    assert "123456789" not in serialized
    assert "987654321" not in serialized
    assert secret not in serialized
    assert "must-not-be-listed" not in serialized
    assert "UNRELATED_PRIVATE_VALUE" not in serialized


def test_missing_env_file_reports_every_variable_missing(tmp_path):
    status = configuration_status(tmp_path / "missing.env")

    assert set(status.values()) == {"missing"}
