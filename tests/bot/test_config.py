from __future__ import annotations

import pytest

from bot.config import BotSettings


def settings_with_admin_ids(value: object) -> BotSettings:
    return BotSettings(
        _env_file=None,
        bot_token="test-token",
        internal_token="test-internal-token-0000000000000001",
        admin_token="test-admin-token-000000000000000002",
        bot_admin_ids=value,
    )


def test_single_numeric_admin_id_is_accepted_after_env_json_decode() -> None:
    settings = settings_with_admin_ids(123456789)

    assert settings.bot_admin_ids == [123456789]


def test_comma_separated_admin_ids_are_accepted() -> None:
    settings = settings_with_admin_ids("123456789, 987654321")

    assert settings.bot_admin_ids == [123456789, 987654321]


def test_privileged_tokens_must_be_strong_and_distinct() -> None:
    with pytest.raises(ValueError):
        BotSettings(
            _env_file=None,
            bot_token="test-token",
            internal_token="secret",
            admin_token="secret",
        )
