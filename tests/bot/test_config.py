from __future__ import annotations

import pytest

from bot.config import BotSettings


@pytest.fixture
def bot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_TOKEN", "test-token")
    monkeypatch.setenv("INTERNAL_TOKEN", "test-internal-token-0000000000000001")
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token-000000000000000002")
    for name in ("BOT_ADMIN_IDS", "BOT_ALLOWED_USER_IDS", "BOT_ALLOWED_CHAT_IDS"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("field_name", ["BOT_ADMIN_IDS", "BOT_ALLOWED_USER_IDS", "BOT_ALLOWED_CHAT_IDS"])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", []),
        ("   ", []),
        ("123456789", [123456789]),
        ("123456789, 987654321", [123456789, 987654321]),
        ("[123456789, 987654321]", [123456789, 987654321]),
    ],
)
def test_telegram_id_lists_from_environment(
    bot_env: None,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    raw: str,
    expected: list[int],
) -> None:
    monkeypatch.setenv(field_name, raw)
    settings = BotSettings(_env_file=None)
    assert getattr(settings, field_name.lower()) == expected


def test_privileged_tokens_must_be_strong_and_distinct() -> None:
    with pytest.raises(ValueError):
        BotSettings(
            _env_file=None,
            bot_token="test-token",
            internal_token="secret",
            admin_token="secret",
        )


def test_allowed_ids_are_separate_from_admin_ids(bot_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_ADMIN_IDS", "10")
    monkeypatch.setenv("BOT_ALLOWED_USER_IDS", "11, 12")
    monkeypatch.setenv("BOT_ALLOWED_CHAT_IDS", "-100, -200")
    settings = BotSettings(_env_file=None)
    assert settings.bot_admin_ids == [10]
    assert settings.bot_allowed_user_ids == [11, 12]
    assert settings.bot_allowed_chat_ids == [-100, -200]
