from __future__ import annotations

from bot.background import broadcast_chat_ids


def test_broadcast_chat_ids_dedupes_group_members() -> None:
    users = [
        {"owner_external_id": "42"},
        {"owner_external_id": "-100:7"},
        {"owner_external_id": "-100:8"},
        {"owner_external_id": "-200:9"},
    ]

    targets, skipped = broadcast_chat_ids(users)

    assert targets == [42, -100, -200]
    assert skipped == 0


def test_broadcast_chat_ids_skips_unparseable_owners() -> None:
    users = [{"owner_external_id": "not-a-number"}, {}, {"owner_external_id": None}]

    targets, skipped = broadcast_chat_ids(users)

    assert targets == []
    assert skipped == 3
