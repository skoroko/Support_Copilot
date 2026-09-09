from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.moderation.service import ModerationService


def write_rules(tmp_path):
    path = tmp_path / "moderation.yaml"
    path.write_text(
        'categories:\n  self-harm:\n    - "\\\\bkill yourself\\\\b"\n',
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_keyword_layer_blocks_before_optional_api(tmp_path) -> None:
    calls = 0

    class Moderations:
        async def create(self, **_kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("OpenAI layer must not run after keyword block")

    service = ModerationService(
        keywords_path=write_rules(tmp_path),
        client=SimpleNamespace(moderations=Moderations()),
        openai_enabled=True,
    )

    result = await service.check_input("Please kill yourself")

    assert result.allowed is False
    assert result.categories == ["self-harm"]
    assert result.blocked_by == "keyword"
    assert calls == 0


@pytest.mark.asyncio
async def test_openai_threshold_can_block_unflagged_result(tmp_path) -> None:
    result = SimpleNamespace(
        flagged=False,
        categories={},
        category_scores={"violence": 0.81},
    )

    class Moderations:
        async def create(self, **kwargs):
            assert kwargs == {"model": "omni-moderation-latest", "input": "borderline"}
            return SimpleNamespace(results=[result])

    service = ModerationService(
        keywords_path=write_rules(tmp_path),
        client=SimpleNamespace(moderations=Moderations()),
        openai_enabled=True,
        thresholds={"violence": 0.8},
    )

    checked = await service.check_output("borderline")

    assert checked.allowed is False
    assert checked.categories == ["violence"]
    assert checked.blocked_by == "openai"


@pytest.mark.asyncio
async def test_optional_api_failure_is_fail_open(tmp_path) -> None:
    class Moderations:
        async def create(self, **_kwargs):
            raise TimeoutError

    service = ModerationService(
        keywords_path=write_rules(tmp_path),
        client=SimpleNamespace(moderations=Moderations()),
        openai_enabled=True,
    )

    result = await service.check_input("normal support question")

    assert result.allowed is True
    assert result.reasons == ["optional moderation API unavailable"]
