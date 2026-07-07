from __future__ import annotations

import os

import pytest

from app.core.config import get_settings


@pytest.fixture(autouse=True)
def test_openai_key() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

