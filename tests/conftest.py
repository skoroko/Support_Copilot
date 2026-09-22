from __future__ import annotations

import os


# Strong, non-production values make fail-closed configuration deterministic
# before test modules import app.main. Individual configuration tests may
# still delete or replace them with monkeypatch.
os.environ["OPENAI_API_KEY"] = "sk-test-only"
os.environ["INTERNAL_TOKEN"] = "test-internal-token-0000000000000001"
os.environ["ADMIN_TOKEN"] = "test-admin-token-000000000000000002"
