from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from support_copilot.config import load_env_file


class ConfigTests(unittest.TestCase):
    def test_load_env_file_strips_bom_from_first_key(self) -> None:
        old_value = os.environ.pop("TEST_OPENAI_API_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                env_path = Path(tmp_dir) / ".env"
                env_path.write_text(
                    "\ufeffTEST_OPENAI_API_KEY=test-key\n",
                    encoding="utf-8",
                )

                load_env_file(env_path)

            self.assertEqual(os.environ.get("TEST_OPENAI_API_KEY"), "test-key")
        finally:
            if old_value is None:
                os.environ.pop("TEST_OPENAI_API_KEY", None)
            else:
                os.environ["TEST_OPENAI_API_KEY"] = old_value

    def test_load_env_file_overrides_empty_environment_value(self) -> None:
        old_value = os.environ.get("TEST_EMPTY_ENV")
        os.environ["TEST_EMPTY_ENV"] = ""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                env_path = Path(tmp_dir) / ".env"
                env_path.write_text("TEST_EMPTY_ENV=filled\n", encoding="utf-8")

                load_env_file(env_path)

            self.assertEqual(os.environ.get("TEST_EMPTY_ENV"), "filled")
        finally:
            if old_value is None:
                os.environ.pop("TEST_EMPTY_ENV", None)
            else:
                os.environ["TEST_EMPTY_ENV"] = old_value


if __name__ == "__main__":
    unittest.main()
