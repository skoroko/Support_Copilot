from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from support_copilot.cli import _parse_image_command, _resolve_image_path
from support_copilot.config import project_root


class ImageCommandParsingTests(unittest.TestCase):
    def test_parse_windows_path_with_spaces_without_quotes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="support cli ") as tmp_dir:
            image_path = Path(tmp_dir) / "chart_response_times.png"
            image_path.write_bytes(b"not empty")

            result = _parse_image_command(f"/image {image_path} почему так медленно?")

        self.assertIsNotNone(result)
        parsed_path, question = result
        self.assertEqual(parsed_path, str(image_path))
        self.assertEqual(question, "почему так медленно?")

    def test_parse_quoted_path(self) -> None:
        result = _parse_image_command('/image "C:\\tmp\\file.png" что видно?')

        self.assertEqual(result, ("C:\\tmp\\file.png", "что видно?"))

    def test_resolve_path_with_repeated_project_folder(self) -> None:
        root = project_root()
        test_image = root / "tmp_cli_path_test.png"
        test_image.write_bytes(b"not empty")
        try:
            path = _resolve_image_path(f"{root.name}\\{test_image.name}")

            self.assertTrue(path.endswith(test_image.name))
            self.assertTrue(Path(path).is_file())
        finally:
            test_image.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
