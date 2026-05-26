from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from support_copilot.multimodal.vision import (
    ConfigurationError,
    VisionConfig,
    _build_chat_payload,
    analyze_image,
    detect_image_size,
    encode_image_as_data_url,
)


TINY_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01"
    b"\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00"
    b"\x90wS\xde"
    b"\x00\x00\x00\x0cIDAT"
    b"\x08\xd7c\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00"
    b"\x18\xdd\x8d\xb0"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class VisionTests(unittest.TestCase):
    def test_detect_png_size(self) -> None:
        self.assertEqual(detect_image_size(TINY_PNG), (1, 1))

    def test_encode_image_as_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "tiny.png"
            image_path.write_bytes(TINY_PNG)

            data_url, mime_type, size, width, height = encode_image_as_data_url(image_path)

        self.assertTrue(data_url.startswith("data:image/png;base64,"))
        self.assertEqual(mime_type, "image/png")
        self.assertEqual(size, len(TINY_PNG))
        self.assertEqual((width, height), (1, 1))

    def test_analysis_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "screenshot_error.png"
            image_path.write_bytes(TINY_PNG)

            with self.assertRaises(ConfigurationError):
                analyze_image(
                    image_path,
                    "Что произошло?",
                    VisionConfig(model="gpt-5.4-mini"),
                )

    def test_gpt5_vision_payload_uses_max_completion_tokens(self) -> None:
        payload = _build_chat_payload(
            "data:image/png;base64,AAA=",
            "Что видно?",
            VisionConfig(model="gpt-5.4-mini", max_tokens=123),
        )

        self.assertEqual(payload["max_completion_tokens"], 123)
        self.assertNotIn("max_tokens", payload)

    def test_legacy_vision_payload_uses_max_tokens(self) -> None:
        payload = _build_chat_payload(
            "data:image/png;base64,AAA=",
            "Что видно?",
            VisionConfig(model="gpt-4.1-mini", max_tokens=123),
        )

        self.assertEqual(payload["max_tokens"], 123)
        self.assertNotIn("max_completion_tokens", payload)


if __name__ == "__main__":
    unittest.main()
