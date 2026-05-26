"""Multimodal helpers for Support Copilot."""

from .vision import (
    ConfigurationError,
    VisionAPIError,
    VisionConfig,
    VisionResult,
    analyze_image,
    append_analysis_log,
    encode_image_as_data_url,
    format_result_for_console,
)

__all__ = [
    "ConfigurationError",
    "VisionAPIError",
    "VisionConfig",
    "VisionResult",
    "analyze_image",
    "append_analysis_log",
    "encode_image_as_data_url",
    "format_result_for_console",
]
