from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Template


PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def load_prompt_file(relative_path: str) -> str:
    path = (PROMPTS_DIR / relative_path).resolve()
    if not path.is_relative_to(PROMPTS_DIR):
        raise ValueError(f"Prompt path escapes prompts directory: {relative_path}")
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=8)
def render_system_prompt(version: str = "v1", **context: object) -> str:
    text = load_prompt_file(f"system_{version}.j2")
    return Template(text).render(**context).strip()


@lru_cache(maxsize=16)
def load_tool_description(tool_name: str) -> str:
    return load_prompt_file(f"tools/{tool_name}.md").strip()

