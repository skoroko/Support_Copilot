"""Domain tools available to Support Copilot."""

from support_copilot.tools.handlers import TOOL_HANDLERS, search_knowledge_base
from support_copilot.tools.schemas import available_tools, search_knowledge_base_tool_schema

__all__ = [
    "TOOL_HANDLERS",
    "available_tools",
    "search_knowledge_base",
    "search_knowledge_base_tool_schema",
]

