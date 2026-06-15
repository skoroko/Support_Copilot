from __future__ import annotations

import unittest

from support_copilot.config import SupportCopilotSettings, load_settings
from support_copilot.infrastructure.tool_calling import build_demo_tool_calling_client
from support_copilot.prompts.loader import render_system_prompt
from support_copilot.tools.handlers import search_knowledge_base
from support_copilot.tools import TOOL_HANDLERS, available_tools
from support_copilot.tools.schemas import search_knowledge_base_tool_schema, validate_tool_schema


class ToolCallingTests(unittest.TestCase):
    def test_settings_reads_provider_values(self) -> None:
        settings = SupportCopilotSettings(
            openai_api_key="sk-test",
            openai_model="test-model",
            ollama_models="qwen3:1.7b,gemma3:4b",
        )
        self.assertEqual(settings.openai_api_key, "sk-test")
        self.assertEqual(settings.openai_model, "test-model")
        self.assertEqual(settings.selected_ollama_model, "qwen3:1.7b")

    def test_load_settings_uses_env_file_defaults(self) -> None:
        settings = load_settings(env_path=None)
        self.assertEqual(settings.openai_base_url, "https://api.openai.com/v1")
        self.assertEqual(settings.ollama_api_key, "ollama")

    def test_search_knowledge_base_schema_is_strict(self) -> None:
        schema = search_knowledge_base_tool_schema()
        validate_tool_schema(schema)
        function = schema["function"]
        self.assertEqual(function["name"], "search_knowledge_base")
        self.assertTrue(function["strict"])
        self.assertFalse(function["parameters"]["additionalProperties"])
        self.assertEqual(
            set(function["parameters"]["required"]),
            set(function["parameters"]["properties"]),
        )

    def test_search_knowledge_base_reads_local_data(self) -> None:
        result = search_knowledge_base(
            query="лицензия не активируется второй день",
            product="licensing",
        )
        self.assertEqual(result["kb_version"], "support-kb-v1")
        self.assertTrue(result["results"])
        self.assertEqual(result["results"][0]["id"], "KB-LIC-001")

    def test_tool_calling_cycle_runs_from_main_package(self) -> None:
        client = build_demo_tool_calling_client()
        result = client.chat_with_tools(
            [
                {
                    "role": "system",
                    "content": render_system_prompt(
                        product_name="MIDAS Support Copilot",
                        kb_version="support-kb-v1",
                    ),
                },
                {
                    "role": "user",
                    "content": "Найди в базе знаний, что запросить если лицензия не активируется.",
                },
            ],
            tools=available_tools(),
            tool_handlers=TOOL_HANDLERS,
            temperature=0,
            max_tokens=500,
        )
        self.assertFalse(result.direct_answer)
        self.assertEqual(result.tool_calls[0].name, "search_knowledge_base")
        self.assertIn("KB-LIC-001", result.answer)


if __name__ == "__main__":
    unittest.main()
