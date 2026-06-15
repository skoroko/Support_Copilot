from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from support_copilot.prompts.loader import load_tool_description


SEARCH_KNOWLEDGE_BASE_NAME = "search_knowledge_base"


SEARCH_KNOWLEDGE_BASE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 2,
            "maxLength": 500,
            "description": "Вопрос пользователя или короткая поисковая формулировка.",
        },
        "product": {
            "type": "string",
            "enum": [
                "support_copilot",
                "licensing",
                "installation",
                "calculation",
                "documentation",
                "security",
            ],
            "description": "Раздел базы знаний, наиболее близкий к вопросу.",
        },
    },
    "required": ["query", "product"],
    "additionalProperties": False,
}


class StrictParametersSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["object"]
    properties: dict[str, Any]
    required: list[str]
    additional_properties: Literal[False] = Field(alias="additionalProperties")

    @field_validator("required")
    @classmethod
    def required_matches_properties(cls, value: list[str], info: Any) -> list[str]:
        properties = info.data.get("properties", {})
        if set(value) != set(properties):
            raise ValueError("strict tool schemas must require every declared property")
        return value


class FunctionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["search_knowledge_base"]
    description: str = Field(min_length=40)
    parameters: StrictParametersSchema
    strict: Literal[True]


class ToolSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: FunctionSchema


def validate_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    ToolSchema.model_validate(schema)
    return schema


def search_knowledge_base_tool_schema() -> dict[str, Any]:
    schema = {
        "type": "function",
        "function": {
            "name": SEARCH_KNOWLEDGE_BASE_NAME,
            "description": load_tool_description(SEARCH_KNOWLEDGE_BASE_NAME),
            "parameters": SEARCH_KNOWLEDGE_BASE_PARAMETERS,
            "strict": True,
        },
    }
    return validate_tool_schema(schema)


def available_tools() -> list[dict[str, Any]]:
    return [search_knowledge_base_tool_schema()]

