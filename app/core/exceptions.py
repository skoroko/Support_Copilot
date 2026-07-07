from __future__ import annotations


class LLMError(Exception):
    code = "llm_error"

    def __init__(self, message: str = "LLM provider error") -> None:
        super().__init__(message)
        self.message = message


class LLMRateLimitError(LLMError):
    code = "llm_rate_limit"


class LLMTimeoutError(LLMError):
    code = "llm_timeout"


class LLMAuthError(LLMError):
    code = "llm_auth"


class LLMOutputBlockedError(LLMError):
    code = "llm_output_blocked"
