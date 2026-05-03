from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import get_settings
from app.schemas import Citation, RetrievedChunk


class BaseLLMProvider(ABC):
    provider_name: str

    @abstractmethod
    def answer_question(
        self,
        question: str,
        retrieved_context: list[RetrievedChunk],
        answer_style: str,
        prompt_template: str,
    ) -> tuple[str, list[Citation]]:
        raise NotImplementedError

    @abstractmethod
    def extract_insights(self, document_title: str, content: str, prompt_template: str) -> dict:
        raise NotImplementedError


def get_llm_provider() -> BaseLLMProvider:
    settings = get_settings()
    if settings.ai_provider.lower() == "openai" and settings.openai_api_key:
        from app.services.openai_provider import OpenAIProvider

        return OpenAIProvider(settings.openai_api_key)

    from app.services.mock_llm_provider import MockLLMProvider

    return MockLLMProvider()

