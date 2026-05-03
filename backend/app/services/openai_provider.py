from __future__ import annotations

import json
import os

from app.schemas import Citation, LLMAnswerPayload, RetrievedChunk
from app.services.llm_provider import BaseLLMProvider
from app.services.mock_llm_provider import MockLLMProvider


class OpenAIProvider(BaseLLMProvider):
    provider_name = "openai"

    def __init__(self, api_key: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.fallback = MockLLMProvider()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def answer_question(
        self,
        question: str,
        retrieved_context: list[RetrievedChunk],
        answer_style: str,
        prompt_template: str,
    ) -> tuple[str, list[Citation]]:
        if not retrieved_context:
            return self.fallback.answer_question(question, retrieved_context, answer_style, prompt_template)

        context_json = json.dumps([chunk.model_dump() for chunk in retrieved_context], indent=2)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt_template},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n"
                            f"Answer style: {answer_style}\n"
                            f"Retrieved context JSON:\n{context_json}"
                        ),
                    },
                ],
                temperature=0,
            )
            raw = response.choices[0].message.content or "{}"
            parsed = LLMAnswerPayload.model_validate_json(raw)
            return parsed.answer, parsed.citations
        except Exception:
            return self.fallback.answer_question(question, retrieved_context, answer_style, prompt_template)

    def extract_insights(self, document_title: str, content: str, prompt_template: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": f"Title: {document_title}\nDocument:\n{content}"},
                ],
                temperature=0,
            )
            return json.loads(response.choices[0].message.content or "{}")
        except Exception:
            return self.fallback.extract_insights(document_title, content, prompt_template)

