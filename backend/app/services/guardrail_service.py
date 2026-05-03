from __future__ import annotations

from app.schemas import Citation, GuardrailResult, InsightResponse, RetrievedChunk
from app.services.embedding_provider import tokenize


class GuardrailService:
    def validate_answer(
        self,
        answer: str,
        citations: list[Citation],
        retrieved_context: list[RetrievedChunk],
    ) -> GuardrailResult:
        warnings: list[str] = []
        errors: list[str] = []

        if not retrieved_context:
            errors.append("No retrieved context was available, so the answer must refuse.")
        if retrieved_context and not citations:
            errors.append("Answer is missing citations.")
        if len({citation.chunk_id for citation in citations}) != len(citations):
            warnings.append("Duplicate citations were removed or should be removed.")

        if retrieved_context and answer:
            context_tokens = set(tokenize(" ".join(chunk.text for chunk in retrieved_context)))
            answer_tokens = {
                token
                for token in tokenize(answer)
                if len(token) > 5
                and token
                not in {
                    "uploaded",
                    "documents",
                    "document",
                    "based",
                    "chunk",
                    "could",
                    "enough",
                    "information",
                }
            }
            unsupported = answer_tokens - context_tokens
            if answer_tokens and len(unsupported) / max(len(answer_tokens), 1) > 0.45:
                warnings.append("Answer may contain claims with weak keyword support in retrieved context.")

        return GuardrailResult(passed=not errors, warnings=warnings, errors=errors)

    def dedupe_citations(self, citations: list[Citation]) -> list[Citation]:
        seen: set[str] = set()
        unique: list[Citation] = []
        for citation in citations:
            if citation.chunk_id in seen:
                continue
            seen.add(citation.chunk_id)
            unique.append(citation)
        return unique

    def validate_insights(self, payload: dict, metadata: dict) -> tuple[InsightResponse | None, GuardrailResult]:
        try:
            response = InsightResponse.model_validate({**payload, "model_metadata": metadata, "guardrail_result": {"passed": True}})
        except Exception as exc:
            return None, GuardrailResult(passed=False, errors=[f"Insight schema validation failed: {exc}"])

        if not response.summary.strip():
            return None, GuardrailResult(passed=False, errors=["Insight summary cannot be empty."])
        response.guardrail_result = GuardrailResult(passed=True, warnings=[], errors=[])
        return response, response.guardrail_result

