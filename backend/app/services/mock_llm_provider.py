from __future__ import annotations

import re

from app.schemas import Citation, RetrievedChunk
from app.services.embedding_provider import tokenize
from app.services.llm_provider import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    provider_name = "mock"

    def answer_question(
        self,
        question: str,
        retrieved_context: list[RetrievedChunk],
        answer_style: str,
        prompt_template: str,
    ) -> tuple[str, list[Citation]]:
        if not retrieved_context:
            return "I could not find enough information in the uploaded documents.", []

        question_tokens = set(tokenize(question))
        selected: list[RetrievedChunk] = []
        for chunk in retrieved_context:
            chunk_tokens = set(tokenize(chunk.text))
            if question_tokens & chunk_tokens:
                selected.append(chunk)
            if len(selected) >= (3 if answer_style == "detailed" else 2):
                break
        if not selected:
            selected = retrieved_context[:1]

        sentences: list[str] = []
        for chunk in selected:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text.strip()):
                if not sentence:
                    continue
                if question_tokens & set(tokenize(sentence)):
                    sentences.append(sentence)
                    break
            if not sentences:
                sentences.append(chunk.text.strip()[:220])

        citation_labels = " ".join(chunk.citation for chunk in selected)
        answer = f"Based on the uploaded documents, {' '.join(sentences)} {citation_labels}"
        citations = [
            Citation(citation=chunk.citation, document_title=chunk.document_title, chunk_id=chunk.chunk_id)
            for chunk in selected
        ]
        return answer, citations

    def extract_insights(self, document_title: str, content: str, prompt_template: str) -> dict:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", content.strip()) if part.strip()]
        summary = sentences[0] if sentences else f"{document_title} contains uploaded knowledge for retrieval."
        if len(sentences) > 1:
            summary = f"{summary} {sentences[1]}"

        action_items = []
        risks = []
        decisions = []
        entities = set(re.findall(r"\b[A-Z][A-Za-z0-9&-]{2,}\b", content))

        for sentence in sentences:
            lowered = sentence.lower()
            if any(marker in lowered for marker in ("action", "todo", "follow up", "must", "owner")):
                owner_match = re.search(r"owner[:\s]+([A-Z][A-Za-z ]+)", sentence)
                deadline_match = re.search(r"(by|before|due)\s+([A-Za-z0-9 ,/-]+)", sentence, flags=re.I)
                action_items.append(
                    {
                        "task": sentence,
                        "owner": owner_match.group(1).strip() if owner_match else "unknown",
                        "deadline": deadline_match.group(2).strip() if deadline_match else "unknown",
                    }
                )
            if any(marker in lowered for marker in ("risk", "blocked", "blocker", "delay", "failure", "security", "compliance")):
                risks.append(sentence)
            if any(marker in lowered for marker in ("decided", "decision", "agreed", "approved")):
                decisions.append(sentence)

        if not action_items:
            action_items.append(
                {
                    "task": f"Review {document_title} with stakeholders and confirm next steps.",
                    "owner": "unknown",
                    "deadline": "unknown",
                }
            )
        if not risks and "policy" in document_title.lower():
            risks.append("Policy exceptions may need manual review if they are not explicitly documented.")

        return {
            "summary": summary,
            "action_items": action_items[:5],
            "risks": risks[:5],
            "decisions": decisions[:5],
            "entities": sorted(entities)[:10],
            "follow_up_questions": [
                f"What source updates would change the guidance in {document_title}?",
                "Are there owners and deadlines missing from the document?",
                "Which sections should be monitored for policy or API drift?",
            ],
        }

