from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.schemas import DocumentCreate, EvalCaseResult, EvalQuestionResult, EvalRunResponse
from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentService
from app.services.embedding_provider import LocalEmbeddingProvider
from app.services.guardrail_service import GuardrailService
from app.services.mock_llm_provider import MockLLMProvider
from app.services.prompt_loader import load_prompt
from app.services.retrieval_service import RetrievalService


EVAL_DIR = Path(__file__).resolve().parents[2] / "evals"


class EvalService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.embedding_provider = LocalEmbeddingProvider()
        self.guardrails = GuardrailService()
        self.llm = MockLLMProvider()

    def run_all(self) -> EvalRunResponse:
        results = [self._run_case(path) for path in sorted(EVAL_DIR.glob("*.json"))]
        return EvalRunResponse(eval_count=len(results), results=results)

    def _run_case(self, path: Path) -> EvalCaseResult:
        data = json.loads(path.read_text(encoding="utf-8"))
        document_service = DocumentService(
            self.db,
            ChunkingService(chunk_size=500, chunk_overlap=80),
            self.embedding_provider,
        )
        document_ids: list[int] = []
        for doc in data["documents"]:
            created, _ = document_service.ingest(
                DocumentCreate(
                    title=doc["title"],
                    content=doc["content"],
                    source_type=doc.get("source_type", "text"),
                    tags=doc.get("tags", []),
                )
            )
            document_ids.append(created.id)

        question_results = [self._score_question(question) for question in data["questions"]]
        for document_id in document_ids:
            self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
            document = self.db.get(Document, document_id)
            if document:
                self.db.delete(document)
        self.db.commit()

        average = round(sum(result.score for result in question_results) / max(len(question_results), 1), 2)
        return EvalCaseResult(name=data["name"], question_results=question_results, average_score=average)

    def _score_question(self, question_data: dict) -> EvalQuestionResult:
        question = question_data["question"]
        expected_terms = [term.lower() for term in question_data.get("expected_answer_contains", [])]
        retrieved = RetrievalService(self.db, self.embedding_provider).retrieve(question, top_k=5)
        answer, citations = self.llm.answer_question(question, retrieved, "concise", load_prompt("rag_qa_v1.txt"))
        guardrail = self.guardrails.validate_answer(answer, citations, retrieved)
        context_text = " ".join(chunk.text.lower() for chunk in retrieved)
        retrieval_hit = bool(retrieved) and all(term in context_text for term in expected_terms)
        answer_contains = all(term in answer.lower() for term in expected_terms)
        citation_present = bool(citations) if question_data.get("expected_citation_required", True) else True
        unsupported = any("weak keyword support" in warning for warning in guardrail.warnings)
        checks = [retrieval_hit, answer_contains, citation_present, not unsupported]
        score = round((sum(1 for check in checks if check) / len(checks)) * 100, 2)
        return EvalQuestionResult(
            question=question,
            retrieval_hit=retrieval_hit,
            answer_contains_expected_terms=answer_contains,
            citation_present=citation_present,
            unsupported_answer_detected=unsupported,
            score=score,
        )
