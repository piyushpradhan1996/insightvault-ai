from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SourceType = Literal["text", "markdown", "policy", "meeting_notes", "api_doc", "report"]
AnswerStyle = Literal["concise", "detailed"]


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    source_type: SourceType = "text"
    tags: list[str] = Field(default_factory=list)


class DocumentIngestionResponse(BaseModel):
    document_id: int
    title: str
    ingestion_status: str
    chunk_count: int


class DocumentListItem(BaseModel):
    id: int
    title: str
    source_type: str
    tags: list[str]
    ingestion_status: str
    chunk_count: int
    created_at: datetime


class DocumentDetail(DocumentListItem):
    content_preview: str


class AsyncDocumentResponse(BaseModel):
    job_id: str
    status: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    document_id: int | None = None
    error: str | None = None


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_title: str
    score: float
    text: str
    citation: str


class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    answer_style: AnswerStyle = "concise"


class Citation(BaseModel):
    citation: str
    document_title: str
    chunk_id: str


class ModelMetadata(BaseModel):
    provider: str
    used_rag: bool = True
    top_k: int | None = None
    prompt_version: str
    embedding_provider: str | None = None


class GuardrailResult(BaseModel):
    passed: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_context: list[RetrievedChunk]
    model_metadata: ModelMetadata
    guardrail_result: GuardrailResult


class ActionItem(BaseModel):
    task: str
    owner: str = "unknown"
    deadline: str = "unknown"


class InsightResponse(BaseModel):
    summary: str = Field(..., min_length=1)
    action_items: list[ActionItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    model_metadata: ModelMetadata
    guardrail_result: GuardrailResult


class EvalQuestionResult(BaseModel):
    question: str
    retrieval_hit: bool
    answer_contains_expected_terms: bool
    citation_present: bool
    unsupported_answer_detected: bool
    score: float


class EvalCaseResult(BaseModel):
    name: str
    question_results: list[EvalQuestionResult]
    average_score: float


class EvalRunResponse(BaseModel):
    eval_count: int
    results: list[EvalCaseResult]


class LLMAnswerPayload(BaseModel):
    answer: str
    citations: list[Citation]

    model_config = ConfigDict(extra="ignore")

