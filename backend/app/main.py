from __future__ import annotations

import json
import uuid

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SessionLocal, get_db, init_db
from app.models import Document, IngestionJob
from app.schemas import (
    AskRequest,
    AskResponse,
    AsyncDocumentResponse,
    DocumentCreate,
    DocumentDetail,
    DocumentIngestionResponse,
    DocumentListItem,
    EvalRunResponse,
    InsightResponse,
    JobResponse,
    ModelMetadata,
    RetrievalRequest,
    RetrievalResponse,
)
from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentService
from app.services.embedding_provider import get_embedding_provider
from app.services.eval_service import EvalService
from app.services.guardrail_service import GuardrailService
from app.services.llm_provider import get_llm_provider
from app.services.markdown_export import render_document_summary_markdown
from app.services.prompt_loader import load_prompt
from app.services.retrieval_service import RetrievalService

settings = get_settings()
app = FastAPI(
    title="InsightVault AI",
    description="RAG-based knowledge intelligence API with citations, structured insights, guardrails, and evals.",
    version="0.1.0",
    contact={
        "name": "piyushpradhan1996",
        "url": "https://github.com/piyushpradhan1996",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def _chunking_service() -> ChunkingService:
    return ChunkingService(settings.chunk_size, settings.chunk_overlap)


def _document_response(document: Document) -> DocumentListItem:
    tags = json.loads(document.tags_json or "[]")
    return DocumentListItem(
        id=document.id,
        title=document.title,
        source_type=document.source_type,
        tags=tags,
        ingestion_status=document.ingestion_status,
        chunk_count=len(document.chunks),
        created_at=document.created_at,
    )


def _ingest_document(db: Session, payload: DocumentCreate) -> DocumentIngestionResponse:
    service = DocumentService(db, _chunking_service(), get_embedding_provider())
    document, chunk_count = service.ingest(payload)
    return DocumentIngestionResponse(
        document_id=document.id,
        title=document.title,
        ingestion_status=document.ingestion_status,
        chunk_count=chunk_count,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": get_llm_provider().provider_name}


@app.post("/api/documents", response_model=DocumentIngestionResponse)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)) -> DocumentIngestionResponse:
    return _ingest_document(db, payload)


@app.post("/api/documents/file", response_model=DocumentIngestionResponse)
async def upload_document_file(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    source_type: str = Form(default="text"),
    tags: str = Form(default=""),
    db: Session = Depends(get_db),
) -> DocumentIngestionResponse:
    if not file.filename or not file.filename.lower().endswith((".txt", ".md", ".json")):
        raise HTTPException(status_code=400, detail="Only .txt, .md, and .json files are supported in the MVP.")
    content = (await file.read()).decode("utf-8")
    parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    payload = DocumentCreate(
        title=title or file.filename,
        content=content,
        source_type=source_type,  # type: ignore[arg-type]
        tags=parsed_tags,
    )
    return _ingest_document(db, payload)


def _process_async_ingestion(job_id: str, payload: dict) -> None:
    db = SessionLocal()
    try:
        job = db.get(IngestionJob, job_id)
        if not job:
            return
        job.status = "processing"
        db.commit()
        result = _ingest_document(db, DocumentCreate.model_validate(payload))
        job.status = "completed"
        job.document_id = result.document_id
        db.commit()
    except Exception as exc:
        job = db.get(IngestionJob, job_id)
        if job:
            job.status = "failed"
            job.error = str(exc)
            db.commit()
    finally:
        db.close()


@app.post("/api/documents/async", response_model=AsyncDocumentResponse)
def create_document_async(
    payload: DocumentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> AsyncDocumentResponse:
    job_id = str(uuid.uuid4())
    db.add(IngestionJob(id=job_id, status="queued"))
    db.commit()
    background_tasks.add_task(_process_async_ingestion, job_id, payload.model_dump())
    return AsyncDocumentResponse(job_id=job_id, status="queued")


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobResponse:
    job = db.get(IngestionJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(job_id=job.id, status=job.status, document_id=job.document_id, error=job.error)


@app.get("/api/documents", response_model=list[DocumentListItem])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentListItem]:
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return [_document_response(document) for document in documents]


@app.get("/api/documents/{document_id}", response_model=DocumentDetail)
def get_document(document_id: int, db: Session = Depends(get_db)) -> DocumentDetail:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    base = _document_response(document)
    return DocumentDetail(**base.model_dump(), content_preview=document.content[:500])


@app.post("/api/retrieve", response_model=RetrievalResponse)
def retrieve(payload: RetrievalRequest, db: Session = Depends(get_db)) -> RetrievalResponse:
    results = RetrievalService(db, get_embedding_provider()).retrieve(payload.query, payload.top_k)
    return RetrievalResponse(query=payload.query, results=results)


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest, db: Session = Depends(get_db)) -> AskResponse:
    embedding_provider = get_embedding_provider()
    retrieved = RetrievalService(db, embedding_provider).retrieve(payload.question, payload.top_k)
    llm = get_llm_provider()
    answer, citations = llm.answer_question(
        payload.question,
        retrieved,
        payload.answer_style,
        load_prompt("rag_qa_v1.txt"),
    )
    guardrails = GuardrailService()
    citations = guardrails.dedupe_citations(citations)
    result = guardrails.validate_answer(answer, citations, retrieved)
    metadata = ModelMetadata(
        provider=llm.provider_name,
        used_rag=True,
        top_k=payload.top_k,
        prompt_version="rag_qa_v1",
        embedding_provider=embedding_provider.name,
    )
    return AskResponse(
        answer=answer,
        citations=citations,
        retrieved_context=retrieved,
        model_metadata=metadata,
        guardrail_result=result,
    )


def _generate_insights(document: Document) -> InsightResponse:
    llm = get_llm_provider()
    raw = llm.extract_insights(document.title, document.content, load_prompt("insight_extraction_v1.txt"))
    metadata = ModelMetadata(
        provider=llm.provider_name,
        used_rag=False,
        top_k=None,
        prompt_version="insight_extraction_v1",
        embedding_provider=None,
    ).model_dump()
    response, guardrail = GuardrailService().validate_insights(raw, metadata)
    if not response:
        raise HTTPException(status_code=500, detail=guardrail.errors)
    return response


@app.post("/api/insights", response_model=InsightResponse)
def insights(payload: dict, db: Session = Depends(get_db)) -> InsightResponse:
    document_id = payload.get("document_id")
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return _generate_insights(document)


@app.get("/api/documents/{document_id}/summary.md", response_class=PlainTextResponse)
def document_summary_markdown(document_id: int, db: Session = Depends(get_db)) -> str:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return render_document_summary_markdown(document.title, _generate_insights(document))


@app.post("/api/evals/run", response_model=EvalRunResponse)
def run_evals(db: Session = Depends(get_db)) -> EvalRunResponse:
    return EvalService(db).run_all()
