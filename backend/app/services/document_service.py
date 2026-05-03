from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.schemas import DocumentCreate
from app.services.chunking_service import ChunkingService
from app.services.embedding_provider import BaseEmbeddingProvider


class DocumentService:
    def __init__(
        self,
        db: Session,
        chunking_service: ChunkingService,
        embedding_provider: BaseEmbeddingProvider,
    ) -> None:
        self.db = db
        self.chunking_service = chunking_service
        self.embedding_provider = embedding_provider

    def ingest(self, payload: DocumentCreate) -> tuple[Document, int]:
        document = Document(
            title=payload.title,
            content=payload.content,
            source_type=payload.source_type,
            tags_json=json.dumps(payload.tags),
            ingestion_status="processing",
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)

        chunks = self.chunking_service.chunk_text(payload.content)
        for chunk in chunks:
            chunk_id = f"doc-{document.id}-chunk-{chunk.chunk_index}"
            citation = f"[{document.title}, chunk {chunk.chunk_index}]"
            self.db.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_id=chunk_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    embedding_json=json.dumps(self.embedding_provider.embed(chunk.text)),
                    character_start=chunk.character_start,
                    character_end=chunk.character_end,
                    citation_label=citation,
                )
            )
        document.ingestion_status = "completed"
        self.db.commit()
        self.db.refresh(document)
        return document, len(chunks)

