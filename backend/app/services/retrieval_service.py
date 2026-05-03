from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk
from app.schemas import RetrievedChunk
from app.services.embedding_provider import BaseEmbeddingProvider, cosine_similarity


class RetrievalService:
    def __init__(self, db: Session, embedding_provider: BaseEmbeddingProvider) -> None:
        self.db = db
        self.embedding_provider = embedding_provider

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = self.embedding_provider.embed(query)
        rows = (
            self.db.query(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .all()
        )
        scored: list[RetrievedChunk] = []
        for chunk, document in rows:
            chunk_embedding = json.loads(chunk.embedding_json)
            score = cosine_similarity(query_embedding, chunk_embedding)
            if score <= 0:
                continue
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_title=document.title,
                    score=round(score, 4),
                    text=chunk.text,
                    citation=chunk.citation_label,
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

