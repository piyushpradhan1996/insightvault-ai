from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod

from app.config import get_settings

TOKEN_RE = re.compile(r"[a-zA-Z0-9_/-]{3,}")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BaseEmbeddingProvider(ABC):
    name: str

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic hashed bag-of-words embeddings for local MVP retrieval."""

    name = "local"

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            input=text,
        )
        return list(response.data[0].embedding)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def get_embedding_provider() -> BaseEmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider.lower() == "openai" and settings.openai_api_key:
        return OpenAIEmbeddingProvider(settings.openai_api_key)
    return LocalEmbeddingProvider()

