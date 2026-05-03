from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ai_provider: str = "mock"
    openai_api_key: str | None = None
    embedding_provider: str = "local"
    database_url: str = "sqlite:///./insightvault.db"
    rag_top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120
    prompt_version: str = "rag_qa_v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

