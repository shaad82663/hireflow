from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    """All app configuration. Values come from environment variables or the .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "HireFlow"
    environment: str = "local"  # local | test | production
    debug: bool = False

    database_url: str = "postgresql+asyncpg://hireflow:hireflow@localhost:5432/hireflow"

    company_name: str = "Acme Robotics"
    company_url: str = "https://example.com"
    company_country: str = "IN" 

    # LLM settings (used from Part 3)
    llm_provider: str = "openai"  # bedrock_converse | openai
    llm_model: str = "set-me-in-.env"
    llm_temperature: float | None = 0.0  # None = use the model's own default
    aws_region: str = "us-east-1"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr  | None = None

    # Embeddings (used from Part 5). "local" runs a small free model on your machine.
    embedding_provider: Literal["local", "openai", "fake"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    max_resume_mb: int = 5
    @property
    def checkpoint_db_url(self) -> str:
        """LangGraph's Postgres checkpointer uses psycopg, which wants a plain postgresql:// URL."""
        return self.database_url.replace("+asyncpg", "")

@lru_cache
def get_settings() -> Settings:
    """Create Settings once and reuse it (like a singleton)."""
    return Settings()