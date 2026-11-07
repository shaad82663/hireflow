from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All app configuration. Values come from environment variables or the .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "HireFlow"
    environment: str = "local"  # local | test | production
    debug: bool = False

    database_url: str = "postgresql+asyncpg://hireflow:hireflow@localhost:5432/hireflow"

    company_name: str = "Acme Robotics"
    company_url: str = "https://example.com"

    # LLM settings (used from Part 3)
    llm_provider: str = "bedrock_converse"  # bedrock_converse | openai
    llm_model: str = "set-me-in-.env"
    llm_temperature: float = 0.0
    aws_region: str = "us-east-1"


@lru_cache
def get_settings() -> Settings:
    """Create Settings once and reuse it (like a singleton)."""
    return Settings()