import os

# Must be set BEFORE hireflow is imported. From Part 5, tests use a real Postgres test database,
# because vector search and full-text search only exist in Postgres.
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://hireflow:hireflow@localhost:5432/hireflow_test"
)
os.environ["ENVIRONMENT"] = "test"
os.environ["EMBEDDING_PROVIDER"] = "fake"  # instant, deterministic, no model download

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from hireflow.db.base import Base
from hireflow.db.session import engine
from hireflow.main import create_app


@pytest.fixture(autouse=True)
async def fresh_db():
    """Every test starts with empty tables."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    """An HTTP client that talks to the app in memory (no real server needed)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def job_payload() -> dict:
    return {
        "title": "Senior Python Engineer",
        "location": "Bengaluru",
        "remote": True,
        "employment_type": "FULL_TIME",
        "description": "Build agentic hiring tools with FastAPI and LangGraph.",
        "must_have": ["Python", " FastAPI ", "python", ""],
        "nice_to_have": ["AWS"],
    }


@pytest.fixture
async def open_job(client, job_payload) -> dict:
    """A job that is already open, so candidates can apply."""
    job = (await client.post("/api/jobs", json=job_payload)).json()
    return (await client.patch(f"/api/jobs/{job['id']}", json={"status": "open"})).json()