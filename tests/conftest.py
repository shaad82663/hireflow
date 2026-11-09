import os

# Must be set BEFORE hireflow is imported, so the app uses a throwaway test database
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["ENVIRONMENT"] = "test"

import pytest
from httpx import ASGITransport, AsyncClient

from hireflow.db.base import Base
from hireflow.db.session import engine
from hireflow.main import create_app


@pytest.fixture(autouse=True)
async def fresh_db():
    """Every test starts with empty tables."""
    async with engine.begin() as conn:
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