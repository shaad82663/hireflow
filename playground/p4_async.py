import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from functools import wraps


async def fetch_resume(candidate_id: int) -> str:
    await asyncio.sleep(0.2)
    return f"resume-{candidate_id}"

async def fetch_all() -> list[str]:
    return await asyncio.gather(*(fetch_resume(i) for i in range(5)))

@asynccontextmanager
async def db_connections(name: str) -> AsyncIterator[str]:
    print(f"Open file  {name}")
    try:
        yield name
    finally:
        print(f"closed {name}")
