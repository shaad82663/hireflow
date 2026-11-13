from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from hireflow.models.application import Application, Candidate, ResumeChunk

CANDIDATE_POOL = 20  # how many results each search method contributes before fusion
RRF_K = 60  # standard constant for Reciprocal Rank Fusion; dampens the effect of rank 1 vs 2


@dataclass
class SearchHit:
    chunk_id: int
    application_id: int
    candidate_name: str
    content: str
    score: float


def _base_query(job_id: int):
    return (
        select(ResumeChunk.id, ResumeChunk.application_id, Candidate.name, ResumeChunk.content)
        .join(Application, ResumeChunk.application_id == Application.id)
        .join(Candidate, Application.candidate_id == Candidate.id)
        .where(Application.job_id == job_id)
    )

async def vector_search(session: AsyncSession, query_vec: list[float], job_id: int) -> list:
    """Meaning-based: closest embeddings first (cosine distance, smaller = more similar)."""
    stmt = (
        _base_query(job_id)
        .order_by(ResumeChunk.embedding.cosine_distance(query_vec))
        .limit(CANDIDATE_POOL)
    )

async def keyword_search(session: AsyncSession, query: str, job_id: int) -> list:
    """Exact-word based: Postgres full-text search. Great for names like 'FastAPI' or 'AWS'."""
    document = func.to_tsvector("english",ResumeChunk.content)
    ts_query = func.websearch_to_tsquery("english", query)
    stmt = (
        _base_query(job_id)
        .where(document.op('@@')(ts_query))
        .order_by(func.ts_rank(document, ts_query).desc())
        .limit(CANDIDATE_POOL)
    )
    return list((await session.execute(stmt)).all())

def reciprocal_rank_fusion(*result_lists: list) -> dict[int, float]:
    """Merge ranked lists: each appearance adds 1 / (RRF_K + rank). Items ranked well by BOTH
    methods win. Ranks are used instead of raw scores because the two scores are not comparable."""
    scores: dict[int, float] = {}
    for results in result_lists:
        for rank, row in enumerate(results, start=1):
            scores[row.id] = scores.get(row.id, 0.0) + 1.0 / (RRF_K + rank)
    return scores

async def hybrid_search(
        session: AsyncSession, embeddings: Embeddings, job_id: int, query: str, k: int = 5
) ->  list[SearchHit]:
    query_vec = await embeddings.aembed_query(query)
    by_vector = await vector_search(session, query_vec, job_id)
    by_keyword = await keyword_search(session, query, job_id)

    scores = reciprocal_rank_fusion(by_vector, by_keyword)
    rows = {row.id: row for row in [*by_vector, *by_keyword]}
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [
        SearchHit(
            chunk_id=cid,
            application_id=rows[cid].application_id,
            candidate_name=rows[cid].name,
            content=rows[cid].content,
            score=round(scores[cid], 5),
        )
        for cid in best
    ]
