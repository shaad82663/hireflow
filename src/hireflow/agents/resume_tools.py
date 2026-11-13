from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


from hireflow.rag.search import hybrid_search


def build_resume_tools(
        session_factory: async_sessionmaker[AsyncSession], embeddings: Embeddings
) -> list[BaseTool]:
    """Agentic RAG: instead of always retrieving, the agent DECIDES when to search resumes."""
    @tool
    async def search_resumes(job_id: int, query: str) -> list[dict]:
        """Search the resumes of applicants to one job. Use specific skills or experience as
        the query, e.g. 'Kubernetes in production' or 'led a team'. Returns the best excerpts
        with the candidate name and application id."""
        async with session_factory() as session:
            hits = await hybrid_search(session, embeddings, job_id, query, k=6)
        if not hits:
            return [{"info": "No matching resume excerpts for this job."}]
        return [
            {
                "application_id": h.application_id,
                "candidate": h.candidate_name,
                "excerpt": h.content,
            }
            for h in hits
        ]
    
    return [search_resumes]