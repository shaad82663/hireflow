import json
import os
import sys

import pytest
from fakes import FakeToolModel, tool_call
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

from hireflow.db.session import SessionLocal
from hireflow.mcp_server import build_server
from hireflow.rag.embeddings import get_embeddings

RESUME = "Backend engineer. Five years of Python and FastAPI. Ran Kubernetes on AWS in production."


@pytest.fixture
async def job_with_applicant(client, open_job) -> dict:
    await client.post(
        f"/careers/{open_job['slug']}/apply",
        data={"name": "Asha Rao", "email": "asha@example.com", "consent": "true"},
        files={"resume": ("asha.txt", RESUME.encode(), "text/plain")},
    )
    return open_job


def server(allow_writes: bool = False):
    return build_server(SessionLocal, get_embeddings(), allow_writes=allow_writes)


def data(result) -> object:
    """The JSON a tool returned. Typed returns (list[dict]) come as structuredContent, wrapped
    as {"result": [...]}; a plain dict comes back as JSON text."""
    assert not result.isError, result.content
    payload = result.structuredContent
    if payload is None:
        return json.loads(result.content[0].text)
    return payload["result"] if set(payload) == {"result"} else payload


async def test_lists_only_read_only_tools_by_default():
    async with create_connected_server_and_client_session(server()) as mcp:
        tools = (await mcp.list_tools()).tools

    assert {t.name for t in tools} == {
        "list_open_jobs",
        "get_job",
        "search_resumes",
        "interview_status",
    }
    assert all(t.annotations.readOnlyHint for t in tools)


async def test_write_tools_need_the_flag(job_with_applicant):
    async with create_connected_server_and_client_session(server(allow_writes=True)) as mcp:
        assert "close_job" in {t.name for t in (await mcp.list_tools()).tools}
        result = await mcp.call_tool("close_job", {"job_id": job_with_applicant["id"]})

    assert data(result) == {"id": job_with_applicant["id"], "status": "closed"}


async def test_read_tools_return_hireflow_data(job_with_applicant):
    job_id = job_with_applicant["id"]
    async with create_connected_server_and_client_session(server()) as mcp:
        jobs = data(await mcp.call_tool("list_open_jobs", {}))
        job = data(await mcp.call_tool("get_job", {"job_id": job_id}))
        hits = data(
            await mcp.call_tool("search_resumes", {"job_id": job_id, "query": "Kubernetes"})
        )
        interviews = data(await mcp.call_tool("interview_status", {"job_id": job_id}))

    assert jobs == [{"id": job_id, "title": "Senior Python Engineer", "location": "Bengaluru"}]
    assert job["must_have"] == ["Python", "FastAPI"]
    assert hits[0]["candidate"] == "Asha Rao"
    assert interviews == []


async def test_errors_come_back_as_tool_errors():
    async with create_connected_server_and_client_session(server()) as mcp:
        missing = await mcp.call_tool("get_job", {"job_id": 999})
        bad_limit = await mcp.call_tool("search_resumes", {"job_id": 1, "query": "x", "limit": 50})
        wrong_type = await mcp.call_tool("get_job", {"job_id": "abc"})

    assert missing.isError and "999" in missing.content[0].text
    assert bad_limit.isError and "between 1 and 10" in bad_limit.content[0].text
    assert wrong_type.isError  # the schema is checked before your code runs


async def test_resource_and_prompt(job_with_applicant):
    job_id = job_with_applicant["id"]
    async with create_connected_server_and_client_session(server()) as mcp:
        resource = await mcp.read_resource(f"hireflow://jobs/{job_id}")
        prompt = await mcp.get_prompt("screening_brief", {"job_id": str(job_id)})

    assert json.loads(resource.contents[0].text)["title"] == "Senior Python Engineer"
    assert f"job {job_id}" in prompt.messages[0].content.text


async def test_a_langchain_agent_can_use_mcp_tools(job_with_applicant):
    """langchain-mcp-adapters turns MCP tools into normal LangChain tools."""
    script = [tool_call("list_open_jobs", {}), AIMessage("One open job: Senior Python Engineer.")]
    async with create_connected_server_and_client_session(server()) as mcp:
        tools = await load_mcp_tools(mcp)
        agent = create_agent(FakeToolModel(messages=iter(script)), tools=tools)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "Open jobs?"}]})

    tool_message = result["messages"][2]
    assert tool_message.type == "tool" and "Senior Python Engineer" in tool_message.text
    assert result["messages"][-1].text == "One open job: Senior Python Engineer."


async def test_real_stdio_server_process(job_with_applicant):
    """Start the server exactly like Claude Desktop does: as a child process over stdio."""
    client = MultiServerMCPClient(
        {
            "hireflow": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "hireflow.mcp_server"],
                # the child process gets the test database and the fake embeddings
                "env": {**os.environ},
            }
        }
    )
    tools = {t.name: t for t in await client.get_tools()}
    assert "close_job" not in tools

    answer = await tools["list_open_jobs"].ainvoke({})
    assert "Senior Python Engineer" in str(answer)