import json
import re


async def test_careers_page_shows_only_open_jobs(client, job_payload):
    draft = (await client.post("/api/jobs", json=job_payload)).json()
    job_payload["title"] = "Data Engineer"
    open_job = (await client.post("/api/jobs", json=job_payload)).json()
    await client.patch(f"/api/jobs/{open_job['id']}", json={"status": "open"})

    html = (await client.get("/careers")).text
    assert "Data Engineer" in html
    assert draft["slug"] not in html


async def test_job_detail_has_google_jobs_jsonld(client, job_payload):
    job = (await client.post("/api/jobs", json=job_payload)).json()
    await client.patch(f"/api/jobs/{job['id']}", json={"status": "open"})

    res = await client.get(f"/careers/{job['slug']}")
    assert res.status_code == 200
    match = re.search(r'<script type="application/ld\+json">(.*?)</script>', res.text, re.S)
    data = json.loads(match.group(1))
    assert data["@type"] == "JobPosting"
    assert data["title"] == "Senior Python Engineer"
    assert data["jobLocationType"] == "TELECOMMUTE"


async def test_draft_job_not_public(client, job_payload):
    job = (await client.post("/api/jobs", json=job_payload)).json()
    res = await client.get(f"/careers/{job['slug']}")
    assert res.status_code == 404