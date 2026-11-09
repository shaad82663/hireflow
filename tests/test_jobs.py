from hireflow.services.jobs import slugify


def test_slugify():
    assert slugify("Senior Python Engineer!") == "senior-python-engineer"
    assert slugify("  C++ / Rust Dev  ") == "c-rust-dev"


async def test_create_job_cleans_skills(client, job_payload):
    res = await client.post("/api/jobs", json=job_payload)
    assert res.status_code == 201
    body = res.json()
    assert body["must_have"] == ["Python", "FastAPI"]  # trimmed, empty and duplicate removed
    assert body["status"] == "draft"
    assert body["slug"].startswith("senior-python-engineer-")


async def test_create_job_validation_error(client, job_payload):
    job_payload["title"] = "x"
    res = await client.post("/api/jobs", json=job_payload)
    assert res.status_code == 422


async def test_get_update_delete(client, job_payload):
    job_id = (await client.post("/api/jobs", json=job_payload)).json()["id"]

    res = await client.patch(f"/api/jobs/{job_id}", json={"status": "open"})
    assert res.status_code == 200
    assert res.json()["status"] == "open"
    assert res.json()["title"] == "Senior Python Engineer"  # unchanged fields stay

    res = await client.get("/api/jobs", params={"status": "open"})
    assert [j["id"] for j in res.json()] == [job_id]

    assert (await client.delete(f"/api/jobs/{job_id}")).status_code == 204
    res = await client.get(f"/api/jobs/{job_id}")
    assert res.status_code == 404
    assert res.json()["detail"] == f"Job {job_id} not found"