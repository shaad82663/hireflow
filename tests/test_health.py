async def test_health(client):
    res = await client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


async def test_health_db(client):
    res = await client.get("/health/db")
    assert res.status_code == 200
    assert res.json() == {"database": "ok"}