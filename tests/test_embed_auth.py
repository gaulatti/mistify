"""gaulatti/mistify#62: /embed is reachable across hosts only with EMBED_BEARER_TOKEN.

The route answers 503 when no token is configured, 401 without the right credential, and
200 with it. Each result is counted with a bounded label, parsed from the real collector.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from prometheus_client import generate_latest

from src.endpoints import embedding

TOKEN = "fictional-embed-token-0123456789"


class _Row:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _Vectors:
    def __init__(self, rows):
        self._rows = rows
        self.ndim = 2
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index):
        return _Row(self._rows[index])


def build(token=TOKEN):
    state = SimpleNamespace(
        config={"EMBED_BEARER_TOKEN": token},
        embedder=object(),
        embedding_lock=asyncio.Lock(),
        embedding_pool=ThreadPoolExecutor(max_workers=1),
    )
    app = FastAPI()

    @app.middleware("http")
    async def attach(request, call_next):
        request.state.app_state = state
        return await call_next(request)

    app.include_router(embedding.router)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return state, client


@pytest.fixture
def vectors(monkeypatch):
    monkeypatch.setattr(
        "src.endpoints.embedding._embed_sync",
        lambda _embedder, _texts, _batch, _normalize: _Vectors([[0.1, 0.2, 0.3]]),
    )


async def post(client, headers=None):
    return await client.post("/embed", json={"id": 1, "content": "Headline"}, headers=headers or {})


def counts():
    result = {}
    for line in generate_latest().decode().splitlines():
        if line.startswith("mistify_embed_requests_total{"):
            label = line.split('outcome="', 1)[1].split('"', 1)[0]
            result[label] = float(line.rsplit(" ", 1)[1])
    return result


@pytest.mark.asyncio
async def test_no_header_is_refused_with_a_bearer_challenge(vectors):
    _, client = build()
    response = await post(client)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_a_wrong_token_is_refused(vectors):
    _, client = build()
    response = await post(client, {"Authorization": "Bearer not-the-token"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_a_non_ascii_header_is_refused_not_a_server_error(vectors):
    _, client = build()
    # httpx refuses a non-ASCII str header, so the raw bytes are sent (the server still sees a non-ASCII value).
    response = await post(client, {"Authorization": "Bearer tökén-ü".encode("utf-8")})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_the_right_token_embeds_the_item(vectors):
    _, client = build()
    response = await post(client, {"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200
    assert response.json()["embeddings"] == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.asyncio
async def test_an_unset_token_turns_the_cross_host_route_off(vectors):
    _, client = build(token=None)
    response = await post(client, {"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 503
    assert response.json()["detail"] == "embed route not configured"


@pytest.mark.asyncio
async def test_the_result_counter_is_bounded_and_exposed(vectors):
    before = counts()
    _, client = build()
    await post(client)
    await post(client, {"Authorization": "Bearer wrong"})
    await post(client, {"Authorization": f"Bearer {TOKEN}"})
    _, unset = build(token=None)
    await post(unset, {"Authorization": f"Bearer {TOKEN}"})
    after = counts()
    assert after.get("unauthorized", 0) - before.get("unauthorized", 0) == 2
    assert after.get("ok", 0) - before.get("ok", 0) == 1
    assert after.get("unconfigured", 0) - before.get("unconfigured", 0) == 1
    assert set(after) <= {"ok", "unauthorized", "unconfigured", "failed"}, "labels stay bounded"


@pytest.mark.asyncio
async def test_the_token_never_appears_in_a_response(vectors):
    _, client = build()
    response = await post(client, {"Authorization": "Bearer wrong"})
    assert TOKEN not in response.text
