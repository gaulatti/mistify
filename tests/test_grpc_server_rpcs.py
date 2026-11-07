"""Covers each RPC of `src/grpc/server.py`.

Every enqueue RPC turns a protobuf request into an `OperationEnvelope`, enqueues
it, records a bounded queue event, and answers with the operation id and whether
it was newly queued. These tests drive the real servicer with real protobuf
messages and a fake queue, and assert the envelope the queue receives.
"""

from types import SimpleNamespace

import pytest

from src import metrics
from src.grpc import server as grpc_server
from src.grpc.mistify import operations_pb2, operations_pb2_grpc
from src.grpc.server import MistifyOperationsService, start_grpc_server


class FakeQueue:
    def __init__(self, queued=True, error=None):
        self.queued = queued
        self.error = error
        self.envelopes = []

    async def enqueue(self, envelope):
        self.envelopes.append(envelope)
        if self.error is not None:
            raise self.error
        return self.queued


@pytest.fixture
def events(monkeypatch):
    recorded = []
    monkeypatch.setattr(metrics, "record_queue_event", lambda op, outcome: recorded.append((op, outcome)))
    return recorded


def service(queue=None, app_state=None):
    return MistifyOperationsService(queue or FakeQueue(), app_state or SimpleNamespace())


def context_message():
    return operations_pb2.OperationContext(
        service="monitor", tenant="acme", request_id="req-1", trace_id="trace-1"
    )


# --- AnalyzePosts -----------------------------------------------------------

def analyze_request(**fields):
    request = operations_pb2.AnalyzePostsRequest(context=context_message(), **fields)
    first = request.items.add(idempotency_key="key-1")
    first.post.update({"id": 1, "content": "first"})
    second = request.items.add()
    second.post.update({"id": 2, "content": "second"})
    return request


@pytest.mark.asyncio
async def test_analyze_posts_enqueues_one_envelope_with_every_item(events):
    queue = FakeQueue()
    request = analyze_request()
    request.metadata.update({"batch": "b1"})
    request.callback.url = "https://monitor.example/cb"
    request.callback.headers["x-token"] = "t"
    request.grpc_callback.target = "monitor:50051"
    request.grpc_callback.service = "svc.Ingest"
    request.grpc_callback.method = "Receive"

    response = await service(queue).AnalyzePosts(request, None)

    envelope = queue.envelopes[0]
    assert envelope.operation_type == "analyze_posts"
    assert envelope.payload == {
        "items": [
            {"post": {"id": 1.0, "content": "first"}, "idempotency_key": "key-1"},
            {"post": {"id": 2.0, "content": "second"}, "idempotency_key": None},
        ]
    }
    assert envelope.metadata == {"batch": "b1"}
    assert envelope.context.model_dump() == {
        "service": "monitor", "tenant": "acme", "request_id": "req-1", "trace_id": "trace-1",
    }
    assert (envelope.callback.url, envelope.callback.headers) == ("https://monitor.example/cb", {"x-token": "t"})
    assert (envelope.grpc_callback.target, envelope.grpc_callback.service, envelope.grpc_callback.method) == (
        "monitor:50051", "svc.Ingest", "Receive",
    )
    assert response.operation_id == envelope.operation_id and response.queued is True
    assert events == [("analyze_posts", "enqueued")]


@pytest.mark.asyncio
async def test_analyze_posts_forwards_force_and_labels_only_when_set(events):
    queue = FakeQueue()
    await service(queue).AnalyzePosts(analyze_request(), None)
    await service(queue).AnalyzePosts(
        analyze_request(force=True, classification_labels=["sports", "politics"]), None
    )

    plain, forced = queue.envelopes
    assert "force" not in plain.payload and "classification_labels" not in plain.payload
    assert forced.payload["force"] is True
    assert forced.payload["classification_labels"] == ["sports", "politics"]


@pytest.mark.asyncio
async def test_analyze_posts_without_callbacks_leaves_them_unset(events):
    queue = FakeQueue()
    await service(queue).AnalyzePosts(analyze_request(), None)

    envelope = queue.envelopes[0]
    assert envelope.callback is None and envelope.grpc_callback is None


@pytest.mark.asyncio
async def test_a_duplicate_analysis_is_reported_as_not_queued(events):
    response = await service(FakeQueue(queued=False)).AnalyzePosts(analyze_request(), None)

    assert response.queued is False
    assert events == [("analyze_posts", "duplicate")]


@pytest.mark.asyncio
async def test_an_enqueue_failure_is_counted_and_propagated(events):
    with pytest.raises(RuntimeError, match="valkey down"):
        await service(FakeQueue(error=RuntimeError("valkey down"))).AnalyzePosts(analyze_request(), None)

    assert events == [("analyze_posts", "error")]


# --- the simple enqueue RPCs ------------------------------------------------

@pytest.mark.asyncio
async def test_detect_language_defaults_k_to_one(events):
    queue = FakeQueue()
    await service(queue).DetectLanguage(operations_pb2.DetectLanguageRequest(text="hola"), None)
    await service(queue).DetectLanguage(operations_pb2.DetectLanguageRequest(text="hola", k=3), None)

    assert [e.payload for e in queue.envelopes] == [
        {"text": "hola", "k": 1}, {"text": "hola", "k": 3},
    ]
    assert {e.operation_type for e in queue.envelopes} == {"detect_language"}


@pytest.mark.asyncio
async def test_classify_content_forwards_labels(events):
    queue = FakeQueue()
    request = operations_pb2.ClassifyContentRequest(text="match", labels=["sports", "misc"])

    await service(queue).ClassifyContent(request, None)

    assert queue.envelopes[0].operation_type == "classify_content"
    assert queue.envelopes[0].payload == {"text": "match", "labels": ["sports", "misc"]}


@pytest.mark.asyncio
async def test_translate_text_applies_language_defaults(events):
    queue = FakeQueue()
    await service(queue).TranslateText(operations_pb2.TranslateTextRequest(text="hola"), None)
    await service(queue).TranslateText(
        operations_pb2.TranslateTextRequest(text="hola", source_language="spa", target_language="fra"), None
    )

    assert [e.payload for e in queue.envelopes] == [
        {"text": "hola", "source_language": None, "target_language": "eng"},
        {"text": "hola", "source_language": "spa", "target_language": "fra"},
    ]


@pytest.mark.asyncio
async def test_embed_text_forwards_the_content(events):
    queue = FakeQueue()

    await service(queue).EmbedText(operations_pb2.EmbedTextRequest(content="story"), None)

    assert queue.envelopes[0].operation_type == "embed_text"
    assert queue.envelopes[0].payload == {"content": "story"}


@pytest.mark.asyncio
async def test_cluster_post_forwards_the_post_struct(events):
    queue = FakeQueue()
    request = operations_pb2.ClusterPostRequest()
    request.post.update({"content": "story", "hash": "h"})

    await service(queue).ClusterPost(request, None)

    assert queue.envelopes[0].operation_type == "cluster_post"
    assert queue.envelopes[0].payload == {"content": "story", "hash": "h"}


@pytest.mark.asyncio
async def test_idempotency_key_context_metadata_and_callbacks_travel_with_the_envelope(events):
    queue = FakeQueue()
    request = operations_pb2.EmbedTextRequest(
        content="x", idempotency_key="idem-9", context=context_message()
    )
    request.metadata.update({"source": "scout"})
    request.callback.url = "https://cb.example/"

    await service(queue).EmbedText(request, None)

    envelope = queue.envelopes[0]
    assert envelope.idempotency_key == "idem-9"
    assert envelope.context.tenant == "acme"
    assert envelope.metadata == {"source": "scout"}
    assert envelope.callback.url == "https://cb.example/"
    assert envelope.grpc_callback is None  # these request types carry no gRPC callback


@pytest.mark.asyncio
async def test_a_missing_idempotency_key_becomes_none(events):
    queue = FakeQueue()
    await service(queue).EmbedText(operations_pb2.EmbedTextRequest(content="x"), None)

    assert queue.envelopes[0].idempotency_key is None


@pytest.mark.asyncio
async def test_cluster_post_forwards_a_grpc_callback(events):
    queue = FakeQueue()
    request = operations_pb2.ClusterPostRequest()
    request.grpc_callback.target = "monitor:50051"

    await service(queue).ClusterPost(request, None)

    callback = queue.envelopes[0].grpc_callback
    assert callback.target == "monitor:50051"
    assert callback.service == ""  # explicit empty values are forwarded, not defaulted


@pytest.mark.asyncio
async def test_duplicate_and_failed_enqueues_are_reported_for_the_simple_rpcs(events):
    duplicate = await service(FakeQueue(queued=False)).EmbedText(
        operations_pb2.EmbedTextRequest(content="x"), None
    )
    with pytest.raises(ValueError, match="boom"):
        await service(FakeQueue(error=ValueError("boom"))).DetectLanguage(
            operations_pb2.DetectLanguageRequest(text="x"), None
        )

    assert duplicate.queued is False
    assert events == [("embed_text", "duplicate"), ("detect_language", "error")]


# --- scout RPCs -------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_scout_queries_maps_request_and_response(monkeypatch):
    calls = []

    async def fake_generate(app_state, title, source_language, max_queries):
        calls.append((app_state, title, source_language, max_queries))
        return ["query one", "query two"], "Translated title", "spa"

    monkeypatch.setattr(grpc_server, "generate_scout_queries", fake_generate)
    state = SimpleNamespace()

    response = await service(app_state=state).GenerateScoutQueries(
        operations_pb2.ScoutQueryRequest(title="Titular", source_language="spa", max_queries=4), None
    )

    assert calls == [(state, "Titular", "spa", 4)]
    assert list(response.queries) == ["query one", "query two"]
    assert (response.translated_title, response.source_language) == ("Translated title", "spa")


@pytest.mark.asyncio
async def test_generate_scout_queries_defaults(monkeypatch):
    calls = []

    async def fake_generate(app_state, title, source_language, max_queries):
        calls.append((source_language, max_queries))
        return [], "", ""

    monkeypatch.setattr(grpc_server, "generate_scout_queries", fake_generate)

    await service().GenerateScoutQueries(operations_pb2.ScoutQueryRequest(title="t"), None)

    assert calls == [(None, 2)]


@pytest.mark.asyncio
async def test_rank_scout_candidates_maps_candidates_and_scores(monkeypatch):
    calls = []

    async def fake_rank(app_state, seed_text, candidates, min_score, max_candidates):
        calls.append((seed_text, candidates, min_score, max_candidates))
        return [("b", 0.9), ("a", 0.75)]

    monkeypatch.setattr(grpc_server, "rank_scout_candidates", fake_rank)
    request = operations_pb2.ScoutRankRequest(seed_text="seed", min_score=0.5, max_candidates=3)
    request.candidates.add(id="a", title="Title A")
    request.candidates.add(id="b", title="Title B")

    response = await service().RankScoutCandidates(request, None)

    assert calls == [("seed", [("a", "Title A"), ("b", "Title B")], 0.5, 3)]
    assert [(c.id, c.score) for c in response.candidates] == [("b", pytest.approx(0.9)), ("a", pytest.approx(0.75))]


@pytest.mark.asyncio
async def test_rank_scout_candidates_defaults(monkeypatch):
    calls = []

    async def fake_rank(app_state, seed_text, candidates, min_score, max_candidates):
        calls.append((min_score, max_candidates))
        return []

    monkeypatch.setattr(grpc_server, "rank_scout_candidates", fake_rank)

    response = await service().RankScoutCandidates(operations_pb2.ScoutRankRequest(seed_text="s"), None)

    assert calls == [(0.55, 15)]
    assert list(response.candidates) == []


# --- helpers ----------------------------------------------------------------

def test_callbacks_without_a_destination_are_dropped():
    svc = service()

    assert svc._callback(None) is None
    assert svc._callback(operations_pb2.HttpCallback()) is None
    assert svc._callback(operations_pb2.HttpCallback(headers={"x": "y"})) is None
    assert svc._grpc_callback(None) is None
    assert svc._grpc_callback(operations_pb2.GrpcCallback(service="s")) is None


# --- server start-up --------------------------------------------------------

@pytest.mark.asyncio
async def test_start_grpc_server_registers_the_servicer_and_listens(monkeypatch):
    class FakeServer:
        def __init__(self):
            self.ports, self.started = [], False

        def add_insecure_port(self, address):
            self.ports.append(address)

        async def start(self):
            self.started = True

    fake = FakeServer()
    registered = []
    monkeypatch.setattr(grpc_server.grpc.aio, "server", lambda: fake)
    monkeypatch.setattr(
        operations_pb2_grpc,
        "add_MistifyOperationsServicer_to_server",
        lambda servicer, server: registered.append((servicer, server)),
    )
    monkeypatch.setattr(grpc_server, "GRPC_PORT", 51234)
    queue, state = FakeQueue(), SimpleNamespace()

    result = await start_grpc_server(queue, state)

    assert result is fake and fake.started is True
    assert fake.ports == ["[::]:51234"]
    servicer, server = registered[0]
    assert server is fake
    assert isinstance(servicer, MistifyOperationsService)
    assert (servicer.operation_queue, servicer.app_state) == (queue, state)
