"""Covers `src/operations/worker.py`'s dispatch, callback and payload handling.

The existing worker tests cover analysis coalescing and retry exhaustion. These
cover operation dispatch, each endpoint delegation, callback channel selection,
error formatting, and the protobuf item builder.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import grpc
import httpx
import pytest

from src.operations.models import (
    GrpcCallback,
    HttpCallback,
    OperationEnvelope,
    QueuedOperation,
)
from src.operations.worker import CALLBACK_MAX_ATTEMPTS, OperationWorker


class _Stop:
    """Sentinel that makes run_forever exit via cancellation."""


def app_state(**overrides):
    config = {"PROCESSING_TRANSLATE_TO_ENGLISH": True}
    config.update(overrides.pop("config", {}))
    return SimpleNamespace(config=config, **overrides)


CLUSTER_PAYLOAD = {
    "id": 1,
    "uuid": "11111111-1111-1111-1111-111111111111",
    "content": "text",
    "source": "rss",
    "createdAt": "2026-08-29T12:00:00Z",
    "hash": "h1",
}


def worker(queue=None, state=None):
    return OperationWorker(queue or SimpleNamespace(), state or app_state())


def envelope(operation_type, payload=None, **kwargs):
    return OperationEnvelope(
        operation_type=operation_type, payload=payload or {}, **kwargs
    )


# --- _run_operation dispatch ------------------------------------------------

@pytest.mark.asyncio
async def test_each_operation_type_reaches_its_own_handler(monkeypatch):
    instance = worker()
    called = []

    for name in (
        "_analyze_posts",
        "_detect_language",
        "_classify_content",
        "_translate_text",
        "_embed_text",
        "_cluster_post",
    ):
        async def handler(_envelope, _name=name):
            called.append(_name)
            return {}

        monkeypatch.setattr(instance, name, handler)

    for operation_type in (
        "analyze_posts",
        "detect_language",
        "classify_content",
        "translate_text",
        "embed_text",
        "cluster_post",
    ):
        await instance._run_operation(envelope(operation_type))

    assert called == [
        "_analyze_posts",
        "_detect_language",
        "_classify_content",
        "_translate_text",
        "_embed_text",
        "_cluster_post",
    ]


@pytest.mark.asyncio
async def test_an_unsupported_operation_type_is_rejected():
    with pytest.raises(ValueError, match="Unsupported operation_type"):
        await worker()._run_operation(envelope("teleport"))


# --- endpoint delegation ----------------------------------------------------

@pytest.mark.asyncio
async def test_detect_language_delegates_and_returns_a_plain_dict(monkeypatch):
    async def fake_detect(request, _http_request):
        assert request.text == "hola"
        return SimpleNamespace(model_dump=lambda: {"languages": ["es"]})

    monkeypatch.setattr("src.operations.worker.language.detect_language", fake_detect)

    result = await worker()._detect_language(envelope("detect_language", {"text": "hola", "k": 1}))
    assert result == {"languages": ["es"]}


@pytest.mark.asyncio
async def test_classify_content_delegates(monkeypatch):
    async def fake_classify(request, _http_request):
        assert request.labels == ["a"]
        return SimpleNamespace(model_dump=lambda: {"labels": ["a"]})

    monkeypatch.setattr("src.operations.worker.classification.classify_content", fake_classify)

    result = await worker()._classify_content(
        envelope("classify_content", {"text": "t", "labels": ["a"]})
    )
    assert result == {"labels": ["a"]}


@pytest.mark.asyncio
async def test_translate_text_delegates(monkeypatch):
    async def fake_translate(request, _http_request):
        assert request.source_language == "es"
        return SimpleNamespace(model_dump=lambda: {"translated_text": "hi"})

    monkeypatch.setattr("src.operations.worker.translation.translate_text", fake_translate)

    result = await worker()._translate_text(
        envelope("translate_text", {"text": "hola", "source_language": "es"})
    )
    assert result == {"translated_text": "hi"}


@pytest.mark.asyncio
async def test_embed_text_returns_the_endpoint_result_unchanged(monkeypatch):
    async def fake_embed(_item, _http_request):
        return {"embedding": [0.1, 0.2]}

    monkeypatch.setattr("src.operations.worker.embedding.embed_items", fake_embed)

    result = await worker()._embed_text(envelope("embed_text", {"content": "text"}))
    assert result == {"embedding": [0.1, 0.2]}


@pytest.mark.asyncio
async def test_cluster_post_wraps_its_result_as_a_single_item(monkeypatch):
    async def fake_cluster(_post, _http_request):
        return SimpleNamespace(model_dump=lambda: {"cluster": 3})

    monkeypatch.setattr("src.operations.worker.clustering.cluster_texts", fake_cluster)

    result = await worker()._cluster_post(
        envelope(
            "cluster_post",
            CLUSTER_PAYLOAD,
            idempotency_key="key-1",
        )
    )

    assert result == {"items": [{"post": {"cluster": 3}, "idempotency_key": "key-1"}]}


@pytest.mark.asyncio
async def test_cluster_post_tolerates_a_missing_idempotency_key(monkeypatch):
    async def fake_cluster(_post, _http_request):
        return SimpleNamespace(model_dump=lambda: {"cluster": 3})

    monkeypatch.setattr("src.operations.worker.clustering.cluster_texts", fake_cluster)

    result = await worker()._cluster_post(
        envelope("cluster_post", CLUSTER_PAYLOAD)
    )
    assert result["items"][0]["idempotency_key"] == ""


# --- analyze payload shaping ------------------------------------------------

@pytest.mark.asyncio
async def test_a_string_author_is_promoted_to_an_object(monkeypatch):
    captured = {}

    async def fake_analysis(request, _http_request):
        captured["item"] = request.items[0]
        return SimpleNamespace(results=[])

    monkeypatch.setattr("src.operations.worker.analysis.unified_analysis", fake_analysis)

    await worker()._analyze_posts(envelope("analyze_posts", {
        "items": [{"post": {
            "source": "rss", "uri": "https://e.test/a", "content": "Headline",
            "createdAt": "2026-08-29T12:00:00Z", "hash": "h1", "author": "Jane Doe",
        }}],
    }))

    author = captured["item"].author
    name = author.get("name") if isinstance(author, dict) else author.name
    assert name == "Jane Doe", (
        "Monitor sends author as a bare string; it must be promoted to an object "
        f"before analysis, got {author!r}"
    )


@pytest.mark.asyncio
async def test_an_invalid_analyze_payload_is_rejected_with_context(monkeypatch):
    monkeypatch.setattr(
        "src.operations.worker.analysis.unified_analysis",
        AsyncMock(side_effect=AssertionError("must not be reached")),
    )

    with pytest.raises(ValueError, match="Invalid analyze payload"):
        await worker()._analyze_posts(
            envelope("analyze_posts", {"items": [{"post": {"content": 42}}]})
        )


def test_extract_items_accepts_every_payload_shape():
    instance = worker()

    assert instance._extract_items({"items": [{"a": 1}]}) == [{"a": 1}]
    assert instance._extract_items({"input": [{"b": 2}]}) == [{"b": 2}]
    assert instance._extract_items({"input": {"c": 3}}) == [{"c": 3}]
    # A bare payload is treated as a single item.
    assert instance._extract_items({"d": 4}) == [{"d": 4}]


# --- callback channel selection --------------------------------------------

@pytest.mark.asyncio
async def test_a_grpc_callback_takes_precedence_over_http(monkeypatch):
    instance = worker()
    calls = []
    monkeypatch.setattr(instance, "_deliver_grpc_callback",
                        AsyncMock(side_effect=lambda *a: calls.append("grpc")))
    monkeypatch.setattr(instance, "_deliver_http_callback",
                        AsyncMock(side_effect=lambda *a: calls.append("http")))

    await instance._deliver_callback(
        envelope("detect_language",
                 grpc_callback=GrpcCallback(target="monitor:50055"),
                 callback=HttpCallback(url="https://e.test/cb")),
        "succeeded",
    )

    assert calls == ["grpc"]


@pytest.mark.asyncio
async def test_an_http_callback_is_used_when_no_grpc_callback_is_set(monkeypatch):
    instance = worker()
    calls = []
    monkeypatch.setattr(instance, "_deliver_http_callback",
                        AsyncMock(side_effect=lambda *a: calls.append("http")))

    await instance._deliver_callback(
        envelope("detect_language", callback=HttpCallback(url="https://e.test/cb")),
        "succeeded",
    )
    assert calls == ["http"]


@pytest.mark.asyncio
async def test_no_callback_configured_is_reported_as_delivered(monkeypatch):
    # Nothing to deliver to is success, not failure: the operation must be acked.
    assert await worker()._deliver_callback_safely(envelope("detect_language"), "succeeded") is True


@pytest.mark.asyncio
async def test_callback_target_names_the_channel():
    instance = worker()

    assert instance._callback_target(
        envelope("x", grpc_callback=GrpcCallback(target="monitor:50055"))
    ) == "grpc://monitor:50055"
    assert instance._callback_target(
        envelope("x", callback=HttpCallback(url="https://e.test/cb"))
    ) == "https://e.test/cb"
    assert instance._callback_target(envelope("x")) == "none"


# --- error formatting -------------------------------------------------------

def test_callback_errors_are_formatted_per_transport():
    instance = worker()

    status_error = httpx.HTTPStatusError(
        "bad", request=httpx.Request("POST", "https://e.test/cb"),
        response=httpx.Response(503, text="upstream down"),
    )
    assert instance._format_callback_error(status_error) == "HTTP 503: upstream down"

    request_error = httpx.ConnectError("connection refused")
    assert instance._format_callback_error(request_error) == "connection refused"

    assert instance._format_callback_error(RuntimeError("something else")) == "something else"


def test_http_status_error_text_is_truncated():
    instance = worker()
    error = httpx.HTTPStatusError(
        "bad", request=httpx.Request("POST", "https://e.test/cb"),
        response=httpx.Response(500, text="x" * 500),
    )

    formatted = instance._format_callback_error(error)
    assert formatted.startswith("HTTP 500: ")
    assert len(formatted) < 500, "an upstream body must not flood the log line"


def test_grpc_errors_are_formatted_with_their_status_code():
    instance = worker()

    class FakeAioRpcError(grpc.aio.AioRpcError):
        def __init__(self):
            pass

        def code(self):
            return grpc.StatusCode.UNAVAILABLE

        def details(self):
            return "monitor is down"

    assert instance._format_callback_error(FakeAioRpcError()) == "UNAVAILABLE: monitor is down"


# --- _build_analysis_result_items ------------------------------------------

def test_result_items_are_built_from_wrapped_posts():
    instance = worker()

    items = instance._build_analysis_result_items({
        "items": [{"post": {"hash": "h1", "content": "text"}, "idempotency_key": "key-1"}],
    })

    assert len(items) == 1
    assert items[0].hash == "key-1", "the idempotency key is preferred over the post hash"
    assert items[0].payload["content"] == "text"


def test_result_items_fall_back_to_the_post_hash():
    instance = worker()

    items = instance._build_analysis_result_items({
        "items": [{"post": {"hash": "h1"}}],
    })
    assert items[0].hash == "h1"


def test_result_items_accept_an_unwrapped_post():
    instance = worker()

    items = instance._build_analysis_result_items({"items": [{"hash": "h1", "lang": "es"}]})

    assert items[0].hash == "h1"
    assert items[0].payload["lang"] == "es"


def test_result_items_skip_non_dict_entries():
    instance = worker()

    items = instance._build_analysis_result_items({"items": ["not a dict", 42, {"hash": "h1"}]})
    assert len(items) == 1


def test_result_items_are_empty_without_a_result():
    instance = worker()

    assert instance._build_analysis_result_items(None) == []
    assert instance._build_analysis_result_items({}) == []
    assert instance._build_analysis_result_items({"items": []}) == []


def test_result_items_tolerate_a_post_with_no_hash_at_all():
    instance = worker()

    items = instance._build_analysis_result_items({"items": [{"post": {"content": "t"}}]})
    assert items[0].hash == ""


# --- _merge_processed_items -------------------------------------------------

def _result(**fields):
    base = {
        "content": "merged content",
        "language_detection": None,
        "content_classification": None,
        "translation": None,
        "newsworthiness": None,
        "urgency": None,
        "embedding": None,
        "timings": None,
    }
    base.update(fields)
    return SimpleNamespace(**base)


def _dumpable(payload):
    return SimpleNamespace(model_dump=lambda: payload)


def test_merge_leaves_items_without_a_matching_result_untouched():
    instance = worker()

    merged = instance._merge_processed_items(
        [{"hash": "a"}, {"hash": "b"}], SimpleNamespace(results=[_result()])
    )

    assert merged[0]["content"] == "merged content"
    assert merged[1] == {"hash": "b"}, "an item with no result must pass through unchanged"


def test_merge_without_a_response_returns_the_raw_items():
    instance = worker()

    assert instance._merge_processed_items([{"hash": "a"}], None) == [{"hash": "a"}]


def test_merge_adopts_the_detected_language_only_when_absent():
    instance = worker()
    detection = _dumpable({"languages": ["es"], "probabilities": [0.99]})

    adopted = instance._merge_processed_items(
        [{"hash": "a"}], SimpleNamespace(results=[_result(language_detection=detection)])
    )
    assert adopted[0]["lang"] == "es"

    # An upstream-supplied language must win over detection.
    kept = instance._merge_processed_items(
        [{"hash": "a", "lang": "tr"}],
        SimpleNamespace(results=[_result(language_detection=detection)]),
    )
    assert kept[0]["lang"] == "tr"


def test_merge_ignores_a_detection_with_no_languages():
    instance = worker()
    detection = _dumpable({"languages": [], "probabilities": []})

    merged = instance._merge_processed_items(
        [{"hash": "a"}], SimpleNamespace(results=[_result(language_detection=detection)])
    )
    assert "lang" not in merged[0]


@pytest.mark.parametrize("label", ["error", "timeout", "uncertain"])
def test_merge_does_not_promote_a_non_answer_classification_label(label):
    instance = worker()
    classification = _dumpable({"label": label, "score": 0.1})

    merged = instance._merge_processed_items(
        [{"hash": "a"}], SimpleNamespace(results=[_result(content_classification=classification)])
    )

    assert "content_classification" in merged[0]
    assert "classification_labels" not in merged[0], (
        f"{label!r} is a non-answer and must not become a stored label"
    )


def test_merge_promotes_a_real_classification_label():
    instance = worker()
    classification = _dumpable({"label": "politics", "score": 0.9})

    merged = instance._merge_processed_items(
        [{"hash": "a"}], SimpleNamespace(results=[_result(content_classification=classification)])
    )
    assert merged[0]["classification_labels"] == ["politics"]


def test_merge_carries_every_optional_analysis_field():
    instance = worker()
    result = _result(
        translation=_dumpable({"translated_text": "hi"}),
        newsworthiness=7.5,
        urgency=3.0,
        embedding=[0.1, 0.2],
        timings=_dumpable({"total_ms": 12.0}),
    )

    merged = instance._merge_processed_items([{"hash": "a"}], SimpleNamespace(results=[result]))

    assert merged[0]["translation"] == {"translated_text": "hi"}
    assert merged[0]["newsworthiness"] == 7.5
    assert merged[0]["urgency"] == 3.0
    assert merged[0]["embeddings"] == [0.1, 0.2]
    assert merged[0]["analysis_timings"] == {"total_ms": 12.0}


def test_merge_distinguishes_a_zero_score_from_an_absent_one():
    instance = worker()

    merged = instance._merge_processed_items(
        [{"hash": "a"}], SimpleNamespace(results=[_result(newsworthiness=0.0, urgency=0.0)])
    )

    assert merged[0]["newsworthiness"] == 0.0, "zero is a score, not a missing value"
    assert merged[0]["urgency"] == 0.0


# --- HTTP callback delivery -------------------------------------------------

class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient inside the worker module."""

    captured = {}

    def __init__(self, timeout=None):
        _FakeAsyncClient.captured["timeout"] = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, url, headers=None, json=None):
        _FakeAsyncClient.captured.update({"url": url, "headers": headers, "json": json})
        return httpx.Response(
            _FakeAsyncClient.captured.get("status", 200),
            request=httpx.Request("POST", url),
        )


@pytest.mark.asyncio
async def test_http_callback_posts_the_operation_envelope(monkeypatch):
    _FakeAsyncClient.captured = {"status": 200}
    monkeypatch.setattr("src.operations.worker.httpx.AsyncClient", _FakeAsyncClient)

    await worker()._deliver_http_callback(
        envelope(
            "detect_language",
            {"text": "hola"},
            idempotency_key="key-1",
            callback=HttpCallback(url="https://e.test/cb", headers={"x-token": "t"}),
        ),
        "succeeded",
        {"languages": ["es"]},
        None,
    )

    captured = _FakeAsyncClient.captured
    assert captured["url"] == "https://e.test/cb"
    assert captured["headers"] == {"x-token": "t"}
    assert captured["json"]["status"] == "succeeded"
    assert captured["json"]["idempotencyKey"] == "key-1"
    assert captured["json"]["result"] == {"languages": ["es"]}
    assert captured["json"]["error"] is None
    assert "operationId" in captured["json"] and "operationType" in captured["json"]


@pytest.mark.asyncio
async def test_http_callback_raises_on_an_error_status(monkeypatch):
    _FakeAsyncClient.captured = {"status": 503}
    monkeypatch.setattr("src.operations.worker.httpx.AsyncClient", _FakeAsyncClient)

    with pytest.raises(httpx.HTTPStatusError):
        await worker()._deliver_http_callback(
            envelope("detect_language", callback=HttpCallback(url="https://e.test/cb")),
            "failed",
            None,
            "boom",
        )


# --- gRPC callback delivery -------------------------------------------------

class _FakeChannel:
    def __init__(self, target):
        _FakeGrpc.captured["target"] = target

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeStub:
    def __init__(self, _channel):
        pass

    async def ReceiveAnalysisResult(self, request, timeout=None):
        _FakeGrpc.captured.update({"request": request, "timeout": timeout})
        return SimpleNamespace(ok=_FakeGrpc.captured.get("ok", True))


class _FakeGrpc:
    captured = {}


@pytest.mark.asyncio
async def test_grpc_callback_sends_the_analysis_result(monkeypatch):
    _FakeGrpc.captured = {"ok": True}
    monkeypatch.setattr("src.operations.worker.grpc.aio.insecure_channel", _FakeChannel)
    monkeypatch.setattr(
        "src.operations.worker.operations_pb2_grpc.MonitorIngestStub", _FakeStub
    )

    await worker()._deliver_grpc_callback(
        envelope("analyze_posts", grpc_callback=GrpcCallback(target="monitor:50055")),
        "succeeded",
        {"items": [{"post": {"hash": "h1", "lang": "es"}, "idempotency_key": "key-1"}]},
        None,
    )

    captured = _FakeGrpc.captured
    assert captured["target"] == "monitor:50055"
    assert captured["request"].status == "succeeded"
    assert captured["request"].operation_type == "analyze_posts"
    assert captured["request"].error == "", "a success must send an empty error, not null"
    assert len(captured["request"].items) == 1
    assert captured["request"].items[0].hash == "key-1"
    assert captured["timeout"] is not None, "the callback must be bounded"


@pytest.mark.asyncio
async def test_grpc_callback_raises_when_monitor_rejects_it(monkeypatch):
    _FakeGrpc.captured = {"ok": False}
    monkeypatch.setattr("src.operations.worker.grpc.aio.insecure_channel", _FakeChannel)
    monkeypatch.setattr(
        "src.operations.worker.operations_pb2_grpc.MonitorIngestStub", _FakeStub
    )

    with pytest.raises(RuntimeError, match="Monitor rejected callback"):
        await worker()._deliver_grpc_callback(
            envelope("analyze_posts", grpc_callback=GrpcCallback(target="monitor:50055")),
            "succeeded",
            None,
            None,
        )


@pytest.mark.asyncio
async def test_grpc_callback_forwards_a_failure_reason(monkeypatch):
    _FakeGrpc.captured = {"ok": True}
    monkeypatch.setattr("src.operations.worker.grpc.aio.insecure_channel", _FakeChannel)
    monkeypatch.setattr(
        "src.operations.worker.operations_pb2_grpc.MonitorIngestStub", _FakeStub
    )

    await worker()._deliver_grpc_callback(
        envelope("analyze_posts", grpc_callback=GrpcCallback(target="monitor:50055")),
        "failed",
        None,
        "model unavailable",
    )

    assert _FakeGrpc.captured["request"].error == "model unavailable"


# --- run_forever ------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_forever_skips_idle_polls_and_processes_claims(monkeypatch):
    processed = []
    claims = [None, "claim", _Stop()]

    class Queue:
        queue_name = "mistify:operations"

        async def dequeue(self, timeout_seconds=None):
            value = claims.pop(0)
            if isinstance(value, _Stop):
                raise asyncio.CancelledError
            return value

        async def requeue_later(self, _queued):
            processed.append("requeued")

    instance = worker(Queue())
    monkeypatch.setattr(instance, "_coalesce_analysis", AsyncMock(side_effect=lambda q: q))
    monkeypatch.setattr(
        instance, "process", AsyncMock(side_effect=lambda q: processed.append(q))
    )

    with pytest.raises(asyncio.CancelledError):
        await instance.run_forever()

    assert processed == ["claim"], "an idle poll must be skipped, not treated as work"


@pytest.mark.asyncio
async def test_run_forever_requeues_a_claim_whose_processing_raised(monkeypatch):
    requeued = []
    claims = ["claim", _Stop()]

    class Queue:
        queue_name = "mistify:operations"

        async def dequeue(self, timeout_seconds=None):
            value = claims.pop(0)
            if isinstance(value, _Stop):
                raise asyncio.CancelledError
            return value

        async def requeue_later(self, queued):
            requeued.append(queued)

    instance = worker(Queue())
    monkeypatch.setattr(instance, "_coalesce_analysis", AsyncMock(side_effect=lambda q: q))
    monkeypatch.setattr(instance, "process", AsyncMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr("src.operations.worker.asyncio.sleep", AsyncMock())

    with pytest.raises(asyncio.CancelledError):
        await instance.run_forever()

    assert requeued == ["claim"], "a failed claim must go back to the queue, not vanish"


@pytest.mark.asyncio
async def test_run_forever_survives_a_queue_failure(monkeypatch):
    attempts = []

    class Queue:
        queue_name = "mistify:operations"

        async def dequeue(self, timeout_seconds=None):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("redis unavailable")
            raise asyncio.CancelledError

    slept = AsyncMock()
    monkeypatch.setattr("src.operations.worker.asyncio.sleep", slept)

    with pytest.raises(asyncio.CancelledError):
        await worker(Queue()).run_forever()

    assert len(attempts) == 2, "the loop must retry rather than exit on a queue error"
    slept.assert_awaited()
