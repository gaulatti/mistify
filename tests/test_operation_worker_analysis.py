from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.operations.models import GrpcCallback, OperationEnvelope, QueuedOperation
from src.operations.worker import OperationWorker
from src.operations.queue import OperationQueue


@pytest.mark.asyncio
async def test_analyze_posts_uses_hash_when_monitor_payload_has_no_id(monkeypatch):
    captured = {}

    async def fake_analysis(request, http_request):
        captured["item"] = request.items[0]
        return SimpleNamespace(results=[])

    monkeypatch.setattr("src.operations.worker.analysis.unified_analysis", fake_analysis)
    app_state = SimpleNamespace(
        config={"PROCESSING_TRANSLATE_TO_ENGLISH": True},
    )
    worker = OperationWorker(SimpleNamespace(), app_state)
    envelope = OperationEnvelope(
        operation_type="analyze_posts",
        payload={
            "items": [
                {
                    "post": {
                        "source": "youtube",
                        "uri": "https://youtube.example/video",
                        "content": "Video title",
                        "createdAt": "2026-08-06T00:00:00Z",
                        "hash": "video-hash",
                    }
                }
            ]
        },
    )

    await worker._analyze_posts(envelope)

    assert captured["item"].id == "video-hash"


@pytest.mark.asyncio
async def test_forced_analysis_bypasses_idempotency_filter():
    redis = SimpleNamespace(rpush=AsyncMock())
    queue = OperationQueue(redis)
    envelope = OperationEnvelope(
        operation_type="analyze_posts",
        payload={"force": True, "items": [{"idempotency_key": "same-hash"}]},
    )

    queued = await queue._enqueue_analyze_posts(envelope, "serialized")

    assert queued is True
    redis.rpush.assert_awaited_once_with(queue.queue_name, "serialized")


@pytest.mark.asyncio
async def test_callback_delivery_retries_temporary_failure(monkeypatch):
    worker = OperationWorker(SimpleNamespace(), SimpleNamespace())
    worker._deliver_callback = AsyncMock(
        side_effect=[RuntimeError("monitor restarting"), None]
    )
    monkeypatch.setattr("src.operations.worker.asyncio.sleep", AsyncMock())
    envelope = OperationEnvelope(operation_type="analyze_posts", payload={})

    await worker._deliver_callback_safely(envelope, "succeeded", result={})

    assert worker._deliver_callback.await_count == 2


@pytest.mark.asyncio
async def test_worker_coalesces_compatible_analysis_operations(monkeypatch):
    callback = GrpcCallback(target="monitor:50055")
    first = QueuedOperation(envelope=OperationEnvelope(
        operation_type="analyze_posts",
        payload={"items": [{"post": {"id": "one"}}]},
        grpc_callback=callback,
    ))
    second = QueuedOperation(envelope=OperationEnvelope(
        operation_type="analyze_posts",
        payload={"items": [{"post": {"id": "two"}}]},
        grpc_callback=callback,
    ))
    queue = SimpleNamespace(
        dequeue_nowait=AsyncMock(side_effect=[second, None]),
        requeue_next=AsyncMock(),
    )
    worker = OperationWorker(queue, SimpleNamespace())

    combined = await worker._coalesce_analysis(first)

    assert [item["post"]["id"] for item in combined.envelope.payload["items"]] == [
        "one",
        "two",
    ]
    queue.requeue_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_does_not_coalesce_forced_analysis():
    queued = QueuedOperation(envelope=OperationEnvelope(
        operation_type="analyze_posts",
        payload={"force": True, "items": [{"post": {"id": "one"}}]},
    ))
    queue = SimpleNamespace(dequeue_nowait=AsyncMock())
    worker = OperationWorker(queue, SimpleNamespace())

    assert await worker._coalesce_analysis(queued) is queued
    queue.dequeue_nowait.assert_not_awaited()
