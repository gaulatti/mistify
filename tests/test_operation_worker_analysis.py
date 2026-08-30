from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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

    delivered = await worker._deliver_callback_safely(
        envelope,
        "succeeded",
        result={},
    )

    assert delivered is True
    assert worker._deliver_callback.await_count == 2


@pytest.mark.asyncio
async def test_callback_exhaustion_keeps_operation_for_retry(monkeypatch):
    queue = SimpleNamespace(
        acknowledge=AsyncMock(),
        requeue_later=AsyncMock(),
    )
    worker = OperationWorker(queue, SimpleNamespace())
    worker._run_operation = AsyncMock(return_value={"items": []})
    worker._deliver_callback = AsyncMock(side_effect=RuntimeError("monitor down"))
    monkeypatch.setattr("src.operations.worker.asyncio.sleep", AsyncMock())
    queued = QueuedOperation(
        envelope=OperationEnvelope(operation_type="analyze_posts", payload={}),
        receipts=["durable-receipt"],
    )

    await worker.process(queued)

    assert worker._deliver_callback.await_count == 6
    queue.requeue_later.assert_awaited_once_with(queued)
    queue.acknowledge.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_callback_acknowledges_operation():
    queue = SimpleNamespace(
        acknowledge=AsyncMock(),
        requeue_later=AsyncMock(),
    )
    worker = OperationWorker(queue, SimpleNamespace())
    worker._run_operation = AsyncMock(return_value={"items": []})
    worker._deliver_callback = AsyncMock(return_value=None)
    queued = QueuedOperation(
        envelope=OperationEnvelope(operation_type="analyze_posts", payload={}),
        receipts=["durable-receipt"],
    )

    await worker.process(queued)

    queue.acknowledge.assert_awaited_once_with(queued)
    queue.requeue_later.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_coalesces_compatible_analysis_operations(monkeypatch):
    callback = GrpcCallback(target="monitor:50055")
    first = QueuedOperation(envelope=OperationEnvelope(
        operation_type="analyze_posts",
        payload={"items": [{"post": {"id": "one"}}]},
        grpc_callback=callback,
    ), receipts=["first"])
    second = QueuedOperation(envelope=OperationEnvelope(
        operation_type="analyze_posts",
        payload={"items": [{"post": {"id": "two"}}]},
        grpc_callback=callback,
    ), receipts=["second"])
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
    assert combined.receipts == ["first", "second"]
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


@pytest.mark.asyncio
async def test_youtube_clustering_uses_priority_end_of_queue():
    redis = SimpleNamespace(rpush=AsyncMock())
    queue = OperationQueue(redis)
    envelope = OperationEnvelope(
        operation_type="cluster_post",
        payload={"source": "youtube", "content": "Video"},
    )

    assert await queue.enqueue(envelope) is True
    redis.rpush.assert_awaited_once()


@pytest.mark.asyncio
async def test_queue_claims_atomically_and_acknowledges_after_delivery():
    raw = QueuedOperation(
        envelope=OperationEnvelope(operation_type="analyze_posts", payload={})
    ).model_dump_json()
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[1])
    redis = SimpleNamespace(
        brpoplpush=AsyncMock(return_value=raw),
        pipeline=MagicMock(return_value=pipe),
    )
    queue = OperationQueue(redis)

    claimed = await queue.dequeue(timeout_seconds=1)
    assert claimed is not None
    assert claimed.receipts == [raw]
    redis.brpoplpush.assert_awaited_once_with(
        queue.queue_name,
        queue.processing_queue_name,
        timeout=1,
    )

    await queue.acknowledge(claimed)
    pipe.lrem.assert_called_once_with(queue.processing_queue_name, 1, raw)
    pipe.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_recover_inflight_returns_every_claim_to_pending_queue():
    redis = SimpleNamespace(
        rpoplpush=AsyncMock(side_effect=["one", "two", None]),
    )
    queue = OperationQueue(redis)

    assert await queue.recover_inflight() == 2
    assert redis.rpoplpush.await_count == 3
