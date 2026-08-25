import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

os.environ["LOAD_MODELS_ON_STARTUP"] = "false"
os.environ["METRICS_BEARER_TOKEN"] = "bounded-test-scrape-token"

from src import metrics  # noqa: E402
from src.grpc.mistify import operations_pb2  # noqa: E402
from src.grpc.server import MistifyOperationsService  # noqa: E402
from src.operations.models import (  # noqa: E402
    HttpCallback,
    OperationEnvelope,
    QueuedOperation,
)
from src.operations.worker import OperationWorker  # noqa: E402
from src.server import app, app_state  # noqa: E402


def _mistify_samples(payload: str):
    return [
        sample
        for family in text_string_to_metric_families(payload)
        if family.name.startswith("mistify_")
        for sample in family.samples
    ]


def test_metrics_token_is_optional_and_blank_values_are_disabled():
    assert metrics.normalize_metrics_token(None) is None
    assert metrics.normalize_metrics_token("") is None
    assert metrics.normalize_metrics_token("   ") is None
    assert (
        metrics.normalize_metrics_token(" bounded-test-scrape-token ")
        == "bounded-test-scrape-token"
    )


@pytest.mark.asyncio
async def test_metrics_endpoint_is_hidden_when_token_is_unconfigured(monkeypatch):
    monkeypatch.setitem(app_state.config, "METRICS_BEARER_TOKEN", None)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.get("/metrics")
        supplied = await client.get(
            "/metrics", headers={"Authorization": "Bearer arbitrary-token"}
        )

    assert missing.status_code == 404
    assert supplied.status_code == 404


@pytest.mark.asyncio
async def test_metrics_endpoint_requires_bearer_and_parses_real_exposition(monkeypatch):
    monkeypatch.setitem(
        app_state.config, "METRICS_BEARER_TOKEN", "bounded-test-scrape-token"
    )
    monkeypatch.setattr(app_state.operation_queue, "size", AsyncMock(return_value=4))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.get("/metrics")
        incorrect = await client.get(
            "/metrics", headers={"Authorization": "Bearer private-document-id"}
        )
        response = await client.get(
            "/metrics",
            headers={"Authorization": "Bearer bounded-test-scrape-token"},
        )

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert incorrect.status_code == 401
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/plain; version=0.0.4"
    )

    samples = _mistify_samples(response.text)
    assert samples
    assert any(
        sample.name == "mistify_operation_queue_depth" and sample.value == 4
        for sample in samples
    )
    assert any(
        sample.name == "mistify_http_requests_total"
        and sample.labels.get("route") == "/metrics"
        and sample.labels.get("status_code") == "401"
        for sample in samples
    )
    assert "bounded-test-scrape-token" not in response.text
    assert "private-document-id" not in response.text


@pytest.mark.asyncio
async def test_grpc_and_worker_success_failure_labels_are_bounded(monkeypatch):
    successful_queue = SimpleNamespace(enqueue=AsyncMock(return_value=True))
    successful_service = MistifyOperationsService(successful_queue, SimpleNamespace())
    request = operations_pb2.DetectLanguageRequest(
        text="private content that must never be a metric label"
    )
    await successful_service.DetectLanguage(request, None)

    failed_queue = SimpleNamespace(
        enqueue=AsyncMock(side_effect=RuntimeError("private queue error"))
    )
    failed_service = MistifyOperationsService(failed_queue, SimpleNamespace())
    with pytest.raises(RuntimeError, match="private queue error"):
        await failed_service.DetectLanguage(request, None)

    worker = OperationWorker(SimpleNamespace(), SimpleNamespace())
    worker._run_operation = AsyncMock(
        side_effect=[{"ok": True}, RuntimeError("private worker error")]
    )
    worker._deliver_callback_safely = AsyncMock()
    await worker.process(
        QueuedOperation(
            envelope=OperationEnvelope(operation_type="detect_language", payload={})
        )
    )
    await worker.process(
        QueuedOperation(
            envelope=OperationEnvelope(
                operation_type="private-unbounded-operation", payload={}
            )
        )
    )

    exposition = generate_latest().decode()
    samples = _mistify_samples(exposition)
    grpc_samples = [
        sample for sample in samples if sample.name == "mistify_grpc_requests_total"
    ]
    job_samples = [
        sample for sample in samples if sample.name == "mistify_operation_jobs_total"
    ]

    assert {sample.labels["outcome"] for sample in grpc_samples} >= {
        "success",
        "error",
    }
    assert {sample.labels["method"] for sample in grpc_samples} == {
        "DetectLanguage"
    }
    assert {sample.labels["outcome"] for sample in job_samples} >= {
        "success",
        "error",
    }
    assert "private-unbounded-operation" not in exposition
    assert "private queue error" not in exposition
    assert "private worker error" not in exposition
    assert "private content" not in exposition


@pytest.mark.asyncio
async def test_callback_dependency_records_retry_without_target_or_error(monkeypatch):
    worker = OperationWorker(SimpleNamespace(), SimpleNamespace())
    worker._deliver_callback = AsyncMock(
        side_effect=[RuntimeError("private callback error"), None]
    )
    monkeypatch.setattr("src.operations.worker.asyncio.sleep", AsyncMock())
    envelope = OperationEnvelope(
        operation_type="analyze_posts",
        payload={},
        callback=HttpCallback(url="https://private.example/callback"),
    )

    await worker._deliver_callback_safely(envelope, "succeeded", result={})

    exposition = generate_latest().decode()
    samples = _mistify_samples(exposition)
    callback_samples = [
        sample
        for sample in samples
        if sample.name == "mistify_callback_requests_total"
    ]
    retry_samples = [
        sample
        for sample in samples
        if sample.name == "mistify_operation_retries_total"
    ]

    assert {sample.labels["outcome"] for sample in callback_samples} >= {
        "success",
        "error",
    }
    assert {sample.labels["channel"] for sample in callback_samples} == {"http"}
    assert any(
        sample.labels == {"operation": "analyze_posts"} and sample.value >= 1
        for sample in retry_samples
    )
    assert "private.example" not in exposition
    assert "private callback error" not in exposition


def test_custom_metric_label_keys_and_gpu_aggregation_are_bounded():
    allowed_label_keys = {
        "channel",
        "device",
        "endpoint",
        "failure_type",
        "le",
        "method",
        "model",
        "operation",
        "outcome",
        "phase",
        "quantile",
        "route",
        "service",
        "status_code",
        "step",
        "version",
    }
    samples = _mistify_samples(generate_latest().decode())

    assert all(set(sample.labels) <= allowed_label_keys for sample in samples)
    gpu_samples = [
        sample
        for sample in samples
        if sample.name.startswith("mistify_gpu_")
    ]
    assert all("device_id" not in sample.labels for sample in gpu_samples)
