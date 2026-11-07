"""Covers `src/server.py`: configuration, lifecycle, middleware and system routes.

The module builds the process-wide `app` and `app_state` at import time, so the
import-time behaviours (eager model loading, detector selection, the `__main__`
entry point) are exercised by executing the file a second time under another
module name, with `initialize_models`, `uvicorn.run` and the model-heavy
collaborators replaced. The shared `app_state` used by other tests is not touched.
"""

import asyncio
import os
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

os.environ["LOAD_MODELS_ON_STARTUP"] = "false"
os.environ["METRICS_BEARER_TOKEN"] = "server-test-scrape-token"

from src import metrics  # noqa: E402
from src import server  # noqa: E402

SERVER_FILE = str(Path(server.__file__))


# --- import-time behaviour --------------------------------------------------

def execute_server(monkeypatch, tmp_path, run_name="src.server_probe", **env):
    """Run server.py again in a fresh namespace and return its globals."""
    import torch

    # The module sets torch's thread counts at import; torch refuses a second call.
    monkeypatch.setattr(torch, "set_num_threads", lambda *_a: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *_a: None)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("LOAD_MODELS_ON_STARTUP", env.pop("LOAD_MODELS_ON_STARTUP", "false"))
    for key in ("LANGUAGE_DETECTOR", "FASTTEXT_MODEL_PATH", "GLOTLID_MODEL_PATH"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    namespace = runpy.run_path(SERVER_FILE, run_name=run_name)
    state = namespace["app_state"]
    for pool in ("thread_pool", "classification_pool", "translation_pool", "embedding_pool", "clustering_pool"):
        getattr(state, pool).shutdown(wait=False)
    return namespace


def test_models_are_skipped_when_eager_loading_is_disabled(monkeypatch, tmp_path):
    state = execute_server(monkeypatch, tmp_path)["app_state"]

    assert state.fasttext_model is None and state.classifier is None
    assert state.translator is None and state.embedder is None and state.nlp is None
    assert state.translator_model_name == "none"


def test_eager_loading_installs_every_model_from_the_loader(monkeypatch, tmp_path):
    from src.helpers import models as models_module

    seen = []

    def fake_initialize(config):
        seen.append(config)
        return ("fasttext", "classifier", "translator", "embedder", "nlp", "seamless")

    monkeypatch.setattr(models_module, "initialize_models", fake_initialize)

    state = execute_server(monkeypatch, tmp_path, LOAD_MODELS_ON_STARTUP="true")["app_state"]

    assert (state.fasttext_model, state.classifier, state.translator) == ("fasttext", "classifier", "translator")
    assert (state.embedder, state.nlp, state.translator_model_name) == ("embedder", "nlp", "seamless")
    assert seen == [state.config]


def test_default_language_detector_is_lid176(monkeypatch, tmp_path):
    config = execute_server(monkeypatch, tmp_path)["app_state"].config

    assert config["LANGUAGE_DETECTOR"] == "lid176"
    assert config["FASTTEXT_MODEL_PATH"] == "lid.176.bin"
    assert config["FASTTEXT_MODEL_URL"].endswith("lid.176.bin")


def test_glotlid_detector_is_selected_from_the_environment(monkeypatch, tmp_path):
    config = execute_server(
        monkeypatch, tmp_path, LANGUAGE_DETECTOR=" GlotLID ", GLOTLID_MODEL_PATH="/models/glot.bin"
    )["app_state"].config

    assert config["LANGUAGE_DETECTOR"] == "glotlid"
    assert config["FASTTEXT_MODEL_PATH"] == "/models/glot.bin"
    assert "glotlid" in config["FASTTEXT_MODEL_URL"]


def test_an_unknown_language_detector_stops_startup(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Unknown LANGUAGE_DETECTOR"):
        execute_server(monkeypatch, tmp_path, LANGUAGE_DETECTOR="nope")


def test_running_the_module_starts_uvicorn_on_the_configured_port(monkeypatch, tmp_path):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))

    namespace = execute_server(monkeypatch, tmp_path, run_name="__main__", HTTP_PORT="8123")

    app, kwargs = calls[0]
    assert app is namespace["app"]
    assert kwargs == {"host": "0.0.0.0", "port": 8123, "access_log": False}


# --- lifespan and processing loop -------------------------------------------

@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_the_processing_loop(monkeypatch):
    order = []
    monkeypatch.setattr(server, "start_processing_loop", AsyncMock(side_effect=lambda: order.append("start")))
    monkeypatch.setattr(server, "stop_processing_loop", AsyncMock(side_effect=lambda: order.append("stop")))

    async with server.lifespan(server.app):
        order.append("serving")

    assert order == ["start", "serving", "stop"]


@pytest.mark.asyncio
async def test_lifespan_stops_the_loop_even_when_serving_fails(monkeypatch):
    stop = AsyncMock()
    monkeypatch.setattr(server, "start_processing_loop", AsyncMock())
    monkeypatch.setattr(server, "stop_processing_loop", stop)

    with pytest.raises(RuntimeError, match="crash"):
        async with server.lifespan(server.app):
            raise RuntimeError("crash")

    stop.assert_awaited_once()


class FakeWorker:
    def __init__(self):
        self.started = 0

    async def run_forever(self):
        self.started += 1
        await asyncio.Event().wait()


def install_loop_state(monkeypatch, recovered=0, grpc_server=None, task=None):
    state = server.app_state
    queue = SimpleNamespace(recover_inflight=AsyncMock(return_value=recovered))
    worker = FakeWorker()
    monkeypatch.setattr(state, "operation_queue", queue)
    monkeypatch.setattr(state, "operation_worker", worker)
    monkeypatch.setattr(state, "grpc_server", grpc_server)
    monkeypatch.setattr(state, "operation_worker_task", task)
    return queue, worker


@pytest.mark.asyncio
async def test_start_recovers_inflight_work_then_starts_grpc_and_the_worker(monkeypatch, caplog):
    queue, worker = install_loop_state(monkeypatch, recovered=3)
    started_grpc = object()
    start_grpc = AsyncMock(return_value=started_grpc)
    monkeypatch.setattr(server, "start_grpc_server", start_grpc)

    with caplog.at_level("WARNING", logger="mistify"):
        await server.start_processing_loop()
    try:
        await asyncio.sleep(0)
        assert "Recovered 3 interrupted" in caplog.text
        start_grpc.assert_awaited_once_with(queue, server.app_state)
        assert server.app_state.grpc_server is started_grpc
        assert worker.started == 1
        assert not server.app_state.operation_worker_task.done()
    finally:
        server.app_state.operation_worker_task.cancel()


@pytest.mark.asyncio
async def test_start_without_recovered_work_logs_nothing_and_keeps_a_running_setup(monkeypatch, caplog):
    running = asyncio.create_task(asyncio.Event().wait())
    existing_grpc = object()
    install_loop_state(monkeypatch, recovered=0, grpc_server=existing_grpc, task=running)
    start_grpc = AsyncMock()
    monkeypatch.setattr(server, "start_grpc_server", start_grpc)
    try:
        with caplog.at_level("WARNING", logger="mistify"):
            await server.start_processing_loop()

        assert "Recovered" not in caplog.text
        start_grpc.assert_not_awaited()
        assert server.app_state.grpc_server is existing_grpc
        assert server.app_state.operation_worker_task is running
    finally:
        running.cancel()


@pytest.mark.asyncio
async def test_start_replaces_a_finished_worker_task(monkeypatch):
    finished = asyncio.create_task(asyncio.sleep(0))
    await finished
    _queue, worker = install_loop_state(monkeypatch, grpc_server=object(), task=finished)
    try:
        await server.start_processing_loop()
        await asyncio.sleep(0)

        assert server.app_state.operation_worker_task is not finished
        assert worker.started == 1
    finally:
        server.app_state.operation_worker_task.cancel()


class FakePool:
    def __init__(self):
        self.calls = []

    def shutdown(self, **kwargs):
        self.calls.append(kwargs)


def install_stop_state(monkeypatch, grpc_server=None, task=None):
    state = server.app_state
    pools = {name: FakePool() for name in (
        "thread_pool", "classification_pool", "translation_pool", "embedding_pool", "clustering_pool",
    )}
    redis = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(state, "grpc_server", grpc_server)
    monkeypatch.setattr(state, "operation_worker_task", task)
    monkeypatch.setattr(state, "redis_client", redis)
    for name, pool in pools.items():
        monkeypatch.setattr(state, name, pool)
    return redis, pools


@pytest.mark.asyncio
async def test_stop_drains_grpc_cancels_the_worker_and_closes_resources(monkeypatch):
    grpc_server = SimpleNamespace(stop=AsyncMock())
    task = asyncio.create_task(asyncio.Event().wait())
    redis, pools = install_stop_state(monkeypatch, grpc_server=grpc_server, task=task)

    await server.stop_processing_loop()

    grpc_server.stop.assert_awaited_once_with(grace=5)
    assert task.cancelled()
    redis.aclose.assert_awaited_once()
    for pool in pools.values():
        assert pool.calls == [{"wait": False, "cancel_futures": True}]


@pytest.mark.asyncio
async def test_stop_tolerates_no_grpc_server_and_no_running_worker(monkeypatch):
    done = asyncio.create_task(asyncio.sleep(0))
    await done
    redis, pools = install_stop_state(monkeypatch, grpc_server=None, task=done)

    await server.stop_processing_loop()

    redis.aclose.assert_awaited_once()
    assert all(len(pool.calls) == 1 for pool in pools.values())


@pytest.mark.asyncio
async def test_stop_works_when_the_worker_never_started(monkeypatch):
    redis, _pools = install_stop_state(monkeypatch)

    await server.stop_processing_loop()

    redis.aclose.assert_awaited_once()


# --- middleware -------------------------------------------------------------

def counter_value(counter, **labels):
    return counter.labels(**labels)._value.get()


class ScopedRequest:
    def __init__(self, path="/probe", method="GET"):
        self.method = method
        self.scope = {"type": "http", "path": path}
        self.state = SimpleNamespace()


@pytest.mark.asyncio
async def test_a_handled_http_exception_is_counted_with_its_status_and_reraised():
    labels = dict(method="GET", route="unmatched", status_code="418")
    before = counter_value(metrics.HTTP_REQUESTS_TOTAL, **labels)

    async def call_next(_request):
        raise HTTPException(status_code=418, detail="teapot")

    with pytest.raises(HTTPException) as caught:
        await server.prometheus_http_metrics(ScopedRequest(), call_next)

    assert caught.value.status_code == 418
    assert counter_value(metrics.HTTP_REQUESTS_TOTAL, **labels) == before + 1


@pytest.mark.asyncio
async def test_an_unexpected_exception_is_counted_as_a_500_and_reraised():
    request_labels = dict(method="POST", route="unmatched", status_code="500")
    exception_labels = dict(method="POST", route="unmatched", failure_type="unexpected")
    requests_before = counter_value(metrics.HTTP_REQUESTS_TOTAL, **request_labels)
    exceptions_before = counter_value(metrics.HTTP_EXCEPTIONS_TOTAL, **exception_labels)
    inflight_before = metrics.HTTP_INPROGRESS._value.get()

    async def call_next(_request):
        raise RuntimeError("handler bug")

    with pytest.raises(RuntimeError, match="handler bug"):
        await server.prometheus_http_metrics(ScopedRequest(method="POST"), call_next)

    assert counter_value(metrics.HTTP_REQUESTS_TOTAL, **request_labels) == requests_before + 1
    assert counter_value(metrics.HTTP_EXCEPTIONS_TOTAL, **exception_labels) == exceptions_before + 1
    assert metrics.HTTP_INPROGRESS._value.get() == inflight_before


@pytest.mark.asyncio
async def test_a_successful_response_is_counted_with_its_status():
    labels = dict(method="GET", route="/health", status_code="204")
    before = counter_value(metrics.HTTP_REQUESTS_TOTAL, **labels)

    async def call_next(_request):
        return SimpleNamespace(status_code=204)

    response = await server.prometheus_http_metrics(ScopedRequest(path="/health"), call_next)

    assert response.status_code == 204
    assert counter_value(metrics.HTTP_REQUESTS_TOTAL, **labels) == before + 1


@pytest.mark.asyncio
async def test_the_state_middleware_attaches_app_state_to_the_request():
    request = ScopedRequest()

    async def call_next(req):
        assert req.state.app_state is server.app_state
        return "response"

    assert await server.add_state_to_request(request, call_next) == "response"


# --- system routes ----------------------------------------------------------

def client():
    transport = httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_reports_model_availability_and_system_facts(monkeypatch):
    monkeypatch.setattr(server.app_state, "fasttext_model", object())
    monkeypatch.setattr(server.app_state, "classifier", None)
    monkeypatch.setattr(server.app_state, "translator", object())
    monkeypatch.setattr(server.app_state, "embedder", None)
    monkeypatch.setattr(server.app_state, "nlp", object())

    async with client() as http:
        response = await http.get("/health")

    body = response.json()
    assert response.status_code == 200 and body["status"] == "healthy"
    assert body["models"] == {
        "fasttext_loaded": True,
        "classifier_loaded": False,
        "translator_loaded": True,
        "embedder_loaded": False,
        "nlp_loaded": True,
        "media_processor_available": server.app_state.media_operation_runner.available,
    }
    assert body["system"]["device"] in {"cuda", "mps", "cpu"}
    assert body["system"]["threads"] >= 1 and body["system"]["memory_mb"] > 0


@pytest.mark.asyncio
async def test_root_describes_the_service_and_its_endpoints():
    async with client() as http:
        body = (await http.get("/")).json()

    assert body["version"] == server.VERSION
    assert body["operation_queue"] == {"queue_name": server.app_state.operation_queue.queue_name}
    assert body["endpoints"]["language_detection"] == "/detect"
    assert body["endpoints"]["text_clustering"] == "/cluster"
    assert "durable_media_operations" in body["capabilities"]


@pytest.mark.asyncio
async def test_metrics_requires_the_configured_bearer_token(monkeypatch):
    monkeypatch.setitem(server.app_state.config, "METRICS_BEARER_TOKEN", "right-token")
    monkeypatch.setattr(server.app_state, "operation_queue", SimpleNamespace(size=AsyncMock(return_value=4)))
    monkeypatch.setattr(metrics, "update_runtime_metrics", lambda state: None)

    async with client() as http:
        anonymous = await http.get("/metrics")
        wrong = await http.get("/metrics", headers={"Authorization": "Bearer wrong"})
        right = await http.get("/metrics", headers={"Authorization": "Bearer right-token"})

    assert anonymous.status_code == wrong.status_code == 401
    assert anonymous.headers["www-authenticate"] == "Bearer"
    assert right.status_code == 200
    assert right.headers["content-type"].startswith("text/plain")
    assert "mistify_" in right.text
    assert "right-token" not in right.text


@pytest.mark.asyncio
async def test_metrics_scrape_records_the_queue_depth(monkeypatch):
    depths = []
    monkeypatch.setitem(server.app_state.config, "METRICS_BEARER_TOKEN", "right-token")
    monkeypatch.setattr(server.app_state, "operation_queue", SimpleNamespace(size=AsyncMock(return_value=7)))
    monkeypatch.setattr(metrics, "update_runtime_metrics", lambda state: None)
    monkeypatch.setattr(metrics, "record_queue_depth", depths.append)

    async with client() as http:
        await http.get("/metrics", headers={"Authorization": "Bearer right-token"})

    assert depths == [7]
