"""Covers the `/media/operations` API in `src/endpoints/media.py`.

The router is mounted on a minimal app with fake collaborators, so these tests
pin the endpoint contract alone: bearer-token gating, admission (processor
availability, probe failures, queueing failures leave a terminal record),
status lookup and idempotent cancellation.
"""

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from src import metrics
from src.endpoints import media
from src.operations.media_models import (
    MediaOperationOptions,
    MediaOperationRecord,
    MediaOperationState,
    MediaProbeResult,
    MediaSource,
    MediaStageName,
    MediaStageState,
    MediaType,
)
from src.operations.media_probe import MediaProbeError

TOKEN = "media-test-token"
CHECKSUM = "b" * 64
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BODY = {
    "source": {"uri": "file:///media/a.wav", "checksum_sha256": CHECKSUM, "media_type": "audio/wav"},
    "options": {"max_attempts": 4},
}


class FakeStore:
    def __init__(self):
        self.records, self.saves = {}, []

    async def save(self, record):
        self.records[record.operation_id] = record.model_copy(deep=True)
        self.saves.append(record.state)

    async def get(self, operation_id):
        found = self.records.get(operation_id)
        return found.model_copy(deep=True) if found else None


class FakeProbe:
    def __init__(self, error=None):
        self.error = error

    async def probe(self, source):
        if self.error:
            raise self.error
        return MediaProbeResult(
            checksum_sha256=source.checksum_sha256, detected_media_type=source.media_type,
            duration_ms=1500, size_bytes=64,
        )


class FakeQueue:
    def __init__(self, result=True, error=None):
        self.result, self.error, self.envelopes = result, error, []

    async def enqueue(self, envelope):
        self.envelopes.append(envelope)
        if self.error:
            raise self.error
        return self.result


def build(token=TOKEN, available=True, probe=None, queue=None):
    state = SimpleNamespace(
        config={"MEDIA_OPERATION_BEARER_TOKEN": token},
        media_operation_runner=SimpleNamespace(available=available),
        media_probe=probe or FakeProbe(),
        media_operation_store=FakeStore(),
        operation_queue=queue or FakeQueue(),
    )
    app = FastAPI()

    @app.middleware("http")
    async def attach(request, call_next):
        request.state.app_state = state
        return await call_next(request)

    app.include_router(media.router)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return state, client


@pytest.fixture
def events(monkeypatch):
    recorded = []
    monkeypatch.setattr(metrics, "record_queue_event", lambda op, outcome: recorded.append((op, outcome)))
    return recorded


# --- authentication ---------------------------------------------------------

@pytest.mark.asyncio
async def test_the_api_is_hidden_when_no_token_is_configured():
    _state, client = build(token=None)
    async with client:
        anonymous = await client.get("/media/operations/op_x")
        supplied = await client.get("/media/operations/op_x", headers=AUTH)

    assert anonymous.status_code == supplied.status_code == 404


@pytest.mark.asyncio
async def test_a_missing_or_wrong_token_is_a_401_with_a_challenge():
    _state, client = build()
    async with client:
        missing = await client.post("/media/operations", json=BODY)
        wrong = await client.get("/media/operations/op_x", headers={"Authorization": "Bearer nope"})

    for response in (missing, wrong):
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"


# --- submission -------------------------------------------------------------

@pytest.mark.asyncio
async def test_submission_stores_a_probed_record_and_enqueues_the_operation(events):
    state, client = build()
    async with client:
        response = await client.post("/media/operations", json=BODY, headers=AUTH)

    assert response.status_code == 202
    body = response.json()
    assert body["state"] == "queued"
    assert body["status_url"] == f"http://test/media/operations/{body['operation_id']}"
    record = state.media_operation_store.records[body["operation_id"]]
    probe_stage = record.stages[MediaStageName.PROBE]
    assert probe_stage.state == MediaStageState.SUCCEEDED and probe_stage.attempts == 1
    assert probe_stage.provenance.model == "local-media-probe"
    assert probe_stage.provenance.input_checksum_sha256 == CHECKSUM
    assert probe_stage.output["duration_ms"] == 1500
    assert record.options.max_attempts == 4 and record.probe.size_bytes == 64
    envelope = state.operation_queue.envelopes[0]
    assert envelope.operation_id == body["operation_id"]
    assert envelope.operation_type == "process_media"
    assert envelope.payload == {"source_checksum_sha256": CHECKSUM}
    assert envelope.options.max_attempts == 4
    assert events == [("process_media", "enqueued")]


@pytest.mark.asyncio
async def test_submission_is_refused_while_no_processor_is_available():
    state, client = build(available=False)
    async with client:
        response = await client.post("/media/operations", json=BODY, headers=AUTH)

    assert response.status_code == 503 and response.json()["detail"] == "Media processor unavailable"
    assert state.media_operation_store.records == {} and state.operation_queue.envelopes == []


@pytest.mark.asyncio
async def test_probe_failures_surface_their_code_and_status_without_queueing():
    state, client = build(probe=FakeProbe(MediaProbeError("media_too_large", status_code=413)))
    async with client:
        response = await client.post("/media/operations", json=BODY, headers=AUTH)

    assert response.status_code == 413 and response.json()["detail"] == "media_too_large"
    assert state.media_operation_store.records == {} and state.operation_queue.envelopes == []


@pytest.mark.asyncio
async def test_an_invalid_submission_is_rejected_before_any_work():
    state, client = build()
    bad = {**BODY, "source": {**BODY["source"], "checksum_sha256": "short"}}
    async with client:
        response = await client.post("/media/operations", json=bad, headers=AUTH)

    assert response.status_code == 422 and state.operation_queue.envelopes == []


@pytest.mark.asyncio
async def test_a_queue_error_leaves_a_failed_record_and_a_503(events):
    state, client = build(queue=FakeQueue(error=RuntimeError("valkey down")))
    async with client:
        response = await client.post("/media/operations", json=BODY, headers=AUTH)

    assert response.status_code == 503 and response.json()["detail"] == "Media queue unavailable"
    (record,) = state.media_operation_store.records.values()
    assert (record.state, record.error_code) == (MediaOperationState.FAILED, "media_queue_unavailable")
    assert events == [("process_media", "error")]
    assert "valkey" not in response.text


@pytest.mark.asyncio
async def test_a_rejected_enqueue_leaves_a_failed_record_and_a_503(events):
    state, client = build(queue=FakeQueue(result=False))
    async with client:
        response = await client.post("/media/operations", json=BODY, headers=AUTH)

    assert response.status_code == 503
    (record,) = state.media_operation_store.records.values()
    assert (record.state, record.error_code) == (MediaOperationState.FAILED, "media_queue_rejected")
    assert events == []


# --- status and cancel ------------------------------------------------------

def stored_record(state_value=MediaOperationState.QUEUED, running_stage=None):
    rec = MediaOperationRecord(
        operation_id="op_1",
        source=MediaSource(uri="file:///media/a.wav", checksum_sha256=CHECKSUM, media_type=MediaType.WAV),
        options=MediaOperationOptions(),
        probe=MediaProbeResult(
            checksum_sha256=CHECKSUM, detected_media_type=MediaType.WAV, duration_ms=10, size_bytes=1
        ),
        state=state_value,
    )
    if running_stage:
        rec.stages[running_stage].state = MediaStageState.RUNNING
    return rec


@pytest.mark.asyncio
async def test_status_returns_the_stored_record_or_404():
    state, client = build()
    await state.media_operation_store.save(stored_record())
    async with client:
        found = await client.get("/media/operations/op_1", headers=AUTH)
        missing = await client.get("/media/operations/op_2", headers=AUTH)

    assert found.status_code == 200 and found.json()["operation_id"] == "op_1"
    assert missing.status_code == 404 and missing.json()["detail"] == "Media operation not found"


@pytest.mark.asyncio
async def test_cancel_marks_the_operation_and_its_running_stage_canceled(monkeypatch):
    outcomes = []
    monkeypatch.setattr(metrics, "record_media_outcome", outcomes.append)
    state, client = build()
    await state.media_operation_store.save(stored_record(running_stage=MediaStageName.DIARIZATION))
    async with client:
        response = await client.post("/media/operations/op_1/cancel", headers=AUTH)

    body = response.json()
    assert response.status_code == 200 and body["state"] == "canceled"
    assert body["stages"]["diarization"]["state"] == "canceled"
    assert body["stages"]["diarization"]["completed_at"] is not None
    assert body["stages"]["transcription"]["state"] == "pending"
    assert (await state.media_operation_store.get("op_1")).state == MediaOperationState.CANCELED
    assert outcomes == ["canceled"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal", [MediaOperationState.SUCCEEDED, MediaOperationState.FAILED, MediaOperationState.CANCELED]
)
async def test_cancelling_a_finished_operation_changes_nothing(terminal, monkeypatch):
    outcomes = []
    monkeypatch.setattr(metrics, "record_media_outcome", outcomes.append)
    state, client = build()
    await state.media_operation_store.save(stored_record(terminal))
    saves_before = len(state.media_operation_store.saves)
    async with client:
        response = await client.post("/media/operations/op_1/cancel", headers=AUTH)

    assert response.status_code == 200 and response.json()["state"] == terminal.value
    assert len(state.media_operation_store.saves) == saves_before and outcomes == []


@pytest.mark.asyncio
async def test_cancelling_an_unknown_operation_is_a_404():
    _state, client = build()
    async with client:
        response = await client.post("/media/operations/op_missing/cancel", headers=AUTH)

    assert response.status_code == 404
