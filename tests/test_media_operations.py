import hashlib
import os
import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from prometheus_client import generate_latest

os.environ["LOAD_MODELS_ON_STARTUP"] = "false"

from src.operations.media import (  # noqa: E402
    MediaOperationRunner,
    MediaOperationStore,
    MediaProcessingError,
)
from src.operations.media_models import (  # noqa: E402
    DiarizationArtifact,
    DiarizedSegment,
    MediaOperationOptions,
    MediaOperationRecord,
    MediaOperationState,
    MediaProbeResult,
    MediaSource,
    MediaStageName,
    MediaStageState,
    MediaType,
    ModelIdentity,
    SpeakerState,
    StageProvenance,
    SummaryArtifact,
    SummaryPoint,
    TimeRange,
    TranscriptSegment,
    TranscriptionArtifact,
)
from src.operations.media_probe import LocalMediaProbe, MediaProbeError  # noqa: E402
from src.operations.models import OperationEnvelope, QueuedOperation  # noqa: E402
from src.operations.worker import OperationWorker  # noqa: E402
from src import metrics  # noqa: E402
from src.server import app, app_state  # noqa: E402


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    async def set(self, key, value, *, ex):
        self.values[key] = value
        self.ttls[key] = ex
        return True

    async def get(self, key):
        return self.values.get(key)


class FakeQueue:
    def __init__(self):
        self.envelopes = []
        self.enqueue = AsyncMock(side_effect=self._enqueue)
        self.acknowledge = AsyncMock()
        self.requeue_later = AsyncMock()

    async def _enqueue(self, envelope):
        self.envelopes.append(envelope)
        return True


class FixtureMediaBackend:
    available = True

    def __init__(self, *, transient_summary_failures=0, ungrounded=False):
        self.transient_summary_failures = transient_summary_failures
        self.ungrounded = ungrounded
        self.calls = {
            MediaStageName.TRANSCRIPTION: 0,
            MediaStageName.DIARIZATION: 0,
            MediaStageName.SUMMARY: 0,
        }

    def identity(self, stage):
        return ModelIdentity(
            provider="fixture",
            model=f"fixture-{stage.value}",
            version="2026-09-06",
        )

    async def transcribe(self, source, probe):
        self.calls[MediaStageName.TRANSCRIPTION] += 1
        midpoint = probe.duration_ms // 2
        return TranscriptionArtifact(
            language=source.language_hint or "en",
            language_confidence=0.96,
            segments=[
                TranscriptSegment(start_ms=0, end_ms=midpoint, text="Opening line."),
                TranscriptSegment(
                    start_ms=midpoint,
                    end_ms=probe.duration_ms,
                    text="Confirmed fact.",
                ),
            ],
        )

    async def diarize(self, source, probe, transcription):
        self.calls[MediaStageName.DIARIZATION] += 1
        first, second = transcription.segments
        return DiarizationArtifact(
            segments=[
                DiarizedSegment(
                    **first.model_dump(),
                    speaker_state=SpeakerState.UNKNOWN,
                ),
                DiarizedSegment(
                    **second.model_dump(),
                    speaker_state=SpeakerState.IDENTIFIED,
                    speaker_label="speaker-1",
                    speaker_confidence=0.82,
                ),
            ]
        )

    async def summarize(self, source, probe, diarization):
        self.calls[MediaStageName.SUMMARY] += 1
        if self.calls[MediaStageName.SUMMARY] <= self.transient_summary_failures:
            raise MediaProcessingError("summary_provider_busy", retryable=True)
        segment = diarization.segments[-1]
        citation = (
            TimeRange(start_ms=1, end_ms=2)
            if self.ungrounded
            else TimeRange(start_ms=segment.start_ms, end_ms=segment.end_ms)
        )
        return SummaryArtifact(
            points=[
                SummaryPoint(
                    text="A confirmed fact was recorded.",
                    source_timecodes=[citation],
                )
            ]
        )


def write_wav(path: Path, *, seconds=1) -> str:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8_000)
        handle.writeframes(b"\x00\x00" * 8_000 * seconds)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_mp3(path: Path) -> str:
    path.write_bytes(b"\xff\xfb\x90\x64" + b"\x00" * 15_996)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_mp4(path: Path) -> str:
    def box(kind, payload):
        return struct.pack(">I4s", len(payload) + 8, kind) + payload

    mvhd = b"\x00\x00\x00\x00" + struct.pack(">IIII", 0, 0, 1_000, 1_250)
    path.write_bytes(
        box(b"ftyp", b"isom\x00\x00\x02\x00isom")
        + box(b"moov", box(b"mvhd", mvhd))
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: Path, checksum: str, media_type: MediaType) -> MediaSource:
    return MediaSource(
        uri=path.as_uri(),
        checksum_sha256=checksum,
        media_type=media_type,
        language_hint="en",
    )


def operation_record(
    media_source: MediaSource,
    probe: MediaProbeResult,
    *,
    operation_id="op_fixture",
    max_attempts=3,
    retention_seconds=300,
) -> MediaOperationRecord:
    record = MediaOperationRecord(
        operation_id=operation_id,
        source=media_source,
        probe=probe,
        options=MediaOperationOptions(
            max_attempts=max_attempts,
            retention_seconds=retention_seconds,
        ),
    )
    stage = record.stages[MediaStageName.PROBE]
    stage.state = MediaStageState.SUCCEEDED
    stage.attempts = 1
    stage.output = probe.model_dump(mode="json")
    stage.provenance = StageProvenance(
        provider="mistify",
        model="local-media-probe",
        version="1",
        input_checksum_sha256=media_source.checksum_sha256,
    )
    return record


@pytest.mark.asyncio
async def test_probe_accepts_wav_mp3_and_mp4_and_rejects_invalid_inputs(tmp_path):
    wav = tmp_path / "fixture.wav"
    mp3 = tmp_path / "fixture.mp3"
    mp4 = tmp_path / "fixture.mp4"
    checksums = {
        MediaType.WAV: (wav, write_wav(wav)),
        MediaType.MP3: (mp3, write_mp3(mp3)),
        MediaType.MP4: (mp4, write_mp4(mp4)),
    }
    probe = LocalMediaProbe(tmp_path, max_bytes=1_000_000, max_duration_seconds=10)

    results = [
        await probe.probe(source(path, checksum, media_type))
        for media_type, (path, checksum) in checksums.items()
    ]
    assert {result.detected_media_type for result in results} == set(MediaType)
    assert all(result.duration_ms > 0 for result in results)

    with pytest.raises(MediaProbeError, match="checksum_mismatch"):
        await probe.probe(source(wav, "0" * 64, MediaType.WAV))
    with pytest.raises(MediaProbeError, match="media_too_large"):
        await LocalMediaProbe(
            tmp_path, max_bytes=10, max_duration_seconds=10
        ).probe(source(wav, checksums[MediaType.WAV][1], MediaType.WAV))

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not media")
    with pytest.raises(MediaProbeError, match="corrupt_media"):
        await probe.probe(
            source(
                corrupt,
                hashlib.sha256(corrupt.read_bytes()).hexdigest(),
                MediaType.MP4,
            )
        )

    outside = tmp_path.parent / "outside.wav"
    outside_checksum = write_wav(outside)
    try:
        with pytest.raises(MediaProbeError, match="source_outside_media_root"):
            await probe.probe(source(outside, outside_checksum, MediaType.WAV))
    finally:
        outside.unlink()


@pytest.mark.asyncio
async def test_authenticated_submission_runs_end_to_end_with_provenance(
    monkeypatch, tmp_path
):
    wav = tmp_path / "fixture.wav"
    checksum = write_wav(wav)
    redis = FakeRedis()
    store = MediaOperationStore(redis)
    backend = FixtureMediaBackend()
    runner = MediaOperationRunner(store, backend)
    queue = FakeQueue()
    monkeypatch.setitem(
        app_state.config, "MEDIA_OPERATION_BEARER_TOKEN", "media-secret"
    )
    monkeypatch.setattr(
        app_state, "media_probe", LocalMediaProbe(tmp_path, 1_000_000, 10)
    )
    monkeypatch.setattr(app_state, "media_operation_store", store)
    monkeypatch.setattr(app_state, "media_operation_runner", runner)
    monkeypatch.setattr(app_state, "operation_queue", queue)
    request_payload = {
        "source": {
            "uri": wav.as_uri(),
            "checksum_sha256": checksum,
            "media_type": "audio/wav",
            "language_hint": "en",
        },
        "options": {"max_attempts": 3, "retention_seconds": 300},
    }
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized = await client.post("/media/operations", json=request_payload)
        assert unauthorized.status_code == 401
        submitted = await client.post(
            "/media/operations",
            json=request_payload,
            headers={"Authorization": "Bearer media-secret"},
        )
        assert submitted.status_code == 202
        operation_id = submitted.json()["operation_id"]

        worker = OperationWorker(
            queue, SimpleNamespace(media_operation_runner=runner)
        )
        await worker.process(
            QueuedOperation(envelope=queue.envelopes[0], receipts=["receipt"])
        )
        completed = await client.get(
            f"/media/operations/{operation_id}",
            headers={"Authorization": "Bearer media-secret"},
        )

    assert completed.status_code == 200
    body = completed.json()
    assert body["state"] == "succeeded"
    assert body["result"]["source_checksum_sha256"] == checksum
    assert body["result"]["language"] == "en"
    assert body["result"]["transcript"][0]["speaker_state"] == "unknown"
    assert body["result"]["transcript"][0]["speaker_label"] is None
    assert body["result"]["summary"][0]["source_timecodes"] == [
        {"start_ms": 500, "end_ms": 1000}
    ]
    assert set(body["result"]["provenance"]) == {
        "probe",
        "transcription",
        "diarization",
        "summary",
    }
    assert all(
        value["input_checksum_sha256"] == checksum
        for value in body["result"]["provenance"].values()
    )
    queue.acknowledge.assert_awaited_once()
    assert redis.ttls[store.key(operation_id)] == 300

    metrics.update_runtime_metrics(
        SimpleNamespace(media_operation_runner=runner)
    )
    exposition = generate_latest().decode()
    assert 'operation="process_media"' in exposition
    assert 'phase="probe"' in exposition
    assert 'phase="transcription"' in exposition
    assert 'outcome="succeeded"' in exposition
    assert 'mistify_model_available{model="media_processor"} 1.0' in exposition
    assert wav.as_uri() not in exposition
    assert checksum not in exposition
    assert "Confirmed fact" not in exposition


@pytest.mark.asyncio
async def test_corrupt_media_fails_before_queue_or_processor(monkeypatch, tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not media")
    checksum = hashlib.sha256(corrupt.read_bytes()).hexdigest()
    store = MediaOperationStore(FakeRedis())
    backend = FixtureMediaBackend()
    queue = FakeQueue()
    monkeypatch.setitem(
        app_state.config, "MEDIA_OPERATION_BEARER_TOKEN", "media-secret"
    )
    monkeypatch.setattr(
        app_state, "media_probe", LocalMediaProbe(tmp_path, 1_000_000, 10)
    )
    monkeypatch.setattr(app_state, "media_operation_store", store)
    monkeypatch.setattr(
        app_state, "media_operation_runner", MediaOperationRunner(store, backend)
    )
    monkeypatch.setattr(app_state, "operation_queue", queue)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/media/operations",
            headers={"Authorization": "Bearer media-secret"},
            json={
                "source": {
                    "uri": corrupt.as_uri(),
                    "checksum_sha256": checksum,
                    "media_type": "video/mp4",
                }
            },
        )
        unsupported = await client.post(
            "/media/operations",
            headers={"Authorization": "Bearer media-secret"},
            json={
                "source": {
                    "uri": corrupt.as_uri(),
                    "checksum_sha256": checksum,
                    "media_type": "application/octet-stream",
                }
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "corrupt_media"
    assert unsupported.status_code == 422
    queue.enqueue.assert_not_awaited()
    assert not any(backend.calls.values())
    exposition = generate_latest().decode()
    assert 'mistify_media_operation_phase_total{outcome="error",phase="probe"}' in exposition
    assert corrupt.as_uri() not in exposition
    assert checksum not in exposition


@pytest.mark.asyncio
async def test_restart_retries_only_failed_stage_and_reuses_terminal_result(
    tmp_path,
):
    wav = tmp_path / "fixture.wav"
    checksum = write_wav(wav)
    media_source = source(wav, checksum, MediaType.WAV)
    probe = await LocalMediaProbe(tmp_path, 1_000_000, 10).probe(media_source)
    redis = FakeRedis()
    store = MediaOperationStore(redis)
    backend = FixtureMediaBackend(transient_summary_failures=1)
    await store.save(operation_record(media_source, probe))
    queued = QueuedOperation(
        envelope=OperationEnvelope(
            operation_id="op_fixture",
            operation_type="process_media",
            payload={"source_checksum_sha256": checksum},
        ),
        receipts=["durable-receipt"],
    )
    first_queue = FakeQueue()

    await OperationWorker(
        first_queue,
        SimpleNamespace(media_operation_runner=MediaOperationRunner(store, backend)),
    ).process(queued)
    first = await store.get("op_fixture")
    assert first is not None
    assert first.state == MediaOperationState.QUEUED
    first_queue.requeue_later.assert_awaited_once_with(queued)
    first_queue.acknowledge.assert_not_awaited()
    assert backend.calls == {
        MediaStageName.TRANSCRIPTION: 1,
        MediaStageName.DIARIZATION: 1,
        MediaStageName.SUMMARY: 1,
    }

    restarted_runner = MediaOperationRunner(store, backend)
    second_queue = FakeQueue()
    await OperationWorker(
        second_queue,
        SimpleNamespace(media_operation_runner=restarted_runner),
    ).process(queued)
    second = await store.get("op_fixture")
    assert second is not None
    assert second.state == MediaOperationState.SUCCEEDED
    second_queue.acknowledge.assert_awaited_once_with(queued)
    second_queue.requeue_later.assert_not_awaited()
    assert backend.calls == {
        MediaStageName.TRANSCRIPTION: 1,
        MediaStageName.DIARIZATION: 1,
        MediaStageName.SUMMARY: 2,
    }

    replay = await restarted_runner.run("op_fixture")
    assert replay.result == second.result.model_dump(mode="json")
    assert backend.calls[MediaStageName.SUMMARY] == 2


@pytest.mark.asyncio
async def test_invalid_grounding_fails_terminally_without_retry(tmp_path):
    wav = tmp_path / "fixture.wav"
    checksum = write_wav(wav)
    media_source = source(wav, checksum, MediaType.WAV)
    probe = await LocalMediaProbe(tmp_path, 1_000_000, 10).probe(media_source)
    store = MediaOperationStore(FakeRedis())
    await store.save(operation_record(media_source, probe))

    outcome = await MediaOperationRunner(
        store, FixtureMediaBackend(ungrounded=True)
    ).run("op_fixture")

    assert outcome.state == MediaOperationState.FAILED
    assert outcome.error_code == "ungrounded_summary"
    assert outcome.retry is False


@pytest.mark.asyncio
async def test_cancel_is_persistent_and_worker_skips_processors(monkeypatch, tmp_path):
    wav = tmp_path / "fixture.wav"
    checksum = write_wav(wav)
    media_source = source(wav, checksum, MediaType.WAV)
    probe = await LocalMediaProbe(tmp_path, 1_000_000, 10).probe(media_source)
    store = MediaOperationStore(FakeRedis())
    await store.save(operation_record(media_source, probe, operation_id="op_cancel"))
    backend = FixtureMediaBackend()
    runner = MediaOperationRunner(store, backend)
    monkeypatch.setitem(
        app_state.config, "MEDIA_OPERATION_BEARER_TOKEN", "media-secret"
    )
    monkeypatch.setattr(app_state, "media_operation_store", store)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/media/operations/op_cancel/cancel",
            headers={"Authorization": "Bearer media-secret"},
        )

    assert response.status_code == 200
    assert response.json()["state"] == "canceled"
    outcome = await runner.run("op_cancel")
    assert outcome.state == MediaOperationState.CANCELED
    assert not any(backend.calls.values())


@pytest.mark.asyncio
async def test_media_api_is_hidden_without_auth_config(monkeypatch):
    monkeypatch.setitem(app_state.config, "MEDIA_OPERATION_BEARER_TOKEN", None)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/media/operations/op_missing")
        supplied = await client.get(
            "/media/operations/op_missing",
            headers={"Authorization": "Bearer arbitrary"},
        )

    assert response.status_code == 404
    assert supplied.status_code == 404
    metrics.update_runtime_metrics(app_state)
    assert 'mistify_model_available{model="media_processor"} 0.0' in (
        generate_latest().decode()
    )
